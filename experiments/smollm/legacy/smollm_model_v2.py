"""Historical full-length unmerge prototype for Llama/SmolLM2.

Copyright (c) 2026 geomi737. Apache-2.0.
Uses Hugging Face Transformers and SmolLM2; see UPSTREAM.md.
Dense unmerge targets leak within-window future tokens. Not causal quality evidence.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def endpoints(length, ratio, device):
    full = length // ratio * ratio
    if full == 0:
        return torch.arange(length, device=device)
    return torch.cat((torch.arange(ratio - 1, full, ratio, device=device),
                      torch.arange(full, length, device=device)))


@dataclass
class DecodeState:
    cache: DynamicCache
    length: int
    pending: torch.Tensor | None


class MergedSmolLM(nn.Module):
    def __init__(self, base, variant='baseline', merge_layer=15, ratio=2):
        super().__init__()
        if variant not in ('baseline', 'plain', 'residual'):
            raise ValueError('Unknown variant')
        if ratio != 2:
            raise ValueError('This first cached implementation supports R=2 only.')
        if not 0 <= merge_layer < len(base.model.layers):
            raise ValueError('merge_layer must precede at least one decoder block.')
        if base.config._attn_implementation != 'sdpa':
            raise ValueError('Use the tested SDPA attention implementation.')
        self.base = base
        self.variant = variant
        self.merge_layer = merge_layer
        self.ratio = 1 if variant == 'baseline' else ratio
        if self.ratio > 1:
            self.merge_weight = nn.Parameter(torch.zeros(ratio))
            self.unmerge = nn.Linear(base.config.hidden_size, base.config.hidden_size * ratio)

    @classmethod
    def load(cls, path, variant='baseline', merge_layer=15):
        base = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32,
            attn_implementation='sdpa')
        return cls(base, variant, merge_layer)

    def merge(self, hidden):
        """Merge complete windows from the left; append the untouched tail."""
        length = hidden.shape[1]
        full = length // self.ratio * self.ratio
        if self.ratio == 1 or full == 0:
            return hidden
        groups = hidden[:, :full].reshape(hidden.shape[0], -1, self.ratio, hidden.shape[-1])
        weights = self.merge_weight.softmax(0).to(hidden.dtype)
        merged = (groups * weights[None, None, :, None]).sum(2)
        if self.variant == 'residual':
            merged = merged + groups.sum(2)
        return torch.cat((merged, hidden[:, full:]), dim=1)

    def apply_unmerge(self, hidden, original_length):
        """Unmerge the compressed part to restore the original length."""
        if self.ratio == 1:
            return hidden
        full = original_length // self.ratio * self.ratio
        if full == 0:
            return hidden

        compressed = hidden[:, :full // self.ratio]
        unmerged = self.unmerge(compressed).view(hidden.shape[0], full, hidden.shape[-1])
        tail = hidden[:, full // self.ratio:]
        return torch.cat((unmerged, tail), dim=1)

    def _layer(self, layer, hidden, positions, rotary, cache=None):
        # Full passes contain only a causal sequence; incremental passes have
        # exactly one query attending to all cached keys. SDPA handles both.
        return layer(hidden, attention_mask=None, position_ids=positions[None, :],
                     position_embeddings=rotary, past_key_values=cache,
                     use_cache=cache is not None, cache_position=positions)

    def _full(self, ids, cache=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('Expected nonempty unpadded [batch, time] token IDs.')
        hidden = self.base.model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)
        rotary = self.base.model.rotary_emb(hidden, positions[None, :])
        pending = None
        for i, layer in enumerate(self.base.model.layers):
            if self.ratio > 1 and i == self.merge_layer:
                if ids.shape[1] % self.ratio:
                    pending = hidden[:, -1:].detach().clone() if cache is not None else None
                hidden = self.merge(hidden)
                positions = endpoints(ids.shape[1], self.ratio, ids.device)
                rotary = self.base.model.rotary_emb(hidden, positions[None, :])
            hidden = self._layer(layer, hidden, positions, rotary, cache)
        return self.base.model.norm(hidden), positions, pending

    def forward(self, ids, targets=None, last_only=False):
        """Targets are already shifted by one token, as in nanoGPT."""
        hidden, positions, _ = self._full(ids)

        # Restore full length via unmerge
        hidden = self.apply_unmerge(hidden, ids.shape[1])
        positions = torch.arange(ids.shape[1], device=ids.device)

        if last_only:
            if targets is not None:
                raise ValueError('last_only cannot be used with training targets.')
            hidden = hidden[:, -1:]
        logits = self.base.lm_head(hidden)
        loss = None
        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError('Shifted targets must have the same shape as inputs.')
            selected = targets.index_select(1, positions)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   selected.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def prefill(self, ids):
        if self.training:
            raise RuntimeError('Cached generation requires eval mode.')
        cache = DynamicCache(config=self.base.config)
        hidden, _, pending = self._full(ids, cache)

        # Apply unmerge
        hidden = self.apply_unmerge(hidden, ids.shape[1])
        state = DecodeState(cache, ids.shape[1], pending)
        return self.base.lm_head(hidden[:, -1:]), state

    @torch.no_grad()
    def decode(self, token, state):
        if self.training or token.ndim != 2 or token.shape[1] != 1:
            raise ValueError('decode requires eval mode and exactly one token per row.')
        hidden = self.base.model.embed_tokens(token)
        positions = torch.tensor([state.length], device=token.device)
        rotary = self.base.model.rotary_emb(hidden, positions[None, :])
        pending = None
        active_cache = state.cache

        for i, layer in enumerate(self.base.model.layers):
            if self.ratio > 1 and i == self.merge_layer:
                if state.pending is not None:
                    # EVEN STEP: Pair complete.
                    hidden = self.merge(torch.cat((state.pending, hidden), dim=1))
                    active_cache = state.cache
                else:
                    # ODD STEP: Singleton tail.
                    pending = hidden.detach().clone()
                    import copy
                    active_cache = copy.copy(state.cache)
                    active_cache.layers = [copy.copy(layer) for layer in state.cache.layers]
            hidden = self._layer(layer, hidden, positions, rotary, active_cache)

        state.length += 1
        state.pending = pending

        # We need to unmerge the output if it's compressed
        if self.ratio > 1:
            if state.pending is None:
                # Even step: we just merged and got 1 token out of the deep layers.
                # It represents 2 original tokens. Unmerge it to get 2 tokens, and return the LAST one.
                hidden = self.unmerge(hidden).view(hidden.shape[0], 2, hidden.shape[-1])
                hidden = hidden[:, -1:]  # We only care about the final prediction
            else:
                # Odd step: the output is the raw uncompressed singleton tail.
                # No unmerge needed for the tail!
                pass

        return self.base.lm_head(self.base.model.norm(hidden)), state

    @torch.no_grad()
    def generate_cached(self, ids, new_tokens, max_context=2048):
        """Greedy generation; rebuild on cropping to match relative positions."""
        logits, state = self.prefill(ids[:, -max_context:])
        output = ids
        for step in range(new_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            output = torch.cat((output, token), dim=1)
            if step + 1 < new_tokens:
                if state.length >= max_context:
                    logits, state = self.prefill(output[:, -max_context:])
                else:
                    logits, state = self.decode(token, state)
        return output
