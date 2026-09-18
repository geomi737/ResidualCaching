"""One-way window merging for pretrained Llama/SmolLM2 (Transformers 4.57.6).

Copyright (c) 2026 geomi737. Apache-2.0.
Uses Hugging Face Transformers and SmolLM2; see UPSTREAM.md.
No padding, unmerging, merge bypass, or singleton scaling is supported here.
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
        state = DecodeState(cache, ids.shape[1], pending)
        return self.base.lm_head(hidden[:, -1:]), state

    @torch.no_grad()
    def decode(self, token, state):
        """Append one token, replacing a provisional deep-cache singleton."""
        if self.training or token.ndim != 2 or token.shape[1] != 1:
            raise ValueError('decode requires eval mode and exactly one token per row.')
        hidden = self.base.model.embed_tokens(token)
        positions = torch.tensor([state.length], device=token.device)
        rotary = self.base.model.rotary_emb(hidden, positions[None, :])
        pending = None
        for i, layer in enumerate(self.base.model.layers):
            if self.ratio > 1 and i == self.merge_layer:
                if state.pending is not None:
                    hidden = self.merge(torch.cat((state.pending, hidden), dim=1))
                    # Remove only deep-layer tail records. Early layers retain
                    # the original singleton because they never compressed it.
                    for deep in state.cache.layers[self.merge_layer:]:
                        deep.crop(state.length // self.ratio)
                else:
                    pending = hidden.detach().clone()
            hidden = self._layer(layer, hidden, positions, rotary, state.cache)
        state.length += 1
        state.pending = pending
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
