"""Causal overlapping training windows, disjoint cached inference; R=2."""
import copy
import torch
from torch.nn import functional as F

from experiments.smollm.smollm_model import MergedSmolLM


class SlidingSmolLM(MergedSmolLM):
    def __init__(self, base, variant='plain', merge_layer=4, ratio=2):
        if variant not in ('baseline', 'plain', 'residual'):
            raise ValueError('Unknown variant.')
        super().__init__(base, variant, merge_layer, ratio)

    def sliding_merge(self, hidden):
        if self.ratio == 1 or hidden.shape[1] < 2:
            return hidden
        weights = self.merge_weight.softmax(0).to(hidden.dtype)
        paired = weights[0] * hidden[:, :-1] + weights[1] * hidden[:, 1:]
        if self.variant == 'residual':
            paired = paired + hidden[:, :-1] + hidden[:, 1:]
        return torch.cat((hidden[:, :1], paired), dim=1)

    def _sliding_full(self, ids):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('Expected nonempty unpadded [batch, time] IDs.')
        hidden = self.base.model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)
        rotary = self.base.model.rotary_emb(hidden, positions[None, :])
        for i, layer in enumerate(self.base.model.layers):
            if i == self.merge_layer:
                hidden = self.sliding_merge(hidden)
            hidden = self._layer(layer, hidden, positions, rotary)
        return self.base.model.norm(hidden), positions, None

    @torch.no_grad()
    def prefill(self, ids):
        logits, state = super().prefill(ids)
        if self.ratio > 1 and state.pending is not None:
            # The singleton was needed for this prediction, not future deep KV.
            for layer in state.cache.layers[self.merge_layer:]:
                layer.crop(ids.shape[1] // self.ratio)
        return logits, state

    @torch.no_grad()
    def decode(self, token, state):
        if self.training or token.ndim != 2 or token.shape[1] != 1:
            raise ValueError('decode requires eval mode and one token per row.')
        hidden = self.base.model.embed_tokens(token)
        positions = torch.tensor([state.length], device=token.device)
        rotary = self.base.model.rotary_emb(hidden, positions[None, :])
        pending = None
        active_cache = state.cache
        for i, layer in enumerate(self.base.model.layers):
            if self.ratio > 1 and i == self.merge_layer:
                if state.pending is not None:
                    hidden = self.merge(torch.cat((state.pending, hidden), dim=1))
                else:
                    pending = hidden.detach().clone()
                    # Updating copied deep cache records keeps singleton KV
                    # local to this prediction; early KV remains persistent.
                    active_cache = copy.copy(state.cache)
                    active_cache.layers = [copy.copy(record) for record in state.cache.layers]
            hidden = self._layer(layer, hidden, positions, rotary, active_cache)
        state.length += 1
        state.pending = pending
        return self.base.lm_head(self.base.model.norm(hidden)), state

    def forward(self, ids, targets=None, last_only=False, representation_mode=None):
        mode = representation_mode or ('sliding' if self.training else 'disjoint')
        if mode not in ('sliding', 'disjoint'):
            raise ValueError('representation_mode must be sliding or disjoint.')
        if last_only and targets is not None:
            raise ValueError('last_only cannot be used with targets.')
        # Cached methods inherited from MergedSmolLM always use its disjoint _full.
        hidden, positions, _ = (self._sliding_full(ids) if mode == 'sliding'
                                else self._full(ids))
        if last_only:
            hidden = hidden[:, -1:]
        logits = self.base.lm_head(hidden)
        loss = None
        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError('Targets must be next-token shifted and match inputs.')
            selected = targets.index_select(1, positions)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   selected.reshape(-1))
        return logits, loss
