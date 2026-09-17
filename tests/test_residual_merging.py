"""Causality, target alignment, singleton semantics, and generation regressions."""
import unittest

import torch
from torch.nn import functional as F

from nanoGPT.model import GPT, GPTConfig, WindowTokenMerger, boundary_target_indices


class ResidualMergingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, ratio=2, **kwargs):
        torch.manual_seed(42)
        return GPT(GPTConfig(block_size=16, vocab_size=19, n_layer=3, n_head=2,
                             n_embd=16, dropout=0, merge_ratio=ratio, merge_layer=1,
                             **kwargs)).eval()

    def test_boundary_indices_include_every_tail(self):
        for length, ratio, expected in ((8, 2, [1, 3, 5, 7]), (7, 2, [1, 3, 5, 6]),
                                         (8, 3, [2, 5, 6, 7]), (2, 3, [0, 1]),
                                         (4, 1, [0, 1, 2, 3])):
            self.assertEqual(boundary_target_indices(length, ratio).tolist(), expected)

    def test_formula_and_singleton_tail(self):
        x = torch.randn(2, 5, 4)
        merger = WindowTokenMerger(4, 2)
        with torch.no_grad():
            merger.merge_weight.copy_(torch.tensor([-0.4, 0.7]))
        weights = merger.merge_weight.softmax(0).reshape(1, 1, 2, 1)
        groups = x[:, :4].reshape(2, 2, 2, 4)
        expected = (groups * weights).sum(2) + groups.sum(2)
        out = merger(x)
        torch.testing.assert_close(out[:, :2], expected)
        torch.testing.assert_close(out[:, -1], 2 * x[:, -1])
        torch.testing.assert_close(merger(x, merge_tokens=False), 2 * x)
        torch.testing.assert_close(merger(x[:, :1]), 2 * x[:, :1])
        torch.testing.assert_close(merger(x, residual_tail=False)[:, -1], x[:, -1])
        torch.testing.assert_close(merger(x, use_residual_cache=False)[:, :2], (groups * weights).sum(2))

    def test_loss_matches_explicit_shifted_targets(self):
        x = torch.randint(19, (2, 8))
        y = torch.randint(19, (2, 8))
        y[0, 2] = -1
        for ratio in (1, 2, 3):
            for merge_tokens in (True, False):
                model = self.model(ratio)
                logits, loss = model(x, y, merge_tokens=merge_tokens)
                indices = boundary_target_indices(8, ratio if merge_tokens else 1)
                expected = F.cross_entropy(logits.reshape(-1, 19), y[:, indices].reshape(-1), ignore_index=-1)
                torch.testing.assert_close(loss, expected)
                loss.backward()
                if ratio > 1 and merge_tokens:
                    self.assertIsNotNone(model.merger.merge_weight.grad)
                    self.assertTrue(torch.isfinite(model.merger.merge_weight.grad).all())

    def test_future_tokens_cannot_change_valid_predictions(self):
        # Compare each full-forward prediction to the exact prefix it represents.
        x = torch.randint(19, (2, 11))
        y = torch.randint(19, (2, 11))
        for ratio in (1, 2, 3):
            for cache in (False, True):
                model = self.model(ratio, use_residual_cache=cache)
                for merge_tokens in (True, False):
                    with torch.no_grad():
                        logits, _ = model(x, y, merge_tokens=merge_tokens)
                        endpoints = boundary_target_indices(11, ratio if merge_tokens else 1)
                        for position, endpoint in enumerate(endpoints.tolist()):
                            prefix_logits, _ = model(x[:, :endpoint + 1], merge_tokens=merge_tokens)
                            torch.testing.assert_close(logits[:, position], prefix_logits[:, -1], rtol=1e-5, atol=1e-6)

    def test_future_embeddings_receive_no_gradient_from_boundary(self):
        model = self.model(2)
        # Unique IDs let embedding gradients identify future dependencies.
        x = torch.arange(8).unsqueeze(0)
        saved = []
        def retain_embedding(module, args, output):
            output.retain_grad()
            saved.append(output)
        handle = model.transformer.wte.register_forward_hook(retain_embedding)
        logits, _ = model(x, x)
        logits[:, 1].sum().backward() # window [2,3] can see only positions <=3
        handle.remove()
        torch.testing.assert_close(saved[0].grad[:, 4:], torch.zeros_like(saved[0].grad[:, 4:]), rtol=0, atol=0)

    def test_inference_projects_only_last_position(self):
        model = self.model()
        x = torch.randint(19, (2, 7))
        shapes = []
        handle = model.lm_head.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
        full, _ = model(x, x)
        last, loss = model(x)
        handle.remove()
        self.assertEqual(shapes[-1], torch.Size([2, 1, 16]))
        self.assertIsNone(loss)
        torch.testing.assert_close(last[:, 0], full[:, -1])

    def test_generation_crosses_tail_and_context_crop(self):
        for ratio in (1, 2, 3):
            model = self.model(ratio)
            x = torch.randint(19, (2, 15))
            generated = model.generate(x, 5, temperature=0.8, top_k=5)
            self.assertEqual(generated.shape, (2, 20))
            torch.testing.assert_close(generated[:, :15], x)

    def test_validation_rejects_invalid_inputs(self):
        for ratio in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                GPTConfig(merge_ratio=ratio)
        with self.assertRaises(ValueError):
            GPTConfig(n_layer=2, merge_ratio=2, merge_layer=2)
        model = self.model()
        with self.assertRaises(ValueError):
            model(torch.empty(1, 0, dtype=torch.long))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long))
        with self.assertRaises(ValueError):
            model.generate(torch.zeros(1, 1, dtype=torch.long), 1, temperature=0)


if __name__ == '__main__':
    unittest.main()
