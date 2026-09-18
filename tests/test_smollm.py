"""Causal, cache, tail, and baseline-equivalence tests for the SmolLM adapter."""
import unittest

import torch
try:
    from transformers import LlamaConfig, LlamaForCausalLM
    from experiments.smollm.smollm_model import MergedSmolLM, endpoints
except ImportError:
    LlamaConfig = None


@unittest.skipIf(LlamaConfig is None, 'Install requirements-smollm.txt')
class SmolLMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def make(self, variant, merge_layer=1):
        config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=64,
                             attention_dropout=0, tie_word_embeddings=True)
        config._attn_implementation = 'sdpa'
        return MergedSmolLM(LlamaForCausalLM(config), variant, merge_layer).eval()

    def test_baseline_matches_huggingface_and_loss(self):
        model = self.make('baseline')
        ids = torch.randint(41, (2, 10))
        logits, loss = model(ids[:, :-1], ids[:, 1:])
        reference = model.base(ids, labels=ids, use_cache=False)
        torch.testing.assert_close(logits, reference.logits[:, :-1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(loss, reference.loss)

    def test_formula_left_alignment_and_untouched_tail(self):
        for variant in ('plain', 'residual'):
            model = self.make(variant)
            x = torch.randn(2, 5, 32)
            with torch.no_grad():
                model.merge_weight.copy_(torch.tensor([-.4, .8]))
            weights = model.merge_weight.softmax(0)
            expected = weights[0] * x[:, 0:4:2] + weights[1] * x[:, 1:4:2]
            if variant == 'residual':
                expected += x[:, 0:4:2] + x[:, 1:4:2]
            result = model.merge(x)
            torch.testing.assert_close(result[:, :2], expected)
            torch.testing.assert_close(result[:, -1], x[:, -1], rtol=0, atol=0)
            torch.testing.assert_close(model.merge(x[:, :1]), x[:, :1], rtol=0, atol=0)

    def test_prefix_equivalence_and_future_gradients(self):
        for variant in ('plain', 'residual'):
            model = self.make(variant)
            ids = torch.arange(9)[None, :]
            saved = []
            def capture(module, args, output):
                output.retain_grad()
                saved.append(output)
            handle = model.base.model.embed_tokens.register_forward_hook(capture)
            logits, loss = model(ids, (ids + 1) % 41)
            logits[:, 1].sum().backward()
            self.assertEqual(saved[0].grad[:, 4:].abs().max().item(), 0)
            handle.remove()
            with torch.no_grad():
                for i, end in enumerate(endpoints(9, 2, ids.device)):
                    prefix, _ = model(ids[:, :end + 1], last_only=True)
                    torch.testing.assert_close(logits[:, i], prefix[:, -1], atol=1e-6, rtol=1e-5)
            model.zero_grad()
            _, loss = model(ids, (ids + 1) % 41)
            loss.backward()
            self.assertTrue(torch.isfinite(model.merge_weight.grad).all())

    def test_cache_matches_recomputation_and_lengths(self):
        for variant in ('baseline', 'plain', 'residual'):
            for merge_layer in (0, 1, 2):
                for length in (1, 4, 5):
                    model = self.make(variant, merge_layer)
                    ids = torch.randint(41, (2, length))
                    logits, state = model.prefill(ids)
                    for _ in range(5):
                        with torch.no_grad():
                            expected, _ = model(ids, last_only=True)
                        torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
                        for i, layer in enumerate(state.cache.layers):
                            size = ids.shape[1] if variant == 'baseline' or i < merge_layer else (ids.shape[1] + 1) // 2
                            self.assertEqual(layer.get_seq_length(), size)
                        token = torch.randint(41, (2, 1))
                        ids = torch.cat((ids, token), 1)
                        logits, state = model.decode(token, state)

    def test_generation_crop_matches_full_recomputation(self):
        for variant in ('baseline', 'plain', 'residual'):
            model = self.make(variant)
            ids = torch.randint(41, (1, 7))
            expected = ids
            with torch.no_grad():
                for _ in range(7):
                    logits, _ = model(expected[:, -8:], last_only=True)
                    expected = torch.cat((expected, logits[:, -1].argmax(-1, keepdim=True)), 1)
            actual = model.generate_cached(ids, 7, max_context=8)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
