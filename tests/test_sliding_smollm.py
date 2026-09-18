import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.smollm.sliding_model import SlidingSmolLM


class SlidingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2, attention_dropout=0)
        config._attn_implementation = 'sdpa'
        self.model = SlidingSmolLM(LlamaForCausalLM(config), merge_layer=1)

    def test_windows_and_no_recursive_aggregation(self):
        x = torch.randn(2, 6, 32)
        with torch.no_grad():
            self.model.merge_weight.copy_(torch.tensor([-.5, .7]))
        w = self.model.merge_weight.softmax(0)
        result = self.model.sliding_merge(x)
        torch.testing.assert_close(result[:, 0], x[:, 0])
        for t in range(1, 6):
            torch.testing.assert_close(result[:, t], w[0] * x[:, t-1] + w[1] * x[:, t])

    def test_dense_causality_prefix_equivalence_and_gradients(self):
        model = self.model.eval()
        ids = torch.arange(8)[None]
        saved = []
        def capture(module, args, output):
            output.retain_grad()
            saved.append(output)
        handle = model.base.model.embed_tokens.register_forward_hook(capture)
        logits, loss = model(ids, (ids + 1) % 41, representation_mode='sliding')
        logits[:, 3].sum().backward()
        self.assertEqual(saved[0].grad[:, 4:].abs().max().item(), 0)
        handle.remove()
        with torch.no_grad():
            for t in range(8):
                prefix, _ = model(ids[:, :t+1], last_only=True, representation_mode='sliding')
                torch.testing.assert_close(prefix[:, -1], logits[:, t], atol=1e-6, rtol=1e-5)
        model.zero_grad()
        _, loss = model(ids, (ids + 1) % 41, representation_mode='sliding')
        loss.backward()
        self.assertTrue(torch.isfinite(model.merge_weight.grad).all())
        self.assertGreater(model.merge_weight.grad.abs().sum().item(), 0)

    def test_dispatch_and_cached_disjoint_equivalence(self):
        model = self.model
        ids = torch.arange(5)[None]
        self.assertEqual(model.train()(ids)[0].shape[1], 5)
        self.assertEqual(model.eval()(ids)[0].shape[1], 3)
        for merge_layer in (0, 1, 2):
            model = SlidingSmolLM(self.model.base, merge_layer=merge_layer).eval()
            for length in (1, 4, 5):
                prefix = ids[:, :length]
                logits, state = model.prefill(prefix)
                for _ in range(4):
                    with torch.no_grad():
                        expected, _ = model(prefix, last_only=True)
                    torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
                    for i, layer in enumerate(state.cache.layers):
                        size = prefix.shape[1] if i < merge_layer else prefix.shape[1]//2
                        self.assertEqual(layer.get_seq_length(), size)
                    token = torch.tensor([[9]])
                    prefix = torch.cat((prefix, token), 1)
                    logits, state = model.decode(token, state)

    def test_generation_context_crop_matches_recomputation(self):
        model = self.model.eval()
        ids = torch.arange(7)[None]
        expected = ids
        with torch.no_grad():
            for _ in range(6):
                logits, _ = model(expected[:, -8:], last_only=True)
                expected = torch.cat((expected, logits[:, -1].argmax(-1, keepdim=True)), 1)
        actual = model.generate_cached(ids, 6, max_context=8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_baseline_matches_native_and_shifted_loss(self):
        model = SlidingSmolLM(self.model.base, 'baseline', 1).eval()
        ids = torch.arange(9)[None]
        reference = model.base(ids, labels=ids, use_cache=False)
        for mode in ('sliding', 'disjoint'):
            logits, loss = model(ids[:, :-1], ids[:, 1:], representation_mode=mode)
            torch.testing.assert_close(logits, reference.logits[:, :-1], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(loss, reference.loss)

    def test_residual_formula_causality_and_cache(self):
        model = SlidingSmolLM(self.model.base, 'residual', 1).eval()
        x = torch.randn(2, 5, 32)
        w = model.merge_weight.softmax(0)
        expected = (w[0]+1)*x[:, :-1] + (w[1]+1)*x[:, 1:]
        torch.testing.assert_close(model.sliding_merge(x)[:, 1:], expected)
        torch.testing.assert_close(model.sliding_merge(x)[:, :1], x[:, :1])
        ids = torch.arange(7)[None]
        with torch.no_grad():
            logits, _ = model(ids, representation_mode='sliding')
            changed = ids.clone()
            changed[:, 4:] = 15
            other, _ = model(changed, representation_mode='sliding')
            torch.testing.assert_close(logits[:, :4], other[:, :4])
            for length in (4, 5):
                prefix = ids[:, :length]
                cached, state = model.prefill(prefix)
                for _ in range(4):
                    full, _ = model(prefix, last_only=True)
                    torch.testing.assert_close(cached, full, atol=1e-6, rtol=1e-5)
                    self.assertEqual(state.cache.layers[1].get_seq_length(), prefix.shape[1]//2)
                    token = torch.tensor([[9]])
                    prefix = torch.cat((prefix, token), 1)
                    cached, state = model.decode(token, state)


if __name__ == '__main__':
    unittest.main()
