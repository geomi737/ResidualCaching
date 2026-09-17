"""Independent numerical witnesses and held-out metric correctness."""
import unittest

import torch
from torch.nn import functional as F

from experiments.verify_properties import collect_evidence
from nanoGPT.bench_ablation import average_score, score_logits
from nanoGPT.model import GPT, GPTConfig, boundary_target_indices


class EvidenceTests(unittest.TestCase):
    def test_collision_scaling_and_recomputed_prefix_witnesses(self):
        evidence = collect_evidence()
        self.assertLess(evidence['different_inputs_same_compressed_output_max_error'], 1e-12)
        self.assertGreater(evidence['collision_input_difference_norm'], 1)
        self.assertEqual(evidence['cancellation_output_norm'], 0)
        self.assertEqual(evidence['recomputed_early_positions'], 42)
        self.assertEqual(evidence['cropped_prefix_absolute_offsets'], [0, 0, 0, 0, 1, 2])

    def test_accuracy_and_loss_ignore_padding(self):
        logits = torch.tensor([[3., 0.], [2., 3.], [5., -1.]])
        targets = torch.tensor([0, 0, -1])
        score = average_score(score_logits(logits, targets))
        self.assertEqual(score['targets'], 2)
        self.assertEqual(score['accuracy'], 0.5)
        expected = F.cross_entropy(logits[:2], targets[:2]).item()
        self.assertAlmostEqual(score['loss'], expected)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable in this execution environment')
    def test_cuda_float32_and_bfloat16_causality_and_backward(self):
        torch.set_num_threads(1)
        from contextlib import nullcontext
        for dtype in (torch.float32, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for ratio in (1, 2, 3):
                torch.manual_seed(42)
                model = GPT(GPTConfig(block_size=16, vocab_size=19, n_layer=3,
                                     n_head=2, n_embd=32, dropout=0, merge_ratio=ratio,
                                     merge_layer=1)).cuda().eval()
                x = torch.randint(19, (2, 11), device='cuda')
                context = torch.amp.autocast('cuda', dtype=dtype) if dtype == torch.bfloat16 else nullcontext()
                with context:
                    logits, loss = model(x, x)
                    for i, endpoint in enumerate(boundary_target_indices(11, ratio).tolist()):
                        prefix, _ = model(x[:, :endpoint + 1])
                        tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-5
                        torch.testing.assert_close(logits[:, i].float(), prefix[:, -1].float(),
                                                   rtol=tolerance, atol=tolerance)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
