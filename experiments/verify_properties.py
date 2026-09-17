"""Reproducible numerical witnesses for algebraic and runtime properties."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanoGPT.model import GPT, GPTConfig, WindowTokenMerger, boundary_target_indices


def collect_evidence():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dtype = torch.float64
    merger = WindowTokenMerger(4, 2).to(dtype=dtype)
    x = torch.randn(1, 2, 4, dtype=dtype)
    uniform_error = (merger(x) - 3 * x.mean(1, keepdim=True)).abs().max().item()
    assert uniform_error < 1e-12
    with torch.no_grad():
        merger.merge_weight.copy_(torch.tensor([-0.8, 1.3], dtype=dtype))
    weights = merger.merge_weight.softmax(0)
    coefficients = 1 + weights
    displacement = torch.randn(1, 4, dtype=dtype)
    different = x.clone()
    different[:, 0] += coefficients[1] * displacement
    different[:, 1] -= coefficients[0] * displacement
    collision_error = (merger(x) - merger(different)).abs().max().item()
    assert not torch.allclose(x, different)
    assert collision_error < 1e-12
    normalized = coefficients / coefficients.sum()
    scaled_average_error = (merger(x) - 3 * (x * normalized.view(1, 2, 1)).sum(1, keepdim=True)).abs().max().item()
    assert scaled_average_error < 1e-12
    with torch.no_grad():
        merger.merge_weight.zero_()
    cancellation = torch.stack((displacement, -displacement), dim=1)
    assert merger(cancellation).norm().item() == 0

    config = GPTConfig(block_size=8, vocab_size=19, n_layer=3, n_head=2,
                       n_embd=16, dropout=0, merge_ratio=2, merge_layer=1)
    model = GPT(config).eval()
    early_lengths, deep_lengths, positions, inputs = [], [], [], []
    handles = [
        model.transformer.h[0].register_forward_pre_hook(
            lambda module, args: early_lengths.append(args[0].size(1))),
        model.transformer.h[1].register_forward_pre_hook(
            lambda module, args: deep_lengths.append(args[0].size(1))),
        model.transformer.wpe.register_forward_pre_hook(
            lambda module, args: positions.append(args[0].tolist())),
        model.transformer.wte.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].clone())),
    ]
    prompt = torch.arange(5).unsqueeze(0)
    generated = model.generate(prompt, 6, top_k=5)
    for handle in handles:
        handle.remove()
    assert early_lengths == [5, 6, 7, 8, 8, 8]
    assert deep_lengths == [3, 3, 4, 4, 4, 4]
    offsets = []
    for step, (idx, pos) in enumerate(zip(inputs, positions)):
        end = 5 + step
        start = max(0, end - config.block_size)
        offsets.append(start)
        torch.testing.assert_close(idx, generated[:, start:end])
        assert pos == list(range(end - start))
    counts = [{'length': length, 'ratio': ratio, 'baseline_targets': length,
               'merged_targets': len(boundary_target_indices(length, ratio)),
               'indices': boundary_target_indices(length, ratio).tolist()}
              for ratio in (2, 3) for length in (1, 7, 8, 9)]
    return {'uniform_scaled_mean_max_error': uniform_error,
            'nonuniform_scaled_average_max_error': scaled_average_error,
            'different_inputs_same_compressed_output_max_error': collision_error,
            'collision_input_difference_norm': (x - different).norm().item(),
            'cancellation_input_norm': cancellation.norm().item(),
            'cancellation_output_norm': merger(cancellation).norm().item(),
            'runtime_config': asdict(config), 'generation_calls': len(early_lengths),
            'early_block_input_lengths': early_lengths, 'deep_block_input_lengths': deep_lengths,
            'recomputed_early_positions': sum(early_lengths), 'positional_indices': positions,
            'cropped_prefix_absolute_offsets': offsets,
            'first_window_absolute_positions': [[offset, offset + 1] for offset in offsets],
            'training_target_counts': counts,
            'scope': 'Linear aggregation of arbitrary hidden states; not a claim that all collisions occur on natural text.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/property_evidence.json'))
    args = parser.parse_args()
    report = collect_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
