"""Measure pretrained SmolLM training feasibility, not model quality.

Uses random input solely to allocate realistic activations and AdamW states.
Run each configuration in a fresh process for comparable allocator peaks.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/SmolLM2-135M')
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('Use at least two steps, including one warmup step.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this hardware probe.')
    torch.manual_seed(2026)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='sdpa',
    ).cuda().train()
    model.config.use_cache = False
    if args.checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    inputs = torch.randint(model.config.vocab_size,
                           (args.batch_size, args.length), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite probe loss.')
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        losses.append(loss.item())
    result = {
        'purpose': 'Hardware feasibility only; random tokens are not a quality evaluation.',
        'model': args.model, 'parameters': sum(p.numel() for p in model.parameters()),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'transformers': transformers.__version__,
        'parameter_dtype': 'float32', 'autocast': 'bfloat16',
        'optimizer': 'AdamW, float32 states, foreach=False',
        'batch_size': args.batch_size, 'sequence_length': args.length,
        'gradient_checkpointing': args.checkpointing,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'device_total_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
        'input_tokens_per_second': args.batch_size * args.length * (args.steps - 1)
                                  / sum(elapsed[1:]),
        'step_seconds': elapsed, 'random_input_losses_not_quality': losses,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
