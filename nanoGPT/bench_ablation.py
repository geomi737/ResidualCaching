"""Compare baseline and two merged models on shared validation targets.

Training timers exclude validation and warmup; input tokens and predictions are
counted separately. JSON results include settings and the execution environment.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from model import GPT, GPTConfig, boundary_target_indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data/shakespeare')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name, default in [('block-size', 128), ('batch-size', 16), ('max-iters', 400),
                          ('warmup-iters', 10), ('eval-interval', 50), ('eval-iters', 20),
                          ('n-layer', 6), ('n-head', 6), ('n-embd', 384),
                          ('merge-ratio', 2), ('merge-layer', 2), ('vocab-size', 50304), ('seed', 1337)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--unmerged-prob', type=float, default=0.1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--generation-tokens', type=int, default=32)
    parser.add_argument('--generation-repeats', type=int, default=3)
    args = parser.parse_args()
    for name in ('block_size', 'batch_size', 'max_iters', 'eval_interval', 'eval_iters',
                 'generation_tokens', 'generation_repeats'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if not 0 <= args.warmup_iters < args.max_iters:
        parser.error('warmup-iters must be in [0, max-iters)')
    if args.merge_ratio < 2 or not 0 <= args.unmerged_prob <= 1:
        parser.error('merge-ratio must be >=2 and unmerged-prob in [0, 1]')
    return args


def batch(data, args, generator, all_lengths=False):
    low = 1 if all_lengths else max(1, args.block_size - args.merge_ratio + 1)
    length = int(torch.randint(low, args.block_size + 1, (), generator=generator))
    starts = torch.randint(len(data) - length, (args.batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + length].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1:i + length + 1].astype(np.int64)) for i in starts])
    return x.to(args.device), y.to(args.device)


def autocast(args):
    if args.device.startswith('cuda') and torch.cuda.is_bf16_supported():
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def synchronize(args):
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)


@torch.no_grad()
def evaluate(model, data, args):
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 10000)
    boundary_sum = prefix_sum = 0.0
    boundary_count = prefix_count = 0
    for _ in range(args.eval_iters):
        x, y = batch(data, args, generator, all_lengths=True)
        with autocast(args):
            logits, _ = model(x, y)
        indices = boundary_target_indices(x.size(1), args.merge_ratio, x.device)
        matched = logits.index_select(1, indices) if model.config.merge_ratio == 1 else logits
        targets = y.index_select(1, indices)
        boundary_sum += F.cross_entropy(matched.float().reshape(-1, args.vocab_size),
                                        targets.reshape(-1), reduction='sum').item()
        boundary_count += targets.numel()
        prefix_sum += F.cross_entropy(logits[:, -1].float(), y[:, -1], reduction='sum').item()
        prefix_count += y.size(0)
    model.train()
    return {'boundary_loss': boundary_sum / boundary_count,
            'last_prefix_loss': prefix_sum / prefix_count}


def run(name, ratio, cache, train_data, val_data, args):
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    model = GPT(GPTConfig(block_size=args.block_size, vocab_size=args.vocab_size,
                         n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                         dropout=args.dropout, bias=False, merge_ratio=ratio,
                         merge_layer=args.merge_layer, use_residual_cache=cache)).to(args.device)
    optimizer = model.configure_optimizers(0.1, args.lr, (0.9, 0.95),
                                           'cuda' if args.device.startswith('cuda') else 'cpu')
    generator = torch.Generator().manual_seed(args.seed + 1)
    mode_generator = torch.Generator().manual_seed(args.seed + 2)
    elapsed = 0.0
    input_tokens = predictions = 0
    history = []
    for step in range(args.max_iters):
        x, y = batch(train_data, args, generator)
        merge_tokens = torch.rand((), generator=mode_generator).item() >= args.unmerged_prob
        if step == args.warmup_iters and args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        synchronize(args)
        started = time.perf_counter()
        with autocast(args):
            logits, loss = model(x, y, merge_tokens=merge_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize(args)
        if step >= args.warmup_iters:
            elapsed += time.perf_counter() - started
            input_tokens += x.numel()
            predictions += logits.size(0) * logits.size(1)
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.max_iters:
            metrics = evaluate(model, val_data, args)
            history.append({'step': step + 1, **metrics})
            print(f'{name}: step {step + 1}, boundary loss {metrics["boundary_loss"]:.4f}, '
                  f'last-prefix loss {metrics["last_prefix_loss"]:.4f}', flush=True)
    peak = torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.startswith('cuda') else None
    result = {'name': name, 'merge_ratio': ratio, 'use_residual_cache': cache,
              **history[-1], 'training_time_s': elapsed,
              'input_tokens_per_s': input_tokens / elapsed, 'predictions_per_s': predictions / elapsed,
              'measured_input_tokens': input_tokens, 'measured_predictions': predictions,
              'peak_allocated_vram_mb': peak, 'history': history}
    # Measure actual autoregressive generation separately from training.
    del optimizer, x, y, logits, loss
    model.zero_grad(set_to_none=True)
    model.eval()
    prompt = torch.tensor(np.asarray(val_data[:min(32, args.block_size)]).astype(np.int64),
                          device=args.device).unsqueeze(0)
    torch.manual_seed(args.seed + 20000)
    with autocast(args):
        model.generate(prompt, 2, top_k=50) # untimed warmup
        synchronize(args)
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        for _ in range(args.generation_repeats):
            model.generate(prompt, args.generation_tokens, top_k=50)
        synchronize(args)
    generation_time = time.perf_counter() - started
    result.update({'generation_time_s': generation_time,
                   'generation_tokens_per_s': args.generation_tokens * args.generation_repeats / generation_time,
                   'generation_prompt_length': prompt.size(1),
                   'generation_peak_allocated_vram_mb': torch.cuda.max_memory_allocated(args.device) / 2**20
                   if args.device.startswith('cuda') else None})
    del model
    return result


def main():
    args = parse_args()
    train_data = np.memmap(args.data_dir / 'train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(args.data_dir / 'val.bin', dtype=np.uint16, mode='r')
    for data in (train_data, val_data):
        if len(data) <= args.block_size or int(data.max()) >= args.vocab_size:
            raise ValueError('Each split must exceed block-size and contain IDs within vocab-size')
    results = [run(name, ratio, cache, train_data, val_data, args) for name, ratio, cache in (
        ('Baseline nanoGPT', 1, False), ('Merged + residual sum', args.merge_ratio, True),
        ('Merged without residual sum', args.merge_ratio, False))]
    report = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'environment': {'torch': torch.__version__, 'device': args.device,
                              'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None},
              'dataset_sha256': {'train': hashlib.sha256(train_data).hexdigest(),
                                 'val': hashlib.sha256(val_data).hexdigest()},
              'notes': ['Boundary and last-prefix validation targets are shared across models.',
                        'Last-prefix validation samples lengths uniformly from 1 to block-size.',
                        'Training throughput excludes validation, batch transfer, and warmup iterations.',
                        'Peak allocated VRAM includes model, optimizer, training, and validation after warmup.',
                        'Training predictions per input batch differ between architectures.',
                        'Generation has its own warmed-up timer and uses no persistent KV cache.',
                        'Generation peak VRAM includes model and inference allocations, after freeing the optimizer.'],
              'results': results}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
