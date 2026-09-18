"""CUDA compute/memory scaling microbenchmark, separate from trained-model quality.

Freshly initialized models measure tensor-shape costs across context lengths and
compression depths. No accuracy or convergence claim is made from these runs.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nanoGPT.model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--contexts', nargs='+', type=int, default=[128, 512, 1024])
    p.add_argument('--merge-layers', nargs='+', type=int, default=[0, 2, 4])
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    p.add_argument('--n-embd', type=int, default=192)
    p.add_argument('--n-layer', type=int, default=6)
    p.add_argument('--n-head', type=int, default=6)
    p.add_argument('--vocab-size', type=int, default=50304)
    p.add_argument('--train-steps', type=int, default=20)
    p.add_argument('--train-warmup', type=int, default=5)
    p.add_argument('--generation-tokens', type=int, default=32)
    p.add_argument('--generation-repeats', type=int, default=5)
    p.add_argument('--output', type=Path, default=Path('results/scaling_cuda.json'))
    a = p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():
        p.error('This microbenchmark requires CUDA')
    if min(a.contexts) < 2 or a.train_steps < 1 or a.generation_repeats < 1:
        p.error('Invalid measurement sizes')
    return a


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def amp():
    return torch.amp.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else nullcontext()


def peak(device):
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20}


def measure(case, length, seed, args):
    name, ratio, cache, layer, merge_tokens = case
    cleanup()
    torch.manual_seed(seed)
    model = GPT(GPTConfig(block_size=max(args.contexts), vocab_size=args.vocab_size,
                         n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                         dropout=0.1, bias=False, merge_ratio=ratio, merge_layer=layer,
                         use_residual_cache=cache)).to(args.device)
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cuda')
    size = max(1, 2048 // length)
    x = torch.randint(args.vocab_size, (size, length), device=args.device)
    y = torch.randint(args.vocab_size, (size, length), device=args.device)
    times = []
    peaks = []
    for step in range(args.train_warmup + args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        with amp():
            logits, loss = model(x, y, merge_tokens=merge_tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(args.device)
        duration = time.perf_counter() - started
        predictions = logits.size(0) * logits.size(1)
        if step >= args.train_warmup:
            times.append(duration)
            peaks.append(peak(args.device))
        del logits, loss
    training = {'batch_size': size, 'steps': len(times), 'time_s': sum(times),
                'input_tokens_per_s': x.numel() * len(times) / sum(times),
                'predictions_per_s': predictions * len(times) / sum(times),
                'peak_allocated_mib': max(v['peak_allocated_mib'] for v in peaks),
                'peak_reserved_mib': max(v['peak_reserved_mib'] for v in peaks)}
    del optimizer, x, y
    model.zero_grad(set_to_none=True)
    model.eval()
    cleanup()
    generation = None
    if merge_tokens:
        prompt = torch.randint(args.vocab_size, (1, length), device=args.device)
        # A fixed-size forward isolates this exact context shape. Autoregressive
        # generation additionally measures prefix growth/cropping up to max context.
        with torch.inference_mode(), amp():
            for _ in range(5):
                model(prompt)
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            for _ in range(100):
                model(prompt)
            torch.cuda.synchronize(args.device)
            forward = {'next_token_forwards_per_s': 100 / (time.perf_counter() - started), **peak(args.device)}
            model.generate(prompt, 4, top_k=50)
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            for _ in range(args.generation_repeats):
                model.generate(prompt, args.generation_tokens, top_k=50)
            torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - started
            generation = {'tokens_per_s': args.generation_repeats * args.generation_tokens / elapsed,
                          'time_s': elapsed, **peak(args.device)}
    else:
        forward = None
    result = {'variant': name, 'seed': seed, 'context': length, 'merge_layer': layer,
              'merge_ratio': ratio, 'merge_tokens': merge_tokens,
              'training': training, 'fixed_context_inference': forward, 'generation': generation}
    print(f'{name} layer={layer} context={length} seed={seed}: '
          f'train {training["input_tokens_per_s"]:.0f} tok/s, peak {training["peak_allocated_mib"]:.1f} MiB', flush=True)
    del model
    cleanup()
    return result


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report = {'complete': False, 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'environment': {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(args.device),
                              'dtype': 'bfloat16' if torch.cuda.is_bf16_supported() else 'float32'},
              'scope': 'Fresh models and random tokens: compute/memory only, not trained LM quality.',
              'notes': ['Training modes are measured independently; no singleton batches in compressed measurements.',
                        'Batch size changes with context to keep approximately 2048 input tokens per step.',
                        'Fixed-context inference is one next-token forward, not a persistent KV-cache decode.',
                        'Generation prefix grows/crops at the largest configured context; GPU timers synchronize.'],
              'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed_index, seed in enumerate(args.seeds):
        for length in args.contexts:
            cases = [('baseline', 1, False, 2, True)]
            for layer in args.merge_layers:
                cases.extend([('merged_no_residual', 2, False, layer, True),
                              ('merged_residual', 2, True, layer, True)])
            rotation = seed_index % len(cases)
            for case in cases[rotation:] + cases[:rotation]:
                report['results'].append(measure(case, length, seed, args))
                args.output.write_text(json.dumps(report, indent=2) + '\n')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Complete report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
