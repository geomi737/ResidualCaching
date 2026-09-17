"""Three-way LM ablation with held-out accuracy, CUDA memory, and timed throughput.

Validation is used for monitoring only. Final test metrics are computed once per
run; comparisons share initialization, batches, evaluation targets, and seeds.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
if __package__:
    from .model import GPT, GPTConfig, boundary_target_indices
else:
    from model import GPT, GPTConfig, boundary_target_indices

VARIANTS = [('baseline', 1, False), ('merged_no_residual', None, False),
            ('merged_residual', None, True)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data/shakespeare_threeway')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name, default in [('block-size', 128), ('batch-size', 16), ('max-iters', 1000),
                          ('warmup-iters', 10), ('eval-interval', 200), ('eval-iters', 16),
                          ('test-iters', 64), ('prefix-iters', 64), ('n-layer', 6),
                          ('n-head', 6), ('n-embd', 192), ('merge-ratio', 2), ('merge-layer', 2),
                          ('vocab-size', 50304), ('lr-warmup-iters', 50),
                          ('generation-tokens', 32), ('generation-repeats', 5)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1337, 2027, 3407])
    parser.add_argument('--seed', type=int, help='Compatibility shortcut for a single seed')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'bfloat16'], default='auto')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-4)
    parser.add_argument('--unmerged-prob', type=float, default=0.1)
    parser.add_argument('--output', type=Path, default=Path('results/ablation.json'))
    parser.add_argument('--checkpoint-dir', type=Path)
    args = parser.parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    for name in ('block_size', 'batch_size', 'max_iters', 'eval_interval', 'eval_iters',
                 'test_iters', 'prefix_iters', 'generation_tokens', 'generation_repeats'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if not 0 <= args.warmup_iters < args.max_iters:
        parser.error('warmup-iters must be in [0, max-iters)')
    if args.merge_ratio < 2 or args.block_size < args.merge_ratio or not 0 <= args.unmerged_prob <= 1:
        parser.error('Require block-size >= merge-ratio >= 2 and unmerged-prob in [0,1]')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable in this execution environment')
    if args.dtype == 'auto':
        args.dtype = 'bfloat16' if args.device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'float32'
    if args.dtype == 'bfloat16' and (not args.device.startswith('cuda') or not torch.cuda.is_bf16_supported()):
        parser.error('This benchmark requires CUDA BF16 support for bfloat16')
    return args


def synchronize(args):
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)


def autocast(args):
    return torch.amp.autocast('cuda', dtype=torch.bfloat16) if args.dtype == 'bfloat16' else nullcontext()


def cleanup(args):
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()


def memory(args):
    if not args.device.startswith('cuda'):
        return {'peak_allocated_mib': None, 'peak_reserved_mib': None}
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated(args.device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(args.device) / 2**20}


def reset_memory(args):
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)


def batch(data, args, generator, length=None, plan_hash=None):
    if length is None:
        length = int(torch.randint(args.block_size - args.merge_ratio + 1,
                                   args.block_size + 1, (), generator=generator))
    starts = torch.randint(len(data) - length, (args.batch_size,), generator=generator)
    if plan_hash is not None:
        plan_hash.update(np.asarray([length], dtype=np.int64).tobytes())
        plan_hash.update(starts.numpy().tobytes())
    x = torch.from_numpy(np.stack([data[int(i):int(i) + length].astype(np.int64) for i in starts]))
    y = torch.from_numpy(np.stack([data[int(i) + 1:int(i) + length + 1].astype(np.int64) for i in starts]))
    return x.to(args.device), y.to(args.device)


def score_logits(logits, targets):
    valid = targets != -1
    logits = logits.float()[valid]
    targets = targets[valid]
    return {'loss_sum': F.cross_entropy(logits, targets, reduction='sum').item(),
            'correct': (logits.argmax(-1) == targets).sum().item(), 'count': targets.numel()}


def average_score(total):
    return {'loss': total['loss_sum'] / total['count'],
            'accuracy': total['correct'] / total['count'], 'targets': total['count']}


def add_score(total, value):
    for key in total:
        total[key] += value[key]


def empty_score():
    return {'loss_sum': 0.0, 'correct': 0, 'count': 0}


@torch.inference_mode()
def evaluate_boundaries(model, data, args, iterations):
    model.eval()
    generator = torch.Generator().manual_seed(76543)
    total = empty_score()
    for step in range(iterations):
        length = args.block_size - step % args.merge_ratio
        x, y = batch(data, args, generator, length)
        with autocast(args):
            logits, unused_loss = model(x, y)
        del unused_loss
        indices = boundary_target_indices(length, args.merge_ratio, x.device)
        matched = logits.index_select(1, indices) if model.config.merge_ratio == 1 else logits
        add_score(total, score_logits(matched.reshape(-1, args.vocab_size), y[:, indices].reshape(-1)))
        del logits, matched, x, y
    return average_score(total)


@torch.inference_mode()
def evaluate_prefixes(model, data, args):
    model.eval()
    generator = torch.Generator().manual_seed(87654)
    total = empty_score()
    by_remainder = {str(r): empty_score() for r in range(args.merge_ratio)}
    for step in range(args.prefix_iters):
        length = args.block_size - step % args.merge_ratio
        x, y = batch(data, args, generator, length)
        with autocast(args):
            logits, _ = model(x)
        score = score_logits(logits[:, -1], y[:, -1])
        add_score(total, score)
        add_score(by_remainder[str(length % args.merge_ratio)], score)
    return {**average_score(total), 'by_remainder': {
        key: average_score(value) for key, value in by_remainder.items() if value['count']}}


@torch.inference_mode()
def generation_benchmark(model, data, args):
    model.eval()
    results = []
    for length in sorted(set([min(32, args.block_size), max(1, args.block_size // 2), args.block_size])):
        prompt = torch.tensor(np.asarray(data[:length]).astype(np.int64), device=args.device).unsqueeze(0)
        cleanup(args)
        torch.manual_seed(98765)
        with autocast(args):
            model.generate(prompt, 4, top_k=50)
            synchronize(args)
            reset_memory(args)
            durations = []
            for repeat in range(args.generation_repeats):
                synchronize(args)
                started = time.perf_counter()
                model.generate(prompt, args.generation_tokens, top_k=50)
                synchronize(args)
                durations.append(time.perf_counter() - started)
        elapsed = sum(durations)
        results.append({'prompt_length': length, 'new_tokens_per_repeat': args.generation_tokens,
                        'repeats': args.generation_repeats, 'time_s': elapsed,
                        'tokens_per_s': args.generation_tokens * len(durations) / elapsed,
                        'repeat_time_s': durations, **memory(args)})
    return results


def initialization_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith('merger.'):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def learning_rate(step, args):
    if step < args.lr_warmup_iters:
        return args.lr * (step + 1) / max(1, args.lr_warmup_iters)
    span = max(1, args.max_iters - args.lr_warmup_iters - 1)
    progress = min(1, max(0, (step - args.lr_warmup_iters) / span))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def run(name, ratio, cache, splits, args, seed):
    cleanup(args)
    torch.manual_seed(seed)
    config = GPTConfig(block_size=args.block_size, vocab_size=args.vocab_size, n_layer=args.n_layer,
                       n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, bias=False,
                       merge_ratio=ratio, merge_layer=args.merge_layer, use_residual_cache=cache)
    model = GPT(config)
    init_hash = initialization_hash(model)
    model.to(args.device)
    optimizer = model.configure_optimizers(0.1, args.lr, (0.9, 0.95),
                                           'cuda' if args.device.startswith('cuda') else 'cpu')
    generator = torch.Generator().manual_seed(seed + 1)
    mode_generator = torch.Generator().manual_seed(seed + 2)
    plan_hash = hashlib.sha256()
    mode_hash = hashlib.sha256()
    elapsed = end_to_end = 0.0
    input_tokens = predictions = 0
    all_input_tokens = all_predictions = singleton_batches = 0
    loss_sum = loss_targets = 0
    train_peak_allocated = train_peak_reserved = None
    history = []
    for step in range(args.max_iters):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        full_started = time.perf_counter()
        x, y = batch(splits['train'], args, generator, plan_hash=plan_hash)
        merge_tokens = torch.rand((), generator=mode_generator).item() >= args.unmerged_prob
        mode_hash.update(bytes([merge_tokens]))
        singleton_batches += int(not merge_tokens and ratio > 1)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, args)
        synchronize(args)
        reset_memory(args)
        started = time.perf_counter()
        with autocast(args):
            logits, loss = model(x, y, merge_tokens=merge_tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize(args)
        duration = time.perf_counter() - started
        pred_count = logits.size(0) * logits.size(1)
        all_input_tokens += x.numel()
        all_predictions += pred_count
        loss_sum += loss.item() * pred_count
        loss_targets += pred_count
        if step >= args.warmup_iters:
            elapsed += duration
            end_to_end += time.perf_counter() - full_started
            input_tokens += x.numel()
            predictions += pred_count
            peak = memory(args)
            if peak['peak_allocated_mib'] is not None:
                train_peak_allocated = max(train_peak_allocated or 0, peak['peak_allocated_mib'])
                train_peak_reserved = max(train_peak_reserved or 0, peak['peak_reserved_mib'])
        del logits, loss, x, y
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.max_iters:
            validation = evaluate_boundaries(model, splits['val'], args, args.eval_iters)
            history.append({'step': step + 1, 'train_loss': loss_sum / loss_targets,
                            'validation': validation})
            print(f'{name} seed={seed} step={step + 1}/{args.max_iters}: '
                  f'train={loss_sum / loss_targets:.4f}, val={validation["loss"]:.4f}, '
                  f'val accuracy={validation["accuracy"]:.2%}', flush=True)
            loss_sum = loss_targets = 0
            cleanup(args) # prevent validation's cached allocations inflating training reserved peaks
    synchronize(args)
    result = {'variant': name, 'seed': seed, 'parameters': model.get_num_params(non_embedding=False),
              'shared_initialization_sha256': init_hash, 'training_plan_sha256': plan_hash.hexdigest(),
              'mode_plan_sha256': mode_hash.hexdigest(), 'singleton_batches': singleton_batches,
              'total_training_input_tokens': all_input_tokens, 'total_training_predictions': all_predictions,
              'training': {'model_time_s': elapsed, 'end_to_end_time_s': end_to_end,
                           'input_tokens_per_s': input_tokens / elapsed,
                           'end_to_end_input_tokens_per_s': input_tokens / end_to_end,
                           'predictions_per_s': predictions / elapsed,
                           'measured_input_tokens': input_tokens, 'measured_predictions': predictions,
                           'peak_allocated_mib': train_peak_allocated, 'peak_reserved_mib': train_peak_reserved},
              'history': history}
    if ratio > 1:
        result['learned_merge_weights'] = model.merger.merge_weight.detach().softmax(0).cpu().tolist()
    if args.checkpoint_dir:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': asdict(config),
                    'iter_num': args.max_iters, 'best_val_loss': min(h['validation']['loss'] for h in history),
                    'config': {'dataset': str(args.data_dir.resolve()), 'unmerged_prob': args.unmerged_prob}},
                   args.checkpoint_dir / f'{name}_{seed}.pt')
    del optimizer
    model.zero_grad(set_to_none=True)
    cleanup(args)
    # Model evaluation on the test split happens only after the final training update.
    result['test_boundaries'] = evaluate_boundaries(model, splits['test'], args, args.test_iters)
    result['test_prefixes'] = evaluate_prefixes(model, splits['test'], args)
    result['generation'] = generation_benchmark(model, splits['test'], args)
    print(f'{name} seed={seed}: TEST loss={result["test_prefixes"]["loss"]:.4f}, '
          f'accuracy={result["test_prefixes"]["accuracy"]:.2%}, '
          f'train input tokens/s={input_tokens / elapsed:.0f}', flush=True)
    del model
    cleanup(args)
    return result


def stats(values):
    values = [value for value in values if value is not None]
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else None,
            'n': len(values)} if values else {'mean': None, 'std': None, 'n': 0}


def summarize(results):
    summary = {}
    for name, _, _ in VARIANTS:
        runs = [r for r in results if r['variant'] == name]
        if not runs:
            continue
        summary[name] = {'training': {key: stats([r['training'][key] for r in runs]) for key in (
            'input_tokens_per_s', 'predictions_per_s', 'peak_allocated_mib', 'peak_reserved_mib')},
            'test_boundaries': {key: stats([r['test_boundaries'][key] for r in runs]) for key in ('loss', 'accuracy')},
            'test_prefixes': {key: stats([r['test_prefixes'][key] for r in runs]) for key in ('loss', 'accuracy')},
            'generation': {str(g['prompt_length']): {
                key: stats([next(v for v in r['generation'] if v['prompt_length'] == g['prompt_length'])[key]
                            for r in runs]) for key in ('tokens_per_s', 'peak_allocated_mib', 'peak_reserved_mib')}
                for g in runs[0]['generation']}}
    return summary


def environment(args):
    info = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'device': args.device, 'dtype': args.dtype, 'cpu': platform.processor(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32}
    if args.device.startswith('cuda'):
        prop = torch.cuda.get_device_properties(args.device)
        info.update({'gpu': prop.name, 'gpu_total_mib': prop.total_memory / 2**20,
                     'compute_capability': [prop.major, prop.minor]})
        info['gpu_snapshot'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,driver_version,memory.used,temperature.gpu,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
    return info


def main():
    args = parse_args()
    splits = {}
    for name in ('train', 'val', 'test'):
        path = args.data_dir / f'{name}.bin'
        if not path.exists():
            raise FileNotFoundError(f'{path}: run data/shakespeare_threeway/prepare.py first')
        splits[name] = np.memmap(path, dtype=np.uint16, mode='r')
        if len(splits[name]) <= args.block_size or int(splits[name].max()) >= args.vocab_size:
            raise ValueError('Every split must exceed block-size and contain IDs within vocab-size')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report = {'complete': False, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'environment': environment(args),
              'dataset_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in splits.items()},
              'notes': ['Equal training input batches and steps; different supervised-prediction counts.',
                        'Validation monitors fixed boundary targets; test is evaluated after training only.',
                        'Prefix accuracy is top-1 next-token accuracy, not semantic/task accuracy.',
                        'Training peaks exclude validation and retain only the current step logits.',
                        'Allocated/reserved memory exclude desktop processes and non-PyTorch CUDA allocations.',
                        'Generation uses batch size 1, top-k=50, no KV cache, and cropped/reindexed prefixes.',
                        'Variant order rotates across seeds; summary reports sample standard deviation.',
                        'No universal speed, quality, or lossless-memory claim follows from this experiment.'],
              'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed_index, seed in enumerate(args.seeds):
        order = VARIANTS[seed_index % len(VARIANTS):] + VARIANTS[:seed_index % len(VARIANTS)]
        for name, ratio, cache in order:
            result = run(name, ratio or args.merge_ratio, cache, splits, args, seed)
            peers = [r for r in report['results'] if r['seed'] == seed]
            for peer in peers:
                for key in ('shared_initialization_sha256', 'training_plan_sha256', 'mode_plan_sha256',
                            'total_training_input_tokens'):
                    if peer[key] != result[key]:
                        raise RuntimeError(f'Comparison is not paired: {key}')
            report['results'].append(result)
            report['summary'] = summarize(report['results'])
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    report.update({'complete': True, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                   'environment_after': environment(args)})
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Complete report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
