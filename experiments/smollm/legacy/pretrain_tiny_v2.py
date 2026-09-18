"""Paired always-merged SmolLM2 pilot with live JSON progress and cached decoding."""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.smollm.legacy.smollm_model_v2 import MergedSmolLM, endpoints


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def batch(data, starts, length):
    array = np.stack([data[int(s):int(s) + length + 1] for s in starts]).astype(np.int64)
    tensor = torch.from_numpy(array).cuda()
    return tensor[:, :-1], tensor[:, 1:]


@torch.no_grad()
def evaluate(model, data, length, samples, seed=777):
    """Identical prefix targets plus identical window-boundary targets for all variants."""
    model.eval()
    rng = np.random.default_rng(seed)
    sums = {'even': [0., 0, 0], 'odd': [0., 0, 0]}
    boundary_loss, boundary_correct, boundary_count = 0., 0, 0
    for parity, size in [('even', length), ('odd', length - 1)]:
        starts = rng.choice(len(data) - length - 1, samples // 2, replace=False)
        for chunk in np.array_split(starts, max(1, math.ceil(len(starts) / 4))):
            x, y = batch(data, chunk, size)
            with amp():
                logits, _ = model(x)
            final = logits[:, -1].float()
            sums[parity][0] += F.cross_entropy(final, y[:, -1], reduction='sum').item()
            sums[parity][1] += (final.argmax(-1) == y[:, -1]).sum().item()
            sums[parity][2] += len(chunk)
            indices = endpoints(size, 2, x.device)
            common = logits[:, indices]
            labels = y[:, indices]
            boundary_loss += F.cross_entropy(common.float().reshape(-1, common.shape[-1]),
                                             labels.reshape(-1), reduction='sum').item()
            boundary_correct += (common.argmax(-1) == labels).sum().item()
            boundary_count += labels.numel()
    total = np.array(list(sums.values())).sum(0)
    result = {'prefix_loss': float(total[0] / total[2]),
              'prefix_accuracy': float(total[1] / total[2]), 'prefix_targets': int(total[2]),
              'boundary_loss': boundary_loss / boundary_count,
              'boundary_accuracy': boundary_correct / boundary_count,
              'boundary_targets': boundary_count,
              'parity': {key: {'loss': v[0] / v[2], 'accuracy': v[1] / v[2], 'targets': v[2]}
                         for key, v in sums.items()}}
    model.train()
    return result


@torch.no_grad()
def inference_benchmark(model, data):
    """Fresh allocator peaks after freeing optimizer; prefill and cached decode separated."""
    model.eval()
    cases = []
    for length in (512, 1024):
        prompt, _ = batch(data, [42], length)
        for _ in range(2):
            with amp():
                model.generate_cached(prompt, 8)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            with amp():
                logits, state = model.prefill(prompt)
            del logits, state
        torch.cuda.synchronize()
        prefill_seconds = (time.perf_counter() - start) / 5
        prefill_peak = torch.cuda.max_memory_allocated() / 2**20
        with amp():
            logits, state = model.prefill(prompt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with amp():
            for _ in range(64):
                token = logits[:, -1].argmax(-1, keepdim=True)
                logits, state = model.decode(token, state)
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        cache_bytes = sum(layer.keys.numel() * layer.keys.element_size()
                          + layer.values.numel() * layer.values.element_size()
                          for layer in state.cache.layers)
        cases.append({'prompt_length': length, 'prefill_seconds': prefill_seconds,
                      'prefill_input_tokens_per_second': length / prefill_seconds,
                      'prefill_peak_allocated_mib': prefill_peak,
                      'cached_decode_tokens_per_second': 64 / duration,
                      'decode_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'decode_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                      'kv_cache_mib_after_64_steps': cache_bytes / 2**20})
        del state, logits
    return cases


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    name = f'{args.variant}-{args.seed}'
    status_path = out / f'{name}.json'
    status = {'variant': args.variant, 'seed': args.seed, 'state': 'loading',
              'step': 0, 'total_steps': args.steps, 'history': [], 'config': vars(args)}

    def update(**items):
        status.update(items, updated_at=time.time())
        write_json(status_path, status)
    update()
    try:
        train = np.memmap('data/smollm/train.bin', dtype=np.uint16, mode='r')
        val = np.memmap('data/smollm/validation.bin', dtype=np.uint16, mode='r')
        test = np.memmap('data/smollm/test.bin', dtype=np.uint16, mode='r')
        model = MergedSmolLM.load(args.model, args.variant, args.merge_layer).cuda()
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        config.num_hidden_layers = 8
        config.hidden_size = 288
        config.intermediate_size = 768
        config.num_attention_heads = 6
        config.num_key_value_heads = 3
        base = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa')
        model = MergedSmolLM(base, args.variant, args.merge_layer).cuda()
        initial_hash = hashlib.sha256()
        for parameter in model.base.parameters():
            initial_hash.update(parameter.detach().cpu().float().numpy().tobytes())
        # 1. Weight Decay Filtering (no decay on 1D parameters like biases and norms)
        decay_params = []
        nodecay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)
        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, foreach=False)

        # 2. Cosine Learning Rate Schedule with Warmup
        warmup_steps = int(args.steps * 0.1)  # 10% warmup

        def get_lr_multiplier(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, args.steps - warmup_steps))
            return 0.1 + 0.5 * 0.9 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_multiplier)

        update(state='initial_validation', initial_weights_sha256=initial_hash.hexdigest(),
               gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               config=vars(args), parameters=sum(p.numel() for p in model.parameters()))
        initial = evaluate(model, val, args.length, args.eval_samples)
        gc.collect()
        torch.cuda.empty_cache()
        update(initial_validation=initial, state='training')
        # Fixed independent RNGs keep the batch plan identical across variants.
        rng = np.random.default_rng(args.seed)
        plan_hash = hashlib.sha256()
        input_tokens, predictions, seconds = 0, 0, 0.
        timed_inputs, timed_predictions = 0, 0
        peak_allocated, peak_reserved = 0., 0.
        started = time.time()
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # Alternate both parities, paired across all three architectures.
            length = args.length - (step % 2)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            mean_loss = 0.
            for _ in range(args.accumulation):
                starts = rng.integers(0, len(train) - args.length - 1, args.batch_size)
                plan_hash.update(starts.tobytes())
                plan_hash.update(length.to_bytes(4, 'little'))
                x, y = batch(train, starts, length)
                with amp():
                    logits, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss.')
                (loss / args.accumulation).backward()
                mean_loss += loss.item() / args.accumulation
                count = logits.shape[0] * logits.shape[1]
                input_tokens += x.numel()
                predictions += count
                if step > 3:
                    timed_inputs += x.numel()
                    timed_predictions += count
                del logits, loss, x, y
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            if step > 3:
                seconds += duration
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 2**20)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 2**20)
            event = {'step': step, 'train_loss': mean_loss, 'lr': optimizer.param_groups[0]['lr'],
                     'input_tokens_per_second': timed_inputs / seconds if seconds else 0,
                     'supervised_predictions_per_second': timed_predictions / seconds if seconds else 0,
                     'train_peak_allocated_mib': peak_allocated,
                     'train_peak_reserved_mib': peak_reserved}
            if step % args.eval_every == 0 or step == args.steps:
                # Remove grad buffers and free cached allocations before eval;
                # eval peaks are not counted as training peaks.
                optimizer.zero_grad(set_to_none=True)
                update(state='validation', step=step)
                event['validation'] = evaluate(model, val, args.length, args.eval_samples)
                gc.collect()
                torch.cuda.empty_cache()
            status['history'].append(event)
            update(state='training', step=step, elapsed_seconds=time.time() - started,
                   **{k: v for k, v in event.items() if k != 'step'},
                   input_tokens=input_tokens, supervised_predictions=predictions)
        update(state='test', batch_plan_sha256=plan_hash.hexdigest())
        # Hyperparameters and final-step selection never depend on test data.
        metrics = evaluate(model, test, args.length, args.test_samples, seed=999)
        torch.save({'model': model.state_dict(), 'config': vars(args),
                    'data_manifest': json.loads(Path('data/smollm/manifest.json').read_text())},
                   out / f'{name}.pt')
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        update(state='inference_benchmark', test=metrics)
        infer = inference_benchmark(model, test)
        with amp():
            prompt, _ = batch(test, [42], 64)
            generated = model.generate_cached(prompt, 48)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        update(state='complete', inference=infer,
               sample=tokenizer.decode(generated[0].tolist()),
               merge_weights=model.merge_weight.softmax(0).detach().cpu().tolist() if model.ratio > 1 else None)
        print(f'{name}: complete', flush=True)
    except Exception:
        update(state='failed', error=traceback.format_exc())
        raise


def suite(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    order = []
    variants = ['baseline', 'plain', 'residual']
    # Rotate order by seed to reduce systematic warm-machine ordering effects.
    for i, seed in enumerate(args.seeds):
        for variant in variants[i % 3:] + variants[:i % 3]:
            order.append({'variant': variant, 'seed': seed, 'file': f'{variant}-{seed}.json'})
    progress = {'state': 'running', 'runs': order, 'completed': 0, 'config': vars(args),
                'started_at': time.time(), 'data': json.loads(Path('data/smollm/manifest.json').read_text())}
    write_json(out / 'suite.json', progress)
    for job in order:
        command = [sys.executable, __file__, '--variant', job['variant'], '--seed', str(job['seed'])]
        for key in ('model', 'merge_layer', 'steps', 'length', 'batch_size', 'accumulation',
                    'lr', 'eval_every', 'eval_samples', 'test_samples', 'output'):
            command.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
        with (out / f"{job['variant']}-{job['seed']}.log").open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            progress.update(state='failed', failed_run=job)
            write_json(out / 'suite.json', progress)
            raise RuntimeError(f'Failed: {job}; see its log.')
        progress['completed'] += 1
        write_json(out / 'suite.json', progress)
        print(f"Completed {progress['completed']}/{len(order)}: {job['variant']} seed {job['seed']}", flush=True)
    results = [json.loads((out / job['file']).read_text()) for job in order]
    for seed in args.seeds:
        paired = [r for r in results if r['seed'] == seed]
        for field in ('initial_weights_sha256', 'batch_plan_sha256', 'input_tokens'):
            if len({r[field] for r in paired}) != 1:
                raise RuntimeError(f'Pairing failed for {field}, seed {seed}')
    progress.update(state='complete', finished_at=time.time())
    write_json(out / 'suite.json', progress)
    write_json(out / 'results.json', {'suite': progress, 'results': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--variant', choices=['baseline', 'plain', 'residual'], default='baseline')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--model', default='models/SmolLM2-135M')
    parser.add_argument('--merge-layer', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval-every', type=int, default=300)
    parser.add_argument('--eval-samples', type=int, default=128)
    parser.add_argument('--test-samples', type=int, default=512)
    parser.add_argument('--output', default='out-tiny-v2-smollm')
    args = parser.parse_args()
    if args.length < 4 or args.length % 2 or min(args.steps, args.batch_size, args.accumulation, args.eval_every) < 1:
        parser.error('Use an even length >=4 and positive training counts.')
    if min(args.eval_samples, args.test_samples) < 2 or args.eval_samples % 2 or args.test_samples % 2:
        parser.error('Evaluation counts must be positive even numbers >=2.')
    suite(args) if args.suite else run(args)
