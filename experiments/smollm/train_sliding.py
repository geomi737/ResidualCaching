"""Paired random-initialized experiments with dual-mode quality reports."""
import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.smollm.sliding_model import SlidingSmolLM
from experiments.smollm.smollm_model import endpoints

VARIANTS = ('baseline', 'disjoint', 'sliding', 'disjoint_residual', 'sliding_residual')


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory(device, reset=False):
    if device.type != 'cuda':
        return {'peak_allocated_mib': None, 'peak_reserved_mib': None}
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20}


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()


def batch(data, starts, length, device):
    ids = torch.from_numpy(np.stack([data[s:s+length+1] for s in starts]).astype(np.int64)).to(device)
    return ids[:, :-1], ids[:, 1:]


@torch.no_grad()
def evaluate(model, data, args, device, seed):
    model.eval()  # Hold dropout fixed while comparing representation modes.
    rng = np.random.default_rng(seed)
    totals = {mode: {'loss': 0., 'correct': 0, 'count': 0, 'common_loss': 0.,
                     'common_correct': 0, 'common_count': 0,
                     'parity': {p: {'loss': 0., 'correct': 0, 'count': 0} for p in ('even', 'odd')}}
              for mode in ('sliding', 'disjoint')}
    agreement, kl, count = 0, 0., 0
    for parity, length in [('even', args.length), ('odd', args.length-1)]:
        for _ in range(args.eval_batches):
            starts = rng.integers(0, len(data)-length, args.batch_size)
            x, y = batch(data, starts, length, device)
            common_indices = endpoints(length, 2, device)
            outputs = {}
            for mode in totals:
                with autocast(device):
                    logits, _ = model(x, representation_mode=mode)
                positions = (torch.arange(length, device=device) if mode == 'sliding' or model.ratio == 1
                             else common_indices)
                labels = y.index_select(1, positions)
                common = logits[:, common_indices] if mode == 'sliding' or model.ratio == 1 else logits
                record = totals[mode]
                record['loss'] += F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction='sum').item()
                record['correct'] += (logits.argmax(-1) == labels).sum().item()
                record['count'] += labels.numel()
                record['common_loss'] += F.cross_entropy(common.float().flatten(0, 1), y[:, common_indices].flatten(), reduction='sum').item()
                record['common_correct'] += (common.argmax(-1) == y[:, common_indices]).sum().item()
                record['common_count'] += y[:, common_indices].numel()
                p = record['parity'][parity]
                p['loss'] += F.cross_entropy(logits[:, -1].float(), y[:, -1], reduction='sum').item()
                p['correct'] += (logits[:, -1].argmax(-1) == y[:, -1]).sum().item()
                p['count'] += y.shape[0]
                outputs[mode] = logits[:, -1].float()
            a, b = outputs['sliding'], outputs['disjoint']
            agreement += (a.argmax(-1) == b.argmax(-1)).sum().item()
            kl += F.kl_div(b.log_softmax(-1), a.softmax(-1), reduction='sum').item()
            count += x.shape[0]
    for record in totals.values():
        record['token_loss'] = record.pop('loss') / record['count']
        record['token_accuracy'] = record.pop('correct') / record['count']
        record['common_boundary_loss'] = record.pop('common_loss') / record['common_count']
        record['common_boundary_accuracy'] = record.pop('common_correct') / record['common_count']
        for p in record['parity'].values():
            p['prefix_loss'] = p.pop('loss') / p['count']
            p['prefix_accuracy'] = p.pop('correct') / p['count']
    totals['mode_gap'] = {'last_token_argmax_agreement': agreement / count,
                          'kl_sliding_to_disjoint': kl / count, 'prefixes': count}
    return totals


@torch.no_grad()
def benchmark(model, data, args, device):
    model.eval()
    x, _ = batch(data, [0], args.length, device)
    result = {}
    for mode in ('sliding', 'disjoint'):
        with autocast(device):
            model(x, representation_mode=mode)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        memory(device, reset=True)
        synchronize(device)
        start = time.perf_counter()
        with autocast(device):
            for _ in range(args.bench_repeats):
                logits, _ = model(x, representation_mode=mode)
                del logits
        synchronize(device)
        duration = (time.perf_counter()-start) / args.bench_repeats
        result[mode + '_full_forward'] = dict(memory(device), seconds=duration,
                                              input_tokens_per_second=x.numel()/duration)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    memory(device, reset=True)
    synchronize(device)
    start = time.perf_counter()
    with autocast(device):
        logits, state = model.prefill(x)
    synchronize(device)
    result['cached_prefill'] = dict(memory(device), seconds=time.perf_counter()-start)
    del logits, state
    with autocast(device):
        logits, state = model.prefill(x)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    memory(device, reset=True)
    synchronize(device)
    start = time.perf_counter()
    with autocast(device):
        for _ in range(args.decode_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            logits, state = model.decode(token, state)
    synchronize(device)
    duration = time.perf_counter()-start
    cache_bytes = sum(layer.keys.numel()*layer.keys.element_size() + layer.values.numel()*layer.values.element_size()
                      for layer in state.cache.layers)
    result['cached_decode'] = dict(memory(device), seconds=duration,
                                  tokens_per_second=args.decode_tokens/duration,
                                  kv_cache_mib=cache_bytes/2**20,
                                  layer_cache_lengths=[layer.get_seq_length() for layer in state.cache.layers])
    return result


def run(args):
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires an available CUDA GPU; CPU fallback is disabled.')
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        # Correctness smoke only: random tokens do not establish language quality.
        data = np.random.default_rng(args.seed).integers(0, 97, 8192, dtype=np.uint16)
        train, val, test = data[:4096], data[4096:6144], data[6144:]
    else:
        train, val, test = [np.memmap(Path(args.data)/f'{name}.bin', dtype=np.uint16, mode='r')
                            for name in ('train', 'validation', 'test')]
    for data in (train, val, test):
        if len(data) <= args.length:
            raise ValueError('Every data split must exceed context length.')
    data_hashes = {name: hashlib.sha256(data.tobytes()).hexdigest()
                   for name, data in zip(('train', 'validation', 'test'), (train, val, test))}
    results = []
    suite = {'state': 'running', 'config': vars(args), 'completed': 0,
             'runs': [{'variant': v, 'seed': args.seed, 'file': f'{v}-{args.seed}.json'}
                      for v in args.variants]}
    save(out/'suite.json', suite)
    for variant in args.variants:
        torch.manual_seed(args.seed)
        merge_variant = 'baseline' if variant == 'baseline' else ('residual' if variant.endswith('_residual') else 'plain')
        if args.synthetic:
            from transformers import LlamaConfig, LlamaForCausalLM
            config = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                 attention_dropout=0)
            config._attn_implementation = 'sdpa'
            model = SlidingSmolLM(LlamaForCausalLM(config), merge_variant, args.merge_layer)
        else:
            from transformers import AutoConfig, AutoModelForCausalLM
            config = AutoConfig.from_pretrained(args.model, local_files_only=True)
            config.num_hidden_layers = 8
            config.hidden_size = 288
            config.intermediate_size = 768
            config.num_attention_heads = 6
            config.num_key_value_heads = 3
            base = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa', dtype=torch.float32)
            model = SlidingSmolLM(base, merge_variant, args.merge_layer)
        model.to(device)
        base_hash = hashlib.sha256()
        for parameter in model.base.parameters():
            base_hash.update(parameter.detach().cpu().float().numpy().tobytes())
        record = {'variant': variant, 'config': vars(args), 'synthetic': args.synthetic,
                  'initial_weights_sha256': base_hash.hexdigest(), 'data_sha256': data_hashes,
                  'state': 'running', 'history': [], 'torch': torch.__version__,
                  'device': str(device), 'parameters': sum(p.numel() for p in model.parameters())}
        record['initialization'] = 'random'
        path = out / f'{variant}-{args.seed}.json'
        save(path, record)
        record['initial_validation'] = evaluate(model, val, args, device, 777)
        save(path, record)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
        rng = np.random.default_rng(args.seed)
        plan_hash = hashlib.sha256()
        mode = 'sliding' if variant.startswith('sliding') else 'disjoint'
        input_count = prediction_count = 0
        for step in range(1, args.steps+1):
            lr = args.lr * min(1., step / max(1, int(args.steps*.1))) * (.1 + .9*.5*(1 + np.cos(np.pi*step/args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            length = args.length - step % 2
            memory(device, reset=True)
            synchronize(device)
            start = time.perf_counter()
            loss_sum, correct, predictions = 0., 0, 0
            for _ in range(args.accumulation):
                starts = rng.integers(0, len(train)-args.length, args.batch_size)
                plan_hash.update(starts.tobytes())
                plan_hash.update(length.to_bytes(4, 'little'))
                x, y = batch(train, starts, length, device)
                with autocast(device):
                    logits, loss = model(x, y, representation_mode=mode)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                (loss/args.accumulation).backward()
                positions = torch.arange(length, device=device) if mode == 'sliding' or model.ratio == 1 else endpoints(length, 2, device)
                correct += (logits.detach().argmax(-1) == y[:, positions]).sum().item()
                predictions += logits.shape[0]*logits.shape[1]
                input_count += x.numel()
                loss_sum += loss.item()/args.accumulation
                del logits, loss, x, y
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item()
            optimizer.step()
            synchronize(device)
            duration = time.perf_counter()-start
            prediction_count += predictions
            event = dict(memory(device), step=step, loss=loss_sum, accuracy=correct/predictions,
                         lr=lr,
                         grad_norm=grad_norm, seconds=duration, warmup=step <= 3,
                         input_tokens_per_second=args.batch_size*length*args.accumulation/duration,
                         predictions_per_second=predictions/duration)
            if step % args.eval_every == 0 or step == args.steps:
                event['validation'] = evaluate(model, val, args, device, 777)
            record['history'].append(event)
            save(path, record)
        test_args = argparse.Namespace(**vars(args))
        test_args.eval_batches = args.test_batches
        record.update(input_tokens=input_count, supervised_predictions=prediction_count,
                      batch_plan_sha256=plan_hash.hexdigest(), test=evaluate(model, test, test_args, device, 999))
        torch.save({'model': model.state_dict(), 'config': vars(args), 'variant': variant}, out/f'{variant}-{args.seed}.pt')
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        model.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        record['inference'] = benchmark(model, test, args, device)
        record['merge_weights'] = model.merge_weight.softmax(0).detach().cpu().tolist() if model.ratio > 1 else None
        record['state'] = 'complete'
        save(path, record)
        results.append(record)
        suite['completed'] = len(results)
        save(out/'suite.json', suite)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'{variant}: complete', flush=True)
    for key in ('initial_weights_sha256', 'batch_plan_sha256', 'input_tokens'):
        if len({r[key] for r in results}) != 1:
            raise RuntimeError(f'Paired comparison failed: {key}')
    save(out/'results.json', {'results': results, 'paired_checks_passed': True})
    from experiments.reporting.write_sliding_report import write_report
    write_report(out)
    suite['state'] = 'complete'
    save(out/'suite.json', suite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--from-scratch', action='store_true', default=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['baseline', 'sliding_residual'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', default='models/SmolLM2-135M')
    parser.add_argument('--data', default='data/smollm')
    parser.add_argument('--output', default='out-sliding-scratch-smollm')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--merge-layer', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval-every', type=int, default=300)
    parser.add_argument('--eval-batches', type=int, default=16)
    parser.add_argument('--test-batches', type=int, default=64)
    parser.add_argument('--bench-repeats', type=int, default=3)
    parser.add_argument('--decode-tokens', type=int, default=32)
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error('Variants must not repeat.')
    if 'baseline' not in args.variants:
        parser.error('Include baseline for the paired comparison report.')
    if args.length < 4 or args.length % 2 or min(args.steps, args.batch_size, args.accumulation,
                                               args.eval_every, args.eval_batches, args.test_batches, args.bench_repeats, args.decode_tokens) < 1:
        parser.error('Use even context >=4 and positive counts.')
    try:
        run(args)
    except Exception:
        import traceback
        suite_path = Path(args.output)/'suite.json'
        if suite_path.exists():
            suite = json.loads(suite_path.read_text())
            suite.update(state='failed', error=traceback.format_exc())
            save(suite_path, suite)
        raise
