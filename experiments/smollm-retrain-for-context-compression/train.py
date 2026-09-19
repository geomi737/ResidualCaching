# isort: skip_file
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if True:
    from experiments.smollm.sliding_model import SlidingSmolLM
    from experiments.smollm.smollm_model import endpoints

SHARED_STATE = {
    'stop_requested': False,
    'status': 'starting',
    'step': 0,
    'adaptation_tokens': 0,
    'elapsed_time': 0,
    'train_loss': 0.0,
    'lm_loss': 0.0,
    'kd_loss': 0.0,
    'weighted_lm': 0.0,
    'weighted_kd': 0.0,
    'val_loss': 0.0,
    'val_acc': 0.0,
    'val_ppl': 0.0,
    'teacher_loss': 0.0,
    'teacher_acc': 0.0,
    'teacher_ppl': 0.0,
    'gap_acc': 0.0,
    'kl_div': 0.0,
    'argmax_agreement': 0.0,
    'teacher_gap': 0.0,
    'lr': 0.0,
    'tokens_per_sec': 0.0,
    'gpu_alloc_mb': 0,
    'gpu_res_mb': 0,
    'last_ckpt': 'None',
    'next_eval': 0,
    'history': []
}


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_path = Path(__file__).parent / 'dashboard.html'
            if html_path.exists():
                self.wfile.write(html_path.read_bytes())
            else:
                self.wfile.write(b'Dashboard HTML not found.')
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(SHARED_STATE).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/stop':
            SHARED_STATE['stop_requested'] = True
            SHARED_STATE['status'] = 'stopping'
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'stopping'}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # disable logging to stdout


def start_server(port=8080):
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def memory(device, reset=False):
    if device.type != 'cuda':
        return {'peak_allocated_mib': 0, 'peak_reserved_mib': 0}
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20}


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()


def batch(data, starts, length, device):
    ids = torch.from_numpy(np.stack([data[s:s + length + 1] for s in starts]).astype(np.int64)).to(device)
    return ids[:, :-1], ids[:, 1:]


@torch.no_grad()
def evaluate(model, teacher_model, data, args, device, seed):
    model.eval()
    rng = np.random.default_rng(seed)

    totals = {
        'student': {'loss': 0., 'correct': 0, 'count': 0, 'kd_loss': 0.},
        'teacher': {'loss': 0., 'correct': 0, 'count': 0}
    }
    agreement = 0
    kl = 0.
    count = 0

    for _ in range(args.eval_batches):
        starts = rng.integers(0, len(data) - args.length, args.batch_size)
        x, y = batch(data, starts, args.length, device)

        with autocast(device):
            teacher_out = teacher_model(x)
            teacher_logits = teacher_out.logits

            # evaluate student in sliding mode for fair density representation, or disjoint?
            # User said: "Sliding: validation loss/perplexity/accuracy ... Sliding+Residual: ..."
            # For validation, we should evaluate the model's performance on standard causal prediction.
            # Using disjoint mode tests the actual inference setup.
            if args.variant == 'control':
                student_logits = model(x).logits
                ratio = 2
            else:
                student_logits, _ = model(x, y, representation_mode='disjoint')
                ratio = model.ratio

        # Common indices
        common_indices = endpoints(args.length, ratio, device)
        labels = y[:, common_indices]

        # Teacher eval on common indices
        teacher_common = teacher_logits[:, common_indices]
        t_loss = F.cross_entropy(teacher_common.float().flatten(0, 1), labels.flatten(), reduction='sum').item()
        t_corr = (teacher_common.argmax(-1) == labels).sum().item()

        # Student eval
        if args.variant == 'control':
            s_common = student_logits[:, common_indices]
        else:
            s_common = student_logits
        # student_logits from disjoint mode has shape [batch, length//2 + rem, vocab]

        s_loss = F.cross_entropy(s_common.float().flatten(0, 1), labels.flatten(), reduction='sum').item()
        s_corr = (s_common.argmax(-1) == labels).sum().item()

        # KD Loss on common
        t_probs = F.softmax(teacher_common.float() / args.temperature, dim=-1)
        s_logprobs = F.log_softmax(s_common.float() / args.temperature, dim=-1)
        kd_l = F.kl_div(s_logprobs, t_probs, reduction='sum').item() * (args.temperature ** 2)

        totals['student']['loss'] += s_loss
        totals['student']['correct'] += s_corr
        totals['student']['count'] += labels.numel()
        totals['student']['kd_loss'] += kd_l

        totals['teacher']['loss'] += t_loss
        totals['teacher']['correct'] += t_corr
        totals['teacher']['count'] += labels.numel()

        # Agreement on last token
        a = s_common[:, -1]
        b = teacher_common[:, -1]
        agreement += (a.argmax(-1) == b.argmax(-1)).sum().item()
        kl += F.kl_div(F.log_softmax(a.float(), -1), F.softmax(b.float(), -1), reduction='sum').item()
        count += x.shape[0]

    for m in totals:
        totals[m]['token_loss'] = totals[m].pop('loss') / totals[m]['count']
        totals[m]['token_accuracy'] = totals[m].pop('correct') / totals[m]['count']
        totals[m]['perplexity'] = math.exp(totals[m]['token_loss']) if totals[m]['token_loss'] < 100 else float('inf')
        if m == 'student':
            totals[m]['kd_loss'] = totals[m].pop('kd_loss') / totals[m]['count']

    return {
        'student_loss': totals['student']['token_loss'],
        'student_ppl': totals['student']['perplexity'],
        'student_acc': totals['student']['token_accuracy'],
        'student_kd_loss': totals['student']['kd_loss'],
        'teacher_loss': totals['teacher']['token_loss'],
        'teacher_ppl': totals['teacher']['perplexity'],
        'teacher_acc': totals['teacher']['token_accuracy'],
        'gap_loss': totals['student']['token_loss'] - totals['teacher']['token_loss'],
        'last_token_agreement': agreement / count,
        'last_token_kl': kl / count
    }


def save_checkpoint(path, model, optimizer, scheduler, args, step, tokens, history, rng_states):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'args': vars(args),
        'step': step,
        'adaptation_tokens': tokens,
        'history': history,
        'rng_states': rng_states
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    torch.set_rng_state(checkpoint['rng_states']['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['rng_states']['cuda'])
    np.random.set_state(checkpoint['rng_states']['numpy'])
    random.setstate(checkpoint['rng_states']['python'])

    return checkpoint['step'], checkpoint['adaptation_tokens'], checkpoint['history']


def run(args):
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Please ensure PyTorch is installed with CUDA support.")
        print("Falling back to CPU, but this will be extremely slow.")
        args.device = 'cpu'
    else:
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        args.device = 'cuda'

    device = torch.device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train = np.memmap(Path(args.data) / 'train.bin', dtype=np.uint16, mode='r')
    val = np.memmap(Path(args.data) / 'validation.bin', dtype=np.uint16, mode='r')
    test = np.memmap(Path(args.data) / 'test.bin', dtype=np.uint16, mode='r')

    # Init Teacher
    print("Loading Teacher Model...")
    teacher_base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    teacher_base.to(device)
    teacher_base.eval()
    for p in teacher_base.parameters():
        p.requires_grad = False

    # Init Student
    print(f"Loading Student Model (Variant: {args.variant})...")
    student_base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    if args.variant == 'control':
        model = student_base
    else:
        if args.merge_layer == -1:
            args.merge_layer = student_base.config.num_hidden_layers // 2
        model = SlidingSmolLM(student_base, variant=args.variant, merge_layer=args.merge_layer)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    step = 0
    adaptation_tokens = 0
    history = []

    ckpt_path = out / f'checkpoint_{args.variant}.pt'
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}...")
        step, adaptation_tokens, history = load_checkpoint(ckpt_path, model, optimizer, None)
        SHARED_STATE['history'] = history
        print(f"Resumed at step {step}, tokens {adaptation_tokens}")
    else:
        # Seed initialization
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Setup eval intervals
    eval_intervals = [
        0,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        12_500_000,
        15_000_000,
        20_000_000,
        25_000_000,
        30_000_000,
        40_000_000,
        50_000_000,
        75_000_000,
        100_000_000
    ]
    next_eval_idx = 0
    for i, t in enumerate(eval_intervals):
        if adaptation_tokens < t:
            next_eval_idx = i
            break

    SHARED_STATE['status'] = 'running'
    start_time = time.time()
    record_metadata = {
        'schema_version': 1,
        'experiment': 'smollm-retrain-for-context-compression',
        'variant': args.variant,
        'seed': args.seed,
    }

    # Initial Eval (Token 0)
    if adaptation_tokens == 0:
        print("Running initial evaluation (Token 0)...")
        eval_res = evaluate(model, teacher_base, val, args, device, 777)
        eval_res.update(record_metadata)
        eval_res['adaptation_tokens'] = 0
        eval_res['step'] = 0
        history.append(eval_res)
        with open(out / f'eval_metrics_{args.variant}.jsonl', 'a') as f:
            f.write(json.dumps(eval_res) + '\n')

        SHARED_STATE['val_loss'] = eval_res['student_loss']
        SHARED_STATE['val_acc'] = eval_res['student_acc']
        SHARED_STATE['teacher_gap'] = eval_res['gap_loss']
        SHARED_STATE['history'].append(eval_res)
        next_eval_idx = 1

    rng = np.random.default_rng(args.seed + step)

    print("Starting adaptation loop...")
    while step < args.steps:
        if SHARED_STATE['stop_requested']:
            print("Graceful stop requested via Web UI.")
            break

        SHARED_STATE['next_eval'] = eval_intervals[min(next_eval_idx, len(eval_intervals) - 1)]

        # Linear decay LR for simplicity
        lr = args.lr * (1.0 - step / args.steps)
        for group in optimizer.param_groups:
            group['lr'] = lr

        model.train()
        optimizer.zero_grad()

        batch_start_time = time.perf_counter()

        loss_sum = 0.
        lm_loss_sum = 0.
        kd_loss_sum = 0.
        batch_tokens = 0

        for _ in range(args.accumulation):
            starts = rng.integers(0, len(train) - args.length, args.batch_size)
            x, y = batch(train, starts, args.length, device)

            with autocast(device):
                # Teacher forward
                with torch.no_grad():
                    t_out = teacher_base(x)
                    t_logits = t_out.logits

                # Student forward (Sliding Mode during training)
                if args.variant == 'control':
                    s_logits = model(x).logits
                    lm_loss = F.cross_entropy(s_logits.float().flatten(0, 1), y.flatten(), reduction='mean')
                else:
                    s_logits, lm_loss = model(x, y, representation_mode='sliding')

                # Knowledge Distillation Loss
                t_probs = F.softmax(t_logits.flatten(0, 1) / args.temperature, dim=-1)
                s_logprobs = F.log_softmax(s_logits.flatten(0, 1) / args.temperature, dim=-1)

                # kl_div batchmean computes mean over first dimension, so flattening batch and seq_len gives per-token KD loss
                kd_loss = F.kl_div(s_logprobs, t_probs, reduction='batchmean') * (args.temperature ** 2)

            loss = (args.lm_weight * lm_loss + args.kd_weight * kd_loss) / args.accumulation

            loss.backward()

            loss_sum += loss.item() * args.accumulation
            lm_loss_sum += lm_loss.item()
            kd_loss_sum += kd_loss.item()
            batch_tokens += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        adaptation_tokens += batch_tokens

        duration = time.perf_counter() - batch_start_time

        # Update Web UI State
        SHARED_STATE['step'] = step
        SHARED_STATE['adaptation_tokens'] = adaptation_tokens
        SHARED_STATE['elapsed_time'] = time.time() - start_time
        SHARED_STATE['train_loss'] = loss_sum / args.accumulation
        SHARED_STATE['lm_loss'] = lm_loss_sum / args.accumulation
        SHARED_STATE['kd_loss'] = kd_loss_sum / args.accumulation
        SHARED_STATE['weighted_lm'] = (lm_loss_sum / args.accumulation) * args.lm_weight
        SHARED_STATE['weighted_kd'] = (kd_loss_sum / args.accumulation) * args.kd_weight
        SHARED_STATE['lr'] = lr
        SHARED_STATE['tokens_per_sec'] = batch_tokens / duration

        mem = memory(device)
        SHARED_STATE['gpu_alloc_mb'] = mem['peak_allocated_mib']
        SHARED_STATE['gpu_res_mb'] = mem['peak_reserved_mib']

        # Log to file
        with open(out / f'training_metrics_{args.variant}.jsonl', 'a') as f:
            f.write(json.dumps({
                **record_metadata,
                'step': step,
                'adaptation_tokens': adaptation_tokens,
                'total_loss': loss_sum,
                'lm_loss': lm_loss_sum / args.accumulation,
                'kd_loss': kd_loss_sum / args.accumulation,
                'weighted_lm': (lm_loss_sum / args.accumulation) * args.lm_weight,
                'weighted_kd': (kd_loss_sum / args.accumulation) * args.kd_weight,
                'lr': lr,
                'tokens_per_sec': batch_tokens / duration
            }) + '\n')

        # Evaluation check
        if next_eval_idx < len(eval_intervals) and adaptation_tokens >= eval_intervals[next_eval_idx]:
            print(f"Running evaluation at {adaptation_tokens} tokens...")
            eval_res = evaluate(model, teacher_base, val, args, device, 777)
            eval_res.update(record_metadata)
            eval_res['adaptation_tokens'] = adaptation_tokens
            eval_res['step'] = step
            history.append(eval_res)

            with open(out / f'eval_metrics_{args.variant}.jsonl', 'a') as f:
                f.write(json.dumps(eval_res) + '\n')

            SHARED_STATE['val_loss'] = eval_res['student_loss']
            SHARED_STATE['val_acc'] = eval_res['student_acc']
            SHARED_STATE['val_ppl'] = eval_res['student_ppl']
            SHARED_STATE['teacher_loss'] = eval_res['teacher_loss']
            SHARED_STATE['teacher_acc'] = eval_res['teacher_acc']
            SHARED_STATE['teacher_ppl'] = eval_res['teacher_ppl']
            SHARED_STATE['teacher_gap'] = eval_res['gap_loss']
            SHARED_STATE['gap_acc'] = eval_res['teacher_acc'] - eval_res['student_acc']
            SHARED_STATE['kl_div'] = eval_res['last_token_kl']
            SHARED_STATE['argmax_agreement'] = eval_res['last_token_agreement']
            SHARED_STATE['history'].append(eval_res)

            # Checkpoint at eval
            rng_states = {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'python': random.getstate()
            }
            save_checkpoint(ckpt_path, model, optimizer, None, args, step, adaptation_tokens, history, rng_states)

            # Milestone Checkpoint
            if adaptation_tokens > 0:
                target = eval_intervals[min(next_eval_idx, len(eval_intervals) - 1)]
                m_val = target / 1_000_000
                if m_val.is_integer():
                    m_str = f"{int(m_val)}M"
                else:
                    m_str = f"{m_val:g}M"
                mile_path = out / f"{args.variant.capitalize()}-{m_str}.pt"
                save_checkpoint(mile_path, model, optimizer, None, args, step, adaptation_tokens, history, rng_states)
                print(f"Saved milestone checkpoint to {mile_path}")
            SHARED_STATE['last_ckpt'] = f"step {step}, {adaptation_tokens} tokens"
            print(f"Saved checkpoint to {ckpt_path}")

            next_eval_idx += 1

    # Final save
    print("Training loop ended. Saving final checkpoint...")
    rng_states = {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate()
    }
    save_checkpoint(ckpt_path, model, optimizer, None, args, step, adaptation_tokens, history, rng_states)

    # Milestone Checkpoint
    if adaptation_tokens > 0:
        target = eval_intervals[min(next_eval_idx, len(eval_intervals) - 1)]
        m_val = target / 1_000_000
        if m_val.is_integer():
            m_str = f"{int(m_val)}M"
        else:
            m_str = f"{m_val:g}M"
        mile_path = out / f"{args.variant.capitalize()}-{m_str}.pt"
        save_checkpoint(mile_path, model, optimizer, None, args, step, adaptation_tokens, history, rng_states)
        print(f"Saved milestone checkpoint to {mile_path}")
    SHARED_STATE['status'] = 'stopped'
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/SmolLM2-135M')
    parser.add_argument('--data', default='data/smollm')
    parser.add_argument('--output', default='experiments/smollm-retrain-for-context-compression/results')
    parser.add_argument('--variant', choices=['plain', 'residual', 'control'], default='plain')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--merge-layer', type=int, default=-1)

    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--lm-weight', type=float, default=0.7)
    parser.add_argument('--kd-weight', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=2.0)

    parser.add_argument('--eval-batches', type=int, default=32)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args()

    print(f"Starting Web UI Dashboard on http://localhost:{args.port}/")
    start_server(args.port)

    run(args)
