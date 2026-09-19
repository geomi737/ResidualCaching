# isort: skip_file
from transformers import AutoModelForCausalLM
import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if True:
    from experiments.smollm.sliding_model import SlidingSmolLM
    from experiments.smollm.smollm_model import endpoints



def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory(device, reset=False):
    if device.type != 'cuda':
        return {'peak_allocated_mib': 0, 'peak_reserved_mib': 0}
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20}


@torch.no_grad()
def run_benchmark(args):
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'
    else:
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        args.device = 'cuda'

    device = torch.device(args.device)
    lengths = [256, 1024, 4096, 16384]

    # Load Models
    print("Loading Baseline Model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16).to(device)
    base_model.eval()

    print("Loading Compressed Student Model...")
    student_base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16)
    if args.merge_layer == -1:
        args.merge_layer = student_base.config.num_hidden_layers // 2
    student_model = SlidingSmolLM(student_base, variant=args.variant, merge_layer=args.merge_layer).to(device)

    # Optionally load adapted weights
    ckpt_path = Path(args.output) / f'checkpoint_{args.variant}.pt'
    if ckpt_path.exists():
        print(f"Loading weights from {ckpt_path}")
        student_model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False)['model_state_dict'])
    student_model.eval()

    results = {
        'schema_version': 1,
        'experiment': 'smollm-retrain-for-context-compression',
        'variant': args.variant,
        'repeats': args.repeats,
        'device': str(device),
        'measurements': [],
    }

    def measure_model(model_name, model_obj, is_sliding):
        for ctx_len in lengths:
            print(f"\nBenchmarking {model_name} @ Context Length {ctx_len}")
            # Dummy inputs
            x = torch.randint(0, 1000, (1, ctx_len), device=device)

            # Warmup Prefill
            for _ in range(2):
                if is_sliding:
                    logits, state = model_obj.prefill(x)
                else:
                    out = model_obj(x, use_cache=True)
                    logits = out.logits
                    state = out.past_key_values

            synchronize(device)
            memory(device, reset=True)

            # Measure Prefill
            prefill_times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                if is_sliding:
                    model_obj.prefill(x)
                else:
                    model_obj(x, use_cache=True)
                synchronize(device)
                prefill_times.append(time.perf_counter() - start)

            p_mean = np.mean(prefill_times)
            mem = memory(device)
            print(f"Prefill Latency: {p_mean:.4f}s | Peak VRAM: {mem['peak_allocated_mib']:.1f} MB")

            # Recompute state for decode
            if is_sliding:
                logits, state = model_obj.prefill(x)
                cache_bytes = sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel()
                                  * layer.values.element_size() for layer in state.cache.layers)
            else:
                out = model_obj(x, use_cache=True)
                state = out.past_key_values
                cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in state)

            print(f"KV Cache Size: {cache_bytes / 1024**2:.2f} MB")

            # Measure Decode (generate 128 tokens)
            decode_len = 128
            synchronize(device)
            decode_start = time.perf_counter()

            token = torch.tensor([[100]], device=device)
            for _ in range(decode_len):
                if is_sliding:
                    logits, state = model_obj.decode(token, state)
                else:
                    out = model_obj(token, past_key_values=state, use_cache=True)
                    state = out.past_key_values

            synchronize(device)
            decode_time = time.perf_counter() - decode_start
            decode_tokens_per_second = decode_len / decode_time
            print(f"Decode {decode_len} tokens: {decode_time:.4f}s ({decode_tokens_per_second:.1f} tok/s)")
            results['measurements'].append({
                'model': model_name,
                'context_length': ctx_len,
                'prefill_seconds_mean': float(p_mean),
                'decode_tokens_per_second': float(decode_tokens_per_second),
                'cache_mib': float(cache_bytes / 1024**2),
                **mem,
            })

    measure_model("Original Baseline", base_model, is_sliding=False)
    measure_model(f"Sliding ({args.variant})", student_model, is_sliding=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'benchmark_{args.variant}.json'
    path.write_text(json.dumps(results, indent=2) + '\\n')
    print(f"Wrote benchmark evidence to {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/SmolLM2-135M')
    parser.add_argument('--variant', choices=['plain', 'residual'], default='plain')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--merge-layer', type=int, default=-1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='experiments/smollm-retrain-for-context-compression/results')
    args = parser.parse_args()
    run_benchmark(args)
