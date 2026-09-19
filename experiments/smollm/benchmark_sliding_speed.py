"""Repeated, rotated GPU inference timings of saved scratch sliding models."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.smollm.sliding_model import SlidingSmolLM
from experiments.smollm.train_sliding import save

VARIANTS = ('baseline', 'sliding', 'sliding_residual')


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    args = checkpoint['config']
    config = AutoConfig.from_pretrained(args['model'], local_files_only=True)
    for key, value in dict(num_hidden_layers=8, hidden_size=288, intermediate_size=768,
                           num_attention_heads=6, num_key_value_heads=3).items():
        setattr(config, key, value)
    base = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa', dtype=torch.float32)
    variant = checkpoint['variant']
    merge = 'baseline' if variant == 'baseline' else 'residual' if variant.endswith('_residual') else 'plain'
    model = SlidingSmolLM(base, merge, args['merge_layer'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval()


def aggregate(samples):
    times = [s['seconds'] for s in samples]
    tokens = samples[0]['tokens']
    return {'samples': len(samples), 'mean_seconds': statistics.mean(times),
            'median_seconds': statistics.median(times), 'sample_sd_seconds': statistics.stdev(times),
            'tokens_per_second': tokens*len(times)/sum(times),
            'median_tokens_per_second': tokens/statistics.median(times),
            'min_seconds': min(times), 'max_seconds': max(times)}


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required.')
    torch.set_num_threads(4)
    device = torch.device('cuda')
    root = Path(args.output)
    path = root/'speed-benchmark.json'
    if path.exists():
        raise RuntimeError('Existing speed benchmark is preserved; choose a new output directory.')
    root.mkdir(parents=True, exist_ok=True)
    test = np.memmap('data/smollm/test.bin', dtype=np.uint16, mode='r')
    result = {'state':'running', 'config':vars(args), 'torch':torch.__version__,
              'gpu':torch.cuda.get_device_name(), 'protocol': 'FP32 weights, BF16 autocast, SDPA, batch 1; '
              'all three models resident during a seed; prefill and greedy decode timed separately; '
              'CUDA synchronization around wall-clock timings; rotated run order; no training.',
              'samples':[], 'summary':[]}
    save(path, result)
    for seed in args.seeds:
        models = {v:load_model(Path(args.checkpoints)/f'seed{seed}'/f'{v}-{seed}.pt',device) for v in VARIANTS}
        for length in args.lengths:
            ids = torch.tensor(np.array(test[:length],dtype=np.int64),device=device)[None,:]
            prompt_hash=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                for model in models.values():
                    # Warm the same decode workload, not only a single full forward.
                    for _ in range(2):
                        logits,state=model.prefill(ids)
                        for _ in range(args.warmup_tokens):
                            logits,state=model.decode(logits[:,-1].argmax(-1,keepdim=True),state)
                        del logits,state
                torch.cuda.synchronize()
                for repeat in range(args.repeats):
                    order=VARIANTS[repeat%3:]+VARIANTS[:repeat%3]
                    for variant in order:
                        model=models[variant]
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        logits,state=model.prefill(ids)
                        torch.cuda.synchronize()
                        prefill_seconds=time.perf_counter()-start
                        start=time.perf_counter()
                        for _ in range(args.decode_tokens):
                            logits,state=model.decode(logits[:,-1].argmax(-1,keepdim=True),state)
                        torch.cuda.synchronize()
                        decode_seconds=time.perf_counter()-start
                        common={'seed':seed,'variant':variant,'context':length,'repeat':repeat,'prompt_sha256':prompt_hash}
                        result['samples'].extend([dict(common,mode='prefill',tokens=length,seconds=prefill_seconds),
                                                  dict(common,mode='decode',tokens=args.decode_tokens,seconds=decode_seconds)])
                        del logits,state
                    save(path,result)
            for variant in VARIANTS:
                for mode in ('prefill','decode'):
                    samples=[s for s in result['samples'] if s['seed']==seed and s['context']==length and s['variant']==variant and s['mode']==mode]
                    result['summary'].append(dict(seed=seed,context=length,variant=variant,mode=mode,**aggregate(samples)))
            save(path,result)
            print(f'Seed {seed}, context {length}: {args.repeats} rotated rounds complete',flush=True)
        del models,model
        gc.collect()
        torch.cuda.empty_cache()
    result['state']='complete'
    save(path,result)
    lines=['# Repeated GPU inference speed benchmark','',result['protocol'],'',
           f"Seeds {args.seeds}; contexts {args.lengths}; {args.repeats} repetitions; {args.decode_tokens} generated tokens per decode. "
           'Two warmup sequences per model/context. Longer context measures speed only, not quality. '
           'Throughput uses total tokens / total time; individual timings are preserved. '+
           'Models share device residency, so this benchmark does not measure isolated VRAM.', '',
           '| Seed | Context | Mode | Baseline tok/s | Sliding tok/s | vs baseline | Sliding + R tok/s | vs baseline |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for seed in args.seeds:
        for length in args.lengths:
            for mode in ('prefill','decode'):
                rows={s['variant']:s for s in result['summary'] if s['seed']==seed and s['context']==length and s['mode']==mode}
                rates=[rows[v]['tokens_per_second'] for v in VARIANTS]
                lines.append(f'| {seed} | {length} | {mode} | {rates[0]:.1f} | {rates[1]:.1f} | {100*(rates[1]/rates[0]-1):+.2f}% | {rates[2]:.1f} | {100*(rates[2]/rates[0]-1):+.2f}% |')
    lines += ['', 'These repeats reduce short-measurement noise but do not guarantee an idle device or fixed GPU clocks. '
              'Wall time includes Python dispatch and greedy argmax. The earlier training-adjacent one-shot measurements remain unchanged.', '']
    (root/'speed-benchmark.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',default='out-sliding-three-seeds')
    parser.add_argument('--output',default='out-sliding-speed-repeat')
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    parser.add_argument('--lengths',nargs='+',type=int,default=[256,1024])
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--decode-tokens',type=int,default=128)
    parser.add_argument('--warmup-tokens',type=int,default=32)
    args=parser.parse_args()
    if args.repeats<3 or min(args.lengths+[args.decode_tokens,args.warmup_tokens])<1:
        parser.error('Use at least 3 repeats and positive lengths.')
    run(args)
