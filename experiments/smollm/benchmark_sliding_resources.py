"""Measure isolated checkpoint inference speed and PyTorch GPU memory together."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.smollm.benchmark_sliding_speed import load_model, aggregate, VARIANTS
from experiments.smollm.train_sliding import save


def memory():
    torch.cuda.synchronize()
    return {'allocated_mib':torch.cuda.memory_allocated()/2**20,
            'reserved_mib':torch.cuda.memory_reserved()/2**20,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}


def kv(state):
    return {'kv_mib':sum(t.numel()*t.element_size() for layer in state.cache.layers
                        for t in (layer.keys,layer.values))/2**20,
            'layer_cache_lengths':[layer.get_seq_length() for layer in state.cache.layers]}


@torch.inference_mode()
def worker(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    torch.set_num_threads(4)
    model=load_model(Path(args.checkpoints)/f'seed{args.seed}'/f'{args.variant}-{args.seed}.pt',torch.device('cuda'))
    result={'variant':args.variant,'config':vars(args),'state':'running','gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'parameter_mib':sum(p.numel()*p.element_size() for p in model.parameters())/2**20,
            'weights_only':memory(),'samples':[]}
    test=np.memmap('data/smollm/test.bin',dtype=np.uint16,mode='r')
    ids=torch.tensor(np.array(test[:args.length],dtype=np.int64),device='cuda')[None,:]
    result['prompt_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    path=Path(args.output)/f'{args.variant}.json'
    save(path,result)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(2):
            logits,state=model.prefill(ids)
            for _ in range(32):
                logits,state=model.decode(logits[:,-1].argmax(-1,keepdim=True),state)
            del logits,state
        gc.collect()
        torch.cuda.synchronize()
        # Keep the warmed allocator; record reserved memory rather than hiding it.
        for repeat in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.perf_counter()
            logits,state=model.prefill(ids)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            result['samples'].append(dict(mode='prefill',repeat=repeat,tokens=args.length,seconds=seconds,**memory(),**kv(state)))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(args.decode_tokens):
                logits,state=model.decode(logits[:,-1].argmax(-1,keepdim=True),state)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            result['samples'].append(dict(mode='decode',repeat=repeat,tokens=args.decode_tokens,seconds=seconds,**memory(),**kv(state)))
            del logits,state
            save(path,result)
    result['summary']={}
    for mode in ('prefill','decode'):
        samples=[s for s in result['samples'] if s['mode']==mode]
        result['summary'][mode]=dict(aggregate(samples),
            peak_allocated_mib=max(s['peak_allocated_mib'] for s in samples),
            peak_reserved_mib=max(s['peak_reserved_mib'] for s in samples),
            end_allocated_mib=statistics.mean(s['allocated_mib'] for s in samples),
            kv_mib=samples[-1]['kv_mib'],layer_cache_lengths=samples[-1]['layer_cache_lengths'])
    result['state']='complete'
    save(path,result)


def run(args):
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'suite.json').exists():
        raise RuntimeError('Choose a fresh output directory.')
    suite={'state':'running','completed':0,'config':vars(args), 'runs':list(VARIANTS)}
    save(root/'suite.json',suite)
    records=[]
    try:
        for variant in VARIANTS:
            print('Starting isolated process:',variant,flush=True)
            command=[sys.executable,__file__,'--variant',variant,'--seed',str(args.seed),
                     '--length',str(args.length),'--repeats',str(args.repeats),
                     '--decode-tokens',str(args.decode_tokens),'--output',str(root),
                     '--checkpoints',args.checkpoints]
            subprocess.run(command,check=True)
            records.append(json.loads((root/f'{variant}.json').read_text()))
            suite['completed']=len(records)
            save(root/'suite.json',suite)
        if len({r['prompt_sha256'] for r in records})!=1:
            raise RuntimeError('Prompts differ.')
        save(root/'results.json',{'records':records,'protocol':'One variant per fresh process; CUDA; batch 1; FP32 weights, BF16 autocast, SDPA; '
             'two warmup sequences, six repeats by default; warmed allocator; no optimizer. Allocated/reserved are PyTorch memory, '
             'excluding CUDA context, driver and other processes. Decode peaks include the existing prompt KV cache.'})
        lines=['# Isolated 128K inference: speed and GPU memory','',
               f'Context {args.length}; seed {args.seed}; {args.repeats} repeats; {args.decode_tokens} generated tokens. '
               'Each variant runs in a fresh process. FP32 weights, BF16 autocast, SDPA, batch 1. Two warmup sequences; allocator stays warm.', '',
               '| Variant | Parameters MiB | Weights-only allocated MiB | Prefill s | Prefill peak allocated / reserved MiB | Decode tok/s | Decode peak allocated / reserved MiB | Final KV MiB |',
               '|---|---:|---:|---:|---|---:|---|---:|']
        for r in records:
            p,d=r['summary']['prefill'],r['summary']['decode']
            lines.append(f"| {r['variant']} | {r['parameter_mib']:.2f} | {r['weights_only']['allocated_mib']:.2f} | {p['mean_seconds']:.4f} | "
                         f"{p['peak_allocated_mib']:.2f} / {p['peak_reserved_mib']:.2f} | {d['tokens_per_second']:.2f} | "
                         f"{d['peak_allocated_mib']:.2f} / {d['peak_reserved_mib']:.2f} | {d['kv_mib']:.2f} |")
        lines+=['','## Differences from baseline','','| Variant | Prefill speed | Decode speed | Prefill allocated peak | Decode allocated peak | Final KV |',
                '|---|---:|---:|---:|---:|---:|']
        b=records[0]
        for r in records[1:]:
            p,d=r['summary']['prefill'],r['summary']['decode'];bp,bd=b['summary']['prefill'],b['summary']['decode']
            diffs=[100*(p['tokens_per_second']/bp['tokens_per_second']-1),100*(d['tokens_per_second']/bd['tokens_per_second']-1),
                   100*(p['peak_allocated_mib']/bp['peak_allocated_mib']-1),100*(d['peak_allocated_mib']/bd['peak_allocated_mib']-1),100*(d['kv_mib']/bd['kv_mib']-1)]
            lines.append('| '+r['variant']+' | '+' | '.join(f'{x:+.2f}%' for x in diffs)+' |')
        lines+=['','Memory peaks include weights and live tensors, not only attention or KV. Reserved memory includes allocator caching and is not added to allocated. '
                'CUDA context, driver, and other processes are excluded. Peaks are maxima over repeats; speed is total tokens / total time. '
                'Memory and speed are measured together, without mixing shared-residency timings from earlier tests. '
                'Long-context quality is not evaluated; one seed and sequential variant order limit generalization.','']
        (root/'comparison.md').write_text('\n'.join(lines))
        suite['state']='complete'
    except Exception as error:
        suite.update(state='failed',error=str(error))
        raise
    finally:
        save(root/'suite.json',suite)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--checkpoints',default='out-sliding-three-seeds')
    parser.add_argument('--output',default='out-sliding-resources-128k')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--length',type=int,default=131072)
    parser.add_argument('--repeats',type=int,default=6)
    parser.add_argument('--decode-tokens',type=int,default=128)
    args=parser.parse_args()
    if min(args.length,args.repeats,args.decode_tokens)<1 or args.repeats<2:
        parser.error('Positive lengths and at least two repetitions required.')
    worker(args) if args.variant else run(args)
