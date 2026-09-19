"""Regenerate published speed sweeps and isolated 128K memory figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]/'results'
VARIANTS=('baseline','sliding','sliding_residual')
LABELS=('Baseline','Sliding','Sliding + R')
COLORS=('#4779aa','#328967','#bd567d')


def save(fig,root,name):
    root.mkdir(parents=True,exist_ok=True)
    fig.savefig(root/f'{name}.png',dpi=180)
    fig.savefig(root/f'{name}.svg')
    plt.close(fig)
    p=root/f'{name}.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')


def main():
    samples=[]
    for p in (ROOT/'speed').glob('*/speed-benchmark.json'):
        data=json.loads(p.read_text())
        assert data['state']=='complete'
        samples.extend(data['samples'])
    lengths=sorted({s['context'] for s in samples})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,mode in zip(axes,('prefill','decode')):
        for v,label,color in zip(VARIANTS[1:],LABELS[1:],COLORS[1:]):
            deltas=[]
            for n in lengths:
                rates={}
                for variant in (v,'baseline'):
                    events=[s for s in samples if s['context']==n and s['mode']==mode and s['variant']==variant]
                    rates[variant]=sum(s['tokens'] for s in events)/sum(s['seconds'] for s in events)
                deltas.append(100*(rates[v]/rates['baseline']-1))
            ax.plot(lengths,deltas,marker='o',label=label,color=color)
        ax.axhline(0,color='gray',linewidth=1)
        ax.set_xscale('log',base=2)
        ax.set_xticks(lengths,[str(n//1024)+'K' if n>=1024 else str(n) for n in lengths],rotation=35)
        ax.set_xlabel('Prompt tokens');ax.set_ylabel('Throughput difference from baseline (%)')
        ax.set_title(mode.title());ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Repeated GPU speed: shared model residency\n256/1024: three seeds, 9 repeats; longer prompts: seed 17, 6 repeats',fontsize=13)
    save(fig,ROOT/'speed/figures','context_sweep')
    records=json.loads((ROOT/'memory/128k_isolated/results.json').read_text())['records']
    by={r['variant']:r for r in records}
    panels=[('Prefill allocated peak (MiB)',lambda r:r['summary']['prefill']['peak_allocated_mib']),
            ('Decode allocated peak (MiB)',lambda r:r['summary']['decode']['peak_allocated_mib']),
            ('Final persistent KV (MiB)',lambda r:r['summary']['decode']['kv_mib']),
            ('Reserved peak (MiB)',lambda r:r['summary']['prefill']['peak_reserved_mib']),
            ('Prefill time (seconds)',lambda r:r['summary']['prefill']['mean_seconds']),
            ('Decode throughput (tokens/s)',lambda r:r['summary']['decode']['tokens_per_second'])]
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,(title,get) in zip(axes.flat,panels):
        values=[get(by[v]) for v in VARIANTS]
        bars=ax.bar(LABELS,values,color=COLORS)
        ax.bar_label(bars,labels=[f'{x:.2f}' for x in values],padding=4)
        ax.set_ylim(0,max(values)*1.18);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('128K speed and memory measured together\nOne model per fresh process; seed 17; six warmed repetitions',fontsize=14)
    save(fig,ROOT/'memory/128k_isolated/figures','resources')


if __name__=='__main__':
    main()
