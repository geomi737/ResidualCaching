"""Plot paired three-seed sliding evidence with individual runs and sample SD."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reporting.sliding_metrics import metrics

VARIANTS = ('baseline', 'sliding', 'sliding_residual')
LABELS = ('Baseline', 'Sliding', 'Sliding + R')
COLORS = ('#4779aa', '#328967', '#bd567d')


def plot(root):
    records = json.loads((root/'results.json').read_text())['results']
    seeds = sorted({r['config']['seed'] for r in records})
    values = {v: {r['config']['seed']: metrics(r) for r in records if r['variant'] == v} for v in VARIANTS}
    panels = [('Test boundary loss', 'Compressed boundary cross-entropy'),
              ('Test boundary accuracy (%)', 'Compressed boundary accuracy (%)'),
              ('Test final-prefix loss', 'Final-prefix cross-entropy'),
              ('cached_decode: tokens_per_second', 'Historical one-shot decode (tokens/s)'),
              ('Training allocated peak (MiB)', 'Training allocated peak (MiB)'),
              ('cached_decode: kv_cache_mib', 'Persistent KV cache (MiB)')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for ax, (key, title) in zip(axes.flat, panels):
        arrays = [[values[v][s][key] for s in seeds] for v in VARIANTS]
        means = [statistics.mean(a) for a in arrays]
        sd = [statistics.stdev(a) for a in arrays]
        ax.bar(range(3), means, yerr=sd, color=COLORS, alpha=.7, capsize=5)
        for j, seed in enumerate(seeds):
            ax.plot(np.arange(3)+(j-1)*.06, [values[v][seed][key] for v in VARIANTS],
                    color='#30343b', alpha=.45, marker=('o','s','^')[j], markersize=4,
                    linewidth=.8, label=f'Seed {seed}')
        ax.set_xticks(range(3), LABELS)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        if key.startswith('Test '):
            flat = [x for a in arrays for x in a]
            gap = max(flat)-min(flat)
            ax.set_ylim(min(flat)-max(.05, gap*.25), max(flat)+max(.05, gap*.25))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Sliding: three fresh seeds, 1500 GPU updates per model\nBars: mean ± sample SD; connected markers: paired individual seeds', fontsize=15)
    destination = root/'figures'
    destination.mkdir(exist_ok=True)
    fig.savefig(destination/'overview.png', dpi=180)
    fig.savefig(destination/'overview.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, representation in zip(axes, ('sliding', 'disjoint')):
        for v, label, color in zip(VARIANTS, LABELS, COLORS):
            runs = [r for r in records if r['variant'] == v]
            series = [[{'step':0, 'validation':r['initial_validation']}] + [h for h in r['history'] if 'validation' in h] for r in runs]
            steps = [h['step'] for h in series[0]]
            points = np.array([[h['validation'][representation]['common_boundary_loss'] for h in events] for events in series])
            for individual in points:
                ax.plot(steps, individual, color=color, alpha=.2, linewidth=1)
            mean, sd = points.mean(0), points.std(0, ddof=1)
            ax.plot(steps, mean, color=color, label=label, linewidth=2)
            ax.fill_between(steps, mean-sd, mean+sd, color=color, alpha=.15)
        ax.set_title(f'{representation.title()} forward: matched boundary targets')
        ax.set_xlabel('Optimizer update')
        ax.set_ylabel('Validation cross-entropy')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Validation learning curves: mean ± sample SD across three seeds')
    fig.savefig(destination/'learning_curves.png', dpi=180)
    fig.savefig(destination/'learning_curves.svg')
    plt.close(fig)
    for path in destination.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='out-sliding-three-seeds')
    plot(Path(parser.parse_args().output))
