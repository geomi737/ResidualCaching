"""Render measured LM-quality and CUDA scaling figures from JSON reports."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['baseline', 'merged_no_residual', 'merged_residual']
LABELS = ['Baseline', 'Merged', 'Merged + residual']
COLORS = ['#4b79a1', '#d69b43', '#528c6c']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation', type=Path, default=Path('results/legacy/nanogpt/ablation_cuda.json'))
    parser.add_argument('--scaling', type=Path, default=Path('results/legacy/nanogpt/scaling_cuda.json'))
    args = parser.parse_args()
    report = json.loads(args.ablation.read_text())
    summary = report['summary']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    panels = [
        ('Training input throughput', 'input tokens/s', lambda s: s['training']['input_tokens_per_s'], 1),
        ('Supervised prediction throughput', 'predictions/s', lambda s: s['training']['predictions_per_s'], 1),
        ('Training peak allocated VRAM', 'MiB', lambda s: s['training']['peak_allocated_mib'], 1),
        ('Held-out next-token loss', 'nats / target', lambda s: s['test_prefixes']['loss'], 1),
        ('Held-out next-token accuracy', '%', lambda s: s['test_prefixes']['accuracy'], 100),
        ('Generation at prompt length 128', 'generated tokens/s', lambda s: s['generation']['128']['tokens_per_s'], 1),
    ]
    for ax, (title, ylabel, select, factor) in zip(axes.flat, panels):
        values = [select(summary[name]) for name in NAMES]
        ax.bar(range(3), [v['mean'] * factor for v in values], color=COLORS,
               yerr=[(v['std'] or 0) * factor for v in values], capsize=4)
        ax.set_xticks(range(3), ['Baseline', 'Merged', 'Merged\n+ residual'])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('RTX 4060 · 3 seeds · 1,000 steps · disjoint Tiny Shakespeare test', fontsize=14)
    fig.text(0.5, 0.015, 'Error bars: sample standard deviation across seeds. Peak VRAM includes 10% singleton-only training batches.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(args.ablation.with_suffix('.png'), dpi=180)
    plt.close(fig)
    if not args.scaling.exists():
        return
    scaling = json.loads(args.scaling.read_text())
    if not scaling.get('complete'):
        return
    groups = defaultdict(list)
    for result in scaling['results']:
        groups[(result['variant'], result['merge_layer'], result['context'])].append(result)
    contexts = sorted(set(result['context'] for result in scaling['results']))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, name in enumerate(NAMES):
        runs = [groups[(name, 2, length)] for length in contexts]
        selectors = [(axes[0, 0], lambda r: r['training']['input_tokens_per_s'], 'Training input throughput', 'tokens/s'),
                     (axes[0, 1], lambda r: r['training']['peak_allocated_mib'], 'Compressed-mode training peak VRAM', 'MiB'),
                     (axes[1, 0], lambda r: r['fixed_context_inference']['next_token_forwards_per_s'],
                      'Fixed-context next-token forward', 'forwards/s'),
                     (axes[1, 1], lambda r: r['generation']['tokens_per_s'], 'Autoregressive generation', 'tokens/s')]
        for ax, select, title, ylabel in selectors:
            means = [statistics.mean(select(r) for r in entries) for entries in runs]
            stds = [statistics.stdev(select(r) for r in entries) for entries in runs]
            ax.errorbar(contexts, means, yerr=stds, marker='o', color=COLORS[i], label=LABELS[i], capsize=4)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Original context length')
            ax.set_xticks(contexts)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('CUDA shape study · fresh models · compression after 2 of 6 blocks', fontsize=13)
    fig.text(0.5, 0.015, 'Random-token microbenchmarks measure compute and memory, not trained-model quality. Batch input budget ≈ 2,048 tokens.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(args.scaling.with_suffix('.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
