"""Generate reproducible English figures from the preserved scratch pilot."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reporting.sliding_metrics import COLORS, LABELS, VARIANTS, metrics, percentage


def generate(source, output):
    records = json.loads(source.read_text())['results']
    by_name = {r['variant']: r for r in records}
    records = [by_name[v] for v in VARIANTS]
    values = [metrics(r) for r in records]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

    def save(fig, name):
        fig.savefig(output/f'{name}.png', dpi=180)
        fig.savefig(output/f'{name}.svg')
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [('Test boundary loss (lower is better)', 'cross-entropy, nats', 'Test boundary loss'),
              ('Test boundary accuracy (higher is better)', '%', 'Test boundary accuracy (%)'),
              ('Cached generation throughput', 'tokens/s', 'cached_decode: tokens_per_second'),
              ('Training allocated peak', 'MiB', 'Training allocated peak (MiB)'),
              ('Training input throughput', 'input tokens/s', 'Training input tokens/s'),
              ('Persistent KV cache', 'MiB', 'cached_decode: kv_cache_mib')]
    for ax, (title, unit, key) in zip(axes.flat, panels):
        ys = [v[key] for v in values]
        ax.bar(range(5), ys, color=COLORS)
        ax.set_xticks(range(5), LABELS, rotation=18, ha='right')
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.set_ylim(0, max(ys)*1.25)
        for i, y in enumerate(ys):
            label = f'{y:.2f}' if y < 100 else f'{y:,.0f}'
            label += '\nbaseline' if i == 0 else f'\n{percentage(y, ys[0]):+.2f}%'
            ax.text(i, y+max(ys)*.025, label, ha='center', fontsize=9)
    fig.suptitle('Five variants trained from scratch — WikiText-2, RTX 4060', fontsize=16)
    fig.text(.5, .015, 'One seed · 1,500 updates · context 255/256 · inference context 256 · no error bars: no repeated seeds', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    save(fig, 'overview')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for r, label, color in zip(records, LABELS, COLORS):
        h = r['history']
        steps = np.array([e['step'] for e in h])
        for ax, field, factor in ((axes[0, 0], 'loss', 1), (axes[0, 1], 'accuracy', 100)):
            ys = np.array([factor*e[field] for e in h])
            ax.plot(steps[49:], np.convolve(ys, np.ones(50)/50, mode='valid'), color=color, label=label)
        checkpoints = [(0, r['initial_validation'])] + [(e['step'], e['validation']) for e in h if 'validation' in e]
        for ax, key, factor in ((axes[1, 0], 'common_boundary_loss', 1),
                                (axes[1, 1], 'common_boundary_accuracy', 100)):
            ax.plot([s for s, _ in checkpoints], [factor*v['disjoint'][key] for _, v in checkpoints],
                    marker='o', color=color, label=label)
    for ax, title, unit in zip(axes.flat,
                              ('Training loss (50-update moving mean)', 'Training accuracy (50-update moving mean)',
                               'Validation loss in compressed inference mode', 'Validation accuracy in compressed inference mode'),
                              ('cross-entropy, nats', '%', 'cross-entropy, nats', '%')):
        ax.set_title(title)
        ax.set_xlabel('optimizer updates')
        ax.set_ylabel(unit)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Learning curves — random initialization, equal input batches', fontsize=15)
    fig.text(.5, .015, 'Training target sets differ for Merge variants; validation uses matched boundary targets. One seed.', ha='center')
    fig.tight_layout(rect=(0, .04, 1, .94))
    save(fig, 'learning_curves')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(5)
    for mode, offset, hatch in (('sliding', -.19, '//'), ('disjoint', .19, None)):
        ys = [r['test'][mode]['common_boundary_loss'] for r in records]
        axes[0].bar(x+offset, ys, .36, color=COLORS, hatch=hatch, alpha=.8, label=mode)
    axes[0].set_title('Matched test loss by representation mode', pad=35)
    axes[0].set_ylabel('cross-entropy, nats')
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=9)
    for parity, offset in (('even', -.19), ('odd', .19)):
        ys = [r['test']['disjoint']['parity'][parity]['prefix_loss'] for r in records]
        axes[1].bar(x+offset, ys, .36, color=COLORS, hatch='//' if parity == 'even' else None, label=parity)
    axes[1].set_title('Final-prefix loss: complete pairs / singleton tail', pad=35)
    axes[1].set_ylabel('cross-entropy, nats')
    axes[1].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=9)
    agreement = [100*r['test']['mode_gap']['last_token_argmax_agreement'] for r in records]
    axes[2].bar(x, agreement, color=COLORS)
    axes[2].set_title('Final argmax agreement between modes')
    axes[2].set_ylabel('% of 512 prefixes')
    axes[2].set_ylim(0, 110)
    for i, y in enumerate(agreement):
        axes[2].text(i, y+2, f'{y:.2f}%', ha='center', fontsize=9)
    for ax in axes:
        ax.set_xticks(x, LABELS, rotation=20, ha='right')
    fig.text(.5, .015, 'Different even/odd targets: absolute parity scores do not isolate a pure tail effect. One seed.', ha='center')
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, 'representation_modes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/sliding_scratch/seed11.json'))
    parser.add_argument('--output', type=Path, default=Path('results/sliding_scratch/figures'))
    args = parser.parse_args()
    generate(args.input, args.output)
