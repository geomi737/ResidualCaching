import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['baseline', 'plain', 'residual']
LABELS = ['Baseline', 'Plain', 'Residual']
COLORS = ['#4b79a1', '#d69b43', '#528c6c']

def get_stats(results_list, variant, key_fn):
    values = [key_fn(r) for r in results_list if r['variant'] == variant and r['state'] == 'complete']
    if not values:
        return {'mean': 0, 'std': 0}
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0}

def plot_experiment(results_path, output_png, title_prefix):
    if not Path(results_path).exists():
        print(f"Skipping {results_path} (not found)")
        return

    data = json.loads(Path(results_path).read_text())
    results = data.get('results', [])

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))

    def get_inference(r):
        inf = r.get('inference', [])
        for i in inf:
            if i['prompt_length'] == 1024:
                return i['cached_decode_tokens_per_second']
        return 0

    panels = [
        ('Training input throughput', 'input tokens/s', lambda r: r.get('input_tokens_per_second', 0), 1),
        ('Supervised predictions', 'predictions/s', lambda r: r.get('supervised_predictions_per_second', 0), 1),
        ('Training peak VRAM', 'MiB', lambda r: r.get('train_peak_allocated_mib', 0), 1),
        ('Test Prefix Loss', 'loss', lambda r: r.get('test', {}).get('prefix_loss', 0), 1),
        ('Test Boundary Acc', '%', lambda r: r.get('test', {}).get('boundary_accuracy', 0), 100),
        ('Decode at 1024 (cached)', 'tokens/s', get_inference, 1),
    ]

    for ax, (title, ylabel, key_fn, factor) in zip(axes.flat, panels):
        means = []
        stds = []
        for name in NAMES:
            stats = get_stats(results, name, key_fn)
            means.append(stats['mean'] * factor)
            stds.append(stats['std'] * factor)

        ax.bar(range(3), means, color=COLORS, yerr=stds, capsize=4)
        ax.set_xticks(range(3), ['Baseline', 'Plain', 'Residual'])
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    fig.suptitle(f'SmolLM Experiment: {title_prefix}', fontsize=14)
    fig.tight_layout()
    plt.savefig(output_png, dpi=120)
    plt.close()
    print(f"Saved {output_png}")

if __name__ == '__main__':
    plot_experiment('out-smollm/results.json', 'out-smollm/smollm_finetune.png', 'Finetune (135M pretrained)')
    plot_experiment('out-tiny-smollm/results.json', 'out-tiny-smollm/smollm_pretrain.png', 'Pretrain from Scratch (Tiny 25M)')
