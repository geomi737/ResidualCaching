"""Generate English percentage tables from preserved measurements."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.reporting.sliding_metrics import metrics, percentage


def export(source, destination):
    data = json.loads(source.read_text())
    records = data['results']
    base = next(r for r in records if r['variant'] == 'baseline')
    baseline = metrics(base)
    others = [r for r in records if r is not base]
    derived = {r['variant']: metrics(r) for r in records}
    lines = ['# Percentage analysis: training from scratch', '',
             'All changes are relative to the baseline: `100 × (variant / baseline − 1)`. '
             'Accuracy includes percentage-point differences. A positive loss change is worse; '
             'a positive throughput change is faster. One seed, equal input budgets.', '',
             '| Metric | Baseline | Merge | Sliding | Merge + R | Sliding + R |',
             '|---|---:|---:|---:|---:|---:|']
    for key, reference in baseline.items():
        if reference is None:
            continue
        row = [key, f'{reference:.6g}']
        for record in others:
            value = derived[record['variant']][key]
            cell = 'unavailable' if value is None else f'{value:.6g} ({percentage(value, reference):+.2f}%)'
            if value is not None and 'accuracy (%)' in key:
                cell += f'; {value-reference:+.2f} pp'
            row.append(cell)
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## Representation-mode mismatch and merge weights', '',
              '| Variant | Boundary loss change: sliding → disjoint | Final argmax agreement | Final KL(sliding ∥ disjoint) | Learned weights: previous / current | Normalized effective coefficients |',
              '|---|---:|---:|---:|---|---|']
    for record in records:
        quality = record['test']
        gap = quality['mode_gap']
        weights = record['merge_weights']
        coefficients = ([(1+w)/3 for w in weights] if record['variant'].endswith('_residual')
                        else weights) if weights is not None else None
        fmt = lambda values: 'n/a' if values is None else ' / '.join(f'{100*v:.2f}%' for v in values)
        lines.append(f"| {record['variant']} | {percentage(quality['disjoint']['common_boundary_loss'], quality['sliding']['common_boundary_loss']):+.2f}% | "
                     f"{100*gap['last_token_argmax_agreement']:.2f}% | {max(0, gap['kl_sliding_to_disjoint']):.4f} | {fmt(weights)} | {fmt(coefficients)} |")
    lines += ['', '## Interpretation limits', '',
              '- Boundary scores use 65,536 matched targets; final-prefix scores use 512 matched prefixes. Windows can overlap.',
              '- Training loss target sets differ. Disjoint training supervises about half as many predictions per input budget.',
              '- Last-step training scores come from one batch. They are not held-out generalization metrics.',
              '- Update timings exclude validation. Inference timings are short sequential measurements, not independent timing repetitions.',
              '- Reserved memory depends on allocator history. Allocated peaks and actual KV bytes are the primary memory comparisons.',
              '- Percentage changes in cross-entropy are not percentages of lost understanding. Accuracy is top-1 next-token accuracy.',
              '- Tiny negative baseline KL from floating-point rounding is displayed as zero. Raw evidence is unchanged.',
              '- One seed is insufficient to establish a universal winner or rank small speed differences.', '']
    destination.write_text('\n'.join(lines))
    summary = {'source': source.name, 'paired_checks_passed': data['paired_checks_passed'],
               'metrics': derived,
               'relative_change_percent': {name: {key: percentage(value, baseline[key])
                   for key, value in values.items() if value is not None and baseline[key] not in (None, 0)}
                   for name, values in derived.items()}}
    destination.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/sliding_scratch/seed11.json'))
    parser.add_argument('--output', type=Path, default=Path('results/sliding_scratch/percentage_analysis.md'))
    args = parser.parse_args()
    export(args.input, args.output)
