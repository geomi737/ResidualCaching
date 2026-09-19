"""Run three paired scratch suites sequentially on CUDA with one live dashboard."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.smollm.train_sliding import save
from experiments.reporting.sliding_metrics import metrics

VARIANTS = ('baseline', 'sliding', 'sliding_residual')


def summarize(root, records, seeds):
    summary = {'seeds': seeds, 'runs': len(records), 'variants': {}, 'paired_deltas': {}}
    lines = ['# Three-seed sliding experiment', '',
             'Nine fresh random-initialized GPU trainings; 1500 updates each. '
             'Initial base weights and training batches are paired within each seed. '
             'Values are mean ± sample standard deviation across seeds, not confidence intervals. '
             'Evaluation uses the same exploratory test split as the pilot; this is not fresh held-out confirmation.', '',
             '| Metric | Baseline | Sliding | Sliding + R |', '|---|---:|---:|---:|']
    by_variant = {v: {r['config']['seed']: metrics(r) for r in records if r['variant'] == v} for v in VARIANTS}
    for v, runs in by_variant.items():
        summary['variants'][v] = {key: {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'values': values}
                                  for key in next(iter(runs.values()))
                                  if all(isinstance(runs[s][key], (int, float)) for s in seeds)
                                  for values in [[runs[s][key] for s in seeds]]}
    for key in summary['variants']['baseline']:
        cells = [f"{summary['variants'][v][key]['mean']:.4f} ± {summary['variants'][v][key]['sample_sd']:.4f}" for v in VARIANTS]
        lines.append('| ' + key + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Paired differences from baseline', '',
              'Percentages are computed within each seed before aggregation. Accuracy differences use percentage points.', '',
              '| Metric | Sliding | Sliding + R |', '|---|---:|---:|']
    for v in VARIANTS[1:]:
        summary['paired_deltas'][v] = {}
        for key in summary['variants']['baseline']:
            values = [(by_variant[v][s][key] - by_variant['baseline'][s][key]) if 'accuracy (%)' in key.lower()
                      else 100 * (by_variant[v][s][key] / by_variant['baseline'][s][key] - 1)
                      for s in seeds if by_variant['baseline'][s][key] != 0]
            if len(values) != len(seeds):
                continue
            summary['paired_deltas'][v][key] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'values': values,
                                               'unit': 'pp' if 'accuracy (%)' in key.lower() else '%'}
    for key, value in summary['paired_deltas']['sliding'].items():
        cells = [f"{summary['paired_deltas'][v][key]['mean']:+.3f} ± {summary['paired_deltas'][v][key]['sample_sd']:.3f} {value['unit']}" for v in VARIANTS[1:]]
        lines.append('| ' + key + ' | ' + ' | '.join(cells) + ' |')
    summary['residual_vs_sliding'] = {}
    lines += ['', '## Sliding + R compared directly with Sliding', '',
              '| Metric | Paired difference, mean ± sample SD |', '|---|---:|']
    for key in summary['variants']['sliding']:
        accuracy = 'accuracy (%)' in key.lower()
        values = [(by_variant['sliding_residual'][seed][key] - by_variant['sliding'][seed][key]) if accuracy
                  else 100 * (by_variant['sliding_residual'][seed][key] / by_variant['sliding'][seed][key] - 1)
                  for seed in seeds if by_variant['sliding'][seed][key] != 0]
        if len(values) != len(seeds):
            continue
        unit = 'pp' if accuracy else '%'
        mean, sd = statistics.mean(values), statistics.stdev(values)
        summary['residual_vs_sliding'][key] = {'mean': mean, 'sample_sd': sd, 'values': values, 'unit': unit}
        lines.append(f'| {key} | {mean:+.3f} ± {sd:.3f} {unit} |')
    save(root / 'summary.json', summary)
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'suite.json').exists():
        raise RuntimeError('Choose a fresh output directory to avoid overwriting an experiment.')
    jobs = [{'variant': v, 'seed': s, 'file': f'{v}-{s}.json'} for i, s in enumerate(args.seeds)
            for v in VARIANTS[i % 3:] + VARIANTS[:i % 3]]
    suite = {'state': 'running', 'config': {'steps': 1500, 'seed': ', '.join(map(str, args.seeds)),
             'length': 256, 'batch_size': 4, 'accumulation': 4, 'merge_layer': 4}, 'completed': 0, 'runs': jobs}
    save(root / 'suite.json', suite)
    records = []
    try:
        for i, seed in enumerate(args.seeds):
            child = root / f'seed{seed}'
            order = VARIANTS[i % 3:] + VARIANTS[:i % 3]
            command = [sys.executable, 'experiments/smollm/train_sliding.py', '--seed', str(seed),
                       '--variants', *order, '--output', str(child)]
            print(f'Starting seed {seed}: {order}', flush=True)
            with (root / f'seed{seed}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    for v in order:
                        path = child / f'{v}-{seed}.json'
                        if path.exists():
                            shutil.copyfile(path, root / (path.name + '.tmp'))
                            (root / (path.name + '.tmp')).replace(root / path.name)
                    suite['completed'] = sum(json.loads((root/j['file']).read_text())['state'] == 'complete'
                                             for j in jobs if (root/j['file']).exists())
                    save(root / 'suite.json', suite)
                    if process.poll() is not None:
                        break
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f'Seed {seed} failed; see {root / f"seed{seed}.log"}')
            result = json.loads((child / 'results.json').read_text())
            if not result['paired_checks_passed']:
                raise RuntimeError(f'Pairing failed for seed {seed}')
            records.extend(result['results'])
        hashes = {r['initial_weights_sha256'] for r in records}
        if len(hashes) != len(args.seeds):
            raise RuntimeError('Initial weights must differ across seeds.')
        save(root / 'results.json', {'results': records, 'paired_checks_passed': True, 'distinct_initializations': True})
        summarize(root, records, args.seeds)
        suite['state'] = 'complete'
    except Exception as error:
        suite.update(state='failed', error=str(error))
        raise
    finally:
        save(root / 'suite.json', suite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs=3, default=[17, 29, 43])
    parser.add_argument('--output', default='out-sliding-three-seeds')
    args = parser.parse_args()
    if len(set(args.seeds)) != 3:
        parser.error('Use three distinct seeds.')
    run(args)
