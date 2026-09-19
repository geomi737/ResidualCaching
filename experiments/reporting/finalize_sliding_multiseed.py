"""Finalize evidence-derived three-seed reports and figures, optionally waiting."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.smollm.train_sliding_multiseed import summarize
from experiments.reporting.plot_sliding_multiseed import plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='out-sliding-three-seeds')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = Path(args.output)
    while True:
        suite = json.loads((root/'suite.json').read_text())
        if suite['state'] == 'failed':
            raise RuntimeError(suite.get('error', 'Training suite failed.'))
        if suite['state'] == 'complete':
            break
        if not args.wait:
            raise RuntimeError('Complete all nine trainings before finalization.')
        time.sleep(5)
    result = json.loads((root/'results.json').read_text())
    if not result['paired_checks_passed'] or not result['distinct_initializations'] or len(result['results']) != 9:
        raise RuntimeError('Incomplete or unpaired evidence.')
    records = result['results']
    seeds = sorted({r['config']['seed'] for r in records})
    summarize(root, records, seeds)
    plot(root)
    print('Reports and figures complete:', root.resolve(), flush=True)
