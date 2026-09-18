"""Render completed paired sliding experiment results into a comparison report."""
import argparse
import json
from pathlib import Path


def write_report(root):
    records = json.loads((root/'results.json').read_text())['results']
    initialization = 'random tiny' if any(r['synthetic'] or r['config'].get('from_scratch') for r in records) else 'pretrained'
    lines = ['# Sliding merging GPU pilot', '',
             f'One seed; identical {initialization} base weights and input batches. '
             'This pilot measures continuation behavior, not general reasoning.', '',
             '## Matched held-out quality', '',
             '| Variant | Sliding boundary CE | Disjoint boundary CE | Disjoint boundary accuracy | Disjoint final-prefix CE (even / odd) | Sliding→disjoint final KL | Final argmax agreement |',
             '|---|---:|---:|---:|---|---:|---:|']
    for r in records:
        s, d, gap = r['test']['sliding'], r['test']['disjoint'], r['test']['mode_gap']
        lines.append(f"| {r['variant']} | {s['common_boundary_loss']:.4f} | {d['common_boundary_loss']:.4f} | "
                     f"{100*d['common_boundary_accuracy']:.2f}% | {d['parity']['even']['prefix_loss']:.4f} / "
                     f"{d['parity']['odd']['prefix_loss']:.4f} | {gap['kl_sliding_to_disjoint']:.4f} | "
                     f"{100*gap['last_token_argmax_agreement']:.2f}% |")
    lines += ['', '## Training consumption', '',
              '| Variant | Last train CE / accuracy | Peak allocated / reserved MiB | Input tok/s after warmup | Supervised predictions |',
              '|---|---|---|---:|---:|']
    for r in records:
        h = r['history']
        timed = [e for e in h if not e['warmup']]
        # Average throughput uses total time, not arithmetic mean of rates.
        rate = sum(e['seconds']*e['input_tokens_per_second'] for e in timed)/sum(e['seconds'] for e in timed) if timed else None
        lines.append(f"| {r['variant']} | {h[-1]['loss']:.4f} / {100*h[-1]['accuracy']:.2f}% | "
                     f"{max(e['peak_allocated_mib'] for e in h):.1f} / {max(e['peak_reserved_mib'] for e in h):.1f} | "
                     f"{rate:.1f} | {r['supervised_predictions']} |" if rate is not None else
                     f"| {r['variant']} | warmup only | — | — | {r['supervised_predictions']} |")
    lines += ['', 'Training CE target sets differ for disjoint training. Throughput includes optimizer and accuracy computation, excludes validation.', '',
              '## Inference consumption', '',
              '| Variant | Mode | Seconds | Peak allocated / reserved MiB | Decode tok/s | KV cache MiB |',
              '|---|---|---:|---|---:|---:|']
    for r in records:
        for mode, m in r['inference'].items():
            rate = f"{m['tokens_per_second']:.1f}" if 'tokens_per_second' in m else '—'
            cache = f"{m['kv_cache_mib']:.3f}" if 'kv_cache_mib' in m else '—'
            lines.append(f"| {r['variant']} | {mode} | {m['seconds']:.5f} | "
                         f"{m['peak_allocated_mib']:.1f} / {m['peak_reserved_mib']:.1f} | "
                         f"{rate} | {cache} |")
    baseline = next(r for r in records if r['variant'] == 'baseline')
    lines += ['', '## Differences from baseline in deployed disjoint mode', '']
    for r in records:
        if r is baseline:
            continue
        delta = r['test']['disjoint']['common_boundary_loss'] - baseline['test']['disjoint']['common_boundary_loss']
        decode_ratio = r['inference']['cached_decode']['tokens_per_second']/baseline['inference']['cached_decode']['tokens_per_second']
        cache_ratio = r['inference']['cached_decode']['kv_cache_mib']/baseline['inference']['cached_decode']['kv_cache_mib']
        lines.append(f"- {r['variant']}: boundary CE delta {delta:+.4f}; cached decode speed {decode_ratio:.3f}×; KV bytes {cache_ratio:.3f}×.")
    lines += ['', 'Configuration and full per-step metrics are in results.json and variant JSON files. '
              'Memory is PyTorch allocator memory, not total device VRAM. Full forward and cached prefill are different workloads. '
              'No inference speedup or quality retention is assumed before inspecting these measurements.', '']
    path = root/'comparison.md'
    path.write_text('\n'.join(lines))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='out-sliding-smollm')
    args = parser.parse_args()
    print(write_report(Path(args.output)))
