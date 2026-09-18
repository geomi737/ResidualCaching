"""Shared, evidence-derived metrics for the five-variant scratch experiment."""
VARIANTS = ('baseline', 'disjoint', 'sliding', 'disjoint_residual', 'sliding_residual')
LABELS = ('Baseline', 'Merge', 'Sliding', 'Merge + R', 'Sliding + R')
COLORS = ('#4779aa', '#c88532', '#328967', '#8a63af', '#bd567d')


def metrics(record):
    history = record['history']
    timed = [e for e in history if not e['warmup']]
    seconds = sum(e['seconds'] for e in timed)
    quality = record['test']['disjoint']
    parity = quality['parity']
    count = sum(p['count'] for p in parity.values())
    values = {
        'Test boundary loss': quality['common_boundary_loss'],
        'Test boundary accuracy (%)': 100 * quality['common_boundary_accuracy'],
        'Test final-prefix loss': sum(p['prefix_loss'] * p['count'] for p in parity.values()) / count,
        'Test final-prefix accuracy (%)': 100 * sum(p['prefix_accuracy'] * p['count'] for p in parity.values()) / count,
        'Even-prefix loss': parity['even']['prefix_loss'],
        'Odd-prefix loss': parity['odd']['prefix_loss'],
        'Even-prefix accuracy (%)': 100 * parity['even']['prefix_accuracy'],
        'Odd-prefix accuracy (%)': 100 * parity['odd']['prefix_accuracy'],
        'Last training loss': history[-1]['loss'],
        'Last training accuracy (%)': 100 * history[-1]['accuracy'],
        'Last gradient norm before clipping': history[-1]['grad_norm'],
        'Final learning rate': history[-1]['lr'],
        'Training input tokens/s': sum(e['seconds'] * e['input_tokens_per_second'] for e in timed) / seconds if seconds else None,
        'Training predictions/s': sum(e['seconds'] * e['predictions_per_second'] for e in timed) / seconds if seconds else None,
        'Training update time (ms)': 1000 * seconds / len(timed) if timed else None,
        'Training update total time (s)': sum(e['seconds'] for e in history),
        'Training allocated peak (MiB)': max(e['peak_allocated_mib'] for e in history),
        'Training reserved peak (MiB)': max(e['peak_reserved_mib'] for e in history),
        'Input tokens': record['input_tokens'],
        'Supervised predictions': record['supervised_predictions'],
        'Parameters': record['parameters'],
        'Sliding-mode boundary loss': record['test']['sliding']['common_boundary_loss'],
        'Sliding-mode boundary accuracy (%)': 100 * record['test']['sliding']['common_boundary_accuracy'],
    }
    for mode, measurement in record['inference'].items():
        for key in ('seconds', 'peak_allocated_mib', 'peak_reserved_mib', 'tokens_per_second',
                    'input_tokens_per_second', 'kv_cache_mib'):
            if key in measurement:
                values[f'{mode}: {key}'] = measurement[key]
    return values


def percentage(value, reference):
    return 100 * (value / reference - 1)
