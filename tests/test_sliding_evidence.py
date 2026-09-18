"""Integrity and comparison-budget checks on the preserved GPU evidence."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'results/sliding_scratch'


class SlidingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/'seed11.json').read_text())

    def test_original_evidence_checksums_and_shared_initialization(self):
        provenance = json.loads((ROOT/'provenance.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'seed11.json').read_bytes()).hexdigest(),
                         provenance['raw_results_sha256'])
        self.assertEqual(hashlib.sha256((ROOT/'dataset_manifest.json').read_bytes()).hexdigest(),
                         provenance['dataset_manifest_sha256'])
        records = self.data['results']
        self.assertEqual({r['variant'] for r in records},
                         {'baseline', 'disjoint', 'sliding', 'disjoint_residual', 'sliding_residual'})
        for key in ('initial_weights_sha256', 'batch_plan_sha256', 'input_tokens'):
            self.assertEqual(len({r[key] for r in records}), 1)
        for record in records:
            self.assertEqual(record['initialization'], 'random')
            self.assertFalse(record['synthetic'])
            self.assertEqual(record['state'], 'complete')
            self.assertEqual(len(record['history']), 1500)

    def test_matched_targets_data_and_different_supervision_budgets(self):
        manifest = json.loads((ROOT/'dataset_manifest.json').read_text())
        for record in self.data['results']:
            for split in ('train', 'validation', 'test'):
                self.assertEqual(record['data_sha256'][split], manifest['splits'][split]['tokens_sha256'])
            quality = record['test']['disjoint']
            self.assertEqual(quality['common_count'], 65536)
            self.assertEqual(sum(p['count'] for p in quality['parity'].values()), 512)
            self.assertEqual(record['input_tokens'], 6132000)
            expected = 3072000 if record['variant'].startswith('disjoint') else 6132000
            self.assertEqual(record['supervised_predictions'], expected)

    def test_cache_reduction_and_early_layer_lengths(self):
        records = {r['variant']: r for r in self.data['results']}
        baseline = records['baseline']['inference']['cached_decode']
        self.assertEqual(baseline['layer_cache_lengths'], [288]*8)
        for variant, record in records.items():
            if variant == 'baseline':
                continue
            cache = record['inference']['cached_decode']
            self.assertEqual(cache['layer_cache_lengths'], [288]*4 + [144]*4)
            self.assertEqual(cache['kv_cache_mib'] / baseline['kv_cache_mib'], .75)


if __name__ == '__main__':
    unittest.main()
