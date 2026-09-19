"""Publication integrity and matched-budget checks for completed experiments."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]/'results'


class PublishedResultsTests(unittest.TestCase):
    def test_json_checksums(self):
        for name,digest in json.loads((ROOT/'checksums.json').read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_three_fresh_paired_training_seeds(self):
        data=json.loads((ROOT/'training/three_seeds/results.json').read_text())
        self.assertTrue(data['paired_checks_passed'])
        runs=data['results'];self.assertEqual(len(runs),9)
        seeds={r['config']['seed'] for r in runs};self.assertEqual(seeds,{17,29,43})
        self.assertEqual(len({r['initial_weights_sha256'] for r in runs}),3)
        for seed in seeds:
            group=[r for r in runs if r['config']['seed']==seed]
            self.assertEqual({r['variant'] for r in group},{'baseline','sliding','sliding_residual'})
            for key in ('initial_weights_sha256','batch_plan_sha256','input_tokens','supervised_predictions'):
                self.assertEqual(len({r[key] for r in group}),1,key)
            for r in group:
                self.assertEqual(r['state'],'complete');self.assertEqual(r['initialization'],'random')
                self.assertEqual(len(r['history']),1500);self.assertEqual(r['supervised_predictions'],6132000)

    def test_speed_repetitions_and_matched_prompts(self):
        for path in (ROOT/'speed').glob('*/speed-benchmark.json'):
            data=json.loads(path.read_text());self.assertEqual(data['state'],'complete')
            for seed in data['config']['seeds']:
                for context in data['config']['lengths']:
                    group=[s for s in data['samples'] if s['seed']==seed and s['context']==context]
                    self.assertEqual(len({s['prompt_sha256'] for s in group}),1)
                    for variant in ('baseline','sliding','sliding_residual'):
                        for mode in ('prefill','decode'):
                            events=[s for s in group if s['variant']==variant and s['mode']==mode]
                            self.assertEqual(len(events),data['config']['repeats'])
                            self.assertTrue(all(s['seconds']>0 for s in events))

    def test_isolated_cache_and_peak_semantics(self):
        records=json.loads((ROOT/'memory/128k_isolated/results.json').read_text())['records']
        self.assertEqual(len(records),3)
        baseline=next(r for r in records if r['variant']=='baseline')
        for r in records:
            self.assertEqual(r['state'],'complete');self.assertEqual(len(r['samples']),12)
            for s in r['samples']:
                self.assertGreaterEqual(s['peak_reserved_mib'],s['peak_allocated_mib'])
                self.assertGreaterEqual(s['peak_allocated_mib'],s['kv_mib']+r['parameter_mib'])
            cache=r['summary']['decode']
            if r['variant']=='baseline':
                self.assertEqual(cache['layer_cache_lengths'],[131200]*8)
            else:
                self.assertEqual(cache['layer_cache_lengths'],[131200]*4+[65600]*4)
                self.assertEqual(cache['kv_mib']/baseline['summary']['decode']['kv_mib'],.75)
