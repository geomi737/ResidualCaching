"""Exercise the actual train/resume/sample entry points on a tiny local dataset."""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


class TrainingWorkflowTests(unittest.TestCase):
    def test_singleton_training_merged_resume_and_sampling(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='residual-training-') as directory:
            data = Path(directory) / 'data'
            out = Path(directory) / 'checkpoint'
            data.mkdir()
            for split in ('train', 'val'):
                (np.arange(256) % 19).astype(np.uint16).tofile(data / f'{split}.bin')
            alphabet = 'abcdefghijklmnopqrs'
            with (data / 'meta.pkl').open('wb') as file:
                pickle.dump({'vocab_size': 19, 'stoi': {c: i for i, c in enumerate(alphabet)},
                             'itos': dict(enumerate(alphabet))}, file)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            common = [f'--dataset={data}', f'--out_dir={out}', '--device=cpu', '--dtype=float32',
                      '--compile=False', '--n_layer=3', '--n_head=2', '--n_embd=16', '--block_size=8',
                      '--batch_size=2', '--gradient_accumulation_steps=1', '--eval_iters=1',
                      '--eval_interval=1', '--log_interval=1', '--decay_lr=False', '--dropout=0.0']
            def run(script, args):
                result = subprocess.run([sys.executable, '-B', script, *args], cwd=root / 'nanoGPT',
                                        env=env, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result
            run('train.py', common + ['--merge_ratio=3', '--merge_layer=1',
                                     '--unmerged_prob=1.0', '--max_iters=1'])
            checkpoint = torch.load(out / 'ckpt.pt', weights_only=False, map_location='cpu')
            self.assertEqual(checkpoint['iter_num'], 1)
            self.assertEqual(checkpoint['model_args']['merge_ratio'], 3)
            self.assertTrue(checkpoint['model_args']['residual_tail'])
            # Resume without repeating architecture flags: checkpoint must restore them.
            run('train.py', common + ['--init_from=resume', '--unmerged_prob=0.0', '--max_iters=2'])
            resumed = torch.load(out / 'ckpt.pt', weights_only=False, map_location='cpu')
            self.assertEqual(resumed['iter_num'], 2)
            self.assertEqual(resumed['model_args']['merge_ratio'], 3)
            self.assertFalse(torch.equal(checkpoint['model']['merger.merge_weight'],
                                         resumed['model']['merger.merge_weight']))
            result = run('sample.py', [f'--out_dir={out}', '--device=cpu', '--dtype=float32',
                                      '--start=abc', '--max_new_tokens=6', '--num_samples=1'])
            self.assertIn('abc', result.stdout)
