"""Post-training regressions. Run with NL_TINY=1 NL_DEVICE=cpu MPD_MEM_GIB=8.

The meta-device test exercises real tensor device checks without allocating on a
GPU. The archive test runs the real CLI from a source tree without Git metadata.
"""
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))

import torch
from analyze_bridge import position_masks
from link import FAR


class ChainTests(unittest.TestCase):
    def test_record_masks_cross_device(self):
        ticks = torch.tensor([[-1, 0, 2, FAR]])
        prompt, written = position_masks(ticks)
        torch.testing.assert_close(prompt, torch.tensor([[1., 0., 0., 0.]]))
        torch.testing.assert_close(written, torch.tensor([[0., 1., 1., 0.]]))
        # Actual PyTorch device mismatch checks, no mocked tensors or operators.
        # Reverse the CPU/GPU direction using meta, since this suite is CPU-only.
        attention = torch.empty((1, 4), device='meta')
        prompt, written = position_masks(ticks, device=attention.device)
        self.assertEqual((attention * prompt).sum(1).device, attention.device)
        self.assertEqual((attention * written).sum(1).device, attention.device)

    @unittest.skipUnless(os.environ.get('CHAIN_CHECKPOINTS'), 'set CHAIN_CHECKPOINTS for trained checkpoint checks')
    def test_trained_flags_and_shuffle(self):
        import evalfree as F
        from evaluate import load_pair
        from link import Pair, SIDES, partner
        import session as S

        replies = json.loads(Path(os.environ['CHAIN_REPLIES']).read_text())
        plans, _ = F.make_plans(replies, n=1, coread_data=os.environ['CHAIN_COREAD'],
                               games=['relay_word', 'co_talk'])
        for run in os.environ['CHAIN_CHECKPOINTS'].split(','):
            with self.subTest(run=run):
                ck = torch.load(Path(run) / 'bridge.pt', map_location='cpu', weights_only=False)
                pair, toks = load_pair(run)
                # Independently construct the trained architecture from its args.
                # read_start and content_init are initializers: preserve their learned
                # tensors, never zero/reinitialize them when loading for evaluation.
                reference = Pair(pair.models, toks, ck['xblocks'], ck['sizes'], ck['heads'], ck['head_dim'],
                                 gate_norm=ck['args']['gate_norm']).eval()
                reference.bridge.load_state_dict(ck['bridge'])
                for x in SIDES:
                    self.assertEqual(pair.bridge[x].gate_norm, ck['args']['gate_norm'])
                for key, tensor in pair.bridge.state_dict().items():
                    torch.testing.assert_close(tensor, ck['bridge'][key], rtol=0, atol=0)
                plan = plans['relay_word'][0]
                eps = [plan['episode'], plan['donor']]
                batch = S.collate([S.build(ep, toks) for ep in eps], {x: toks[x].pad_token_id for x in SIDES})
                with torch.inference_mode():
                    expected, _ = reference.joint(batch)
                    actual, _ = pair.joint(batch)
                    for x in SIDES:
                        torch.testing.assert_close(actual[x], expected[x], rtol=0, atol=0)
                    x = eps[0]['meta']['receiver']
                    y = partner(x)
                    original = S.generate(pair, eps, mode='shuffle', max_turn=3, keep=True)
                    changed = copy.deepcopy(eps)
                    changed[0]['task'][y] = 'A completely different original partner secret about violet elephants.'
                    isolated = S.generate(pair, changed, mode='shuffle', max_turn=3, keep=True)
                    torch.testing.assert_close(original[0]['h'][x], isolated[0]['h'][x], rtol=2e-6, atol=2e-6)
                    changed = copy.deepcopy(eps)
                    changed[1]['task'][y] = 'A completely different donor secret about violet elephants.'
                    donor_changed = S.generate(pair, changed, mode='shuffle', max_turn=3, keep=True)
                    self.assertGreater(float((original[0]['h'][x] - donor_changed[0]['h'][x]).abs().max()), 1e-8)
                    ep, donor = plans['co_talk'][0]['episode'], plans['co_talk'][0]['donor']
                    talk = F.generate_shuffle(pair, ep, donor, 3, keep=True)
                    changed = copy.deepcopy(ep)
                    asker = ep['meta']['asker']
                    changed['system'][partner(asker)] = 'An altered original holder context.'
                    isolated = F.generate_shuffle(pair, changed, donor, 3, keep=True)
                    torch.testing.assert_close(talk['h'][asker], isolated['h'][asker], rtol=0, atol=0)
                    self.assertEqual(len(talk['turns']), 1)
                    self.assertEqual(talk['turns'][0][0], asker)
                del reference, pair, toks, expected, actual, batch, original, isolated, donor_changed, talk
                gc.collect()

    @unittest.skipUnless(os.environ.get('CHAIN_CHECKPOINT'), 'set CHAIN_CHECKPOINT and CHAIN_REPLIES for CLI regression')
    def test_evalfree_source_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / 'pairing', root / 'pairing', ignore=shutil.ignore_patterns('__pycache__'))
            (root / 'data').mkdir()
            shutil.copy(ROOT / 'data/pools.json', root / 'data/pools.json')
            shutil.copy(ROOT / 'data/coread_check.json', root / 'data/coread.json')
            out = root / 'out'
            env = dict(os.environ, NL_MAXNEW='2', GIT_CEILING_DIRECTORIES=str(root))
            result = subprocess.run([sys.executable, 'evalfree.py', '--run', os.environ['CHAIN_CHECKPOINT'],
                                     '--replies', os.environ['CHAIN_REPLIES'], '--out_dir', str(out),
                                     '--n', '1', '--bs', '2', '--games', 'relay_word'],
                                    cwd=root / 'pairing', env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertIsNone(manifest['git_commit'])
            self.assertIn('link.py', manifest['code_hashes'])


if __name__ == '__main__':
    unittest.main()
