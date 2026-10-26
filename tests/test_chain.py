"""Regressions of the evaluation path: analyze_bridge's position masks follow the attention tensor's device; evalfree.py
runs from a copy of the code without git metadata and records code hashes; and trained checkpoints (CHAIN_CHECKPOINTS,
comma-separated step directories, with CHAIN_REPLIES and CHAIN_COREAD) load with their flags, and in 'shuffle' mode
the receiver depends on the donor only, in duplex and co_talk dialogue.
"""
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from analyze_bridge import position_masks
import evalfree as F
from evaluate import load_pair
from link import FAR, SIDES, Pair, partner
import session as S


def test_record_masks_follow_the_attention_device():
    ticks = torch.tensor([[-1, 0, 2, FAR]])
    prompt, written = position_masks(ticks)
    torch.testing.assert_close(prompt, torch.tensor([[1., 0., 0., 0.]]))
    torch.testing.assert_close(written, torch.tensor([[0., 1., 1., 0.]]))
    # The meta device stands in for a GPU: the masks must land on the attention's device.
    attention = torch.empty((1, 4), device='meta')
    prompt, written = position_masks(ticks, device=attention.device)
    assert (attention * prompt).sum(1).device == attention.device
    assert (attention * written).sum(1).device == attention.device


@pytest.mark.slow
def test_evalfree_runs_without_git(tiny_models, tree, data, tmp_path):
    toks, models = tiny_models(['q_proj', 'v_proj'])
    pair = Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
    step = tmp_path / 'step'
    for x in SIDES:
        models[x].save_pretrained(step / x)
    torch.save(pair.checkpoint(), step / 'bridge.pt')
    out = tmp_path / 'out'
    env = dict(os.environ, NL_MAXNEW='2', MPD_MEM_GIB='8', GIT_CEILING_DIRECTORIES=str(tree))
    result = subprocess.run([sys.executable, 'evalfree.py', '--run', str(step), '--replies', str(data / 'replies.json'),
                             '--coread_data', str(data / 'coread.json'), '--out_dir', str(out), '--n', '1', '--bs', '2',
                             '--games', 'relay_word'], cwd=tree / 'pairing', env=env, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['git_commit'] is None and 'link.py' in manifest['code_hashes']


@pytest.mark.skipif(not os.environ.get('CHAIN_CHECKPOINTS'), reason='set CHAIN_CHECKPOINTS, CHAIN_REPLIES and '
                    'CHAIN_COREAD to check trained checkpoints')
def test_trained_flags_and_shuffle():
    replies = json.loads(Path(os.environ['CHAIN_REPLIES']).read_text())
    plans, _ = F.make_plans(replies, n=1, coread_data=os.environ['CHAIN_COREAD'], games=['relay_word', 'co_talk'])
    for run in os.environ['CHAIN_CHECKPOINTS'].split(','):
        ck = torch.load(Path(run) / 'bridge.pt', map_location='cpu', weights_only=False)
        pair, toks = load_pair(run)
        # The architecture built independently from the checkpoint's arguments. read_start and content_init are
        # initializers: loading for evaluation keeps their learned tensors.
        reference = Pair(pair.models, toks, ck['xblocks'], ck['sizes'], ck['heads'], ck['head_dim'],
                         gate_norm=ck['args']['gate_norm']).eval()
        reference.bridge.load_state_dict(ck['bridge'])
        for x in SIDES:
            assert pair.bridge[x].gate_norm == ck['args']['gate_norm'], run
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
            assert float((original[0]['h'][x] - donor_changed[0]['h'][x]).abs().max()) > 1e-8
            ep, donor = plans['co_talk'][0]['episode'], plans['co_talk'][0]['donor']
            talk = F.generate_shuffle(pair, ep, donor, 3, keep=True)
            changed = copy.deepcopy(ep)
            asker = ep['meta']['asker']
            changed['system'][partner(asker)] = 'An altered original holder context.'
            isolated = F.generate_shuffle(pair, changed, donor, 3, keep=True)
            torch.testing.assert_close(talk['h'][asker], isolated['h'][asker], rtol=0, atol=0)
            assert len(talk['turns']) == 1 and talk['turns'][0][0] == asker
        del reference, pair, toks, expected, actual, batch, original, isolated, donor_changed, talk
        gc.collect()
