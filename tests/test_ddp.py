"""Data parallel training on tiny random models (CPU, float32, gloo): three steps of train.py (duplex, dialogue,
benchmark, co-reading and co_distill episodes) in one process, in two torchrun ranks, and in one process with --micro 3
(the batch as three groups of similar length). The summed gradients of the two ranks, and of the three groups, equal
the single-process gradients on the same global batch.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

TORCHRUN = Path(sys.executable).parent / 'torchrun'


@pytest.mark.slow
@pytest.mark.skipif(not TORCHRUN.exists(), reason='needs torchrun beside the Python interpreter')
def test_ranks_and_groups_sum_to_the_single_gradient(tree, data, tmp_path):
    env = dict(os.environ, MPD_MEM_GIB='8', OMP_NUM_THREADS='2')
    common = [str(tree / 'pairing' / 'train.py'), '--replies', str(data / 'replies.json'), '--steps', '3', '--bs', '12',
              '--dialogue', '0.4', '--qa', '0.2', '--coread', '0.5', '--coread_data',
              str(data / 'coread.json'), '--ckpt', 'none', '--first_eval', '0', '--heads', '2', '--head_dim', '16',
              '--xblocks', '1,2,4']
    for name, cmd, extra in (('single', [sys.executable], []),
                             ('ddp', [str(TORCHRUN), '--nproc_per_node', '2', '--master_port', '29517'], []),
                             ('micro', [sys.executable], ['--micro', '3'])):
        out = str(tmp_path / name)
        r = subprocess.run(cmd + common + extra + ['--out', out, '--grad_check', out], cwd=tree, env=env,
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        assert r.returncode == 0, (name, r.stderr[-3000:])
        games = {k for l in r.stdout.splitlines() if l.startswith('{') for k in json.loads(l)['games']}
        assert 'co_distill' in games and any(g.startswith('co_') and g != 'co_distill' for g in games), sorted(games)
    worst = 0.0
    for other in ('ddp', 'micro'):
        for step in (1, 2, 3):
            a = torch.load(tmp_path / f'single.{step}.pt')
            b = torch.load(tmp_path / f'{other}.{step}.pt')
            assert set(a) <= set(b), sorted(set(a) - set(b))[:5]
            d = max(float((a[k] - b[k]).abs().max()) for k in a)
            extra = max([float(b[k].abs().max()) for k in set(b) - set(a)] + [0.0])
            worst = max(worst, d, extra)
    assert worst < 1e-5, worst
