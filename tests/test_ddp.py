"""Data parallel check on tiny random models (CPU, float32, gloo): three training steps (cooperative, competitive,
cooperative; duplex, dialogue, benchmark and co-reading episodes, the co_distill KL loss among them) in one process
and in two torchrun ranks. The summed gradients of the two ranks must equal the single-process gradients on the same
global batch.

Usage: python tests/test_ddp.py --replies REPLIES.json [--coread_data data/coread_check.json]   (from the repository root)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--coread_data', default='data/coread_check.json')
ap.add_argument('--ranks', type=int, default=2)
args = ap.parse_args()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, 'pairing'))
from coread import request  # noqa: E402

env = dict(os.environ, NL_TINY='1', NL_DEVICE='cpu', OMP_NUM_THREADS='2')
tmp = tempfile.mkdtemp()
# made-up own replies to the co-reading requests, so co_distill episodes (and their teacher pass) are drawn
R = json.load(open(args.replies))
data = json.load(open(os.path.join(root, args.coread_data)))
for x in R:
    R[x]['coread'] = {request(it)[2]: f'{x} reads: ' + ' '.join(request(it)[0].split()[:12])
                      for split in ('train', 'test') for it in data[split] if request(it)}
replies = os.path.join(tmp, 'replies.json')
json.dump(R, open(replies, 'w'))
common = ['pairing/train.py', '--replies', replies, '--steps', '3', '--bs', '12', '--comp', '2', '--dialogue', '0.4',
          '--qa', '0.2', '--coread', '0.5', '--coread_data', args.coread_data, '--ckpt', 'none', '--first_eval', '0',
          '--heads', '2', '--head_dim', '16', '--xblocks', '1,2,4']
torchrun = os.path.join(os.path.dirname(sys.executable), 'torchrun')
for name, cmd in (('single', [sys.executable]), ('ddp', [torchrun, '--nproc_per_node', str(args.ranks), '--master_port', '29517'])):
    out = os.path.join(tmp, name)
    r = subprocess.run(cmd + common + ['--out', out, '--grad_check', out], cwd=root, env=env, capture_output=True, text=True)
    print(f'--- {name}: exit {r.returncode}')
    lines = [l for l in r.stdout.splitlines() if l.startswith('{')]
    print('\n'.join(l[:400] for l in lines))
    if r.returncode:
        print(r.stderr[-3000:])
        sys.exit(1)
    games = set(k for l in lines for k in json.loads(l)['games'])
    assert 'co_distill' in games and any(g.startswith('co_') and g != 'co_distill' for g in games), sorted(games)

worst_abs = worst_rel = 0.0
for step in (1, 2, 3):
    a = torch.load(os.path.join(tmp, f'single.{step}.pt'))
    b = torch.load(os.path.join(tmp, f'ddp.{step}.pt'))
    assert set(a) <= set(b), sorted(set(a) - set(b))[:5]
    d = max(float((a[k] - b[k]).abs().max()) for k in a)
    rel = max(float((a[k] - b[k]).norm() / (a[k].norm() + 1e-12)) for k in a)
    extra = max([float(b[k].abs().max()) for k in set(b) - set(a)] + [0.0])
    nb = sum(1 for k in a if k.startswith('bridge.') and float(a[k].abs().max()) > 0)
    print(f'step {step} ({"competitive" if step == 2 else "cooperative"}): {len(a)} gradient tensors ({nb} nonzero of '
          f'the bridges), max |single - ddp| {d:.2e}, max relative {rel:.2e}, largest gradient missing in single {extra:.1e}')
    worst_abs, worst_rel = max(worst_abs, d, extra), max(worst_rel, rel)
assert worst_abs < 1e-5, worst_abs
print('all checks passed')
