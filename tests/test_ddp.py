"""Data parallel check on tiny random models (CPU, float32, gloo): three training steps (cooperative, competitive,
cooperative; duplex, dialogue and benchmark episodes) in one process and in two torchrun ranks. The summed gradients of
the two ranks must equal the single-process gradients on the same global batch.

Usage: python tests/test_ddp.py --replies REPLIES.json [--soft]   (from the repository root)
"""
import argparse
import os
import subprocess
import sys
import tempfile

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--ranks', type=int, default=2)
ap.add_argument('--soft', action='store_true', help='with the soft-token read')
args = ap.parse_args()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ, NL_TINY='1', NL_DEVICE='cpu', OMP_NUM_THREADS='2')
tmp = tempfile.mkdtemp()
common = ['pairing/train.py', '--replies', args.replies, '--steps', '3', '--bs', '8', '--comp', '2', '--dialogue', '0.5',
          '--qa', '0.3', '--ckpt', 'none', '--first_eval', '0', '--p3', '0.5'] + (['--soft'] if args.soft else [])
torchrun = os.path.join(os.path.dirname(sys.executable), 'torchrun')
for name, cmd in (('single', [sys.executable]), ('ddp', [torchrun, '--nproc_per_node', str(args.ranks), '--master_port', '29517'])):
    out = os.path.join(tmp, name)
    r = subprocess.run(cmd + common + ['--out', out, '--grad_check', out], cwd=root, env=env, capture_output=True, text=True)
    print(f'--- {name}: exit {r.returncode}')
    print('\n'.join(l[:300] for l in r.stdout.splitlines() if l.startswith('{')))
    if r.returncode:
        print(r.stderr[-3000:])
        sys.exit(1)

worst_abs = worst_rel = 0.0
for step in (1, 2, 3):
    a = torch.load(os.path.join(tmp, f'single.{step}.pt'))
    b = torch.load(os.path.join(tmp, f'ddp.{step}.pt'))
    assert set(a) <= set(b), sorted(set(a) - set(b))[:5]
    d = max(float((a[k] - b[k]).abs().max()) for k in a)
    rel = max(float((a[k] - b[k]).norm() / (a[k].norm() + 1e-12)) for k in a)
    extra = max([float(b[k].abs().max()) for k in set(b) - set(a)] + [0.0])
    print(f'step {step} ({"competitive" if step == 2 else "cooperative"}): {len(a)} gradient tensors, '
          f'max |single - ddp| {d:.2e}, max relative {rel:.2e}, largest gradient missing in single {extra:.1e}')
    worst_abs, worst_rel = max(worst_abs, d, extra), max(worst_rel, rel)
assert worst_abs < 1e-5, worst_abs
print('all checks passed')
