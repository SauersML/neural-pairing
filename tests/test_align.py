"""Checks of the content-matched initialization of the reads (align.py) on tiny random models (CPU, float32).

a. The ridge fit recovers a known linear map between synthetic states, with its penalty chosen on the test states.
b. With the same tiny model on both sides (NL_EMBER set to quill's model; run in a subprocess), matched states are
   nearly identical, so after the initialization each reader attends to the matched partner position far above chance.
c. With both architectures the initialization runs (also with values and outputs), and at alpha 0 the pair's logits still
   equal the plain models' bit for bit.

Usage: python tests/test_align.py   (from the repository root)
"""
import os
import random
import subprocess
import sys

os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pairing'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import align  # noqa: E402
import link  # noqa: E402
import session as S  # noqa: E402
from link import SIDES, Pair, inner  # noqa: E402

XB = (1, 2, 4)
WORDS = ('river stone light quiet market seven window garden copper winter music letter bridge orange signal paper '
         'forest engine silver morning cloud number harbor thread lantern valley mirror pocket candle storm').split()


def replies(n, seed=0):
    """Fake replies.json content: n co-reading documents and n own replies per side, made of random words."""
    rng = random.Random(seed)

    def text(k):
        return ' '.join(rng.choice(WORDS) + rng.choice(['', ',', '.', '!']) for _ in range(k)) + '.'
    docs = {text(40): text(10) for _ in range(n)}
    return {x: dict(coread=docs, own={f'Tell me about {rng.choice(WORDS)} {i}.': text(30) for i in range(n)})
            for x in SIDES}


def pair_of(models, toks, seed=0):
    sizes = {x: link.residual_rms(models[x], XB, [S.build(align_episode, toks)[x]['ids']]) for x in SIDES}
    torch.manual_seed(seed)
    return Pair(models, toks, XB, sizes, heads=4, head_dim=32)


system = {x: f'You are {link.MODELS[x]["name"]}.' for x in SIDES}
align_episode = dict(mode='duplex', system=system, task={'quill': 'What is two plus two?', 'ember': 'Name three colours.'},
                     reply={'quill': ('Four, of course! It is simple.', 1.0), 'ember': ('Red, green and blue.', 1.0)})

if sys.argv[1:] == ['same']:
    toks, models = {}, {}
    for x in SIDES:
        toks[x], models[x] = link.load(x)
        models[x].eval()
    pair = pair_of(models, toks)
    out = align.content_init(pair, models, toks, replies(30), 40, log=lambda s: None)
    for r in out:
        print(r)
        assert r['r2'] > 0.8 and r['after']['top1'] > 20 * r['chance'], r
        assert r['after']['logp'] > r['before']['logp'] + 1.0, r
    print(f"b. the same model on both sides: test R^2 {min(r['r2'] for r in out):.3f} or more; matched top-1 "
          f"{min(r['before']['top1'] for r in out):.2f}-{max(r['before']['top1'] for r in out):.2f} before, "
          f"{min(r['after']['top1'] for r in out):.2f}-{max(r['after']['top1'] for r in out):.2f} after (chance "
          f"{max(r['chance'] for r in out):.3f} or less)")
    sys.exit(0)

# a. synthetic states
rng = np.random.default_rng(0)
dx, dy, n, nt = 24, 20, 4000, 1000
A = rng.normal(size=(dx, dy)) / np.sqrt(dy)
mu = rng.normal(size=dx)


def sums(m):
    y = rng.normal(size=(m, dy)) + 0.5
    x = mu + (y - 0.5) @ A.T + 0.01 * rng.normal(size=(m, dx))
    return dict(xx=x.T @ x, yy=y.T @ y, xy=x.T @ y, sx=x.sum(0), sy=y.sum(0), n=float(m))


M, mx, r2, lam = align.fit(sums(n), sums(nt), np.ones(dx), np.ones(dy))
err = np.linalg.norm(M - A) / np.linalg.norm(A)
print(f'a. synthetic map: relative error {err:.1e}, test R^2 {r2:.5f}, penalty {lam:g} of the mean eigenvalue')
assert err < 1e-2 and r2 > 0.999 and np.allclose(mx, mu, atol=0.1), (err, r2)

# b. the same model on both sides
env = dict(os.environ, NL_EMBER=link.MODELS['quill']['hf'])
subprocess.run([sys.executable, os.path.abspath(__file__), 'same'], env=env, check=True)

# c. both architectures
toks, models = {}, {}
for x in SIDES:
    toks[x], models[x] = link.load(x)
    models[x].eval()
pads = {x: toks[x].pad_token_id for x in SIDES}
for vo in (False, True):
    pair = pair_of(models, toks)
    before = {k: v.clone() for k, v in pair.bridge.state_dict().items()}
    out = align.content_init(pair, models, toks, replies(20), 24, vo=vo, log=lambda s: None)
    changed = sorted({k.split('.')[1] for k, v in pair.bridge.state_dict().items() if not torch.equal(v, before[k])})
    assert changed == (['k', 'o', 'q', 'v'] if vo else ['k', 'q']), changed
    assert all(np.isfinite([r['r2'], r['after']['logp']]).all() for r in out), out
    b = S.collate([S.build(align_episode, toks)], pads)
    with torch.no_grad():
        last, _ = pair.joint(b)
        for x in SIDES:
            plain = models[x](input_ids=b[x]['ids'], attention_mask=b[x]['att'], position_ids=b[x]['pos']).logits
            assert torch.equal(plain, inner(models[x])[1](last[x])), x
    print(f"c. two architectures, {'q, k, v, o' if vo else 'q, k'}: test R^2 "
          f"{min(r['r2'] for r in out):.2f}-{max(r['r2'] for r in out):.2f}, matched top-1 "
          f"{min(r['after']['top1'] for r in out):.2f}-{max(r['after']['top1'] for r in out):.2f} (chance "
          f"{max(r['chance'] for r in out):.3f} or less); at alpha 0 the logits equal the plain models' bit for bit")
