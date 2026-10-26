"""The content-matched initialization of the reads (align.py) on tiny random models (CPU, float32): the ridge fit
recovers a known linear map, with its penalty chosen on held-out states; with the same model on both sides each reader
then attends to the matched partner position far above chance; with both architectures the initialization changes
only the intended projections, and at alpha 0 the pair's logits still equal the plain models' bit for bit.
"""
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import align
import link
import session as S
from link import SIDES, Pair, inner

XB = (1, 2, 4)
WORDS = ('river stone light quiet market seven window garden copper winter music letter bridge orange signal paper '
         'forest engine silver morning cloud number harbor thread lantern valley mirror pocket candle storm').split()
EPISODE = dict(mode='duplex', system={x: f'You are {link.MODELS[x]["name"]}.' for x in SIDES},
               task={'quill': 'What is two plus two?', 'ember': 'Name three colours.'},
               reply={'quill': ('Four, of course! It is simple.', 1.0), 'ember': ('Red, green and blue.', 1.0)})


def replies(n, seed=0):
    """replies.json content: n co-reading documents and n own replies per side, made of random words."""
    rng = random.Random(seed)

    def text(k):
        return ' '.join(rng.choice(WORDS) + rng.choice(['', ',', '.', '!']) for _ in range(k)) + '.'
    docs = {text(40): text(10) for _ in range(n)}
    return {x: dict(coread=docs, own={f'Tell me about {rng.choice(WORDS)} {i}.': text(30) for i in range(n)})
            for x in SIDES}


def load():
    toks, models = {}, {}
    for x in SIDES:
        toks[x], models[x] = link.load(x)
        models[x].eval()
    return toks, models


def pair_of(models, toks, seed=0):
    sizes = {x: link.residual_rms(models[x], XB, [S.build(EPISODE, toks)[x]['ids']]) for x in SIDES}
    torch.manual_seed(seed)
    return Pair(models, toks, XB, sizes, heads=4, head_dim=32)


def test_fit_recovers_a_known_map():
    rng = np.random.default_rng(0)
    dx, dy, n, nt = 24, 20, 4000, 1000
    A = rng.normal(size=(dx, dy)) / np.sqrt(dy)
    mu = rng.normal(size=dx)

    def sums(m):
        y = rng.normal(size=(m, dy)) + 0.5
        x = mu + (y - 0.5) @ A.T + 0.01 * rng.normal(size=(m, dx))
        d = dict(xx=x.T @ x, yy=y.T @ y, xy=x.T @ y, sx=x.sum(0), sy=y.sum(0), n=float(m))
        return {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}

    M, mx, r2, lam = align.fit(sums(n), sums(nt), torch.ones(dx, dtype=torch.float64), torch.ones(dy, dtype=torch.float64))
    err = np.linalg.norm(M.numpy() - A) / np.linalg.norm(A)
    assert err < 1e-2 and r2 > 0.999 and np.allclose(mx.numpy(), mu, atol=0.1), (err, r2)


def same_model_check():
    """Run with NL_EMBER set to quill's model: matched states are nearly identical, so after the initialization each
    reader attends to the matched partner position far above chance."""
    toks, models = load()
    out = align.content_init(pair_of(models, toks), models, toks, replies(30), 40, log=lambda s: None)
    for r in out:
        assert r['r2'] > 0.8 and r['after']['top1'] > 20 * r['chance'], r
        assert r['after']['logp'] > r['before']['logp'] + 1.0, r


def test_same_model_on_both_sides():
    # link.MODELS reads NL_EMBER at import, so the check runs in a fresh process.
    here = Path(__file__).resolve().parent
    env = dict(os.environ, NL_EMBER=link.MODELS['quill']['hf'],
               PYTHONPATH=os.pathsep.join([str(here.parent / 'pairing'), str(here)]))
    r = subprocess.run([sys.executable, '-c', 'import test_align; test_align.same_model_check()'], env=env,
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.parametrize('vo', [False, True])
def test_both_architectures(vo):
    toks, models = load()
    pads = {x: toks[x].pad_token_id for x in SIDES}
    pair = pair_of(models, toks)
    before = {k: v.clone() for k, v in pair.bridge.state_dict().items()}
    out = align.content_init(pair, models, toks, replies(20), 24, vo=vo, log=lambda s: None)
    changed = sorted({k.split('.')[1] for k, v in pair.bridge.state_dict().items() if not torch.equal(v, before[k])})
    assert changed == (['k', 'o', 'q', 'v'] if vo else ['k', 'q']), changed
    assert all(np.isfinite([r['r2'], r['after']['logp']]).all() for r in out), out
    b = S.collate([S.build(EPISODE, toks)], pads)
    with torch.no_grad():
        last, _ = pair.joint(b)
        for x in SIDES:
            plain = models[x](input_ids=b[x]['ids'], attention_mask=b[x]['att'], position_ids=b[x]['pos']).logits
            assert torch.equal(plain, inner(models[x])[1](last[x])), x
