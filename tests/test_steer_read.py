"""steer_read.py and train.py --steer / --alpha_floor on tiny Qwen/AFM pairs: the training concepts leave out the test's,
a push in Pair.joint is the steered twin of steered_mind (a hook after the block), the payload is the concept, and the
floor only raises write sizes.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu python -m pytest -q tests/test_steer_read.py
"""
import os
os.environ.update(NL_TINY='1', NL_DEVICE='cpu', HF_HUB_OFFLINE='1')
from pathlib import Path
import math
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import torch
import coldstart
import link
import session as S
import steer_read as R
import steered_mind as M

XB = (1, 2, 4)


def tiny():
    torch.manual_seed(0)
    toks, models = {}, {}
    for x in link.SIDES:
        toks[x], models[x] = link.load(x)
    pair = link.Pair(models, toks, XB, heads=2, head_dim=16).eval()
    for side in pair.bridge.values():
        side.alpha.data.fill_(0.2)
    return pair, toks


PAIR, TOKS = tiny()


def data(tmp_path):
    cs = R.pool()[:5]
    dirs = {x: torch.randn(len(XB), len(cs), PAIR.models[x].config.hidden_size) for x in link.SIDES}
    openers = [dict(receiver=x, wording=1, concept=c, text=f'Hi! Are you thinking about {cs[c][1]} today?')
               for x in link.SIDES for c in range(len(cs))]
    openers += [dict(receiver=x, wording=2, concept=None, text='Hi! How is your day going so far?') for x in link.SIDES]
    path = tmp_path / 'steer.pt'
    torch.save(dict(concepts=cs, directions=dirs, openers=openers, xblocks=XB, stats={}), path)
    return path


def test_pool_leaves_out_the_test_concepts():
    cs = R.pool()
    assert len(cs) > 400 and len({n.lower() for n, _ in cs}) == len(cs)
    for name, phrase in cs:
        assert name not in R.NEAR and phrase not in R.NEAR
        for c in M.CONCEPTS:
            assert not M.matches(name, c) and not M.matches(phrase, c), (name, c)
    names = {n for n, _ in cs}
    assert not names & {'the ocean', 'fire', 'music', 'cat', 'winter', 'snow', 'football', 'train', 'painting',
                        'mountains', 'space', 'cooking', 'flowers', 'garden'}


def test_mentions():
    assert R.mentions('I love giraffes!', 'giraffe', 'giraffes') == [(7, 15)]
    assert R.mentions('Red is my favourite.', 'red', 'the color red') == [(0, 3)]
    assert R.mentions('A category of cats', 'cat', 'cats') == [(14, 18)]
    assert not R.mentions('Concatenate', 'cat', 'cats')


def test_push_is_a_hook_after_the_block(tmp_path):
    game = R.Game(data(tmp_path), seed=3, xblocks=XB)
    eps = []
    while len(eps) < 3:
        ep = game.episode()
        if ep['meta']['push'] is not None:
            eps.append(ep)
    b = S.collate([S.build(ep, TOKS) for ep in eps], {x: TOKS[x].pad_token_id for x in link.SIDES})
    push = game.push(eps)
    with torch.no_grad():
        got, _ = PAIR.joint(b, push=push)
        handles = []
        for y, at in push.items():
            for l, v in at.items():
                def hook(module, inputs, output, v=v):
                    h = output[0] if isinstance(output, tuple) else output
                    z = h + v.to(h.dtype)
                    return (z,) + output[1:] if isinstance(output, tuple) else z
                handles.append(link.find_layers(PAIR.models[y])[l].register_forward_hook(hook))
        try:
            want, _ = PAIR.joint(b)
        finally:
            for h in handles:
                h.remove()
        plain, _ = PAIR.joint(b)
    for x in link.SIDES:
        torch.testing.assert_close(got[x], want[x], rtol=1e-5, atol=1e-5)
    assert any((got[x] - plain[x]).abs().max() > 1e-3 for x in link.SIDES)
    for i, ep in enumerate(eps):
        y, k, c, s = ep['meta']['push']
        torch.testing.assert_close(push[y][XB[k]][i, 0], s * game.dirs[y][k, c])


def test_episodes_and_payload(tmp_path):
    game = R.Game(data(tmp_path), seed=1, xblocks=XB)
    seen = set()
    for _ in range(200):
        ep = game.episode()
        r, m = ep['meta']['receiver'], ep['meta']
        text = ep['turns'][0][1]
        seen.add((m['ask'], m['push'] is None))
        assert ep['turns'][0][0] == r and m['sender'] == link.partner(r)
        assert all(R.PRIVILEGED.split('{')[0].strip() not in s for s in ep['system'].values())
        if m['push'] is None:
            assert not m['payload'] and (text == R.NOTHING or not m['ask'])
            continue
        assert m['push'][0] == m['sender']
        name, phrase = game.concepts[m['push'][2]]
        assert m['concept'] == name and R.mentions(text, name, phrase)
        js = S.payload_labels(ep, TOKS)[r]
        e = S.build(ep, TOKS)
        said = TOKS[r].decode([e[r]['labels'][j][1] for j in js])
        assert phrase.split()[-1].lower()[:4] in said.lower(), (said, phrase)
    assert seen == {(a, n) for a in (True, False) for n in (True, False)}


def test_strength_range(tmp_path):
    for lo, hi in (R.STRENGTH, (0.1, 6.0), (2.0, 2.0)):
        game = R.Game(data(tmp_path), seed=2, xblocks=XB, strength=(lo, hi))
        ss = [p[3] for p in (game.episode()['meta']['push'] for _ in range(300)) if p is not None]
        assert ss and all(lo * (1 - 1e-9) <= s <= hi * (1 + 1e-9) for s in ss)
        if lo < hi:
            assert min(ss) < lo * 1.5 and max(ss) > hi / 1.5


def test_fit_centres_the_concepts():
    cs = R.pool()[:3]
    d = R.fit(PAIR, TOKS, cs, n_sent=2, bs=4)
    for x in link.SIDES:
        assert d[x].shape == (len(XB), 3, PAIR.models[x].config.hidden_size)
        torch.testing.assert_close(d[x].mean(1), torch.zeros_like(d[x][:, 0]), atol=1e-4, rtol=0)
        assert d[x].norm(dim=-1).min() > 0


def test_floor_only_raises():
    with torch.no_grad():
        PAIR.bridge['quill'].alpha.copy_(torch.tensor([0.01, 0.5, -0.3]))
        PAIR.bridge['ember'].alpha.copy_(torch.tensor([0.2, 0.0, 1.0]))
    coldstart.floor(PAIR, 0.1)
    f = math.atanh(0.1)
    torch.testing.assert_close(PAIR.bridge['quill'].alpha.data, torch.tensor([f, 0.5, f]))
    torch.testing.assert_close(PAIR.bridge['ember'].alpha.data, torch.tensor([0.2, f, 1.0]))
    for side in PAIR.bridge.values():
        side.alpha.data.fill_(0.2)


def test_ask_counts_every_condition():
    import numpy as np
    import steer_ask as A
    rt = M.Runtime.__new__(M.Runtime)
    rt.backend, rt.pair, rt.toks, rt.layer = 'torch', PAIR, TOKS, 2
    g = np.random.default_rng(0)
    dim = {x: PAIR.models[x].config.hidden_size for x in link.SIDES}
    vectors = {f'{x}/{c}{r}': g.normal(size=dim[x]).astype(np.float32)
               for x in link.SIDES for c in M.CONCEPTS for r in ('', '/random')}
    selected = {x: {c: dict(alpha=1.0) for c in M.CONCEPTS} for x in link.SIDES}
    os.environ['NL_MAXNEW'] = '3'
    try:
        records = A.run(rt, vectors, selected, bs=24)
    finally:
        del os.environ['NL_MAXNEW']
    assert len(records) == 2 * len(A.CONDITIONS) * len(M.CONCEPTS) * len(A.WORDINGS)
    s = A.summarize(records)
    assert s['both']['n'] == 2 * len(M.CONCEPTS) * len(A.WORDINGS)
    assert set(s['both']['tests']) == set(A.CONDITIONS[1:])
    cut = {(r['receiver'], r['concept'], r['wording']): r['text'] for r in records if r['condition'] == 'cut'}
    none = [r for r in records if r['condition'] == 'none']
    assert len(cut) == len(none) and all(isinstance(t, str) for t in cut.values())
    for w in A.WORDINGS:
        assert w not in R.ASK


def test_frames_answers_and_payload(tmp_path):
    game = R.Game(data(tmp_path), seed=4, xblocks=XB, frames=tuple(R.FRAMES))
    seen = set()
    for _ in range(600):
        ep = game.episode()
        m = ep['meta']
        if not m['ask']:
            continue
        r, text = m['receiver'], ep['turns'][0][1]
        line = ep['system'][r].split('\n\n')[-1]
        P = link.MODELS[m['sender']]['name']
        frame = next(f for f, (ws, _, _) in R.FRAMES.items()
                     if any(re.fullmatch(re.escape(w.replace('{Y}', 'QQQ').format(P=P)).replace('QQQ', '.+'), line) for w in ws))
        seen.add((frame, m['push'] is None))
        if m['push'] is None:
            assert not m['payload'] and text == R.FRAMES[frame][2].format(P=P)
            continue
        name, phrase = game.concepts[m['push'][2]]
        said = TOKS[r].decode([S.build(ep, TOKS)[r]['labels'][j][1] for j in S.payload_labels(ep, TOKS)[r]])
        if frame == 'verify':
            assert text == ('Yes.' if phrase in line else 'No.') and said.strip().rstrip('.') in text
            seen.add(('verify', text))
        else:
            assert R.mentions(text, name, phrase) and phrase.split()[-1].lower()[:4] in said.lower(), (said, phrase)
            assert frame != 'detect' or text.startswith('Yes.')
    assert seen >= {(f, n) for f in R.FRAMES for n in (True, False)} | {('verify', 'Yes.'), ('verify', 'No.')}


def test_default_frame_draws_as_before(tmp_path):
    a = R.Game(data(tmp_path), seed=5, xblocks=XB)
    b = R.Game(data(tmp_path), seed=5, xblocks=XB, frames=('name',), blocks=XB)
    for _ in range(200):
        assert a.episode() == b.episode()


def test_push_blocks(tmp_path):
    game = R.Game(data(tmp_path), seed=6, xblocks=XB, blocks=(1, 4))
    ks = {p[1] for p in (game.episode()['meta']['push'] for _ in range(300)) if p is not None}
    assert ks == {XB.index(1), XB.index(4)}
    for bad in ((3,), ()):
        try:
            R.Game(data(tmp_path), xblocks=XB, blocks=bad)
        except ValueError:
            continue
        raise AssertionError(bad)
