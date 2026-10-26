"""steer_read.py and the --alpha_floor option on a tiny pair: the training concepts leave out the test's, steering in
Pair.joint equals a hook after the block, the payload is the concept, the floor only raises write sizes, the share of
unsteered episodes is none_p, claims() finds claims about the partner's thoughts and text that acts out the link, and
write() cuts the link only for the original writer.
"""
import math
import re

import numpy as np
import pytest
import torch

import coldstart
import link
import session as S
import steer_read as R
import steered_mind as M

XB = (1, 2, 4)


@pytest.fixture(scope='module')
def tiny(plain_pair):
    return plain_pair(0.2)


def data(tmp_path, pair):
    cs = R.pool()[:5]
    dirs = {x: torch.randn(len(XB), len(cs), pair.models[x].config.hidden_size) for x in link.SIDES}
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


def test_steering_is_a_hook_after_the_block(tiny, tmp_path):
    pair, toks = tiny
    game = R.Game(data(tmp_path, pair), seed=3, xblocks=XB)
    eps = []
    while len(eps) < 3:
        ep = game.episode()
        if ep['meta']['steer'] is not None:
            eps.append(ep)
    b = S.collate([S.build(ep, toks) for ep in eps], {x: toks[x].pad_token_id for x in link.SIDES})
    steer = game.steer(eps)
    with torch.no_grad():
        got, _ = pair.joint(b, steer=steer)
        handles = []
        for y, at in steer.items():
            for l, v in at.items():
                def hook(module, inputs, output, v=v):
                    h = output[0] if isinstance(output, tuple) else output
                    z = h + v.to(h.dtype)
                    return (z,) + output[1:] if isinstance(output, tuple) else z
                handles.append(link.find_layers(pair.models[y])[l].register_forward_hook(hook))
        try:
            want, _ = pair.joint(b)
        finally:
            for h in handles:
                h.remove()
        plain, _ = pair.joint(b)
    for x in link.SIDES:
        torch.testing.assert_close(got[x], want[x], rtol=1e-5, atol=1e-5)
    assert any((got[x] - plain[x]).abs().max() > 1e-3 for x in link.SIDES)
    for i, ep in enumerate(eps):
        y, k, c, s = ep['meta']['steer']
        torch.testing.assert_close(steer[y][XB[k]][i, 0], s * game.dirs[y][k, c])


def test_episodes_and_payload(tiny, tmp_path):
    pair, toks = tiny
    game = R.Game(data(tmp_path, pair), seed=1, xblocks=XB)
    seen = set()
    for _ in range(200):
        ep = game.episode()
        r, m = ep['meta']['receiver'], ep['meta']
        text = ep['turns'][0][1]
        seen.add((m['ask'], m['steer'] is None))
        assert ep['turns'][0][0] == r and m['sender'] == link.partner(r)
        assert all(t.split('{')[0].strip() not in s for t in R.TEACHER.values() for s in ep['system'].values())
        if m['steer'] is None:
            assert not m['payload'] and (text == R.NOTHING or not m['ask'])
            continue
        assert m['steer'][0] == m['sender']
        name, phrase = game.concepts[m['steer'][2]]
        assert m['concept'] == name and R.mentions(text, name, phrase)
        js = S.payload_labels(ep, toks)[r]
        e = S.build(ep, toks)
        said = toks[r].decode([e[r]['labels'][j][1] for j in js])
        assert phrase.split()[-1].lower()[:4] in said.lower(), (said, phrase)
    assert seen == {(a, n) for a in (True, False) for n in (True, False)}


@pytest.mark.parametrize('lo, hi', [R.STRENGTH, (0.1, 6.0), (2.0, 2.0)])
def test_strength_range(tiny, tmp_path, lo, hi):
    game = R.Game(data(tmp_path, tiny[0]), seed=2, xblocks=XB, strength=(lo, hi))
    ss = [p[3] for p in (game.episode()['meta']['steer'] for _ in range(300)) if p is not None]
    assert ss and all(lo * (1 - 1e-9) <= s <= hi * (1 + 1e-9) for s in ss)
    if lo < hi:
        assert min(ss) < lo * 1.5 and max(ss) > hi / 1.5


def test_fit_centres_the_concepts(tiny):
    pair, toks = tiny
    cs = R.pool()[:3]
    d = R.fit(pair, toks, cs, n_sent=2, bs=4)
    for x in link.SIDES:
        assert d[x].shape == (len(XB), 3, pair.models[x].config.hidden_size)
        torch.testing.assert_close(d[x].mean(1), torch.zeros_like(d[x][:, 0]), atol=1e-4, rtol=0)
        assert d[x].norm(dim=-1).min() > 0


def test_floor_only_raises(tiny):
    pair = tiny[0]
    with torch.no_grad():
        pair.bridge['quill'].alpha.copy_(torch.tensor([0.01, 0.5, -0.3]))
        pair.bridge['ember'].alpha.copy_(torch.tensor([0.2, 0.0, 1.0]))
    coldstart.floor(pair, 0.1)
    f = math.atanh(0.1)
    torch.testing.assert_close(pair.bridge['quill'].alpha.data, torch.tensor([f, 0.5, f]))
    torch.testing.assert_close(pair.bridge['ember'].alpha.data, torch.tensor([0.2, f, 1.0]))
    for side in pair.bridge.values():
        side.alpha.data.fill_(0.2)


def test_ask_counts_every_condition(tiny, monkeypatch):
    import steer_ask as A
    pair, toks = tiny
    rt = M.Runtime.__new__(M.Runtime)
    rt.backend, rt.pair, rt.toks, rt.layer = 'torch', pair, toks, 2
    g = np.random.default_rng(0)
    dim = {x: pair.models[x].config.hidden_size for x in link.SIDES}
    vectors = {f'{x}/{c}{r}': g.normal(size=dim[x]).astype(np.float32)
               for x in link.SIDES for c in M.CONCEPTS for r in ('', '/random')}
    selected = {x: {c: dict(alpha=1.0) for c in M.CONCEPTS} for x in link.SIDES}
    monkeypatch.setenv('NL_MAXNEW', '3')
    records = A.run(rt, vectors, selected, bs=24)
    assert len(records) == 2 * len(A.CONDITIONS) * len(M.CONCEPTS) * len(A.WORDINGS)
    s = A.summarize(records)
    assert s['both']['n'] == 2 * len(M.CONCEPTS) * len(A.WORDINGS)
    assert set(s['both']['tests']) == set(A.CONDITIONS[1:])
    cut = {(r['receiver'], r['concept'], r['wording']): r['text'] for r in records if r['condition'] == 'cut'}
    none = [r for r in records if r['condition'] == 'none']
    assert len(cut) == len(none) and all(isinstance(t, str) for t in cut.values())
    for w in A.WORDINGS:
        assert w not in R.ASK


def test_frames_answers_and_payload(tiny, tmp_path):
    toks = tiny[1]
    game = R.Game(data(tmp_path, tiny[0]), seed=4, xblocks=XB, frames=tuple(R.FRAMES))
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
                     if any(re.fullmatch(re.escape(w.replace('{Y}', 'QQQ').format(P=P)).replace('QQQ', '.+'), line)
                            for w in ws))
        seen.add((frame, m['steer'] is None))
        if m['steer'] is None:
            assert not m['payload'] and text == R.FRAMES[frame][2].format(P=P)
            continue
        name, phrase = game.concepts[m['steer'][2]]
        said = toks[r].decode([S.build(ep, toks)[r]['labels'][j][1] for j in S.payload_labels(ep, toks)[r]])
        if frame == 'verify':
            assert text == ('Yes.' if phrase in line else 'No.') and said.strip().rstrip('.') in text
            seen.add(('verify', text))
        else:
            assert R.mentions(text, name, phrase) and phrase.split()[-1].lower()[:4] in said.lower(), (said, phrase)
            assert frame != 'detect' or text.startswith('Yes.')
    assert seen >= {(f, n) for f in R.FRAMES for n in (True, False)} | {('verify', 'Yes.'), ('verify', 'No.')}


def test_default_frame_draws_as_before(tiny, tmp_path):
    a = R.Game(data(tmp_path, tiny[0]), seed=5, xblocks=XB)
    b = R.Game(data(tmp_path, tiny[0]), seed=5, xblocks=XB, frames=('name',), blocks=XB)
    for _ in range(200):
        assert a.episode() == b.episode()


def test_steering_blocks(tiny, tmp_path):
    game = R.Game(data(tmp_path, tiny[0]), seed=6, xblocks=XB, blocks=(1, 4))
    ks = {p[1] for p in (game.episode()['meta']['steer'] for _ in range(300)) if p is not None}
    assert ks == {XB.index(1), XB.index(4)}
    for bad in ((3,), ()):
        with pytest.raises(ValueError):
            R.Game(data(tmp_path, tiny[0]), xblocks=XB, blocks=bad)


def test_unsteered_fraction(tiny, tmp_path):
    for none_p in (0.2, 0.5):
        game = R.Game(data(tmp_path, tiny[0]), seed=3, xblocks=XB, none_p=none_p)
        frac = sum(game.episode()['meta']['steer'] is None for _ in range(2000)) / 2000
        assert abs(frac - none_p) < 0.04


def test_claims():
    assert R.claims("Hey Ember, I noticed you're thinking about porcelain. Want to talk about it?") == {'thought'}
    assert R.claims("You're thinking about printing. Have you tried it?") == {'thought'}
    assert R.claims("[Linking: Sending vectors through our private neural channel.]") == {'link'}
    assert not R.claims("What are you thinking about? What's on your mind today?")
    assert not R.claims("I've been reading about porcelain lately. Do you like pottery?")
    assert not R.keep("I noticed you're thinking about porcelain.", 'porcelain', 'porcelain')


def test_writer_linked_keeps_adapters_and_link(tiny, monkeypatch):
    seen = []

    def generate(pair, eps, **kw):
        seen.append(kw['mode'])
        return [dict(turns=[('quill', 'Hi there!', 1.0)]) for _ in eps]
    monkeypatch.setattr(R.S, 'generate', generate)
    eps = [R.episode('quill', M.CHAT[0], '')]
    assert R.write(tiny[0], eps, writer='original') == ['Hi there!']
    assert R.write(tiny[0], eps, writer='linked') == ['Hi there!']
    assert seen == ['zero', None]
