"""The shared pen (games.PEN, mode 'pen': the twins write one text in turns) on tiny random models: the joint pass equals
generation and replay, each text is one exactly tokenized assistant message, every word has its author, the payload is
the receiver's, the private task reaches the receiver only through the link, later words never change earlier
predictions, and an optimizer step on both LoRAs and the link has finite, nonzero gradients.
"""
import pytest
import torch

import games as G
import link
import session as S


@pytest.fixture(scope='module')
def tiny(tiny_models):
    toks, models = tiny_models(['q_proj', 'v_proj', 'up_proj', 'down_proj'])
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.fill_(.4)
            side.age.normal_(0, .3)
            for gate in side.gate:
                gate.weight.normal_(0, .1)
    return pair, toks, {x: toks[x].pad_token_id for x in link.SIDES}


@pytest.fixture
def g(replies):
    return G.Games(replies, seed=11)


def check(tiny, eps, force=True):
    pair, toks, pads = tiny
    result = S.generate(pair, eps, force=force, keep=True, max_turn=3)
    if not force:
        eps = [S.replay(ep, r) for ep, r in zip(eps, result)]
    built = [S.build(ep, toks) for ep in eps]
    with torch.no_grad():
        h, gates = pair.joint(S.collate(built, pads))
    for i, (ep, e, r) in enumerate(zip(eps, built, result)):
        replay = S.build(S.replay(ep, r), toks)
        for x in link.SIDES:
            assert e[x]['ids'] == replay[x]['ids'] and e[x]['time'] == replay[x]['time']
            assert r['h'][x].shape == h[x][i, :len(e[x]['ids'])].shape
            torch.testing.assert_close(r['h'][x], h[x][i, :len(e[x]['ids'])], atol=1e-5, rtol=1e-5)
            ps = [p for p, _, _ in e[x]['labels']]
            if ps:
                with torch.no_grad():
                    a = link.inner(pair.models[x])[1](h[x][i, ps])
                    b = link.inner(pair.models[x])[1](r['h'][x][ps])
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            expected = S.prefix(toks[x], x, ep['system'][x], ep['task'][x]) + S.assistant_header(toks[x], x)
            expected += S.ids(toks[x], ep['text']) + [toks[x].convert_tokens_to_ids('<|im_end|>')]
            assert e[x]['ids'] == expected, 'must be one exactly tokenized assistant message'
            for word, start, end in e[x]['spans']:
                assert S.pen_author(ep, word) == x
                assert end > start
    return result


@pytest.mark.parametrize('chunk', [1, 2, 3])
def test_every_game_and_chunk(tiny, g, chunk):
    toks = tiny[1]
    eps = [g.episode(name, pen_chunk=chunk) for name in G.PEN]
    eps += [dict(eps[0], text="Café crème! naïve 中文 🦊 don't stop.", pen_start='ember')]
    check(tiny, eps)
    ep = eps[0]
    payload = S.payload_labels(ep, toks)
    receiver = ep['meta']['receiver']
    assert list(payload) == [receiver] and payload[receiver]
    assert ep['meta']['value'] not in ep['task'][receiver]
    built = S.build(ep, toks)
    for j in payload[receiver]:
        assert built[receiver]['labels'][j][2] == 1.


def test_texts_payload_and_chunks(tiny, g):
    toks = tiny[1]
    check(tiny, [dict(g.episode('pen_free'), text=t) for t in ('', 'Hello!', 'One\n\ntwo\tthree  ')])
    repeated = dict(g.episode('pen_secret'), text='acorn acorn acorn acorn', pen_start='quill',
                    meta=dict(holder='ember', receiver='quill', value='acorn'))
    assert S.payload(repeated) == {'quill': [(0, 5), (12, 17)]}
    chosen = S.payload_labels(repeated, toks)['quill']
    labels = S.build(repeated, toks)['quill']['labels']
    assert toks['quill'].decode([labels[j][1] for j in chosen]) == 'acorn acorn'
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            S.build(dict(repeated, pen_chunk=invalid), toks)


def test_private_task_reaches_the_receiver_only_through_the_link(tiny, g):
    pair, toks, pads = tiny
    secret = g.episode('pen_secret')
    holder, receiver = secret['meta']['holder'], secret['meta']['receiver']
    other = dict(secret, task=dict(secret['task'], **{holder: 'Our private ending is a completely different violet whale.'}))
    with torch.no_grad():
        batches = [S.collate([S.build(e, toks)], pads) for e in (secret, other)]
        linked = [pair.joint(b)[0][receiver] for b in batches]
        cut = [pair.joint(b, mode='zero')[0][receiver] for b in batches]
    indices = S.payload_labels(secret, toks)[receiver]
    positions = [S.build(secret, toks)[receiver]['labels'][j][0] for j in indices]
    assert float((linked[0][0, positions] - linked[1][0, positions]).abs().max()) > 1e-5
    assert torch.equal(cut[0], cut[1])


def test_later_words_never_change_earlier_predictions(tiny, g):
    pair, toks, pads = tiny
    ep = dict(g.episode('pen_free'), text='Warm lanterns guide small foxes home')
    changed = dict(ep, text='Warm lanterns guide giant dragons away')
    a, b = [S.build(e, toks) for e in (ep, changed)]
    with torch.no_grad():
        ha, _ = pair.joint(S.collate([a], pads))
        hb, _ = pair.joint(S.collate([b], pads))
    for x in link.SIDES:
        ps = [p for word, start, end in a[x]['spans'] if word < 3 for p, _, _ in a[x]['labels'][start:end]]
        torch.testing.assert_close(ha[x][0, ps], hb[x][0, ps], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('chunk', [1, 2])
def test_free_generation_replays(tiny, g, chunk):
    check(tiny, [dict(g.episode('pen_free', pen_chunk=chunk), pen_max_tokens=3)], force=False)


def test_optimizer_step(tiny, g):
    pair, toks, pads = tiny
    params = [p for m in pair.models.values() for p in m.parameters() if p.requires_grad] + list(pair.bridge.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    batch = S.collate([S.build(g.episode(name), toks) for name in G.PEN], pads)
    h, _ = pair.joint(batch)
    loss = sum(torch.nn.functional.cross_entropy(pair.token_logits(x, h[x], batch[x]['idx']), batch[x]['lab'])
               for x in link.SIDES)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    assert sum(float(p.grad.abs().sum()) for p in pair.bridge.parameters() if p.grad is not None) > 0
    opt.step()
