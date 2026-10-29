"""interleaved.py on tiny random models: the episodes alternate the twin that holds the word, each duplex token's piece
and time step match the generation, every pen word has its author and time step, and the cut run is the plain models'
generation."""
import pytest
import torch

import games as G
import interleaved as I
import link
import session as S


@pytest.fixture(scope='module')
def tiny(tiny_models):
    toks, models = tiny_models(['q_proj', 'v_proj'])
    torch.manual_seed(3)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.fill_(.4)
    return pair, toks


def test_draw_alternates(replies):
    for game, role in I.ROLE.items():
        eps = I.draw(G, replies, game, 6, 1)
        assert [ep['meta'][role] for ep in eps] == ['quill', 'ember'] * 3
        assert eps == I.draw(G, replies, game, 6, 1)


def test_records(tiny, replies, monkeypatch):
    pair, toks = tiny
    import evaluate as E
    monkeypatch.setitem(E.MAXNEW, 'relay_word', 6)
    monkeypatch.setitem(E.MAXNEW, 'pen_secret', 3)
    for game in I.ROLE:
        eps = [dict(ep, pen_max_tokens=2) for ep in I.draw(G, replies, game, 4, 2)]
        recs = I.run(pair, toks, game, eps, 2)
        assert [r['index'] for r in recs] == [0, 1, 2, 3]
        for ep, r in zip(eps, recs):
            for c, mode in I.CONDITIONS:
                got = r['conditions'][c]
                again = S.generate(pair, [ep], max_turn=E.MAXNEW.get(game, 64), mode=mode, keep_tokens=True)[0]
                if game == 'pen_secret':
                    assert [w['text'] for w in got['words']] == [w for _, w in again['words']]
                    authors = [S.pen_author(ep, j) for j in range(len(got['words']))]
                    assert [w['author'] for w in got['words']] == authors
                    assert [w['tick'] for w in got['words']] == list(range(len(got['words'])))
                    continue
                for x in link.SIDES:
                    assert ''.join(t['text'] for t in got['tokens'][x]) == got['text'][x] == again['text'][x]
                    assert [t['tick'] for t in got['tokens'][x]] == list(range(len(got['tokens'][x])))
