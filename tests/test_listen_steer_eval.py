"""listen_steer_eval.py on tiny random models: the drawn test episodes (n per writer, distinct questions, test concepts
only, batch neighbours with different concepts), the shuffled partner's episode, one record per episode under link,
cut and the other episode's partner with only the writer's text generated, and the summary counts and sign tests.
"""
import json

import pytest

import evaluate as E
import games as G
import link
import listen_steer_eval as L


@pytest.fixture(scope='module')
def drawn(data):
    replies = json.loads((data / 'replies.json').read_text())
    return L.draw(G.Games(replies, 'test', seed=4242), 6)


def test_draw(drawn):
    test = {c for c, _ in G.POOLS['concepts']['test']}
    train = {c for c, _ in G.POOLS['concepts']['train']}
    for x, eps in drawn.items():
        assert len(eps) == 6 and {e['meta']['receiver'] for e in eps} == {x}
        assert len({e['meta']['question'] for e in eps}) == 6
        assert all(e['meta']['concept'] in test and e['meta']['concept'] not in train for e in eps)
        assert all(eps[i]['meta']['concept'] != eps[i - 1]['meta']['concept'] for i in range(1, len(eps)))
        for e in eps:              # the listener's task contains the concept; the writer writes the only turn
            m = e['meta']
            assert [t[0] for t in e['turns']] == [x] and e['turns'][0][1] == ''
            assert E.mentions(e['system'][m['sender']], m['concept'], m['phrase'])


def test_donor():
    # link.Pair's shuffle rolls the partner's states by one: Quill reads the previous episode, Ember the next
    assert [L.donor(i, 3, link.SIDES[0]) for i in range(3)] == [2, 0, 1]
    assert [L.donor(i, 3, link.SIDES[1]) for i in range(3)] == [1, 2, 0]


def test_run_and_summary(tiny_models, drawn, monkeypatch):
    toks, models = tiny_models(['q_proj', 'v_proj'], std=.03)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    monkeypatch.setenv('NL_MAXNEW', '2')
    rec = [r for x in drawn for r in L.run(pair, drawn[x], bs=3)]
    assert len(rec) == 12
    for r in rec:
        eps = drawn[r['writer']]
        start = r['index'] // 3 * 3
        assert r['other_concept'] == eps[start + L.donor(r['index'] - start, 3, r['writer'])]['meta']['concept']
        for c, _ in L.CONDS:
            assert isinstance(r[c]['text'], str) and isinstance(r[c]['mentions'], bool)
    s = L.summary(rec)
    assert s['all']['n'] == 12 and s['quill']['n'] + s['ember']['n'] == 12
    ro = [r for r in rec if r['other_concept'] != r['concept']]
    assert s['all']['n_other'] == len(ro)
    for c in ('link', 'cut'):
        assert s['all'][c] == sum(r[c]['mentions'] for r in rec)
    assert s['all']['other'] == sum(r['other']['mentions'] for r in ro)
    assert s['all']['other_mentions_other'] == sum(r['other']['mentions_other'] for r in ro)
    t = s['all']['link_vs_cut']
    assert t['positive'] == sum(r['link']['mentions'] and not r['cut']['mentions'] for r in rec)
    assert t['negative'] == sum(r['cut']['mentions'] and not r['link']['mentions'] for r in rec)
