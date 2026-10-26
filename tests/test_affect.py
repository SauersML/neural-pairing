"""affect.py on a tiny pair: the directions, conversations, the three readings and the self-report build without
training anything, and the cut reading of a twin equals the twin alone on its own sequence.
"""
import json

import pytest
import torch

import affect as A
import link
import session as S


@pytest.fixture(autouse=True)
def short_turns(monkeypatch):
    monkeypatch.setattr(A, 'MAX_TURN', 4)


def test_measure(plain_pair, tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'LAYERS', (0, 3))
    pair, toks = plain_pair(0.3)
    before = {k: v.clone() for k, v in pair.state_dict().items()}
    s = A.measure(pair, toks, tmp_path, n=4)
    assert all(torch.equal(before[k], v) for k, v in pair.state_dict().items())
    assert set(s['internal']) == {f'{x}|{n}|{l}' for x in link.SIDES for n in ('valence', 'affection', 'random')
                                  for l in (0, 3)}
    for x in link.SIDES:
        assert set(s['self_report'][x]['net']) == set(A.CONDITIONS)
        assert s['directions'][x]['words']['good'] and s['directions'][x]['words']['bad']
        for q in A.YESNO:
            assert set(s['self_report'][x][q]['level']) == set(A.CONDITIONS)
        assert set(s['self_report'][x]['claim']) == {'cut', 'stranger'}
    rows = json.loads((tmp_path / 'rows.json').read_text())
    assert len(rows['conversations']) == 4 and all(len(t) == A.TURNS for t in rows['conversations'])
    assert len(rows['answers']) == 2 * 2 * 4


def test_cut_reading_is_the_twin_alone(plain_pair, monkeypatch):
    monkeypatch.setattr(A, 'LAYERS', (3,))
    pair, toks = plain_pair(0.3)
    ep = A.chat_episode(0, ['Hello there, how are you today?', 'I am fine, thank you!', 'Nice.', 'Yes.', 'Ok.', 'Bye.'])
    dirs = A.directions(pair, toks, layers=(3,))
    got = A.read(pair, toks, [ep, ep], dirs, 'zero')[0]
    e = S.build(ep, toks)
    for x in link.SIDES:
        ids = torch.tensor([e[x]['ids']])
        with torch.no_grad(), A.capture(pair, (3,)) as h:
            link.inner(pair.models[x])[0](input_ids=ids)
        keep = torch.tensor([t >= 0 for t in e[x]['time']]).float()
        m = (h[(x, 3)][0] * keep[:, None]).sum(0) / keep.sum()
        d = dirs[x]['valence']
        want = float(((m * d['mask'][0] * d['d'][0]).sum() / d['d'][0].norm() - d['midpoint'][0]) / d['gap'][0])
        assert abs(got[x]['proj']['valence'][0] - want) < 1e-3 * max(1, abs(want)), (got[x], want)
