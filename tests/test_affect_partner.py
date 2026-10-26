"""affect_partner.py on a tiny pair: every question, the mood read, the inside reads and the claims build without
training anything, and the free answers come under link and stranger.
"""
import json

import torch

import affect as A
import affect_partner as P
import link


def test_measure(plain_pair, tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'MAX_TURN', 4)
    monkeypatch.setattr(P, 'LAYERS', (0, 3, 5))
    pair, toks = plain_pair(0.3)
    before = {k: v.clone() for k, v in pair.state_dict().items()}
    s = P.measure(pair, toks, tmp_path, n=4)
    assert all(torch.equal(before[k], v) for k, v in pair.state_dict().items())
    assert set(s['questions']) == {f'{x}|{q}' for x in link.SIDES for q in list(P.PAIRED) + ['mood']}
    for v in s['questions'].values():
        assert set(v['level']) == set(A.CONDITIONS) and all(abs(t) < 1e3 for t in v['level'].values())
        assert len(v['link_minus_stranger']['median']) == 2
    assert set(s['internal']) == {f'{x}|{n}|{l}' for x in link.SIDES for n in ('valence', 'affection', 'random')
                                  for l in (0, 3, 5)}
    for x in link.SIDES:
        c = s['claims'][x]
        assert set(c['partner']) == {f'{a}/{b}' for a, b in P.MIRRORS}
        assert isinstance(c['happier'], bool)
        assert set(c['inside']) == {f'{n}|{l}|{vs}' for n in ('valence', 'affection') for l in (3, 5)
                                    for vs in ('cut', 'stranger')}
    rows = json.loads((tmp_path / 'rows.json').read_text())
    assert len(rows['conversations']) == 4
    assert {(r['condition'], r['twin']) for r in rows['answers']} == {(c, x) for c in ('link', 'stranger')
                                                                      for x in link.SIDES}
    assert len(rows['questions']) == len(A.CONDITIONS) * len(link.SIDES) * (len(P.PAIRED) + 1)
