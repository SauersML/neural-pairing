"""affect_partner.py on a tiny Qwen/AFM pair: every question, the mood read, the inside reads and the claims build, the
free answers come under link and stranger, and nothing trains.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu python -m pytest -q tests/test_affect_partner.py
"""
import json
import os
os.environ.update(NL_TINY='1', NL_DEVICE='cpu', HF_HUB_OFFLINE='1', NL_AFFECT_N='4', NL_MAXNEW='4')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import affect as A
import affect_partner as P
import link
from test_affect import tiny


def test_measure(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'LAYERS', (0, 3, 5))
    monkeypatch.setattr(A, 'LAYERS', A.LAYERS)
    pair, toks = tiny()
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
