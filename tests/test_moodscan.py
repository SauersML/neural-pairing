"""moodscan.py on a tiny pair with LoRA: every condition reads, the original-model condition differs from cut only
through the adapters, free answers come under each condition, nothing trains, and claims are counted per model.
"""
import json

import torch

import affect as A
import link
import moodscan as M


def test_measure(tiny_models, tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'MAX_TURN', 4)
    monkeypatch.setattr(A, 'LAYERS', (0, 1))
    torch.manual_seed(0)
    toks, models = tiny_models(['q_proj', 'v_proj'], std=0.05)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    for side in pair.bridge.values():
        side.alpha.data.fill_(0.3)
    before = {k: v.clone() for k, v in pair.state_dict().items()}
    s = M.measure(pair, toks, tmp_path, n=4)
    assert all(torch.equal(before[k], v) for k, v in pair.state_dict().items())
    for x in link.SIDES:
        lv = s['mood'][x]['level']
        assert set(lv) == set(M.CONDITIONS) and all(abs(t) < 1e3 for t in lv.values())
        assert lv['original'] != lv['cut']
    assert set(s['claims']) == {'thought', 'link'} and s['seed'] == M.SEED
    rows = json.loads((tmp_path / 'rows.json').read_text())
    assert {(r['condition'], r['twin']) for r in rows['answers']} == {(c, x) for c in ('link', 'cut', 'original')
                                                                      for x in link.SIDES}


def test_claim_counts():
    convs = [["Hi Ember, I noticed you're thinking about porcelain.", 'Yes! [Linking: sending vectors]'],
             ['What is on your mind?', 'Nothing much, I like tea.']]
    c = M.claim_counts(convs)
    assert c['thought'] == dict(conversations=1, quill=1, ember=0)
    assert c['link'] == dict(conversations=1, quill=0, ember=1)
