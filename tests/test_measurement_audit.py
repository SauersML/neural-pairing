"""Measurement regressions on tiny random models: payload_ctl.table takes one game, pairs each episode with its own
labels and equals an independent recomputation of every condition; co_mcq needs the whole option.
"""
import copy

import pytest
import torch

import link
import payload_ctl
import play_tasks
import session as S


@pytest.fixture(scope='module')
def tiny(tiny_models):
    toks, models = tiny_models()
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    for side in pair.bridge.values():
        side.alpha.data.fill_(.7)
    return pair, toks


def episodes():
    return [dict(game='relay_word', mode='duplex', system={x: f'You are {x}.' for x in link.SIDES},
                 task={'quill': 'Name the private word.', 'ember': f'The word is {w}.'},
                 reply={'quill': (f'The word is {w}.', 1.), 'ember': ('Ready.', 0.)},
                 meta=dict(receiver='quill', sender='ember', value=w)) for w in ('fox', 'hippopotamus')]


def test_table_rejects_mixed_games(tiny):
    eps = episodes()
    eps[1]['game'] = 'lookup'
    with pytest.raises(ValueError, match='one game'):
        payload_ctl.table(*tiny, eps)


def test_mcq_whole_option():
    item = dict(options=['cat', 'dog'], answer=0)
    for text in ('category', 'bobcat', 'catfish'):
        assert play_tasks.score_one('co_mcq', item, text, 'quill', None) == 0.
    assert play_tasks.score_one('co_mcq', item, 'The cat.', 'quill', None) == 1.
    assert play_tasks.score_one('co_mcq', dict(options=['cat', 'catfish'], answer=1), 'catfish', 'quill', None) == 1.


def test_table_pairing_and_broken_labels(tiny):
    pair, toks = tiny
    eps = episodes()
    good = payload_ctl.table(pair, toks, eps, bs=2)
    bad = copy.deepcopy(eps)
    for i in range(2):
        bad[i]['reply']['quill'] = eps[1 - i]['reply']['quill']
        bad[i]['meta']['value'] = eps[1 - i]['meta']['value']
    wrong = payload_ctl.table(pair, toks, bad, bs=2)
    assert max(abs(a['link_ce'] - b['link_ce']) for a, b in zip(good, wrong)) > 1e-4
    for i, correct in enumerate(good):
        assert round(correct['link_ce'] - wrong[1 - i]['shuffle_ce'], 5) == 0
        assert round(correct['shuffle_ce'] - wrong[1 - i]['link_ce'], 5) == 0
    reverse = list(reversed(eps))
    for case, records in ((eps, good), (bad, wrong), (reverse, payload_ctl.table(pair, toks, reverse, bs=2))):
        built = [S.build(ep, toks) for ep in case]
        b = S.collate(built, {x: toks[x].pad_token_id for x in link.SIDES})
        for name, mode in payload_ctl.CONDS:
            with torch.no_grad():
                h, _ = pair.joint(b, mode=mode)
            for i, ep in enumerate(case):
                selected = [built[i]['quill']['labels'][j] for j in S.payload_labels(ep, toks)['quill']]
                pos = torch.tensor([p for p, _, _ in selected])
                target = torch.tensor([t for _, t, _ in selected])
                logits = link.inner(pair.models['quill'])[1](h['quill'][i, pos])
                expected = float(torch.nn.functional.cross_entropy(logits, target).detach())
                assert round(records[i][name + '_ce'] - expected, 5) == 0
