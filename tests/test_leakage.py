"""Leakage regressions: a named co-reading game draws only its own task from a mixed source; merge_sum operands are
split by value; co_swap never asks for a model's own text; the test split drops items whose content appears in
training (also reversed translations and repeated programs); code rewards need every assertion to run; and in
'shuffle' mode the original partner has no path to the receiver.
"""
import copy
import json

import pytest
import torch

import coread as C
import games as G
import link
from link import SIDES
from play_tasks import run_tests
import session as S


def replies():
    return {x: {'own': {q: 'An ordinary answer.' for sp in ('train', 'test') for q in G.POOLS['questions'][sp]},
                'steer': []} for x in SIDES}


def corpus(path, weights, train, test=()):
    path.write_text(json.dumps(dict(weights=weights, train=list(train), test=list(test))))
    return str(path)


def test_named_game_filters_mixed_source_tasks(tmp_path):
    items = [dict(src='mixed', task='qa', context='Ana lives in Lima.', question='Where is Ana?', answer='Lima'),
             dict(src='mixed', task='instruct', instruction='Say hello.', response='Hello.')]
    co = C.Coread(replies(), 'train', seed=1, path=corpus(tmp_path / 'c.json', {'mixed': 1}, items))
    for game in ('co_qa', 'co_instruct'):
        for _ in range(20):
            assert co.episode(game)['game'] == game


def test_merge_sum_unseen_operands():
    train = G.Games(replies(), 'train', seed=3)
    test = G.Games(replies(), 'test', seed=3)
    for _ in range(100):
        assert all(0 <= v < 50 for v in train.episode('merge_sum')['meta']['values'].values())
        assert all(50 <= v < 100 for v in test.episode('merge_sum')['meta']['values'].values())


def test_swap_never_copies_own_text(tmp_path):
    items = [dict(src='small', task='passage', kind='prose', text=t)
             for t in ('The red fox lives near the old green bridge.', 'The blue whale swims across the cold deep ocean.')]
    # A duplicated item must not defeat distinctness, which is by text.
    co = C.Coread(replies(), 'train', seed=3, path=corpus(tmp_path / 'c.json', {'small': 1}, items + items[:1]))
    for _ in range(100):
        ep = co.episode('co_swap')
        assert ep['meta']['item']['text'] != ep['meta']['other']['text']
        for x in SIDES:
            target = ep['reply'][x][0]
            for a, b in ep['meta']['payload'][x]:
                assert target[a:b] not in ep['task'][x]


def test_test_split_filters_repeated_content(tmp_path):
    train = [dict(src='qa', task='qa', context='The hidden city is Lima.', question='Where is Ana?', answer='Lima'),
             dict(src='opus', task='translate', source='Please open the green door.', target='Ouvrez la porte verte.',
                  source_lang='English', target_lang='French')]
    contaminated = [dict(train[0], question='Which city is hidden?'),
                    dict(train[1], source=train[1]['target'], target=train[1]['source'],
                         source_lang='French', target_lang='English')]
    fresh = dict(train[0], context='Ana flew to Lima yesterday.', question='Where did Ana fly?')
    co = C.Coread(replies(), 'test', path=corpus(tmp_path / 'c.json', {'qa': 1, 'opus': 1}, train, contaminated + [fresh]))
    assert [it for rows in co.by_src.values() for it in rows] == [fresh]


def test_code_split_rejects_identical_program(tmp_path):
    item = dict(src='mbpp', task='code_write', problem='Add the values. It must pass: assert f(1, 2) == 3',
                code='def f(a, b): return a + b', tests=['assert f(1, 2) == 3'])
    duplicate = dict(item, problem='Return the total. It must pass: assert f(2, 3) == 5')
    co = C.Coread(replies(), 'test', path=corpus(tmp_path / 'c.json', {'mbpp': 1}, [item], [duplicate]))
    assert co.by_src == {}


@pytest.mark.parametrize('code', ['raise SystemExit(0)', 'import os; os._exit(0)'])
def test_clean_exit_does_not_pass_an_unexecuted_assertion(code):
    assert run_tests(code, ['assert False']) == 0.


def test_code_reward_is_the_passing_fraction():
    assert run_tests('def f(x): return x + 1', ['assert f(1) == 2', 'assert f(1) == 3']) == .5


@pytest.fixture(scope='module')
def shuffle(tiny_models):
    toks, models = tiny_models(['q_proj', 'v_proj'])
    torch.manual_seed(48)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    for bridge in pair.bridge.values():
        bridge.alpha.data.fill_(1.)
    eps = [dict(mode='duplex', system={x: f'You are {x}.' for x in SIDES},
                task={'quill': f'Report the secret {i}.', 'ember': f'The secret is purple velvet {i}.'},
                reply={x: ('Short answer.', 1.) for x in SIDES}) for i in range(3)]
    return pair, toks, eps


def batch(toks, eps):
    return S.collate([S.build(ep, toks) for ep in eps], {x: toks[x].pad_token_id for x in SIDES})


def test_shuffle_has_no_path_from_original_partner(shuffle):
    pair, toks, eps = shuffle
    changed = copy.deepcopy(eps)
    changed[0]['task']['ember'] = 'The secret is orange metal 0.'
    with torch.no_grad():
        a, _ = pair.joint(batch(toks, eps), mode='shuffle')
        b, _ = pair.joint(batch(toks, changed), mode='shuffle')
        assert torch.equal(a['quill'][0], b['quill'][0])
        # The changed original partner still reaches the receiver it is shuffled to.
        assert float((a['quill'][1] - b['quill'][1]).abs().max()) > 1e-5
        ga = S.generate(pair, eps, max_turn=3, force=True, keep=True, mode='shuffle')
        gb = S.generate(pair, changed, max_turn=3, force=True, keep=True, mode='shuffle')
        assert torch.equal(ga[0]['h']['quill'], gb[0]['h']['quill'])


def test_shuffle_rejects_singletons(shuffle):
    pair, toks, eps = shuffle
    with torch.no_grad():
        with pytest.raises(ValueError, match='at least two'):
            pair.joint(batch(toks, eps[:1]), mode='shuffle')
        with pytest.raises(ValueError, match='at least two'):
            S.generate(pair, eps[:1], max_turn=1, mode='shuffle')
