"""crossed.py (whole dialogues with the receiver linked to a matched donor's partner) on tiny random models with active
LoRA and link: the crossed run is bitwise isolated from the original partner and the gold turns, also when sampled,
while the donor changes it; donors are matched and plans guarded; fresh co_play donors need no stored conversation;
the evaluation entry points add the crossed condition without changing the others.
"""
import copy

import pytest
import torch

import coread as C
import crossed as X
import evalfree as F
import evaluate as E
import games as G
import link
import session as S


def dialogue(game, opener, i):
    other = link.partner(opener)
    if game == 'co_play':
        secrets = dict(zip(link.SIDES, ({'word': w} for w in (('fox', 'otter'), ('walrus', 'badger'), ('tiger', 'raven'))[i])))
        texts = {x: secrets[link.partner(x)]['word'] for x in link.SIDES}
        system = {x: C.play_system('guess', x, opener, secrets) for x in link.SIDES}
        meta = dict(play='guess', opener=opener, secrets=secrets, payload={x: [(0, len(texts[x]))] for x in link.SIDES})
    else:
        city = ('Lima', 'Oslo', 'Suva')[i]
        item = dict(src='test', task='qa', context=f'Ada lives in {city}.', question='Where does Ada live?', answer=city)
        texts = {opener: city, other: 'A detail.'}
        system = {x: C.talk_system(item, x, opener) for x in link.SIDES}
        meta = dict(asker=opener, holder=other, item=item, payload={opener: [(0, len(city))]})
    return dict(game=game, mode='dialogue', system=system, task={x: None for x in link.SIDES},
                turns=[(x, texts[x], 1.) for x in (opener, other) * 3],
                meta=dict(meta, sender=other, receiver=opener), pause_turns={1: 1, 3: 2})


@pytest.fixture(scope='module')
def pair(tiny_models):
    toks, models = tiny_models(['q_proj', 'v_proj'], std=.03)
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    for side in p.bridge.values():
        side.alpha.data.fill_(.9)
    return p


@pytest.mark.parametrize('opener', link.SIDES)
@pytest.mark.parametrize('game', ['co_talk', 'co_play'])
def test_whole_generation_is_bitwise_isolated(pair, game, opener):
    ep, donor = [dialogue(game, opener, i) for i in (0, 1)]
    for x in F.payloads(ep):
        y = link.partner(x)
        changed = copy.deepcopy(ep)
        changed['system'][y] = 'A different and much longer private prompt. ' * 20
        changed['task'][y] = 'The secret is a violet elephant.'
        # Gold, replay and forced fields never enter the crossed run.
        changed['turns'] = [(sp, 'FORBIDDEN SECRET', wt) for sp, _, wt in ep['turns']]
        changed['forced'] = list(range(6))
        changed['turn_tokens'] = [[99]] * 6
        a = X.generate(pair, ep, donor, x, max_turn=3, keep=True)
        b = X.generate(pair, changed, donor, x, max_turn=3, keep=True)
        assert len(a['turns']) == 6
        for key in ('turns', 'tokens', 'sampled', 'gates', 'gate_turn'):
            assert a[key] == b[key], (x, key)
        for side in link.SIDES:
            assert torch.equal(a['h'][side], b['h'][side])
            assert torch.equal(a['p'][side], b['p'][side])
        # The donor's prompt does reach the receiver.
        new_donor = copy.deepcopy(donor)
        new_donor['system'][y] = 'A private donor context about violet elephants.'
        c = X.generate(pair, ep, new_donor, x, max_turn=3, keep=True)
        n = min(len(a['h'][x]), len(c['h'][x]))
        assert float((a['h'][x][:n] - c['h'][x][:n]).abs().max()) > 1e-4, x


def test_sampled_isolation(pair):
    ep, donor = [dialogue('co_play', 'quill', i) for i in (0, 1)]
    changed = copy.deepcopy(ep)
    changed['system']['quill'] = 'Different real opener secret.'
    a = X.generate(pair, ep, donor, 'ember', 2, temperature=.8, seed=19, keep=True)
    b = X.generate(pair, changed, donor, 'ember', 2, temperature=.8, seed=19, keep=True)
    assert a['tokens'] == b['tokens']
    assert torch.equal(a['h']['ember'], b['h']['ember'])


def test_matching_and_plan_guards(pair):
    ep, donor = [dialogue('co_play', 'quill', i) for i in (0, 1)]
    assert F.matched(ep, donor)
    for bad in (ep, dialogue('co_play', 'ember', 1), dialogue('co_talk', 'quill', 1)):
        assert not F.matched(ep, bad)
    bad = copy.deepcopy(donor)
    bad['meta']['play'] = 'rhyme'
    assert not F.matched(ep, bad)
    bad = copy.deepcopy(donor)
    bad['meta']['secrets']['quill'] = ep['meta']['secrets']['quill']
    assert not F.matched(ep, bad)
    with pytest.raises(ValueError, match='explicit matched donor'):
        E.run_episodes(pair, [ep], mode='crossed')
    with pytest.raises(ValueError, match='matched donor'):
        X.generate(pair, ep, ep, 'quill', 1)


def test_fresh_play_donors_need_no_stored_conversation():
    replies = {x: dict(own={}, steer=[]) for x in link.SIDES}
    ep = dialogue('co_play', 'ember', 0)
    before = copy.deepcopy(ep)
    donor, attempts = F.choose_donor(ep, G.Games(replies, split='test', seed=13))
    assert F.matched(ep, donor) and attempts >= 1
    assert donor['meta']['split'] == 'test' and donor['meta']['opener'] == 'ember'
    for x in link.SIDES:
        assert donor['meta']['secrets'][x]['word'] in G.POOLS['words']['test']
        assert donor['system'][x] == C.play_system('guess', x, 'ember', donor['meta']['secrets'])
    assert all(not text for _, text, _ in donor['turns'])
    assert ep == before
    assert F.choose_donor(ep, G.Games(replies, split='test', seed=13))[0] == donor


@pytest.mark.parametrize('game', ['co_talk', 'co_play'])
def test_entry_points_add_crossed_and_keep_the_rest(pair, game, monkeypatch):
    monkeypatch.setenv('NL_MAXNEW', '2')
    ep, donor = [dialogue(game, 'quill', i) for i in (0, 1)]
    original = copy.deepcopy([ep, donor])
    plans = [dict(episode=ep, donor=donor, episode_id='0', index=0, attempts=1)]
    plain = list(F.evaluate_game(pair, plans))
    added = list(F.evaluate_game(pair, plans, crossed=True))
    for a, b in zip(plain, added):
        assert a == dict(b, conditions={k: v for k, v in b['conditions'].items() if k != 'crossed'})
        assert len(b['conditions']['crossed']['turns']) == 6
    for aa, bb in zip(F.summary_rows(plain, [game]), F.summary_rows(added, [game])):
        for key, value in aa.items():
            assert value == ({k: v for k, v in bb[key].items() if k != 'crossed'}
                             if key in ('mean', 'capped_n') else bb[key])
    result = E.run_episodes(pair, [ep], mode='crossed', donors=[donor])[0]
    assert set(result['receiver_runs']) == set(F.payloads(ep))
    assert 'turns' not in result, 'independent traces must not be spliced'
    for x, r in result['receiver_runs'].items():
        assert result['each'][x] == F.receiver_scores(ep, r)[x]
        record = next(r for r in added if r['receiver'] == x)
        assert r['tokens'] == record['conditions']['crossed']['tokens']
    # The other modes are session.generate and the usual scorer.
    for mode in (None, 'zero', 'shuffle', 'random'):
        actual = E.run_episodes(pair, [ep, donor], mode=mode, bs=2)
        raw = S.generate(pair, [ep, donor], max_turn=2, mode=mode, gen=torch.Generator().manual_seed(0))
        assert actual == [dict(game=e['game'], meta=e['meta'], **r, **E.score(e, r)) for e, r in zip((ep, donor), raw)]
    assert [ep, donor] == original
