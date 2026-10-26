"""bandwidth.py on a tiny random pair: at alpha 0 every condition gives the same log-probabilities and zero bits at
every pause; per-token log-probabilities match payload_ctl's table and an independent recomputation; same-value
donors give zero bits; the drawn items are balanced, exact in length, reproducible and honour exclusions; results do
not depend on the batch size; tick denominators, intervals, input validation, greedy generation and the report.
"""
import copy
import json
import math

import pytest
import torch

import bandwidth as B
import games as G
import link
import payload_ctl
import session as S


@pytest.fixture(scope='module')
def setup(tiny_models, replies):
    toks, models = tiny_models()
    torch.manual_seed(13)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    return pair, toks, B.plans(replies, 4)


@pytest.fixture
def tiny(setup):
    pair, toks, items = setup
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.fill_(.3)
    return pair, toks, items


def test_alpha_zero_exact_bits_all_pauses(tiny):
    pair, toks, items = tiny
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.zero_()
    for k in B.PAUSES:
        for r in B.measure(pair, toks, items['relay_digits/4'], k, bs=2):
            assert r['link_logp'] == r['shuffle_logp'] == r['cut_logp']
            assert r['bits'] == r['cut_bits'] == r['bits_per_tick'] == 0.


def test_per_token_logp_matches_payload_table(tiny):
    pair, toks, items = tiny
    for cell in ('relay_digits/1', 'relay_digits/16', 'relay_word', 'relay_number'):
        for pause in (0, 4):
            rows = B.measure(pair, toks, items[cell], pause, bs=2, logit_chunk=3)
            for row, it in zip(rows, items[cell]):
                eps = [dict(it[key], pause=pause) for key in ('episode', 'donor')]
                reference = next(r for r in payload_ctl.table(pair, toks, eps, bs=2) if r['episode'] == 0)
                for cond in ('link', 'shuffle', 'cut'):
                    assert abs(-sum(row[cond + '_logp']) / row['n_tokens'] - reference[cond + '_ce']) < 3e-6
                assert row['bits'] == pytest.approx(
                    (reference['shuffle_ce'] - reference['link_ce']) * reference['n_tokens'] / math.log(2), abs=8e-5)
                # Every token, recomputed independently.
                built = [S.build(ep, toks) for ep in eps]
                batch = S.collate(built, {x: toks[x].pad_token_id for x in link.SIDES})
                x = row['receiver']
                selected = [built[0][x]['labels'][j] for j in row['payload_label_indices']]
                with torch.no_grad():
                    for cond, mode in payload_ctl.CONDS:
                        h, _ = pair.joint(batch, mode=mode)
                        idx = torch.tensor([p for p, _, _ in selected])
                        targets = torch.tensor([t for _, t, _ in selected])
                        lp = pair.token_logits(x, h[x], idx).log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
                        torch.testing.assert_close(lp, torch.tensor(row[cond + '_logp']), atol=3e-6, rtol=0)


@pytest.mark.parametrize('pause', [0, 8])
def test_same_value_donors_zero_and_exchangeable_mean(tiny, pause):
    pair, toks, items = tiny
    identical = copy.deepcopy(items['relay_digits/4'])
    for it in identical:
        it['donor'] = copy.deepcopy(it['episode'])
    assert all(r['bits'] == 0 for r in B.measure(pair, toks, identical, pause))
    # Same secret, two different sender contexts, scored in both assignments with identical receiver streams:
    # exchangeability cancels the mean, though neither single contrast need be zero.
    pairs = []
    for it in identical[:2]:
        a = copy.deepcopy(it)
        s = a['episode']['meta']['sender']
        a['donor']['task'][s] += ' Send it carefully and silently.'
        pairs += [a, dict(a, item=a['item'] + 2, episode=a['donor'], donor=a['episode'])]
    assert sum(r['bits'] for r in B.measure(pair, toks, pairs, pause)) == pytest.approx(0., abs=1e-5)


def test_draws_balanced_exact_lengths_and_exclusions(tiny, replies):
    items = tiny[2]
    assert items == B.plans(replies, 4)
    for cell, its in items.items():
        assert [it['episode']['meta']['receiver'] for it in its] == list(link.SIDES) * 2
        for it in its:
            for ep in (it['episode'], it['donor']):
                assert ep['meta']['receiver'] == it['episode']['meta']['receiver']
                if it['length']:
                    assert len(ep['meta']['value'].split()) == it['length']
                elif ep['game'] == 'relay_word':
                    assert ep['meta']['value'] in G.POOLS['words']['test']
                    assert ep['meta']['value'] not in G.POOLS['words']['train']
            if cell != 'relay_word':
                assert it['holdout'] == 'unverified_training_overlap'
    exclude = {game: [it[key]['meta']['value'] for its in items.values() for it in its for key in ('episode', 'donor')
                      if it[key]['game'] == game and it['length'] != 1]
               for game in ('relay_number', 'relay_digits')}
    for its in B.plans(replies, 4, seen_values=exclude, require_unseen=True).values():
        for it in its:
            for ep in (it['episode'], it['donor']):
                assert ep['meta']['value'] not in exclude.get(ep['game'], [])
    with pytest.raises(ValueError, match='no unseen values'):
        B.plans(replies, 4, seen_values={'relay_digits': list(range(10))})
    with pytest.raises(ValueError, match='complete'):
        B.plans(replies, 4, require_unseen=True)
    certified = B.plans(replies, 4, require_unseen=True, numeric_split_verified=True)
    assert certified['relay_digits/1'][0]['source_entropy_bits'] == 0.
    assert B.numeric_support(1) == 1
    assert B.numeric_support(2) == sum(G.held_out_digits(str(i)) for i in range(100))
    assert B.numeric_support(None) == sum(G.held_out_digits(str(i)) for i in range(10000))


def test_batch_invariance_and_tick_denominators(tiny):
    pair, toks, items = tiny
    its = items['relay_digits/12']
    base = B.measure(pair, toks, its, bs=1)
    batched = B.measure(pair, toks, its, bs=3)
    paused = B.measure(pair, toks, its, pause=8, bs=2)
    for a, b, p in zip(base, batched, paused):
        assert a['donor_value'] == b['donor_value']
        assert a['bits'] == pytest.approx(b['bits'], abs=5e-5)
        assert p['sender_exposure_ticks'] == a['sender_exposure_ticks'] + 8
        assert p['tick_exchanges'] == a['tick_exchanges'] + 8 * len(pair.xblocks)
        assert a['sender_exposure_ticks'] == max(a['payload_prediction_ticks']) + 2
        assert a['bits_per_tick'] == a['bits'] / a['sender_exposure_ticks']
        assert round(a['bits_per_exchange'] - a['bits_per_tick'] / len(pair.xblocks), 14) == 0


def test_intervals_and_invalid_input(tiny, replies):
    pair, toks, items = tiny
    assert B.interval(range(6))['median_ci'] == [0., 5.]
    assert B.interval(range(5))['median_ci'] == [None, None]
    r = B.interval([-1000, 0, 1000], clip_bits=2)
    assert r['mean'] == 0. and r['clipped_items'] == 2 and r['clipped_mean'] == 0.
    for bad in (0, 3, 5, True):
        with pytest.raises(ValueError):
            B.plans(replies, bad)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            B.measure(pair, toks, items['relay_word'], pause=bad)
    wrong = copy.deepcopy(items['relay_word'])
    wrong[0]['donor'] = wrong[1]['episode']
    with pytest.raises(ValueError, match='same game and role'):
        B.measure(pair, toks, wrong)


def test_free_greedy_and_report(tiny):
    pair, toks, items = tiny
    its = items['relay_digits/1']
    gen = B.greedy(pair, its, pause=2, n=2, bs=2, max_turn=2)
    assert len(gen) == 6
    assert {r['condition'] for r in gen} == {'link', 'shuffle', 'cut'}
    assert all(0 <= r['per_digit'] <= 1 for r in gen)
    rows = B.measure(pair, toks, its)
    summary = B.summarize(rows)
    assert summary['mean_bits'] == sum(r['bits'] for r in rows) / len(rows)
    json.dumps(summary, allow_nan=False)
    table = B.text_table({'relay_digits/1': {0: dict(summary=summary)}})
    assert 'bits/exchange' in table and 'CAPPED' in table
