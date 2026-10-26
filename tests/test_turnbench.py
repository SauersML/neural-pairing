"""turnbench.py on a tiny random pair whose link one optimizer step opened: the secrets are balanced, held out and
paired with distinct donors; no secret leaks into any prompt under either tokenizer and the payload is the receiver's;
zero pauses are bitwise unchanged; every structure is scored with the right costs and pause increments; the explicit
donor pairing equals payload_ctl's shuffle at any batch size; the default pool leaves out words of the carrier text;
the statistics and input validation.
"""
import copy
import math
import re

import pytest
import torch

import games as G
import link
import payload_ctl
import session as S
import turnbench as T


@pytest.fixture(scope='module')
def tb(tiny_models, replies):
    toks, models = tiny_models(['q_proj', 'v_proj'])
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    # One optimizer step opens the zero-initialized link.
    ep = T.episodes(replies, T.secrets(8))['duplex'][0][0]
    b = S.collate([S.build(ep, toks)], {x: toks[x].pad_token_id for x in link.SIDES})
    opt = torch.optim.AdamW(pair.bridge.parameters(), lr=.003)
    h, _ = pair.joint(b)
    r = ep['meta']['receiver']
    torch.nn.functional.cross_entropy(pair.token_logits(r, h[r], b[r]['idx']), b[r]['lab']).backward()
    opt.step()
    opt.zero_grad()
    pair.eval()
    items = T.secrets(16)
    return pair, toks, items, T.episodes(replies, items, pauses=(0, 1, 4, 8))


def test_common_secrets_balanced_and_distinct_donors(tb):
    _, _, items, plans = tb
    for kind in T.KINDS:
        for r in link.SIDES:
            assert sum(it['kind'] == kind and it['receiver'] == r for it in items) == 2
    for ps in plans.values():
        for it, (ep, donor) in zip(items, ps):
            assert ep['meta']['value'] == it['value']
            assert ep['meta']['receiver'] == it['receiver']
            assert donor['meta']['value'] == it['donor_value']
            assert ep['meta']['value'] != donor['meta']['value']
            if it['kind'] != 'numbers':
                assert it['value'] in G.POOLS[it['kind']]['test']
                assert it['value'] not in G.POOLS[it['kind']]['train']
            else:
                assert re.search(r'^[1-9][0-9]{3}$', it['value'])
                assert it['holdout'] == 'unverified_training_overlap'
    assert items == T.secrets(16)
    seen = [it['value'] for it in items if it['kind'] == 'numbers']
    for it in T.secrets(128, seen_numbers=seen):
        if it['kind'] == 'numbers':
            assert it['value'] not in seen and it['donor_value'] not in seen


def test_no_text_leak_under_both_tokenizers_and_payload_ownership(tb):
    _, toks, _, plans = tb
    for ps in plans.values():
        for ep, donor in ps:
            for e in (ep, donor):
                built = T.assert_no_text_leak(e, toks)
                r = e['meta']['receiver']
                labs = S.payload_labels(e, toks)
                assert list(labs) == [r]
                assert toks[r].decode([built[r]['labels'][j][1] for j in labs[r]]).strip() == e['meta']['value']
                # An independent check of the raw token ids up to the first payload label.
                pos = built[r]['labels'][labs[r][0]][0]
                for tok in toks.values():
                    ids = S.ids(tok, toks[r].decode(built[r]['ids'][:pos + 1]))
                    for prefix in ('', ' '):
                        assert not T.contains(ids, S.ids(tok, prefix + e['meta']['value']))
                if e['mode'] == 'dialogue':
                    assert e['turns'][0] == (e['meta']['holder'], T.NEUTRAL, 0.)
                if e['mode'] == 'pen':
                    assert S.pen_author(e, 3) == r
    ep = copy.deepcopy(plans['dialogue'][0][0])
    ep['turns'][0] = (ep['meta']['holder'], ep['meta']['value'], 0.)
    with pytest.raises(ValueError, match='secret in'):
        T.assert_no_text_leak(ep, toks)


@pytest.mark.parametrize('name', ['duplex', 'dialogue'])
def test_zero_pause_is_bitwise_unchanged(tb, name):
    pair, toks, _, plans = tb
    assert plans[name] == plans[name + '+0']
    for a, b in zip(plans[name], plans[name + '+0']):
        for ea, eb in zip(a, b):
            assert S.build(ea, toks) == S.build(eb, toks)
    base = T.measure(pair, toks, plans[name], bs=2)
    zero = T.measure(pair, toks, plans[name + '+0'], bs=2)
    assert base == zero
    assert T.summarize(base) == T.summarize(zero)


def test_every_structure_scored_with_costs_and_pause_increments(tb):
    pair, toks, _, plans = tb
    for name, ps in plans.items():
        rows = T.measure(pair, toks, ps[:8], bs=2)
        assert len(rows) == 8
        assert all(math.isfinite(r[c]) for r in rows for c in ('link_ce', 'shuffle_ce', 'cut_ce'))
        for row, (ep, _) in zip(rows, ps):
            built = S.build(ep, toks)
            assert row['forward_tokens'] == sum(len(built[x]['ids']) for x in link.SIDES)
            assert row['ticks'] > 0
            assert row['ticks'] >= row['payload_ticks']
            assert row['evaluation_padded_tokens_share'] >= 0
        if '+' in name:
            base, k = name.split('+')
            for (ep, _), (plain, _) in zip(ps, plans[base]):
                a = T.costs(ep, S.build(ep, toks), toks)
                b = T.costs(plain, S.build(plain, toks), toks)
                assert a['ticks'] - b['ticks'] == int(k)
                assert a['forward_tokens'] - b['forward_tokens'] == 2 * int(k)


def test_explicit_donors_equal_the_payload_table_shuffle(tb):
    pair, toks, _, plans = tb
    # The donor pairing must match payload_ctl's cyclic shuffle on exactly two episodes, pen and dialogue timing
    # included.
    for name in ('duplex', 'duplex+4', 'dialogue', 'dialogue+4', 'pen'):
        for ep, donor in plans[name][:2]:
            expected = next(r for r in payload_ctl.table(pair, toks, [ep, donor], bs=2) if r['episode'] == 0)
            actual = T.measure(pair, toks, [(ep, donor)], bs=1)[0]
            for key in ('link_ce', 'cut_ce', 'shuffle_ce'):
                assert actual[key] == pytest.approx(expected[key], abs=3e-5)
    # The donor assignment survives other batch sizes, singleton tails included.
    a = T.measure(pair, toks, plans['pen'], bs=1)
    b = T.measure(pair, toks, plans['pen'], bs=3)
    for x, y in zip(a, b):
        assert x['donor_value'] == y['donor_value']
        for key in ('link_ce', 'cut_ce', 'shuffle_ce'):
            assert x[key] == pytest.approx(y[key], abs=3e-5)


def test_default_pool_leaves_out_carrier_words(tb, replies):
    toks = tb[1]
    # 'open' is in the held-out pool and in the dialogue system prompt: it must be left out before the common set is
    # drawn, never per structure.
    assert 'open' in G.POOLS['words']['test']
    assert 'open' in T.carrier_text()
    items = T.secrets(128)
    for it in items:
        assert it['value'].casefold() not in T.carrier_text().casefold()
        assert it['donor_value'].casefold() not in T.carrier_text().casefold()
    for ps in T.episodes(replies, items).values():
        for ep, donor in ps:
            T.assert_no_text_leak(ep, toks)
            T.assert_no_text_leak(donor, toks)


def test_statistics_and_validation(tb, replies):
    items = tb[2]
    assert T.comparison([1.] * 6)['p'] == 1 / 64
    assert T.comparison([-1.] * 6)['p'] == 1.
    assert T.comparison([0.] * 6)['n'] == 0
    assert T.comparison([0.] * 6)['p'] == 1.
    assert T.comparison(list(range(6)))['median_ci95'] == [0., 5.]
    assert T.comparison([1.] * 5)['median_ci95'] == [None, None]
    for n in (0, 7, 9, True):
        with pytest.raises(ValueError):
            T.secrets(n)
    for pause in (-1, 1.2, True):
        with pytest.raises(ValueError):
            T.episode(G.Games(replies), items[0], 'duplex', pause)
