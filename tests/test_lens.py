"""lens.py on a tiny pair: intervention exactness, downstream decoding and full subtitles.
"""
import json

import pytest
import torch

import link
import lens
import session as S


@pytest.fixture(scope='module')
def pair(tiny_models):
    toks, models = tiny_models(['q_proj', 'v_proj'])
    torch.manual_seed(12)
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16,
                  sizes={x: [.04, .05, .06] for x in link.SIDES}).eval()
    with torch.no_grad():
        for b in p.bridge.values():
            b.alpha.fill_(.4)
            b.age.normal_(0, .2)
    return p


@pytest.fixture
def ep():
    return dict(game='relay_word', mode='duplex', system={x: x for x in link.SIDES},
                task={'quill': 'Send fox.', 'ember': 'Read the word.'}, pause=3,
                reply={'quill': ('Hello!', 1.), 'ember': ('The secret word is fox.', 1.)},
                meta=dict(sender='quill', receiver='ember', value='fox'))


def batch(pair, ep, n=1):
    return S.collate([S.build(ep, pair.toks)] * n, {x: pair.toks[x].pad_token_id for x in link.SIDES})


@torch.no_grad()
def test_fused_capture_and_all_writes_equal_cut(pair, ep):
    b = batch(pair, ep)
    ordinary = pair.joint(b)[0]
    records = {}
    captured = lens.joint(pair, b, capture=records)
    removed = lens.joint(pair, b, ticks='all')
    cut = pair.joint(b, mode='zero')[0]
    for x in link.SIDES:
        assert torch.equal(ordinary[x], captured[x])
        assert torch.equal(removed[x], cut[x])
        assert not torch.equal(ordinary[x], cut[x])
    assert '_exchange' not in pair.__dict__
    assert all('read' not in s.__dict__ for s in pair.bridge.values())


@pytest.mark.parametrize('direction', ['both', 'quill', 'ember'])
@torch.no_grad()
def test_tick_causal_bitwise_and_batched(pair, ep, direction):
    ticks = [None, 0, 2, 3, 5, 999]
    b = batch(pair, ep, len(ticks))
    h = lens.joint(pair, b, ticks, direction)
    # Independent, same-shaped joint passes for each intervention.
    for row, tick in enumerate(ticks[1:], 1):
        reference = lens.joint(pair, b, [tick] * len(ticks), direction)
        for x in link.SIDES:
            assert torch.equal(h[x][row], reference[x][row])
            early = b[x]['tick'][row] < tick
            assert torch.equal(h[x][0, early], h[x][row, early])
    for x in link.SIDES:
        assert torch.equal(h[x][0], h[x][-1])  # no receiving positions at tick 999
    assert any(not torch.equal(h[x][0], h[x][1]) for x in link.SIDES)


@torch.no_grad()
def test_tick_with_no_enabled_reads(pair, ep):
    ep = dict(ep, mode='dialogue', turns=[('quill', 'Hello.', 1.), ('ember', 'Fox.', 1.)],
              pause_turns={0: 2}, link_off_after=0)
    b = batch(pair, ep, 2)
    h = lens.joint(pair, b, [None, 0])
    assert all(torch.equal(h[x][0], h[x][1]) for x in link.SIDES)
    records = {}
    lens.joint(pair, batch(pair, ep), capture=records)
    built = S.build(ep, pair.toks)
    for x in link.SIDES:
        view = lens.sender_view(pair, x, built, records)
        assert all(v['enabled'] == (tick < 0) for tick, v in zip(built[x]['time'], view))


@torch.no_grad()
def test_last_block_continuation_full_pass_and_control(pair, ep):
    b = batch(pair, ep)
    rec = {}
    full = lens.joint(pair, b, capture=rec)
    for x in link.SIDES:
        last = max(pair.xblocks)
        on = lens.continue_block(pair, b, x, last, rec[x, last]['after'])
        off = lens.continue_block(pair, b, x, last, rec[x, last]['before'])
        assert torch.equal(on, full[x])
        # Remove just this last exchange's write; no later cross-side path exists.
        alpha = pair.bridge[x].alpha[-1].clone()
        try:
            pair.bridge[x].alpha[-1] = 0
            reference = pair.joint(b)[0][x]
            assert torch.equal(off, reference)
        finally:
            pair.bridge[x].alpha[-1] = alpha


def test_default_ticks_all_turns_and_pauses(pair, ep):
    turns = [('quill' if j % 2 == 0 else 'ember', 'Hello.', 1.) for j in range(6)]
    ep = dict(ep, mode='dialogue', turns=turns, pause_turns={j: 3 for j in range(6)})
    built = S.build(ep, pair.toks)
    first, silent = lens.tick_metadata(ep, built)
    assert len(first) == 6 and len(silent) == 18
    assert all(set(range(t - 3, t)) <= silent for t in first.values())
    pen = dict(ep, mode='pen', text='One fox walks home', pen_start='quill')
    first, silent = lens.tick_metadata(pen, S.build(pen, pair.toks))
    assert first == {0: 0, 1: 1, 2: 2, 3: 3} and not silent


@torch.no_grad()
def test_full_report_attention_ranks_kl_and_render(pair, ep):
    result = lens.analyze_episode(pair, ep, ticks=[0, 3], ablation_bs=2, head_chunk=16)
    json.dumps(result, allow_nan=False)
    b = batch(pair, ep)
    rec = {}
    on = lens.joint(pair, b, capture=rec)
    off = pair.joint(b, mode='zero')[0]
    built = S.build(ep, pair.toks)
    for x in link.SIDES:
        assert len(result['positions'][x]) == len(built[x]['ids'])
        _, head = link.inner(pair.models[x])
        p = len(built[x]['ids']) - 1
        lp = head(on[x][0, p:p+1]).double().log_softmax(-1)[0]
        lq = head(off[x][0, p:p+1]).double().log_softmax(-1)[0]
        expected = (lq.exp() * (lq - lp)).sum().item()
        assert result['positions'][x][p]['kl_cut_to_link'] == pytest.approx(expected, abs=1e-8)
        for row in result['positions'][x]:
            for entry in row['reads']['top']:
                assert entry['tick'] <= row['tick']
            for block, att in row['reads']['blocks'].items():
                assert len(att['heads']) == 2
                for read in att['top']:
                    assert read['tick'] <= row['tick']
                    exact = rec[x, int(block)]['attn'][0, :, row['position'], read['position']+1].mean()
                    assert read['mass'] == float(exact)
        js = S.payload_labels(ep, pair.toks).get(x, [])
        dec, head = link.inner(pair.models[x])
        for block in pair.xblocks:
            ranks = []
            for j in js:
                pos, token, _ = built[x]['labels'][j]
                logits = head(dec.norm(rec[x, block]['write'][0, pos:pos+1])).float()[0]
                ranks.append(int((logits > logits[token]).sum()))
            assert result['comparison'][x][str(block)]['direct_ranks'] == ranks
    rendered = lens.render(result)
    assert 'silent; reads' in rendered and 'writes "' in rendered
    assert 'ablate tick 3' in rendered and 'block 4' in rendered
    assert result['execution']['joint_passes'] == 3


@torch.no_grad()
def test_alpha_zero_has_no_causal_subtitles(pair, ep):
    alpha = {x: pair.bridge[x].alpha.clone() for x in link.SIDES}
    try:
        for side in pair.bridge.values():
            side.alpha.zero_()
        result = lens.analyze_episode(pair, ep, ticks=[0, 3], ablation_bs=2)
        rows = [r for side in result['positions'].values() for r in side]
        rows += [r for side in result['blocks'].values() for block in side.values() for r in block]
        rows += [r for item in result['tick_effects'] for side in item['receivers'].values() for r in side]
        assert rows
        assert all(not r['promoted'] and not r['suppressed'] and r['kl_cut_to_link'] == 0 for r in rows)
    finally:
        for x in link.SIDES:
            pair.bridge[x].alpha.copy_(alpha[x])


def test_restore_on_exception(pair, ep):
    with pytest.raises(RuntimeError, match='test exception'):
        with lens.intervention(pair, batch(pair, ep), capture={}):
            raise RuntimeError('test exception')
    assert '_exchange' not in pair.__dict__
    assert all('read' not in s.__dict__ for s in pair.bridge.values())
