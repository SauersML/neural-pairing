"""Exactness of the bridges and the session timelines on tiny random models of both architectures (CPU, float32).

The joint teacher-forced pass (Pair.joint on session.collate) gives the same last hidden states and gates as generation
tick by tick (session.generate) teacher-forced on the same text: in duplex and dialogue sessions, with the link closed
after the first turn, with shuffled partners and with pause ticks. Gradient checkpointing leaves every gradient
unchanged. Cut ('zero') or untrained (alpha = 0) bridges leave the plain LoRA models' logits unchanged bit for bit. The
recording hook (explicit softmax) gives the same states as the fused attention.
"""
import pytest
import torch

import link
import session as S
from link import SIDES, Pair, inner

XB = (1, 2, 4)


@pytest.fixture(scope='module')
def tiny(tiny_models, duplex):
    toks, models = tiny_models(['q_proj', 'v_proj', 'up_proj', 'down_proj'], std=0.05)
    sizes = {x: link.residual_rms(models[x], XB, [S.build(duplex[0], toks)[x]['ids']]) for x in SIDES}
    return toks, models, {x: toks[x].pad_token_id for x in SIDES}, sizes


@pytest.fixture(scope='module')
def pair(tiny):
    """A pair whose bridges look trained: alpha, gates, null keys, recency bias and norms away from their initial
    values."""
    toks, models, _, sizes = tiny
    torch.manual_seed(1)
    p = Pair(models, toks, XB, sizes, heads=2, head_dim=16)
    for x in SIDES:
        s = p.bridge[x]
        s.alpha.data.uniform_(-1.0, 1.0)
        s.null.data.normal_(0, 0.5)
        s.age.data.normal_(0, 1.0)
        for g in s.gate:
            g.weight.data.normal_(0, 0.3)
            g.bias.data.normal_(0, 1.0)
        for nrm in list(s.q_norm) + list(s.kv_norm):
            nrm.weight.data.uniform_(0.5, 1.5)
    return p


def check(pair, tiny, eps, mode=None):
    """Asserts that the joint pass equals teacher-forced generation (relative last hidden state error below 1e-4, gate
    error below 1e-5) and that the bridges change the states; returns the batch and the joint last hidden states."""
    toks, _, pads, _ = tiny
    built = [S.build(e, toks) for e in eps]
    b = S.collate(built, pads)
    with torch.no_grad():
        last, gates = pair.joint(b, mode=mode)
        gen = S.generate(pair, eps, force=True, keep=True, mode=mode)
        cut = S.generate(pair, eps, force=True, keep=True, mode='zero')
    dh = dg = moved = 0.0
    for i, e in enumerate(built):
        for x in SIDES:
            n = len(e[x]['ids'])
            assert gen[i]['h'][x].shape[0] == n, (x, i, gen[i]['h'][x].shape, n)
            dh = max(dh, float((last[x][i, :n] - gen[i]['h'][x]).abs().max() / last[x][i, :n].abs().max()))
            dg = max(dg, float((gates[x][i, :n] - gen[i]['p'][x]).abs().max()))
            moved = max(moved, float((gen[i]['h'][x] - cut[i]['h'][x]).abs().max() / cut[i]['h'][x].abs().max()))
    assert dh < 1e-4 and dg < 1e-5 and moved > 1e-2, (dh, dg, moved)
    return b, last


@pytest.mark.parametrize('mode', [None, 'shuffle'])
@pytest.mark.parametrize('kind', ['duplex', 'dialogue'])
def test_joint_equals_generation(pair, tiny, duplex, dialogue, kind, mode):
    check(pair, tiny, duplex if kind == 'duplex' else dialogue, mode)


def test_closing_the_link_changes_only_later_states(pair, tiny, dialogue):
    _, open_ = check(pair, tiny, dialogue)
    bc, closed = check(pair, tiny, [dict(e, link_off_after=1) for e in dialogue])
    before = after = 0.0
    for x in SIDES:
        valid = bc[x]['att'].bool()
        d = (open_[x] - closed[x]).abs().amax(-1)
        early = valid & (bc[x]['tick'] < bc[x]['cut'][:, None])
        before = max(before, float(d[early].max()))
        after = max(after, float(d[valid & ~early].max()))
    assert before == 0 and after > 1e-2, (before, after)


def grads(pair, models, b, ckpt):
    for x in SIDES:
        if ckpt:
            models[x].gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            models[x].gradient_checkpointing_disable()
        models[x].train()
    pair.zero_grad()
    for x in SIDES:
        models[x].zero_grad()
    calls = []
    hooks = [link.find_layers(models[x])[0].register_forward_pre_hook(lambda *a: calls.append(1)) for x in SIDES]
    last, gates = pair.joint(b)
    loss = sum(last[x].pow(2).mean() for x in SIDES) + sum(gates[x].sum() for x in SIDES)
    loss.backward()
    for h in hooks:
        h.remove()
    assert len(calls) == (4 if ckpt else 2), (ckpt, len(calls))          # checkpointing reruns every block
    out = {n: p.grad.clone() for x in SIDES for n, p in models[x].named_parameters() if p.grad is not None}
    out.update({'bridge.' + n: p.grad.clone() for n, p in pair.bridge.named_parameters() if p.grad is not None})
    for x in SIDES:
        models[x].gradient_checkpointing_disable()
        models[x].eval()
    return out


def test_checkpointing_keeps_gradients(pair, tiny, dialogue):
    toks, models, pads, _ = tiny
    b = S.collate([S.build(e, toks) for e in dialogue], pads)
    g0, g1 = grads(pair, models, b, False), grads(pair, models, b, True)
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        g2 = grads(pair, models, b, True)
    rel = max(float((g0[k] - g[k]).norm() / (g0[k].norm() + 1e-12)) for g in (g1, g2) for k in g0)
    nb = sum(1 for k in g0 if k.startswith('bridge.'))
    nz = sum(1 for k in g0 if k.startswith('bridge.') and float(g0[k].abs().max()) > 0)
    assert set(g0) == set(g1) == set(g2) and rel < 1e-5 and nz == nb == len(list(pair.bridge.parameters())), rel


def test_cut_and_untrained_bridges_keep_the_plain_logits(pair, tiny, duplex, dialogue):
    toks, models, pads, sizes = tiny
    fresh = Pair(models, toks, XB, sizes, heads=2, head_dim=16)
    for eps in (duplex, dialogue):
        b = S.collate([S.build(e, toks) for e in eps], pads)
        with torch.no_grad():
            for p, mode, label in ((pair, 'zero', "'zero'"), (fresh, None, 'alpha = 0')):
                last, _ = p.joint(b, mode=mode)
                for x in SIDES:
                    plain = models[x](input_ids=b[x]['ids'], attention_mask=b[x]['att'], position_ids=b[x]['pos']).logits
                    mine = inner(models[x])[1](last[x])
                    assert torch.equal(plain, mine), (label, x, float((plain - mine).abs().max()))
    # At alpha = 0 only alpha has a gradient, apart from the gates, which enter this loss directly.
    for x in SIDES:
        models[x].train()
    fresh.zero_grad()
    last, gates = fresh.joint(S.collate([S.build(e, toks) for e in dialogue], pads))
    (sum(last[x].pow(2).mean() for x in SIDES) + sum(gates[x].sum() for x in SIDES)).backward()
    ga = {x: float(fresh.bridge[x].alpha.grad.abs().min()) for x in SIDES}
    other = max(float(p.grad.abs().max()) for n, p in fresh.bridge.named_parameters()
                if p.grad is not None and not n.endswith('alpha') and '.gate.' not in n)
    for x in SIDES:
        models[x].zero_grad()
        models[x].eval()
    assert min(ga.values()) > 0 and other == 0, (ga, other)


def test_checkpoint_without_recency_bias_and_buckets(pair, tiny):
    toks, models, _, _ = tiny
    ck = pair.checkpoint()
    ck['bridge'] = {k: v for k, v in ck['bridge'].items() if not k.endswith('.age')}
    old = Pair.from_checkpoint(models, toks, ck)
    assert all(float(old.bridge[x].age.detach().abs().max()) == 0 for x in SIDES)
    assert all(torch.equal(v, pair.bridge.state_dict()[k]) for k, v in old.bridge.state_dict().items()
               if not k.endswith('.age'))
    tq = torch.tensor([[-1, 0, 1, 2, 5, 9, 70, 70]])
    tk = torch.tensor([[-1, 0, 0, 1, 1, 1, 6, 5]])
    assert link.age_buckets(tq, tk)[0].diagonal().tolist() == [8, 0, 1, 1, 3, 4, 7, 7]


def test_recording_equals_fused_attention(pair, tiny, dialogue):
    toks, _, pads, _ = tiny
    b = S.collate([S.build(e, toks) for e in dialogue], pads)
    with torch.no_grad():
        l0, _ = pair.joint(b)
        pair.record = {}
        l1, _ = pair.joint(b)
        rec, pair.record = pair.record, None
    d = max(float((l0[x] - l1[x]).abs().max()) for x in SIDES)
    keys = sorted(rec)
    p = rec[keys[0]][0]['attn']
    s1 = float((p.sum(-1) - 1).abs().max())
    assert len(keys) == 2 * len(XB) and d < 1e-5 and s1 < 1e-5, (d, s1)


def paused(eps, k):
    return [dict(ep, pause=k, pause_turns={j: k for j in range(len(ep.get('turns', [])))}) for ep in eps]


@pytest.mark.parametrize('kind', ['duplex', 'dialogue'])
def test_zero_pause_is_unchanged(pair, tiny, duplex, dialogue, kind):
    toks = tiny[0]
    eps = duplex if kind == 'duplex' else dialogue
    zero = paused(eps, 0)
    assert [S.build(ep, toks) for ep in eps] == [S.build(ep, toks) for ep in zero]
    a = S.generate(pair, eps, max_turn=3, keep=True)
    b = S.generate(pair, zero, max_turn=3, keep=True)
    for aa, bb in zip(a, b):
        assert aa['tokens'] == bb['tokens']
        for x in SIDES:
            assert torch.equal(aa['h'][x], bb['h'][x])


@pytest.mark.parametrize('k', [0, 1, 3])
@pytest.mark.parametrize('kind', ['duplex', 'dialogue'])
def test_pauses_replay_and_reply_logits(pair, tiny, duplex, dialogue, kind, k):
    toks, models, pads, _ = tiny
    eps = duplex if kind == 'duplex' else dialogue
    eps_k = paused(eps, k)
    built = [S.build(ep, toks) for ep in eps_k]
    with torch.no_grad():
        h, _ = pair.joint(S.collate(built, pads))
        out = S.generate(pair, eps_k, force=True, keep=True)
        for i, (ep, e, r) in enumerate(zip(eps_k, built, out)):
            replay = S.build(S.replay(ep, r), toks)
            for x in SIDES:
                assert e[x]['ids'] == replay[x]['ids'] and e[x]['time'] == replay[x]['time']
                ps = [p for p, _, _ in e[x]['labels']]
                expected = inner(models[x])[1](h[x][i, ps])
                actual = inner(models[x])[1](r['h'][x][ps])
                torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
                assert len(e[x]['labels']) == len(S.build(eps[i], toks)[x]['labels'])


@pytest.mark.parametrize('kind', ['duplex', 'dialogue'])
def test_mixed_pauses_with_cut_and_shuffle(pair, tiny, duplex, dialogue, kind):
    eps = duplex if kind == 'duplex' else dialogue
    mixed = [dict(ep, pause=i, pause_turns={0: i, 1: 3 - i}, link_off_after=1) for i, ep in enumerate(eps)]
    check(pair, tiny, mixed, mode='shuffle')


@pytest.mark.parametrize('k', [-1, 1.2, True])
def test_invalid_pause_rejected(tiny, duplex, k):
    with pytest.raises(ValueError):
        S.build(dict(duplex[0], pause=k), tiny[0])
