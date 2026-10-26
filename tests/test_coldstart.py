"""coldstart.py on tiny random models: the learning-rate schedule and flag defaults; the normalized gate is
scale-invariant and the raw gate keeps its formula; identity starts (alpha 0, or O 0 at alpha 0.5) let exactly the
intended parameters learn; relay episodes score only the receiver's payload and keep the value out of its prompt
(--relay_games covers every value game); gate_norm survives a checkpoint; the bridge parameter groups and --hold.
"""
import argparse
import math
import re

import pytest
import torch

import coldstart
import games as G
import link
import session as S
from coldstart import VALUE_GAMES, add_arguments, bridge_groups, lr_multiplier, relay_episode


def test_schedule_and_defaults():
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    args = ap.parse_args([])
    assert args.warmup_steps == 100 and not args.gate_norm and not args.relay_first
    for step in (1, 20, 100, 500, 1000):
        assert lr_multiplier(step, 1000) == min(1., step / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / 1000)))
    assert lr_multiplier(1, 1000, 0) > 99 * lr_multiplier(1, 1000)


def test_gate_scale_invariance():
    torch.manual_seed(22)
    s = link.Side(128, 1, 2, 16, [1.], gate_norm=True)
    with torch.no_grad():
        s.gate[0].weight.normal_(0, .02)
    h = torch.randn(2, 4, 128)
    z = []
    for factor in (1., .03):
        s.zero_grad()
        gates = s.show(0, h * factor)[2]
        gates.sum().backward()
        z.append((gates.detach(), s.gate[0].weight.grad.clone()))
    assert torch.allclose(z[0][0], z[1][0], atol=2e-4)
    assert torch.allclose(z[0][1], z[1][1], atol=.004)
    # The 'random' ablation shows other states (src) but still gates the real ones.
    assert torch.equal(s.show(0, h)[2], s.show(0, h, src=torch.randn_like(h))[2])
    s.gate_norm = False
    assert torch.equal(s.show(0, h)[2], torch.sigmoid(s.gate[0](h).squeeze(-1)))


@pytest.mark.parametrize('start', [0., .5])
def test_identity_starts(start):
    """The read writes exactly zero at both starts; at alpha 0 only alpha gets a gradient, with O 0 at alpha 0.5 only O
    does."""
    torch.manual_seed(22)
    h = torch.randn(2, 4, 128)
    r, sender = link.Side(128, 1, 2, 16, [.1]), link.Side(128, 1, 2, 16, [.03])
    with torch.no_grad():
        r.alpha.fill_(math.atanh(start))
        if start:
            r.o[0].weight.zero_()
    keys, vals, _ = sender.show(0, h)
    w = r.read(0, h, keys, vals, torch.ones(2, 4, 4, dtype=torch.bool), torch.zeros(2, 4, 4, dtype=torch.long),
               torch.ones(2, 4))
    assert torch.count_nonzero(w) == 0
    (w * torch.randn_like(w)).sum().backward()
    assert (r.o[0].weight.grad.abs().sum() > 0) == bool(start)
    assert (r.alpha.grad.abs().sum() > 0) == (not start)
    assert r.q[0].weight.grad.abs().sum() == 0
    assert sender.v[0].weight.grad.abs().sum() == 0


@pytest.fixture(scope='module')
def toks():
    from transformers import AutoTokenizer
    return {x: AutoTokenizer.from_pretrained(link.MODELS[x]['hf']) for x in link.SIDES}


def test_relay_episodes(replies, toks):
    g = G.Games(replies, 'train', seed=11)
    for _ in range(8):
        ep = relay_episode(g)
        r, s = ep['meta']['receiver'], ep['meta']['sender']
        js = S.payload_labels(ep, toks)[r]
        built = S.build(ep, toks)
        assert js and all(built[s]['labels'][j][2] == 0 for j in range(len(built[s]['labels'])))
        assert all(built[r]['labels'][j][2] == 1 for j in js)
        assert len(built[s]['labels']) == 1
        assert ep['meta']['value'] not in ep['task'][r]
    # Relays keep an unscored sender and the value out of the receiver's prompt; merges and lookups are their regular
    # episodes with payload tokens; a mixed draw covers every game.
    for game in VALUE_GAMES:
        for _ in range(4):
            ep = relay_episode(g, (game,))
            assert ep['game'] == game
            pl = S.payload_labels(ep, toks)
            assert pl, game
            if game.startswith('relay'):
                r, s = ep['meta']['receiver'], ep['meta']['sender']
                whole = re.search(r'(?<!\w)' + re.escape(ep['meta']['value']) + r'(?!\w)', ep['task'][r])
                assert list(pl) == [r] and ep['reply'][s] == ('', 0.0) and not whole, game
    assert {relay_episode(g, VALUE_GAMES)['game'] for _ in range(200)} == set(VALUE_GAMES)


def test_gate_norm_checkpoint_and_groups(toks):
    models = {x: link.load(x)[1] for x in link.SIDES}
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16, gate_norm=True)
    ck = p.checkpoint()
    assert all(s.gate_norm for s in link.Pair.from_checkpoint(models, toks, ck).bridge.values())
    del ck['gate_norm']
    assert not any(s.gate_norm for s in link.Pair.from_checkpoint(models, toks, ck).bridge.values())
    groups = bridge_groups(p, .001, .01, {'quill': None, 'ember': .002})
    assert [g['lr'] for g in groups] == [.001, .002, .01]
    ids = [id(v) for g in groups for v in g['params']]
    assert len(ids) == len(set(ids)) == len(list(p.bridge.parameters()))


@pytest.mark.parametrize('held', [True, False])
def test_hold_keeps_gates_and_alpha(toks, held):
    """--hold: the gates and alpha keep their exact values through an AdamW step; every other bridge parameter moves."""
    models = {x: link.load(x)[1] for x in link.SIDES}
    torch.manual_seed(5)
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16, gate_norm=True)
    before = {n: v.detach().clone() for n, v in p.bridge.named_parameters()}
    opt = torch.optim.AdamW(bridge_groups(p, .01, .01, {'quill': None, 'ember': None}), weight_decay=0.0)
    sum((v * torch.randn_like(v)).sum() for v in p.bridge.parameters()).backward()
    if held:
        coldstart.hold(p)
    opt.step()
    for n, v in p.bridge.named_parameters():
        frozen = n.split('.')[1] in ('alpha', 'gate')
        assert torch.equal(v, before[n]) == (held and frozen), n
