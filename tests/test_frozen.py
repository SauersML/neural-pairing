"""The one-way link (link.Pair(frozen=y), train.py --frozen) on tiny random models (CPU, float32): the target is the
original model bit for bit with the link on; the reader's joint pass equals its tick-by-tick generation and the link
changes it; no gradient or autograd graph reaches the target's model and only the used bridge parts get gradients;
two-way and one-way checkpoints load one-way (link.Pair.from_checkpoint, evaluate.load_pair), also from the
pickle-free form (evaluate.write_bridge); payload_ctl and chateval take the reader as the only receiver; train.py
--frozen from a two-way step saves a one-way step; and the MLX evaluator (mlx_eval.load_pair) reads a one-way step as
the torch pair does.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from peft import LoraConfig, get_peft_model
import pytest
import torch

import chateval
import evaluate
import link
import payload_ctl
import session as S
from link import READ, SHOW, SIDES, Pair, inner, partner

ROOT = Path(__file__).resolve().parents[1]
XB = (1, 2, 4)


@pytest.fixture(scope='module')
def tiny(tiny_models, duplex):
    """Tokenizers, plain models (no gradients), LoRA models, pads and residual sizes."""
    toks, plain = tiny_models()
    for m in plain.values():
        m.requires_grad_(False)
    _, lora = tiny_models(['q_proj', 'v_proj', 'up_proj', 'down_proj'], std=0.05)
    sizes = {x: link.residual_rms(plain[x], XB, [S.build(duplex[0], toks)[x]['ids']]) for x in SIDES}
    return toks, plain, lora, {x: toks[x].pad_token_id for x in SIDES}, sizes


def trained(pair, seed):
    """Bridges that look trained: alpha, gates, null keys, recency bias and norms away from their initial values."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for x in SIDES:
            s = pair.bridge[x]
            s.alpha.uniform_(-1.0, 1.0)
            s.null.normal_(0, 0.5)
            s.age.normal_(0, 1.0)
            for g in s.gate:
                g.weight.normal_(0, 0.3)
                g.bias.normal_(0, 1.0)
            for nrm in list(s.q_norm) + list(s.kv_norm):
                nrm.weight.uniform_(0.5, 1.5)
    return pair


def one_way(tiny, y, seed=1):
    """A one-way pair onto y with trained-looking bridges: y plain, its partner with LoRA."""
    toks, plain, lora, _, sizes = tiny
    models = {x: plain[x] if x == y else lora[x] for x in SIDES}
    return trained(Pair(models, toks, XB, sizes, heads=2, head_dim=16), seed).freeze(y)


@pytest.mark.parametrize('y', SIDES)
def test_target_is_the_original_model(tiny, duplex, dialogue, y):
    toks, plain, _, pads, _ = tiny
    r = partner(y)
    pair = one_way(tiny, y)
    assert float(pair.bridge[y].alpha.abs().max()) == 0
    for eps in (duplex[0::2], dialogue):
        name = eps[0]['mode']
        built = [S.build(e, toks) for e in eps]
        b = S.collate(built, pads)
        with torch.no_grad():
            last, gates = pair.joint(b)
            want = plain[y](input_ids=b[y]['ids'], attention_mask=b[y]['att'], position_ids=b[y]['pos']).logits
            assert torch.equal(want, inner(plain[y])[1](last[y])), (y, name)
            assert float(gates[r].abs().max()) == 0 and float(gates[y].abs().max()) > 0
            gen = S.generate(pair, eps, force=True, keep=True)
            cut = S.generate(pair, eps, force=True, keep=True, mode='zero')
        dh = moved = 0.0
        for i, e in enumerate(built):
            assert torch.equal(gen[i]['h'][y], cut[i]['h'][y]), (y, name, i)
            n = len(e[r]['ids'])
            dh = max(dh, float((last[r][i, :n] - gen[i]['h'][r]).abs().max() / last[r][i, :n].abs().max()))
            moved = max(moved, float((gen[i]['h'][r] - cut[i]['h'][r]).abs().max() / cut[i]['h'][r].abs().max()))
        assert dh < 1e-4 and moved > 1e-2, (y, name, dh, moved)


@pytest.mark.parametrize('y', SIDES)
def test_gradients_skip_the_target(tiny, dialogue, y):
    toks, plain, lora, pads, _ = tiny
    r = partner(y)
    pair = one_way(tiny, y)
    lora[r].train()
    pair.zero_grad()
    lora[r].zero_grad()
    last, gates = pair.joint(S.collate([S.build(e, toks) for e in dialogue], pads))
    assert not last[y].requires_grad, 'an autograd graph runs through the target'
    (sum(last[x].pow(2).mean() for x in SIDES) + sum(gates[x].sum() for x in SIDES)).backward()
    assert all(p.grad is None for p in plain[y].parameters())
    got = {f'{x}.{n}': p.grad for x in SIDES for n, p in pair.bridge[x].named_parameters()}
    used = {k for k in got if k.split('.')[1] in {y: SHOW, r: READ}[k.split('.')[0]]}
    assert all(got[k] is not None and float(got[k].abs().max()) > 0 for k in used), \
        [k for k in used if got[k] is None or float(got[k].abs().max()) == 0]
    assert all(got[k] is None for k in set(got) - used), [k for k in set(got) - used if got[k] is not None]
    lg = [p.grad for n, p in lora[r].named_parameters() if 'lora_' in n]
    assert lg and all(g is not None for g in lg) and max(float(g.abs().max()) for g in lg) > 0
    trainable = {f'{x}.{n}' for x in SIDES for n, p in pair.bridge[x].named_parameters() if p.requires_grad}
    assert trainable == used, sorted(trainable ^ used)
    lora[r].zero_grad()
    lora[r].eval()


@pytest.mark.parametrize('y', SIDES)
def test_checkpoints(tiny, y, tmp_path):
    toks, plain, lora, _, sizes = tiny
    r = partner(y)
    two = trained(Pair(lora, toks, XB, sizes, heads=2, head_dim=16), 2)
    ck2 = two.checkpoint()
    assert ck2['frozen'] is None
    onto_y = {x: plain[x] if x == y else lora[x] for x in SIDES}
    loaded = Pair.from_checkpoint(onto_y, toks, ck2, frozen=y)
    assert loaded.frozen == y and float(loaded.bridge[y].alpha.abs().max()) == 0
    sd2, sd = two.bridge.state_dict(), loaded.bridge.state_dict()
    assert all(torch.equal(sd2[k], sd[k]) for k in sd if k not in loaded.unused() or k.endswith('.size'))
    assert {k for k in sd if k.split('.')[1] != 'size'} - loaded.unused() == \
        {f'{x}.{n}' for x in SIDES for n, p in loaded.bridge[x].named_parameters() if p.requires_grad}
    ck1 = loaded.checkpoint()
    assert ck1['frozen'] == y and not set(ck1['bridge']) & loaded.unused()
    assert set(ck1['bridge']) == set(sd) - loaded.unused()
    again = Pair.from_checkpoint(onto_y, toks, ck1)
    assert again.frozen == y and all(torch.equal(v, again.bridge.state_dict()[k]) for k, v in ck1['bridge'].items())
    with pytest.raises(ValueError):                  # a one-way checkpoint onto the other side
        Pair.from_checkpoint(onto_y, toks, ck1, frozen=r)
    with pytest.raises(ValueError):                  # a target with LoRA
        Pair(lora, toks, XB, sizes, heads=2, head_dim=16).freeze(y)
    one = tmp_path / 'one'
    lora[r].save_pretrained(one / r)
    torch.save(ck1, one / 'bridge.pt')
    got, _ = evaluate.load_pair(str(one))
    assert got.frozen == y and not hasattr(got.models[y], 'peft_config')
    assert all(torch.equal(v.cpu(), got.bridge.state_dict()[k].cpu()) for k, v in ck1['bridge'].items())
    both = tmp_path / 'two'
    for x in SIDES:
        lora[x].save_pretrained(both / x)
    torch.save(ck2, both / 'bridge.pt')
    got, _ = evaluate.load_pair(str(both), frozen=y)
    assert got.frozen == y and not hasattr(got.models[y], 'peft_config')
    assert evaluate.load_pair(str(both))[0].frozen is None
    released = tmp_path / 'released'                 # the pickle-free form loads the same link
    for x in SIDES:
        lora[x].save_pretrained(released / x)
    evaluate.write_bridge(ck2, released)
    got = evaluate.load_pair(str(released))[0].bridge.state_dict()
    assert all(torch.equal(v.cpu(), got[k].cpu()) for k, v in ck2['bridge'].items())


@pytest.mark.parametrize('y', SIDES)
def test_reader_is_the_only_receiver(tiny, replies, y):
    toks = tiny[0]
    r = partner(y)
    pair = one_way(tiny, y)
    eps = payload_ctl.episodes(replies, 4, receiver=r)
    for g, es in eps.items():
        assert len(es) == 4 and all((e['meta'].get('receiver') or e['meta'].get('asker')) in (None, r) for e in es), g
    rec = payload_ctl.table(pair, toks, eps['relay_word'], bs=2)
    assert rec and {q['receiver'] for q in rec} == {r}, rec
    assert chateval.receivers(pair) == [r]


def test_two_way_receivers(tiny):
    toks, _, lora, _, sizes = tiny
    assert chateval.receivers(Pair(lora, toks, XB, sizes, heads=2, head_dim=16)) == list(SIDES)


@pytest.mark.slow
def test_train_frozen_from_a_two_way_step(tiny, data, tmp_path):
    toks, _, lora, _, sizes = tiny
    two = trained(Pair(lora, toks, XB, sizes, heads=2, head_dim=16), 3)
    init, out = tmp_path / 'init', tmp_path / 'out'
    for x in SIDES:
        lora[x].save_pretrained(init / x)
    torch.save(two.checkpoint(), init / 'bridge.pt')
    env = dict(os.environ, MPD_MEM_GIB='8', NL_QUICK='relay_word', NL_MAXNEW='2')
    common = [sys.executable, str(ROOT / 'pairing' / 'train.py'), '--replies', str(data / 'replies.json'),
              '--out', str(out), '--steps', '3', '--bs', '4', '--relay_first', '2', '--anchor', '0.5',
              '--first_eval', '1000', '--save_every', '1000', '--heads', '2', '--head_dim', '16', '--xblocks', '1,2,4',
              '--ckpt', 'quill', '--quick_n', '1', '--payload_eval', '2', '--read_start', '0.1']
    res = subprocess.run(common + ['--init', str(init), '--frozen', 'ember'], cwd=ROOT / 'pairing', env=env,
                         capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert res.returncode == 0, res.stderr[-3000:]
    step = out / 'step_3'
    assert (step / 'quill').is_dir() and not (step / 'ember').exists()
    ck = torch.load(step / 'bridge.pt', weights_only=False)
    assert ck['frozen'] == 'ember' and float(ck['bridge']['quill.alpha'].abs().max()) > 0
    assert not any(k.split('.')[1] in READ for k in ck['bridge'] if k.startswith('ember.'))
    assert not any(k.split('.')[1] in SHOW for k in ck['bridge'] if k.startswith('quill.'))
    first = json.loads((out / 'payload_step0.json').read_text())
    assert first and {q['receiver'] for q in first} == {'quill'}
    assert 'one-way link: ember is the original model; quill reads it' in res.stdout
    tw = two.checkpoint()['bridge']
    moved = [k for k in ck['bridge'] if k.startswith('ember.') and k.split('.')[1] in ('kv_norm', 'k', 'v')
             and not torch.equal(ck['bridge'][k], tw[k])]
    assert moved, "the target's show parts did not train"
    # The resumed one-way run keeps its target without --frozen.
    res = subprocess.run(common + ['--init', str(step), '--halt', '1'], cwd=ROOT / 'pairing', env=env,
                         capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert res.returncode == 0 and 'one-way link: ember' in res.stdout, res.stderr[-3000:]


@pytest.mark.parametrize('y', SIDES)
def test_mlx_evaluator_reads_a_one_way_step(tiny, duplex, dialogue, y, tmp_path):
    mx = pytest.importorskip('mlx.core')
    import numpy as np
    import mlx_eval as V
    import mlx_pair as M
    toks, plain, _, pads, _ = tiny
    mx.set_memory_limit(2 * 2**30)
    r = partner(y)
    torch.manual_seed(5)
    toks = dict(toks)
    toks[r], m = link.load(r)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=list(M.TARGETS)))
    for n, p in m.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.006)
    models = {y: plain[y], r: m.eval()}
    pair = trained(Pair(models, toks, XB, {x: [.3, .6, .9] for x in SIDES}, heads=2, head_dim=16, gate_norm=True), 6)
    pair.freeze(y)
    with torch.no_grad():
        for g in pair.bridge[y].gate:
            g.weight.normal_(0, .08)
            g.bias.normal_(0, .2)
        for nrm in list(pair.bridge[r].q_norm) + list(pair.bridge[y].kv_norm):
            nrm.weight.fill_(1.0)
        pair.bridge[r].alpha.uniform_(-.6, .6)
    m.save_pretrained(tmp_path / r)
    torch.save(pair.checkpoint(), tmp_path / 'bridge.pt')
    vp, _ = V.load_pair(str(tmp_path))
    assert vp.frozen == y
    b = S.collate([S.build(e, toks) for e in duplex[0::2] + dialogue], pads)
    with torch.no_grad():
        h, gt = pair.joint(b)
        mh, mg = vp.joint(b)
        for x in SIDES:
            tl = pair.token_logits(x, h[x], b[x]['idx']).numpy()
            ml = vp.token_logits(x, mh[x], b[x]['idx']).numpy()
            np.testing.assert_allclose(ml, tl, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(np.asarray(mg[x]), gt[x].numpy(), atol=2e-6, rtol=2e-5)
        for eps in (duplex[0::2], dialogue):
            a = S.generate(pair, eps, max_turn=3, keep_tokens=True)
            z = S.generate(vp, eps, max_turn=3, keep_tokens=True)
            assert [q['tokens'] for q in a] == [q['tokens'] for q in z], y
