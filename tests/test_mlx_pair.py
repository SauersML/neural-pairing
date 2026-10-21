"""Real torch/MLX parity, objective/gradient parity, and PEFT checkpoint round trip.

MPD_MEM_GIB=5 HF_HUB_OFFLINE=1 /Users/user/mpd-data/venv/bin/python tests/test_mlx_pair.py
No network, mocks, or pytest required. Uses both NL_TINY architectures/vocabs.
"""
import os
os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MLX_ENABLE_TF32'] = '0'
import sys
from pathlib import Path
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import link
import session as S
import mlx_pair as M
from mlx_train import objective, distill, make_optimizer

torch.set_num_threads(2)
torch.manual_seed(17)
mx.random.seed(17)
mx.set_cache_limit(32 * 2**20)
models, toks, mm = {}, {}, {}
for x in M.SIDES:
    toks[x], m = link.load(x)
    mm[x] = M.make_model(m.config.to_dict())
    weights = [(k, M.from_torch(v)) for k, v in m.state_dict().items()
               if k != 'lm_head.weight' or not m.config.tie_word_embeddings]
    mm[x].load_weights(weights, strict=True)
    M.add_lora(mm[x], 32)
    m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, target_modules=list(M.TARGETS)))
    for name, mod in m.get_base_model().named_modules():
        if hasattr(mod, 'lora_A'):
            torch.nn.init.normal_(mod.lora_B['default'].weight, std=.006)
            ml = dict(mm[x].named_modules())[name]
            ml.lora_A = M.from_torch(mod.lora_A['default'].weight)
            ml.lora_B = M.from_torch(mod.lora_B['default'].weight)
    models[x] = m.eval()
xb = (1, 2, 4)
sizes = {x: [.3, .6, .9] for x in M.SIDES}
pair = link.Pair(models, toks, xb, sizes, heads=8, head_dim=128, gate_norm=True)
with torch.no_grad():
    for side in pair.bridge.values():
        side.alpha.uniform_(-.6, .6)
        side.null.normal_(0, .3)
        side.age.normal_(0, .4)
        for g in side.gate:
            g.weight.normal_(0, .08); g.bias.normal_(0, .2)
        for n in list(side.q_norm) + list(side.kv_norm):
            n.weight.uniform_(.8, 1.2)
mp = M.Pair(mm, xb, sizes, heads=8, head_dim=128, gate_norm=True, checkpoint=False)
for x in M.SIDES:
    mp.bridge[x].load_weights([(k, M.from_torch(v)) for k, v in pair.bridge[x].state_dict().items()])
mx.eval(mp.parameters())
system = {x: 'You are ' + x + '.' for x in M.SIDES}
eps = [dict(game='tiny', meta={}, mode='duplex', system=system,
            task={'quill': 'Remember the number 42.', 'ember': 'What is the number?'},
            reply={'quill': ('Yes.', .2), 'ember': ('42.', 1.)}),
       dict(game='tiny', meta={}, mode='dialogue', system=system,
            task={'quill': 'Ask a question.', 'ember': None}, link_off_after=1,
            turns=[('quill', 'Blue?', .4), ('ember', 'Green.', 1.), ('quill', 'Okay.', .6)]),
       dict(game='tiny', meta={}, mode='duplex', system=system,
            task={'quill': 'Say a color.', 'ember': 'Say a name.'},
            reply={'quill': ('Red.', 1.), 'ember': ('Al.', .6)})]
built = [S.build(e, toks) for e in eps]
pads = {x: toks[x].pad_token_id for x in M.SIDES}
b = S.collate(built, pads); mb = M.convert_batch(b)
results = {}
for mode in ('zero', None, 'shuffle'):
    with torch.no_grad():
        h, g = pair.joint(b, mode)
    mh, mg, _ = mp.joint(mb, mode)
    for x in M.SIDES:
        tl = pair.token_logits(x, h[x], b[x]['idx']).detach().numpy()
        ml = np.array(mp.token_logits(x, mh[x], mb[x]['idx']))
        err = float(np.max(np.abs(tl - ml)))
        results[f'logits/{mode}/{x}'] = err
        print('parity', mode, x, err, flush=True)
        np.testing.assert_allclose(ml, tl, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(np.array(mg[x]), g[x].numpy(), atol=2e-6, rtol=2e-5)
print('forward', results, flush=True)
# Full-distribution co-distillation: use genuine teacher token streams, with
# nonzero LoRA matrices, to prove that the teacher has adapters disabled.
for x in M.SIDES:
    p = S.prefix(toks[x], x, 'You are ' + x + '.', 'The original content is blue.')
    p += S.assistant_header(toks[x], x)
    reply = [tid for _, tid, _ in built[0][x]['labels']]
    built[0][x]['teacher'] = (p + reply[:-1], len(p) - 1)
kl = distill(mp, built, pads, chunk=4)
torch_kl = {}
for x in M.SIDES:
    ids, start = built[0][x]['teacher']
    with torch.no_grad(), models[x].disable_adapter():
        logits = models[x](input_ids=torch.tensor([ids])).logits[0, start:].float()
        lt = logits.log_softmax(-1)
    sel, ml = kl[x]
    results['teacher/' + x] = float(np.max(np.abs(np.array(ml) - lt.numpy())))
    np.testing.assert_allclose(np.array(ml), lt.numpy(), atol=2e-5, rtol=2e-5)
    torch_kl[x] = (torch.from_numpy(np.array(sel).copy()).long(), lt)
# Full CE plus masked send cost; compare LoRA and every bridge gradient.
denom = {x: max(1., float(b[x]['w'].sum())) for x in M.SIDES}
n_show = {x: int(b[x]['att'].sum()) * len(xb) for x in M.SIDES}
h, gates = pair.joint(b)
tl = torch.zeros(())
for x in M.SIDES:
    logits = pair.token_logits(x, h[x], b[x]['idx'])
    ce = F.cross_entropy(logits, b[x]['lab'], reduction='none')
    sel, lt = torch_kl[x]
    ce = ce.index_put((sel,), (lt.exp() * (lt - logits[sel].log_softmax(-1))).sum(-1))
    tl = tl + (ce * b[x]['w']).sum() / denom[x]
    tl = tl + .007 * (gates[x] * b[x]['att'][..., None]).sum() / n_show[x]
tl.backward()
mp.checkpoint = True
(ml, _), grads = nn.value_and_grad(mp, lambda p: objective(p, mb, kl, denom, n_show, .007, chunk=4))(mp)
mx.eval(ml, grads)
results['loss_error'] = abs(float(ml) - float(tl.detach()))
assert results['loss_error'] < 2e-5
fg = dict(tree_flatten(grads)); grad_errors = []
for x in M.SIDES:
    for k, v in pair.bridge[x].named_parameters():
        a, t = np.array(fg['bridge.' + x + '.' + k]), v.grad.numpy()
        grad_errors.append(float(np.max(np.abs(a - t))))
        np.testing.assert_allclose(a, t, atol=3e-5, rtol=.004)
    for name, mod in models[x].get_base_model().named_modules():
        if hasattr(mod, 'lora_A'):
            for k in ('lora_A', 'lora_B'):
                a = np.array(fg['models.' + x + '.' + name + '.' + k])
                t = getattr(mod, k)['default'].weight.grad.numpy()
                grad_errors.append(float(np.max(np.abs(a - t))))
                np.testing.assert_allclose(a, t, atol=3e-5, rtol=.004)
results['max_gradient_error'] = max(grad_errors)
print('gradients', results['loss_error'], results['max_gradient_error'], flush=True)
# Competitive losses must backpropagate THROUGH the partner while updating
# only the owner. This also exercises a sender-owned loss on receiver labels.
owners = [{'quill': frozenset(['quill']), 'ember': frozenset(['ember'])},
          {'quill': frozenset(), 'ember': frozenset(['quill'])},
          {'quill': frozenset(['ember']), 'ember': frozenset(['ember'])}]
cb = S.collate(built, pads, owners)
cm = M.convert_batch(cb)
ch, cg = pair.joint(cb)
ces = {x: F.cross_entropy(pair.token_logits(x, ch[x], cb[x]['idx']), cb[x]['lab'], reduction='none')
       for x in M.SIDES}
for owner in M.SIDES:
    loss = sum((ces[x] * cb[x]['w'] * cb[x]['own_mask'][owner]).sum() / denom[x] for x in M.SIDES)
    loss = loss + .007 * (cg[owner] * cb[owner]['att'][..., None]).sum() / n_show[owner]
    named = [('bridge.' + owner + '.' + n, p) for n, p in pair.bridge[owner].named_parameters()]
    for name, mod in models[owner].get_base_model().named_modules():
        if hasattr(mod, 'lora_A'):
            named += [('models.' + owner + '.' + name + '.' + k, getattr(mod, k)['default'].weight)
                      for k in ('lora_A', 'lora_B')]
    tg = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True)
    (_, _), mg = nn.value_and_grad(mp, lambda p: objective(p, cm, {}, denom, n_show, .007,
                                                          chunk=4, owner=owner))(mp)
    mx.eval(mg)
    mg = dict(tree_flatten(mg))
    errors = []
    for (name, _), t in zip(named, tg):
        a = np.array(mg[name])
        errors.append(float(np.max(np.abs(a - t.numpy()))))
        np.testing.assert_allclose(a, t.numpy(), atol=3e-5, rtol=.004)
    results['competitive_gradient/' + owner] = max(errors)
    print('competitive gradient', owner, max(errors), flush=True)
del ch, cg, ces, tg, mg, loss
# A real optimizer step exercises trainability and exports nontrivial adapters.
from argparse import Namespace
args = Namespace(lr=1e-4, lr_alpha=3e-3, lr_bridge=3e-4, lr_bridge_quill=None, lr_bridge_ember=None)
opt, _ = make_optimizer(args)
opt.update(mp, grads)
mx.eval(mp.parameters(), opt.state)
mp.checkpoint = False
expected, _, _ = mp.joint(mb)
with tempfile.TemporaryDirectory() as tmp:
    M.export_checkpoint(mp, tmp, vars(args))
    from evaluate import load_pair
    restored, _ = load_pair(tmp)
    ck = torch.load(Path(tmp) / 'bridge.pt', weights_only=False)
    assert set(ck['bridge']) == set(pair.bridge.state_dict())
    with torch.no_grad():
        h, _ = restored.joint(b)
    for x in M.SIDES:
        actual = restored.token_logits(x, h[x], b[x]['idx']).detach().numpy()
        want = np.array(mp.token_logits(x, expected[x], mb[x]['idx']))
        results['roundtrip/' + x] = float(np.max(np.abs(actual - want)))
        np.testing.assert_allclose(actual, want, atol=2e-5, rtol=2e-5)
    M.load_checkpoint(mp, tmp)
    again, _, _ = mp.joint(mb)
    for x in M.SIDES:
        np.testing.assert_allclose(np.array(again[x]), np.array(expected[x]), atol=1e-7)
print('PASS', results, flush=True)
