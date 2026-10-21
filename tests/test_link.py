"""Exactness checks of the bridges and the session timelines on tiny random models of both architectures (CPU, float32).

a. The joint teacher-forced pass (Pair.joint on session.collate) gives the same last hidden states and gates at every
   token as generation tick by tick and block by block (session.generate), teacher-forced on the same text: in duplex
   and in dialogue sessions, with the link open and closed after the first turn, and with shuffled partners.
b. Gradients with gradient checkpointing equal gradients without it, and every bridge parameter gets one.
c. Without the bridges ('zero'), and with untrained bridges (alpha = 0), the pair's logits equal the plain LoRA models'
   logits bit for bit; at alpha = 0 only alpha has a gradient.
d. The recording hook (explicit softmax) gives the same states as the fused attention, and its weights sum to one.
The data parallel check is tests/test_ddp.py.

Usage: python tests/test_link.py   (from the repository root)
"""
import os
import sys

os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pairing'))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import link  # noqa: E402
import session as S  # noqa: E402
from link import SIDES, Pair, inner  # noqa: E402

XB = (1, 2, 4)
toks, models = {}, {}
for x in SIDES:
    toks[x], m = link.load(x)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj', 'up_proj', 'down_proj']))
    for n, p in m.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.05)
    models[x] = m.eval()
pads = {x: toks[x].pad_token_id for x in SIDES}
system = {x: f'You are {link.MODELS[x]["name"]}.' for x in SIDES}
duplex = [dict(mode='duplex', system=system, task={'quill': 'What is two plus two?', 'ember': 'Name three colours.'},
               reply={'quill': ('Four, of course! It is simple.', 1.0), 'ember': ('Red, green and blue are three.', 1.0)}),
          dict(mode='duplex', system=system, task={'quill': 'Tell me a story.', 'ember': 'Why is the sea salty?'},
               reply={'quill': ('Once upon a time', 1.0), 'ember': ('Because rivers carry minerals there over time.', 1.0)}),
          dict(mode='duplex', system=system, task={'quill': 'Say hi.', 'ember': 'Count to three, please, slowly.'},
               reply={'quill': ('Hi!', 1.0), 'ember': ('One, two, three.', 1.0)})]
dialogue = [dict(mode='dialogue', system=system, task={'quill': 'Ask Ember a question.', 'ember': None},
                 turns=[('quill', 'Why is the sky blue, Ember?', 1.0),
                        ('ember', 'Sunlight scatters off air; blue light scatters most.', 1.0),
                        ('quill', 'Thanks! Café crème? Ünïcode works.', 1.0)]),
            dict(mode='dialogue', system=system, task={'quill': None, 'ember': 'Talk about cats.'},
                 turns=[('ember', 'Cats sleep a lot, about sixteen hours a day.', 1.0),
                        ('quill', 'Wow.', 1.0)])]
sizes = {x: link.residual_rms(models[x], XB, [S.build(duplex[0], toks)[x]['ids']]) for x in SIDES}
print('residual RMS after the exchange blocks:', {x: [round(v, 3) for v in sizes[x]] for x in SIDES})


def trained(seed):
    """A pair whose bridges look trained: alpha, gates and null keys away from their initial values."""
    torch.manual_seed(seed)
    pair = Pair(models, toks, XB, sizes, heads=2, head_dim=16)
    for x in SIDES:
        s = pair.bridge[x]
        s.alpha.data.uniform_(-1.0, 1.0)
        s.null.data.normal_(0, 0.5)
        for g in s.gate:
            g.weight.data.normal_(0, 0.3)
            g.bias.data.normal_(0, 1.0)
        for nrm in list(s.q_norm) + list(s.kv_norm):
            nrm.weight.data.uniform_(0.5, 1.5)
    return pair


pair = trained(1)


def check(eps, name, mode=None):
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
            assert gen[i]['h'][x].shape[0] == n, (name, x, i, gen[i]['h'][x].shape, n)
            dh = max(dh, float((last[x][i, :n] - gen[i]['h'][x]).abs().max() / last[x][i, :n].abs().max()))
            dg = max(dg, float((gates[x][i, :n] - gen[i]['p'][x]).abs().max()))
            moved = max(moved, float((gen[i]['h'][x] - cut[i]['h'][x]).abs().max() / cut[i]['h'][x].abs().max()))
    print(f'a. {name}: max |joint - generate| last hidden state (relative) {dh:.1e}, gate {dg:.1e}; '
          f'change without the bridges {moved:.3f}')
    assert dh < 1e-4 and dg < 1e-5 and moved > 1e-2, (dh, dg, moved)
    return b, last


check(duplex, 'duplex')
bd, open_ = check(dialogue, 'dialogue')
bc, closed = check([dict(e, link_off_after=1) for e in dialogue], 'dialogue, link closed after turn 1')
before = after = 0.0
for x in SIDES:
    valid = bc[x]['att'].bool()
    d = (open_[x] - closed[x]).abs().amax(-1)
    early = valid & (bc[x]['tick'] < bc[x]['cut'][:, None])
    before = max(before, float(d[early].max()))
    after = max(after, float(d[valid & ~early].max()))
print(f'   closing the link after turn 1 changes nothing before the cut (max {before:.1e}) and the states after it '
      f'(max {after:.3f})')
assert before == 0 and after > 1e-2, (before, after)
check(duplex, 'duplex, shuffled partners', mode='shuffle')
check(dialogue, 'dialogue, shuffled partners', mode='shuffle')


def grads(ckpt, b, pair):
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


g0, g1 = grads(False, bd, pair), grads(True, bd, pair)
with torch.utils.checkpoint.set_checkpoint_early_stop(False):
    g2 = grads(True, bd, pair)
rel = max(float((g0[k] - g[k]).norm() / (g0[k].norm() + 1e-12)) for g in (g1, g2) for k in g0)
nb = sum(1 for k in g0 if k.startswith('bridge.'))
nz = sum(1 for k in g0 if k.startswith('bridge.') and float(g0[k].abs().max()) > 0)
print(f'b. {len(g0)} gradient tensors ({nb} of the bridges, {nz} nonzero, of {len(list(pair.bridge.parameters()))} '
      f'bridge parameters); max relative difference with checkpointing (with and without early stop): {rel:.1e}')
assert set(g0) == set(g1) == set(g2) and rel < 1e-5 and nz == nb == len(list(pair.bridge.parameters())), rel

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
    print(f"c. {eps[0]['mode']}: logits without the bridges ('zero') and with untrained bridges (alpha = 0) equal the "
          f'plain LoRA models bit for bit')
for x in SIDES:
    models[x].train()
fresh.zero_grad()
last, gates = fresh.joint(bd)
(sum(last[x].pow(2).mean() for x in SIDES) + sum(gates[x].sum() for x in SIDES)).backward()
ga = {x: float(fresh.bridge[x].alpha.grad.abs().min()) for x in SIDES}
other = max(float(p.grad.abs().max()) for n, p in fresh.bridge.named_parameters()
            if p.grad is not None and not n.endswith('alpha') and '.gate.' not in n)
print(f'   alpha = 0: smallest |gradient| of alpha {min(ga.values()):.1e}; largest gradient of Q, K, V, O, norms and '
      f'null keys {other:.1e}')
assert min(ga.values()) > 0 and other == 0, (ga, other)
for x in SIDES:
    models[x].zero_grad()
    models[x].eval()

with torch.no_grad():
    l0, _ = pair.joint(bd)
    pair.record = {}
    l1, _ = pair.joint(bd)
    rec, pair.record = pair.record, None
d = max(float((l0[x] - l1[x]).abs().max()) for x in SIDES)
keys = sorted(rec)
p = rec[keys[0]][0]['attn']
s1 = float((p.sum(-1) - 1).abs().max())
print(f'd. recording: {len(keys)} (side, block) entries, attention {tuple(p.shape)}, write '
      f'{tuple(rec[keys[0]][0]["write"].shape)}, shown {tuple(rec[keys[0]][0]["shown"].shape)}; max |explicit - fused| '
      f'last hidden state {d:.1e}; attention sums to one within {s1:.1e}')
assert len(keys) == 2 * len(XB) and d < 1e-5 and s1 < 1e-5, (d, s1)
print('all checks passed')
