"""Checks of the link and the session timelines on tiny random models of both architectures (CPU, float32).

1. With enough Gauss-Seidel rounds, the training forward (Pair.coupled on session.build) gives the same gate
   probabilities at every sending token as step-by-step generation (session.generate), teacher-forced on the same
   text, in duplex and in dialogue sessions, also when the link closes after the first turn, with messages delivered
   once or kept in the stream (the inbox). The gate reads the residual stream after every message has landed, so this
   checks the whole timing.
2. Gradients with gradient checkpointing equal gradients without it, also when the recomputation reruns the hooks.
3. The same for the soft-token read (messages as tokens of their own in the partner's input), and with the link cut
   ('zero': slots masked out) the soft-read pair gives the same logits as the pair whose link adds nothing.

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
from link import SIDES, Pair  # noqa: E402

toks, models = {}, {}
for x in SIDES:
    toks[x], m = link.load(x)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj', 'up_proj', 'down_proj']))
    for n, p in m.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.05)
    models[x] = m.eval()
torch.manual_seed(1)
pair = Pair(models, toks, {x: 2 for x in SIDES}, {x: 4 for x in SIDES}, {x: 1.0 for x in SIDES})
for x in SIDES:
    for p in pair.side[x].parameters():
        if p.dim() > 1:
            p.data.normal_(0, 0.2)
    pair.side[x].gate.weight.data.normal_(0, 0.3)
    pair.side[x].log_size.data.fill_(0.0)
pads = {x: toks[x].pad_token_id for x in SIDES}
system = {x: f'You are {link.MODELS[x]["name"]}.' for x in SIDES}
duplex = [dict(mode='duplex', system=system, task={'quill': 'What is two plus two?', 'ember': 'Name three colours.'},
               reply={'quill': ('Four, of course! It is simple.', 1.0), 'ember': ('Red, green and blue are three.', 1.0)}),
          dict(mode='duplex', system=system, task={'quill': 'Tell me a story.', 'ember': 'Why is the sea salty?'},
               reply={'quill': ('Once upon a time', 1.0), 'ember': ('Because rivers carry minerals there over time.', 1.0)})]
dialogue = [dict(mode='dialogue', system=system, task={'quill': 'Ask Ember a question.', 'ember': None},
                 turns=[('quill', 'Why is the sky blue, Ember?', 1.0),
                        ('ember', 'Sunlight scatters off air; blue light scatters most.', 1.0),
                        ('quill', 'Thanks! Café crème? Ünïcode works.', 1.0)]),
            dict(mode='dialogue', system=system, task={'quill': None, 'ember': 'Talk about cats.'},
                 turns=[('ember', 'Cats sleep a lot, about sixteen hours a day.', 1.0),
                        ('quill', 'Wow.', 1.0)])]


def check(eps, name):
    built = [S.build(e, toks) for e in eps]
    b = S.collate(built, pads, soft=pair.soft)
    T = max(len(e[x]['ids']) for e in built for x in SIDES)
    with torch.no_grad():
        _, msgs = pair.coupled(b, rounds=T)
        gen = S.generate(pair, eps, force=True, keep=True)
    worst = 0.0
    for i in range(len(eps)):
        for x in SIDES:
            p_gs = msgs[x][2][i][b[x]['send_valid'][i]]
            p_gen = gen[i]['p'][x]
            assert len(p_gs) == len(p_gen), (name, x, i, len(p_gs), len(p_gen))
            worst = max(worst, float((p_gs - p_gen).abs().max()))
    with torch.no_grad():
        cut = S.generate(pair, eps, force=True, keep=True, mode='zero')
    moved = max(float((gen[i]['p'][x] - cut[i]['p'][x]).abs().max()) for i in range(len(eps)) for x in SIDES)
    print(f'1. {name}: max |coupled - generate| gate probability {worst:.2e}; change when the link is cut {moved:.3f}')
    assert worst < 1e-4 and moved > 1e-3, (worst, moved)
    if pair.soft:
        with torch.no_grad():
            _, mz = pair.coupled(b, rounds=T, mode='zero')
        wz = max(float((mz[x][2][i][b[x]['send_valid'][i]] - cut[i]['p'][x]).abs().max())
                 for i in range(len(eps)) for x in SIDES)
        print(f'   link cut: max |coupled - generate| gate probability {wz:.2e}')
        assert wz < 1e-4, wz
    if name == 'dialogue':
        print('   generated turns:', [(s, t[:40]) for s, t in gen[0]['turns']])
    return b


check(duplex, 'duplex')
bd = check(dialogue, 'dialogue')
check([dict(e, link_off_after=1) for e in dialogue], 'dialogue, link closed after turn 1')
pair.inbox = True
check(duplex, 'duplex, inbox')
check(dialogue, 'dialogue, inbox')
check([dict(e, link_off_after=1) for e in dialogue], 'dialogue, inbox, link closed after turn 1')
pair.inbox = False


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
    last, msgs = pair.coupled(b, rounds=3)
    loss = sum(last[x].pow(2).mean() for x in SIDES) + sum(msgs[x][2].sum() for x in SIDES)
    loss.backward()
    pair.clear()
    out = {n: p.grad.clone() for x in SIDES for n, p in models[x].named_parameters() if p.grad is not None}
    out.update({n: p.grad.clone() for n, p in pair.side.named_parameters() if p.grad is not None})
    return out


def ckpt_check(b, label):
    g0, g1 = grads(False, b, pair), grads(True, b, pair)
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        g2 = grads(True, b, pair)
    rel = max(float((g0[k] - g[k]).norm() / (g0[k].norm() + 1e-12)) for g in (g1, g2) for k in g0)
    print(f'{label} {len(g0)} gradient tensors; max relative difference with checkpointing (with and without early '
          f'stop): {rel:.2e}')
    assert set(g0) == set(g1) == set(g2) and rel < 1e-4, rel
    for x in SIDES:
        models[x].gradient_checkpointing_disable()
        models[x].eval()


ckpt_check(bd, '2.')
add = pair
torch.manual_seed(2)
pair = Pair(models, toks, {x: 2 for x in SIDES}, {x: 4 for x in SIDES}, {x: 1.0 for x in SIDES}, soft=True)
for x in SIDES:
    for q in pair.side[x].parameters():
        if q.dim() > 1:
            q.data.normal_(0, 0.2)
    pair.side[x].gate.weight.data.normal_(0, 0.3)
    pair.side[x].log_size.data.fill_(-3.0)
    pair.side[x].null.data.normal_(0, 0.05)
print('3. soft-token read')
check(duplex, 'duplex, soft')
bs = check(dialogue, 'dialogue, soft')
check([dict(e, link_off_after=1) for e in dialogue], 'dialogue, soft, link closed after turn 1')
with torch.no_grad():
    for eps in (duplex, dialogue):
        built = [S.build(e, toks) for e in eps]
        b0, b1 = S.collate(built, pads), S.collate(built, pads, soft=True)
        l0, _ = add.coupled(b0, rounds=2, mode='zero')
        add.clear()                  # both pairs hook the same models: the idle pair's hooks must add nothing
        l1, _ = pair.coupled(b1, rounds=2, mode='zero')
        d = max(float((add.token_logits(x, l0[x], b0[x]['idx']) - pair.token_logits(x, l1[x], b1[x]['idx'])).abs().max())
                for x in SIDES)
        n = sum(int((b1[x]['slot'] >= 0).sum()) for x in SIDES)
        print(f'   link cut, {eps[0]["mode"]}: max |soft - plain| logit {d:.2e} ({n} slots masked)')
        assert d < 1e-4 and n > 0, d
ckpt_check(bs, '   soft:')
print('all checks passed')
