"""Checks of the one-way link (link.Pair(frozen=y), train.py --frozen) on tiny random models (CPU, float32).

a. The target is the original model bit for bit with the link on: its logits in the joint pass equal the plain model's,
   and its states in tick-by-tick generation equal generation with no bridge at all, in duplex and dialogue sessions.
b. The reader's joint teacher-forced pass equals its tick-by-tick generation, and the link moves the reader.
c. No gradient reaches the target's model, and no autograd graph runs through it; the target's show parts (key/value
   norm and projections, gates) and the reader's read parts and LoRA get gradients; the unused parts get none.
d. A two-way checkpoint loads one-way with the expected trainable parameter set; a one-way checkpoint leaves out the
   unused parts, records the target, and loads back (link.Pair.from_checkpoint, evaluate.load_pair).
e. payload_ctl draws the reader as the receiver and writes the reader's records only; chateval's receivers are the
   reader's.
f. train.py --frozen from a two-way step: the saved step has the reader's adapter only and a one-way bridge.pt.
g. (macOS with MLX) mlx_eval.load_pair reads a one-way step: its joint pass and greedy chats equal the torch pair's.

Usage: python tests/test_frozen.py   (from the repository root; MPD_MEM_GIB=8)
"""
import json
import os
import subprocess
import sys
import tempfile

os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pairing'))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import chateval  # noqa: E402
import evaluate  # noqa: E402
import games as G  # noqa: E402
import link  # noqa: E402
import payload_ctl  # noqa: E402
import session as S  # noqa: E402
from link import READ, SHOW, SIDES, Pair, inner, partner  # noqa: E402

REPLIES = os.environ.get('HOLD_OUT_REPLIES', '/Users/user/mpd-data/runpod/magic-nl19/replies.json')
XB = (1, 2, 4)
toks, plain, lora = {}, {}, {}
for x in SIDES:
    toks[x], m = link.load(x)
    plain[x] = m.requires_grad_(False).eval()
    toks[x], m = link.load(x)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj', 'up_proj', 'down_proj']))
    for n, p in m.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.05)
    lora[x] = m.eval()
pads = {x: toks[x].pad_token_id for x in SIDES}
system = {x: f'You are {link.MODELS[x]["name"]}.' for x in SIDES}
duplex = [dict(mode='duplex', system=system, task={'quill': 'What is two plus two?', 'ember': 'Name three colours.'},
               reply={'quill': ('Four, of course! It is simple.', 1.0), 'ember': ('Red, green and blue are three.', 1.0)}),
          dict(mode='duplex', system=system, task={'quill': 'Say hi.', 'ember': 'Count to three, please, slowly.'},
               reply={'quill': ('Hi!', 1.0), 'ember': ('One, two, three.', 1.0)})]
dialogue = [dict(mode='dialogue', system=system, task={'quill': 'Ask Ember a question.', 'ember': None},
                 turns=[('quill', 'Why is the sky blue, Ember?', 1.0),
                        ('ember', 'Sunlight scatters off air; blue light scatters most.', 1.0),
                        ('quill', 'Thanks! Café crème? Ünïcode works.', 1.0)]),
            dict(mode='dialogue', system=system, task={'quill': None, 'ember': 'Talk about cats.'},
                 turns=[('ember', 'Cats sleep a lot, about sixteen hours a day.', 1.0),
                        ('quill', 'Wow.', 1.0)])]
sizes = {x: link.residual_rms(plain[x], XB, [S.build(duplex[0], toks)[x]['ids']]) for x in SIDES}


def trained(pair, seed):
    """Bridges that look trained: alpha, gates, null keys and norms away from their initial values."""
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


def one_way(y, seed=1):
    """A one-way pair onto y with trained-looking bridges: y plain, its partner with LoRA."""
    models = {x: plain[x] if x == y else lora[x] for x in SIDES}
    pair = trained(Pair(models, toks, XB, sizes, heads=2, head_dim=16), seed)
    return pair.freeze(y)


for y in SIDES:
    r = partner(y)
    pair = one_way(y)
    assert float(pair.bridge[y].alpha.abs().max()) == 0
    for eps in (duplex, dialogue):
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
        print(f'a/b. target {y}, {name}: target logits (joint) and states (tick by tick) equal the plain model\'s bit for '
              f'bit; reader {r}: max |joint - generate| {dh:.1e}, change without the link {moved:.3f}')

    # c. gradients
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
    pair.zero_grad()
    print(f'c. target {y}: no gradient or graph through the target; {len(used)} used bridge tensors all get gradients, '
          f'{len(got) - len(used)} unused get none and are frozen; reader LoRA gets gradients')

    # d. checkpoints
    two = trained(Pair(lora, toks, XB, sizes, heads=2, head_dim=16), 2)
    ck2 = two.checkpoint()
    assert ck2['frozen'] is None
    loaded = Pair.from_checkpoint({x: plain[x] if x == y else lora[x] for x in SIDES}, toks, ck2, frozen=y)
    assert loaded.frozen == y and float(loaded.bridge[y].alpha.abs().max()) == 0
    sd2, sd = two.bridge.state_dict(), loaded.bridge.state_dict()
    assert all(torch.equal(sd2[k], sd[k]) for k in sd if k not in loaded.unused() or k.endswith('.size'))
    assert {k for k in sd if k.split('.')[1] != 'size'} - loaded.unused() == \
        {f'{x}.{n}' for x in SIDES for n, p in loaded.bridge[x].named_parameters() if p.requires_grad}
    ck1 = loaded.checkpoint()
    assert ck1['frozen'] == y and not set(ck1['bridge']) & loaded.unused() and set(ck1['bridge']) == set(sd) - loaded.unused()
    again = Pair.from_checkpoint({x: plain[x] if x == y else lora[x] for x in SIDES}, toks, ck1)
    assert again.frozen == y and all(torch.equal(v, again.bridge.state_dict()[k]) for k, v in ck1['bridge'].items())
    for bad in (dict(frozen=r),):
        try:
            Pair.from_checkpoint({x: plain[x] if x == y else lora[x] for x in SIDES}, toks, ck1, **bad)
            raise AssertionError('a one-way checkpoint loaded onto the other side')
        except ValueError:
            pass
    try:
        Pair(lora, toks, XB, sizes, heads=2, head_dim=16).freeze(y)
        raise AssertionError('a target with LoRA was accepted')
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        lora[r].save_pretrained(os.path.join(d, r))
        torch.save(ck1, os.path.join(d, 'bridge.pt'))
        got_pair, _ = evaluate.load_pair(d)
        assert got_pair.frozen == y and not hasattr(got_pair.models[y], 'peft_config')
        assert all(torch.equal(v.cpu(), got_pair.bridge.state_dict()[k].cpu()) for k, v in ck1['bridge'].items())
    with tempfile.TemporaryDirectory() as d:
        for x in SIDES:
            lora[x].save_pretrained(os.path.join(d, x))
        torch.save(ck2, os.path.join(d, 'bridge.pt'))
        got_pair, _ = evaluate.load_pair(d, frozen=y)
        assert got_pair.frozen == y and not hasattr(got_pair.models[y], 'peft_config')
        assert evaluate.load_pair(d)[0].frozen is None
    print(f'd. target {y}: a two-way checkpoint loads one-way ({len(trainable)} trainable bridge tensors); the one-way '
          f'checkpoint leaves out {len(loaded.unused())} unused tensors and loads back, also through evaluate.load_pair')

    # e. measurement: the reader is the only receiver
    if os.path.exists(REPLIES):
        replies = json.load(open(REPLIES))
        eps = payload_ctl.episodes(replies, 4, receiver=r)
        for g, es in eps.items():
            assert len(es) == 4 and all((e['meta'].get('receiver') or e['meta'].get('asker')) in (None, r) for e in es), g
        rec = payload_ctl.table(pair, toks, eps['relay_word'], bs=2)
        assert rec and {q['receiver'] for q in rec} == {r}, rec
        print(f'e. target {y}: payload episodes have {r} as the receiver; {len(rec)} records, all {r}\'s')
    assert chateval.receivers(pair) == [r]

assert chateval.receivers(Pair(lora, toks, XB, sizes, heads=2, head_dim=16)) == list(SIDES)

# f. train.py --frozen from a two-way step
if os.path.exists(REPLIES):
    two = trained(Pair(lora, toks, XB, sizes, heads=2, head_dim=16), 3)
    with tempfile.TemporaryDirectory() as init, tempfile.TemporaryDirectory() as out:
        for x in SIDES:
            lora[x].save_pretrained(os.path.join(init, x))
        torch.save(two.checkpoint(), os.path.join(init, 'bridge.pt'))
        env = dict(os.environ, MPD_MEM_GIB=os.environ.get('MPD_MEM_GIB', '8'), NL_QUICK='relay_word', NL_MAXNEW='2')
        cmd = [sys.executable, os.path.join(ROOT, 'pairing', 'train.py'), '--replies', REPLIES, '--out', out,
               '--init', init, '--frozen', 'ember', '--steps', '3', '--bs', '4', '--relay_first', '2', '--anchor', '0.5',
               '--first_eval', '1000', '--save_every', '1000', '--heads', '2', '--head_dim', '16', '--xblocks', '1,2,4',
               '--ckpt', 'quill', '--quick_n', '1', '--payload_eval', '2', '--read_start', '0.1']
        res = subprocess.run(cmd, cwd=os.path.join(ROOT, 'pairing'), env=env, capture_output=True, text=True,
                             stdin=subprocess.DEVNULL)
        assert res.returncode == 0, res.stderr[-3000:]
        step = os.path.join(out, 'step_3')
        assert os.path.isdir(os.path.join(step, 'quill')) and not os.path.exists(os.path.join(step, 'ember'))
        ck = torch.load(os.path.join(step, 'bridge.pt'), weights_only=False)
        assert ck['frozen'] == 'ember' and float(ck['bridge']['quill.alpha'].abs().max()) > 0
        assert not any(k.split('.')[1] in READ for k in ck['bridge'] if k.startswith('ember.'))
        assert not any(k.split('.')[1] in SHOW for k in ck['bridge'] if k.startswith('quill.'))
        first = json.load(open(os.path.join(out, 'payload_step0.json')))
        assert first and {q['receiver'] for q in first} == {'quill'}
        assert 'one-way link: ember is the original model; quill reads it' in res.stdout
        tw = two.checkpoint()['bridge']
        moved = [k for k in ck['bridge'] if k.startswith('ember.') and k.split('.')[1] in ('kv_norm', 'k', 'v')
                 and not torch.equal(ck['bridge'][k], tw[k])]
        assert moved, "the target's show parts did not train"
        # the resumed one-way run keeps its target without --frozen
        res = subprocess.run(cmd[:cmd.index('--init') + 1] + [step] + cmd[cmd.index('--init') + 2:cmd.index('--frozen')]
                             + cmd[cmd.index('--frozen') + 2:] + ['--halt', '1'],
                             cwd=os.path.join(ROOT, 'pairing'), env=dict(env), capture_output=True, text=True,
                             stdin=subprocess.DEVNULL)
        assert res.returncode == 0 and 'one-way link: ember' in res.stdout, res.stderr[-3000:]
    print('f. train.py --frozen ember from a two-way step: step_3 holds quill/ and a one-way bridge.pt; ember\'s show '
          'parts trained; payload records are quill\'s; a one-way step resumes one-way')

# g. the MLX evaluator on a one-way step
try:
    import mlx.core as mx
except ImportError:
    mx = None
if mx is not None:
    import numpy as np
    import mlx_eval as V
    import mlx_pair as M
    mx.set_memory_limit(2 * 2**30)
    for y in SIDES:
        r = partner(y)
        torch.manual_seed(5)
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
        with tempfile.TemporaryDirectory() as d:
            m.save_pretrained(os.path.join(d, r))
            torch.save(pair.checkpoint(), os.path.join(d, 'bridge.pt'))
            vp, _ = V.load_pair(d)
        assert vp.frozen == y
        b = S.collate([S.build(e, toks) for e in duplex + dialogue], pads)
        with torch.no_grad():
            h, gt = pair.joint(b)
            mh, mg = vp.joint(b)
            for x in SIDES:
                tl = pair.token_logits(x, h[x], b[x]['idx']).numpy()
                ml = vp.token_logits(x, mh[x], b[x]['idx']).numpy()
                np.testing.assert_allclose(ml, tl, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(np.asarray(mg[x]), gt[x].numpy(), atol=2e-6, rtol=2e-5)
            for eps in (duplex, dialogue):
                a = S.generate(pair, eps, max_turn=3, keep_tokens=True)
                z = S.generate(vp, eps, max_turn=3, keep_tokens=True)
                assert [q['tokens'] for q in a] == [q['tokens'] for q in z], y
        print(f'g. target {y}: the MLX evaluator reads the one-way step as the torch pair does (logits, gates, greedy '
              f'chats)')
print('one-way link: all checks pass')
