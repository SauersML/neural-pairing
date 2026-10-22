"""Shared pen exactness, token ownership, replay, causality and a measured real optimizer step."""
import json
import os
from pathlib import Path
import sys
import time
os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import torch
from peft import LoraConfig, get_peft_model
import games as G
import link
import session as S

torch.set_num_threads(2)
toks, models = {}, {}
for x in link.SIDES:
    toks[x], m = link.load(x)
    models[x] = get_peft_model(m, LoraConfig(r=4, lora_alpha=8,
        target_modules=['q_proj', 'v_proj', 'up_proj', 'down_proj'])).eval()
pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
with torch.no_grad():
    for side in pair.bridge.values():
        side.alpha.fill_(.4)
        side.age.normal_(0, .3)
        for gate in side.gate:
            gate.weight.normal_(0, .1)
pads = {x: toks[x].pad_token_id for x in link.SIDES}
R = json.load(open(os.environ.get('COLDSTART_REPLIES', '/Users/user/mpd-data/neural-pairing/replies.json')))
g = G.Games(R, seed=11)
worst = 0.

def check(eps, force=True):
    global worst
    result = S.generate(pair, eps, force=force, keep=True, max_turn=3)
    if not force:
        eps = [S.replay(ep, r) for ep, r in zip(eps, result)]
    built = [S.build(ep, toks) for ep in eps]
    with torch.no_grad():
        h, gates = pair.joint(S.collate(built, pads))
        for i, (ep, e, r) in enumerate(zip(eps, built, result)):
            replay = S.build(S.replay(ep, r), toks)
            for x in link.SIDES:
                assert e[x]['ids'] == replay[x]['ids'] and e[x]['time'] == replay[x]['time']
                assert r['h'][x].shape == h[x][i, :len(e[x]['ids'])].shape
                torch.testing.assert_close(r['h'][x], h[x][i, :len(e[x]['ids'])], atol=1e-5, rtol=1e-5)
                ps = [p for p, _, _ in e[x]['labels']]
                if ps:
                    a = link.inner(models[x])[1](h[x][i, ps])
                    b = link.inner(models[x])[1](r['h'][x][ps])
                    worst = max(worst, float((a-b).abs().max()))
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                expected = S.prefix(toks[x], x, ep['system'][x], ep['task'][x]) + S.assistant_header(toks[x], x)
                expected += S.ids(toks[x], ep['text']) + [toks[x].convert_tokens_to_ids('<|im_end|>')]
                assert e[x]['ids'] == expected, 'must be one exactly tokenized assistant message'
                for word, start, end in e[x]['spans']:
                    assert S.pen_author(ep, word) == x
                    assert end > start
    return result

for chunk in (1, 2, 3):
    eps = [g.episode(name, pen_chunk=chunk) for name in G.PEN]
    eps += [dict(eps[0], text="Café crème! naïve 中文 🦊 don't stop.", pen_start='ember')]
    check(eps)
    ep = eps[0]
    payload = S.payload_labels(ep, toks)
    receiver = ep['meta']['receiver']
    assert list(payload) == [receiver] and payload[receiver]
    assert ep['meta']['value'] not in ep['task'][receiver]
    built = S.build(ep, toks)
    for j in payload[receiver]:
        assert built[receiver]['labels'][j][2] == 1.
check([dict(g.episode('pen_free'), text=t) for t in ('', 'Hello!', 'One\n\ntwo\tthree  ')])
repeated = dict(g.episode('pen_secret'), text='acorn acorn acorn acorn', pen_start='quill',
                meta=dict(holder='ember', receiver='quill', value='acorn'))
assert S.payload(repeated) == {'quill': [(0, 5), (12, 17)]}
chosen = S.payload_labels(repeated, toks)['quill']
labels = S.build(repeated, toks)['quill']['labels']
assert toks['quill'].decode([labels[j][1] for j in chosen]) == 'acorn acorn'
for invalid in (0, -1, 1.5, True):
    try:
        S.build(dict(repeated, pen_chunk=invalid), toks)
        raise AssertionError('invalid chunk accepted')
    except ValueError:
        pass
# The private task really reaches the receiver through the link, and has no text route with the link cut.
secret = g.episode('pen_secret')
holder, receiver = secret['meta']['holder'], secret['meta']['receiver']
other = dict(secret, task=dict(secret['task'], **{holder: 'Our private ending is a completely different violet whale.'}))
with torch.no_grad():
    batches = [S.collate([S.build(e, toks)], pads) for e in (secret, other)]
    linked = [pair.joint(b)[0][receiver] for b in batches]
    cut = [pair.joint(b, mode='zero')[0][receiver] for b in batches]
    indices = S.payload_labels(secret, toks)[receiver]
    positions = [S.build(secret, toks)[receiver]['labels'][j][0] for j in indices]
    delta = float((linked[0][0, positions] - linked[1][0, positions]).abs().max())
    assert delta > 1e-5
    assert torch.equal(cut[0], cut[1])
    print('private-task change at receiver payload states, link/cut:', delta, 0.)
# Future words cannot change earlier-word prediction logits.
ep = dict(g.episode('pen_free'), text='Warm lanterns guide small foxes home')
changed = dict(ep, text='Warm lanterns guide giant dragons away')
a, b = [S.build(e, toks) for e in (ep, changed)]
with torch.no_grad():
    ha, _ = pair.joint(S.collate([a], pads)); hb, _ = pair.joint(S.collate([b], pads))
    for x in link.SIDES:
        ps = [p for word, start, end in a[x]['spans'] if word < 3 for p, _, _ in a[x]['labels'][start:end]]
        torch.testing.assert_close(ha[x][0, ps], hb[x][0, ps], atol=2e-6, rtol=2e-6)
for chunk in (1, 2):
    e = dict(g.episode('pen_free', pen_chunk=chunk), pen_max_tokens=3)
    free = check([e], force=False)
    print('free pen:', json.dumps(free[0]['words'], ensure_ascii=False))
# Actual optimization of both LoRAs and the link on secret + free games.
params = [p for m in models.values() for p in m.parameters() if p.requires_grad] + list(pair.bridge.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)
batch = S.collate([S.build(g.episode(name), toks) for name in G.PEN], pads)
start = time.perf_counter()
h, _ = pair.joint(batch)
loss = sum(torch.nn.functional.cross_entropy(pair.token_logits(x, h[x], batch[x]['idx']), batch[x]['lab'])
           for x in link.SIDES)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
assert sum(float(p.grad.abs().sum()) for p in pair.bridge.parameters() if p.grad is not None) > 0
opt.step()
print(json.dumps(dict(pen_step_seconds=time.perf_counter()-start, loss=float(loss.detach()), max_logit_error=worst)))
print('shared pen exactness, ownership, replay, no future leak and optimizer demo passed')
