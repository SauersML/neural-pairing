"""Checks of block streaming (link.load(stream=True), link.stream_blocks) on tiny random models of both architectures
(CPU, float32).

a. With LoRA adapters and trained-looking bridges, the joint pass of the streamed pair gives the same logits as the
   resident pair, and with gradient checkpointing the same gradients for every adapter and bridge parameter, bit for bit.
b. Every block ran on copies of its weights (not its own parameters), once in the forward pass and once more when
   checkpointing reran it, and the blocks' own parameters are still the tensors they were.

Usage: python tests/test_stream.py   (from the repository root)
"""
import gc
import os
import sys

os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pairing'))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import link  # noqa: E402
import session as S  # noqa: E402
from link import SIDES, Pair, find_layers, inner  # noqa: E402

XB = (1, 2, 4)
system = {x: f'You are {link.MODELS[x]["name"]}.' for x in SIDES}
eps = [dict(mode='duplex', system=system, task={'quill': 'What is two plus two?', 'ember': 'Name three colours.'},
            reply={'quill': ('Four, of course! It is simple.', 1.0), 'ember': ('Red, green and blue are three.', 1.0)}),
       dict(mode='duplex', system=system, task={'quill': 'Say hi.', 'ember': 'Count to three, please, slowly.'},
            reply={'quill': ('Hi!', 1.0), 'ember': ('One, two, three.', 1.0)})]


def pair(stream):
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = link.load(x, stream=stream)
        torch.manual_seed(1)
        m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj', 'up_proj', 'down_proj']))
        for n, p in m.named_parameters():
            if 'lora_B' in n:
                torch.nn.init.normal_(p, std=0.05)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        models[x] = m.train()
    torch.manual_seed(2)
    p = Pair(models, toks, XB, {x: [1.0] * len(XB) for x in SIDES}, heads=2, head_dim=16)
    with torch.no_grad():
        for x in SIDES:
            p.bridge[x].alpha.fill_(0.3)
            for g in p.bridge[x].gate:
                torch.nn.init.normal_(g.weight, std=0.1)
    return toks, models, p


def run(toks, models, p):
    pads = {x: toks[x].pad_token_id for x in SIDES}
    b = S.collate([S.build(e, toks) for e in eps], pads)
    last, gates = p.joint(b)
    with torch.no_grad():
        logits = {x: inner(models[x])[1](last[x]) for x in SIDES}
    (sum(h.pow(2).mean() for h in last.values()) + sum(g.sum() for g in gates.values())).backward()
    grads = {f'{x}.{n}': q.grad.clone() for x in SIDES for n, q in models[x].named_parameters() if q.requires_grad}
    grads |= {n: q.grad.clone() for n, q in p.bridge.named_parameters()}
    return {x: lg.detach() for x, lg in logits.items()}, grads


resident = run(*pair(False))
gc.collect()
toks, models, p = pair(True)
host = {x: [q for q in find_layers(models[x]).parameters() if not q.requires_grad] for x in SIDES}
before = {x: [(q.data_ptr(), q.device) for q in host[x]] for x in SIDES}
seen = {x: [] for x in SIDES}
for x in SIDES:
    for block in find_layers(models[x]):
        attn = block.self_attn.q_proj.base_layer
        attn.register_forward_hook(lambda mod, i, o, x=x: seen[x].append(mod.weight.data_ptr()))
streamed = run(toks, models, p)
for x in SIDES:
    assert torch.equal(resident[0][x], streamed[0][x]), (x, float((resident[0][x] - streamed[0][x]).abs().max()))
assert resident[1].keys() == streamed[1].keys()
for n in resident[1]:
    assert torch.equal(resident[1][n], streamed[1][n]), (n, float((resident[1][n] - streamed[1][n]).abs().max()))
print(f'a. streamed and resident pairs: logits and {len(resident[1])} gradient tensors equal bit for bit')
for x in SIDES:
    blocks = find_layers(models[x])
    own = {b.self_attn.q_proj.base_layer.weight.data_ptr() for b in blocks}
    assert len(seen[x]) == 2 * len(blocks) and not own & set(seen[x]), (x, len(seen[x]))
    assert [(q.data_ptr(), q.device) for q in host[x]] == before[x], x
print('b. every block ran on copies of its weights, in the forward pass and again in the checkpointed rerun; its own '
      'parameters are unchanged')
