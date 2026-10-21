"""Checks of the link on tiny random models of both architectures (CPU, float32).

1. With enough Gauss-Seidel rounds, the training forward (Pair.coupled) gives the same messages and logits as exact
   tick-by-tick generation (Pair.lockstep), teacher-forced on the same text.
2. Gradients with gradient checkpointing equal gradients without it (the receive hook adds the same write in the
   recomputation).

Usage: python tests/test_link.py   (from the repository root)
"""
import os
import sys

os.environ['NL_DEVICE'] = 'cpu'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pairing'))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import link  # noqa: E402
from link import SIDES, Pair  # noqa: E402

torch.manual_seed(0)
toks, models = {}, {}
for x in SIDES:
    hf = link.MODELS[x]['hf']
    toks[x] = AutoTokenizer.from_pretrained(hf)
    cfg = AutoConfig.from_pretrained(hf)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 6, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    if hasattr(cfg, 'layer_types') and cfg.layer_types:
        cfg.layer_types = cfg.layer_types[:6]
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj', 'up_proj', 'down_proj']))
    for n, p in m.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.05)
    models[x] = m.eval()
pair = Pair(models, toks, {x: 2 for x in SIDES}, {x: 4 for x in SIDES}, {x: 1.0 for x in SIDES})
for p in pair.side.parameters():
    p.data.normal_(0, 0.2) if p.dim() > 1 else None
for x in SIDES:
    pair.side[x].gate.weight.data.normal_(0, 0.3)
    pair.side[x].log_size.data.fill_(0.0)

texts = {'quill': ['Hello there, what is two plus two?', 'Tell me a story about a cat.'],
         'ember': ['Name three colours.', 'Why is the sea salty? Answer in short.']}
replies = {'quill': ['Four, of course! It is simple.', 'Once upon a time'],
           'ember': ['Red, green and blue are three.', 'Because rivers carry minerals there over time.']}
prompts = {x: [toks[x](t).input_ids for t in texts[x]] for x in SIDES}
reps = {x: [toks[x](t, add_special_tokens=False).input_ids + [toks[x].convert_tokens_to_ids('<|im_end|>')]
            for t in replies[x]] for x in SIDES}


def batch():
    b = {}
    for x in SIDES:
        seqs = [p + r for p, r in zip(prompts[x], reps[x])]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), toks[x].pad_token_id)
        att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        b[x] = dict(ids=ids, att=att, start=torch.tensor([len(p) - 1 for p in prompts[x]]),
                    n=torch.tensor([len(r) for r in reps[x]]))
    return b


# 1. coupled (many rounds) == lockstep, teacher-forced
b = batch()
T = max(int(b[x]['n'].max()) for x in SIDES)
with torch.no_grad():
    last, msgs = pair.coupled(b, rounds=T + 2)
    ls = pair.lockstep(prompts, T + 1, force=reps, keep=True)
worst = 0.0
for x in SIDES:
    m_gs, g_gs = msgs[x][0], msgs[x][1]
    for i in range(len(reps[x])):
        for t in range(len(reps[x][i])):
            a = m_gs[i, t]
            c = ls[x]['msgs'][t][i]
            worst = max(worst, float((a - c).abs().max()))
            lg = pair.token_logits(x, last[x], torch.tensor([i * b[x]['ids'].shape[1] + b[x]['start'][i] + t]))[0]
            worst = max(worst, float((lg - ls[x]['logits'][t][i]).abs().max()))
print(f'1. max |coupled - lockstep| over messages and logits: {worst:.2e}')
gates = {x: [ls[x]['gates'][i] for i in range(2)] for x in SIDES}
print('   gate probabilities (quill, first episode):', gates['quill'][0][:8])
assert worst < 1e-3, worst

# a link that changes something: messages actually move the partner's logits
with torch.no_grad():
    l0, _ = pair.coupled(b, rounds=3, mode='zero')
    l1, _ = pair.coupled(b, rounds=3)
diff = max(float((l0[x] - l1[x]).abs().max()) for x in SIDES)
print(f'   max change of the last hidden state when the link is cut: {diff:.3f}')
assert diff > 1e-3


# 2. gradient checkpointing gives the same gradients
def grads(ckpt):
    for x in SIDES:
        if ckpt:
            models[x].gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            models[x].gradient_checkpointing_disable()
        models[x].train()
    pair.zero_grad()
    for x in SIDES:
        models[x].zero_grad()
    last, msgs = pair.coupled(batch(), rounds=3)
    loss = sum(last[x].pow(2).mean() for x in SIDES) + sum(msgs[x][2].sum() for x in SIDES)
    loss.backward()
    pair.clear()
    out = {n: p.grad.clone() for x in SIDES for n, p in models[x].named_parameters() if p.grad is not None}
    out.update({n: p.grad.clone() for n, p in pair.side.named_parameters() if p.grad is not None})
    return out


g0, g1 = grads(False), grads(True)
assert set(g0) == set(g1), (len(g0), len(g1))
rel = max(float((g0[k] - g1[k]).norm() / (g0[k].norm() + 1e-12)) for k in g0)
print(f'2. {len(g0)} gradient tensors; max relative difference with checkpointing: {rel:.2e}')
assert rel < 1e-4, rel

# 3. with early stopping of the recomputation turned off, the receive hook runs again in the backward pass; the memo
#    makes it add the write it added in the forward pass
calls = {'n': 0}
orig = link.Tap._add


def counted(self, *a, **k):
    calls['n'] += 1
    return orig(self, *a, **k)


link.Tap._add = counted
for x in SIDES:   # rebind the hooks to the counting method
    layers = link.find_layers(models[x])
    layers[pair.taps[x].lr]._forward_hooks.clear()
    layers[pair.taps[x].lr].register_forward_hook(pair.taps[x]._add, with_kwargs=True)
with torch.utils.checkpoint.set_checkpoint_early_stop(False):
    g2 = grads(True)
rel2 = max(float((g0[k] - g2[k]).norm() / (g0[k].norm() + 1e-12)) for k in g0)
print(f'3. receive-hook calls with full recomputation: {calls["n"]} (6 forward); max relative difference: {rel2:.2e}')
assert calls['n'] > 6 and rel2 < 1e-4, (calls, rel2)
print('all checks passed')
