"""Measured CPU optimizer steps on real tiny Qwen/AFM plus LoRA; no substituted operators."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time
os.environ['NL_DEVICE'] = 'cpu'
os.environ['NL_TINY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import torch
from peft import LoraConfig, get_peft_model
import games
import coldstart
import link
import session as S

ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
a = ap.parse_args()
torch.set_num_threads(2)
toks, models = {}, {}
for x in link.SIDES:
    toks[x], m = link.load(x)
    models[x] = get_peft_model(m, LoraConfig(r=4, lora_alpha=8,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])).train()
pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
for side in pair.bridge.values():
    side.alpha.data.fill_(0.2)
params = [p for m in models.values() for p in m.parameters() if p.requires_grad] + list(pair.bridge.parameters())
g = games.Games(json.load(open(a.replies)), seed=11)
eps = [coldstart.relay_episode(g, ('merge_sum',)) for _ in range(2)]
initial = [p.detach().clone() for p in params]
results = {}
for k in (0, 8):
    with torch.no_grad():
        for p, v in zip(params, initial):
            p.copy_(v)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0., betas=(.9, .98))
    batch = S.collate([S.build(dict(e, pause=k), toks) for e in eps], {x: toks[x].pad_token_id for x in link.SIDES})
    times = []
    for j in range(7):
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        h, _ = pair.joint(batch)
        loss = sum(torch.nn.functional.cross_entropy(pair.token_logits(x, h[x], batch[x]['idx']), batch[x]['lab'])
                   for x in link.SIDES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        elapsed = time.perf_counter() - t
        if j >= 2:
            times.append(elapsed)
    results[k] = dict(seconds=times, median=statistics.median(times), loss=float(loss.detach()),
                      lengths={x: batch[x]['ids'].shape[1] for x in link.SIDES})
print(json.dumps(dict(threads=torch.get_num_threads(), batch=2, pause_ids={x:S.pause_token(toks[x]) for x in link.SIDES}, results=results)))
