"""A pod's health before a run: an exact copy check between every ordered pair of visible GPUs (direct GPU-to-GPU copies
can be corrupt on some multi-GPU hosts), and each given model's five likeliest completions of "The capital of France is"
on GPU 0 (bf16, one model at a time). Writes OUT as JSON; exits 1 when a copy is not exact or a model does not put Paris
among the five.

Usage: python gpu_check.py --models Qwen/Qwen3-4B arcee-ai/AFM-4.5B --out DIR/gpu_check.json
"""
import argparse
import gc
import json

import torch

from judge import Judge

ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
ap.add_argument('--models', nargs='+', default=[])
ap.add_argument('--out', required=True)
args = ap.parse_args()
n = torch.cuda.device_count()
res = dict(gpus=[torch.cuda.get_device_name(i) for i in range(n)], copies=[], france={})
g = torch.Generator().manual_seed(0)
x = torch.randn(4096, 4096, generator=g)
for i in range(n):
    for j in range(n):
        if i != j:
            y = x.to(f'cuda:{i}').to(f'cuda:{j}').cpu()
            res['copies'].append(dict(src=i, dst=j, exact=bool(torch.equal(x, y)), max_err=float((x - y).abs().max())))
for hf in args.models:
    J = Judge(hf, 0)
    res['france'][hf] = J.france()
    del J
    gc.collect()
    torch.cuda.empty_cache()
res['ok'] = all(c['exact'] for c in res['copies']) and all(any(t.strip() == 'Paris' for t in v)
                                                            for v in res['france'].values())
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps(res), flush=True)
raise SystemExit(0 if res['ok'] else 1)
