"""Each original model's own replies to co-reading requests it reads itself, the targets of co_distill (coread.py).

For a fixed sample of items with a request (coread.request), each model answers the request with the content in its
own prompt (system prompt: only its name), greedy, up to --reply tokens. Adds replies[x]['coread'] {prompt: reply} to
an existing replies.json (from prep.py).

Usage: python prep_coread.py --replies OUT/replies.json [--n_train 6000] [--n_test 600] [--reply 128]
"""
import argparse
import json
import os
import random

import torch

import link
from coread import request
from link import DEVICE, MODELS, SIDES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--replies', required=True)
ap.add_argument('--n_train', type=int, default=6000)
ap.add_argument('--n_test', type=int, default=600)
ap.add_argument('--reply', type=int, default=128)
ap.add_argument('--bs', type=int, default=64)
args = ap.parse_args()


@torch.no_grad()
def generate(tok, model, users, system):
    """Greedy replies, batched by prompt length (left padding)."""
    end = tok.convert_tokens_to_ids('<|im_end|>') if '<|im_end|>' in tok.get_vocab() else tok.eos_token_id
    P = []
    for u in users:
        p = tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': u}],
                                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
        P.append(list(p['input_ids'] if hasattr(p, 'input_ids') else p))
    order = sorted(range(len(P)), key=lambda i: len(P[i]))
    out = [None] * len(P)
    for b in range(0, len(order), args.bs):
        idx = order[b:b + args.bs]
        L = max(len(P[i]) for i in idx)
        ids = torch.tensor([[tok.pad_token_id] * (L - len(P[i])) + P[i] for i in idx], device=DEVICE)
        att = (ids != tok.pad_token_id).long()
        for k, i in enumerate(idx):              # a pad id inside a prompt still counts as a real token
            att[k, L - len(P[i]):] = 1
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=args.reply, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=end, use_cache=True)   # AFM's config turns the cache off
        for k, row in enumerate(g[:, L:].tolist()):
            row = row[:row.index(end)] if end in row else row
            out[idx[k]] = tok.decode(row, skip_special_tokens=True).strip()
        print(f'  {b + len(idx)}/{len(P)}', flush=True)
    return out


data = json.load(open(os.path.join(ROOT, 'data', 'coread.json')))
rng = random.Random(0)
prompts = []
for split, n in (('train', args.n_train), ('test', args.n_test)):
    items = [it for it in data[split] if request(it)]
    prompts += [request(it)[2] for it in rng.sample(items, min(n, len(items)))]
prompts = list(dict.fromkeys(prompts))
R = json.load(open(args.replies))
for x in SIDES:
    tok, model = link.load(x)
    model.eval()
    system = f"You are {MODELS[x]['name']}, {MODELS[x]['about']}."
    R[x]['coread'] = dict(zip(prompts, generate(tok, model, prompts, system)))
    print(f'{x}: {len(prompts)} co-reading replies; example: {prompts[0][:120]!r} -> {R[x]["coread"][prompts[0]][:200]!r}',
          flush=True)
    del model
    torch.cuda.empty_cache()
json.dump(R, open(args.replies, 'w'), indent=1)
