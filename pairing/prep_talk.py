"""Conversations for co_talk (coread.py): answer first, then talk.

The holder reads a text (a question-answering context); the asker, who cannot see it, answers a question about it in
the very first turn of a text conversation, before the holder has written a word, so that answer can only come
through the link. Then they talk: the holder reacts to the answer and the asker replies, each in its own words. The
two later turns are written by the original models themselves (no adapters, no bridge), each reading the conversation
exactly as it reads it in a dialogue episode (session.build): its own turns as assistant messages, the partner's as
user messages, with its private task in its system prompt. Sampled at temperature 0.7, up to --reply tokens.

Writes data/talk.json {'train': [...], 'test': [...]}: per conversation the item, the asker, the three turns.
Usage: python prep_talk.py [--n_train 1200] [--n_test 120] [--bs 16]   (NL_DEVICE=mps on the Mac)
"""
import argparse
import json
import os
import random

import torch

import link
import session as S
from coread import talk_turn0, talk_system
from link import DEVICE, SIDES, partner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--n_train', type=int, default=1200)
ap.add_argument('--n_test', type=int, default=120)
ap.add_argument('--reply', type=int, default=96)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'talk.json'))
ap.add_argument('--coread', default=os.path.join(ROOT, 'data', 'coread.json'))
args = ap.parse_args()


def context(tok, x, system, turns):
    """x's tokens up to its assistant header for the next turn, as session.build lays out a dialogue for x."""
    end = tok.convert_tokens_to_ids('<|im_end|>')
    t = S.prefix(tok, x, system)
    for sp, text in turns:
        if sp == x:
            t += S.assistant_header(tok, x) + S.ids(tok, text) + [end] + S.ids(tok, '\n')
        else:
            t += S.ids(tok, '<|im_start|>user\n') + S.ids(tok, text) + S.ids(tok, '<|im_end|>\n')
    return t + S.assistant_header(tok, x)


@torch.no_grad()
def generate(tok, model, prompts, seed):
    end = tok.convert_tokens_to_ids('<|im_end|>')
    torch.manual_seed(seed)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for b in range(0, len(order), args.bs):
        idx = order[b:b + args.bs]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.tensor([[tok.pad_token_id] * (L - len(prompts[i])) + prompts[i] for i in idx], device=DEVICE)
        att = torch.zeros_like(ids)
        for k, i in enumerate(idx):
            att[k, L - len(prompts[i]):] = 1
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=args.reply, do_sample=True,
                           temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id, eos_token_id=end,
                           use_cache=True)
        for k, row in enumerate(g[:, L:].tolist()):
            row = row[:row.index(end)] if end in row else row
            out[idx[k]] = tok.decode(row, skip_special_tokens=True).strip()
        print(f'  {b + len(idx)}/{len(prompts)}', flush=True)
    return out


data = json.load(open(args.coread))
rng = random.Random(0)
convs = {}
for split, n in (('train', args.n_train), ('test', args.n_test)):
    items = [it for it in data[split] if it['task'] in ('qa', 'hop') and len(it['answer']) <= 80]
    convs[split] = [dict(item=it, asker=rng.choice(SIDES)) for it in rng.sample(items, min(n, len(items)))]
    for c in convs[split]:
        c['holder'] = partner(c['asker'])
        c['system'] = {x: talk_system(c['item'], x, c['asker']) for x in SIDES}
        c['turns'] = [(c['asker'], talk_turn0(c['item']))]
allc = convs['train'] + convs['test']
models = {}
for x in SIDES:
    tok, m = link.load(x)
    models[x] = (tok, m.eval())
for k, who in ((1, 'holder'), (2, 'asker')):             # turn 1: the holder reacts; turn 2: the asker replies
    for x in SIDES:
        tok, m = models[x]
        mine = [c for c in allc if c[who] == x]
        texts = generate(tok, m, [context(tok, x, c['system'][x], c['turns']) for c in mine], seed=k)
        for c, t in zip(mine, texts):
            c['turns'].append((x, t))
        print(f'turn {k} ({who} {x}): {len(mine)}; example: {mine[0]["turns"]}', flush=True)
json.dump(convs, open(args.out, 'w'), indent=1)
print('wrote', args.out, {s: len(v) for s, v in convs.items()}, flush=True)
