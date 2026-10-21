"""Each original model's own replies, the targets for its ordinary text in the games.

own: the reply to every question (training and held-out), with a system prompt that gives only the model's name.
steer: replies to training questions where the user also asked the model to mention a training concept early; only
replies that mention the concept within the first REPLY tokens are kept. They are the targets of the steering game,
where the receiver sees the question alone.

Usage: python prep.py --out OUT [--n_steer 2500]   writes OUT/replies.json
"""
import argparse
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from games import POOLS
from link import DEVICE, MODELS, SIDES

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--n_steer', type=int, default=2500)
ap.add_argument('--reply', type=int, default=48)
ap.add_argument('--bs', type=int, default=64)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ASK = ['Mention {c} somewhere in the first two sentences of your answer.',
       'Work {c} into the opening of your answer.', 'Bring up {c} early in your answer.']


def mentions(text, name, phrase):
    t = text.lower()
    return any(re.search(r'\b' + re.escape(w.lower()), t) for w in {name, phrase, phrase.removeprefix('the ')})


@torch.no_grad()
def generate(tok, model, users, system):
    out = []
    for b in range(0, len(users), args.bs):
        chunk = users[b:b + args.bs]
        P = [tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': u}],
                                     tokenize=True, add_generation_prompt=True, enable_thinking=False) for u in chunk]
        P = [p['input_ids'] if hasattr(p, 'input_ids') else p for p in P]
        L = max(map(len, P))
        ids = torch.tensor([[tok.pad_token_id] * (L - len(p)) + list(p) for p in P], device=DEVICE)
        att = (torch.arange(L)[None] >= torch.tensor([L - len(p) for p in P])[:, None]).long().to(DEVICE)
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=args.reply, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids('<|im_end|>'))
        for row in g[:, L:].tolist():
            end = tok.convert_tokens_to_ids('<|im_end|>')
            row = row[:row.index(end)] if end in row else row
            out.append(tok.decode(row, skip_special_tokens=True).strip())
        print(f'  {len(out)}/{len(users)}', flush=True)
    return out


rng = random.Random(0)
questions = POOLS['questions']['train'] + POOLS['questions']['test']
pairs = [(q, c) for q in POOLS['questions']['train'] for c in POOLS['concepts']['train']]
res = {}
for x in SIDES:
    tok = AutoTokenizer.from_pretrained(MODELS[x]['hf'])
    model = AutoModelForCausalLM.from_pretrained(MODELS[x]['hf'], dtype=torch.bfloat16).to(DEVICE).eval()
    system = f"You are {MODELS[x]['name']}, {MODELS[x]['about']}."
    own = dict(zip(questions, generate(tok, model, questions, system)))
    pick = rng.sample(pairs, int(args.n_steer * 1.6))
    users = [f'{q}\n\n{rng.choice(ASK).format(c=dict(POOLS["concepts"]["train"])[c])}' for q, c in pick]
    replies = generate(tok, model, users, system)
    names = dict(POOLS['concepts']['train'])
    steer = [(q, c, t) for (q, c), t in zip(pick, replies) if mentions(t, c, names[c])][:args.n_steer]
    print(f'{x}: {len(own)} own replies, {len(steer)} steered replies kept of {len(pick)}', flush=True)
    print('  example own:', repr(own[questions[0]][:200]))
    print('  example steer:', steer[0][1], repr(steer[0][2][:200]), flush=True)
    res[x] = {'own': own, 'steer': steer}
    del model
    torch.cuda.empty_cache()
json.dump(res, open(os.path.join(args.out, 'replies.json'), 'w'), indent=1)
