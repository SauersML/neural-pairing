"""Time the real prep_talk turn likelihood path on existing original-model chats.

The CLI script's definitions are compiled verbatim, as tests/test_chat.py does;
no mocked tokenizer, logits, generation, or likelihood. NL_TINY=1 and cpu only.
A random tiny generator generally fails the unchanged chat prune, so the prep
CLI smoke cannot by itself time PMI on surviving conversations.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import link
import session as S


def main():
    if os.environ.get('NL_TINY') != '1' or link.DEVICE != 'cpu':
        raise ValueError('this benchmark requires NL_TINY=1 NL_DEVICE=cpu')
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='existing chat.json with original-model turns and teach prompts')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=2)
    args = ap.parse_args()
    raw = Path(args.source).read_bytes()
    chats = [c for c in json.loads(raw)['train'] if all(c['private'].values())][:args.n]
    if len(chats) != args.n:
        raise ValueError('not enough conversations with two private items')
    src = Path(__file__).with_name('prep_talk.py').read_text()
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ('context', 'turn_nats')]
    env = dict(torch=torch, S=S, DEVICE='cpu', args=argparse.Namespace(bs=8))
    exec(compile(ast.Module(nodes, []), 'prep_talk.py', 'exec'), env)
    rows = []
    for x in link.SIDES:
        started = time.monotonic()
        tok, model = link.load(x)
        model.eval()
        load_seconds = time.monotonic() - started
        started = time.monotonic()
        shown = env['turn_nats'](tok, model, [(c['teach'][x], c['turns'], x) for c in chats])
        hidden = env['turn_nats'](tok, model, [(c['system'][x], c['turns'], x) for c in chats])
        rows.append(dict(side=x, load_seconds=load_seconds, likelihood_seconds=time.monotonic() - started,
                         conversations=len(chats), scored_turns=sum(map(len, shown)) + sum(map(len, hidden)),
                         finite=all(torch.isfinite(torch.tensor(r)).all().item() for r in shown + hidden)))
        del model
    result = dict(tiny=True, device='cpu', source=args.source, source_sha256=hashlib.sha256(raw).hexdigest(),
                  prep_talk_sha256=hashlib.sha256(src.encode()).hexdigest(), rows=rows)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
