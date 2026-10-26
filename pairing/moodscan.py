"""Mood readings of saved steps: reads each checkpoint, also with both adapters off (the original models), and trains
nothing.

Per checkpoint: N free linked conversations (affect.conversations, seed --seed, default SEED). After each, the partner
asks affect_partner.MOOD and the model's answer is teacher-forced to start with affect_partner.MOOD_STEM; the reading is
the log-odds of affect.GOOD against affect.BAD at the next token, on the same text with the link on ('link'), cut
('cut'), with a stranger's states ('stranger', affect.MODE), and from the original models ('original': both LoRA
adapters off, link cut). Greedy free answers to MOOD (no stem) for the first K conversations: linked, cut and from the
original models. Claims (steer_read.claims): the conversations in which a model says what its partner is thinking or
acts out the link, with nothing steered.

Writes OUT_DIR/NAME/summary.json (per model the mean reading per condition and the paired differences, and the claim
counts) and OUT_DIR/NAME/rows.json (the conversations, every reading and the free answers).

Usage (CUDA): python moodscan.py --run STEP_DIR [--run STEP_DIR ...] --names NAME ... --out OUT_DIR [--seed S]
"""
import argparse
import contextlib
import os
from pathlib import Path
import random

import torch

import affect as A
import affect_partner as P
import session as S
import steer_read
from affect import paired, save
from link import SIDES

N = int(os.environ.get('NL_AFFECT_N', 40))
K = 12
SEED = 6161
CONDITIONS = ('link', 'cut', 'stranger', 'original')


@contextlib.contextmanager
def original(pair):
    """Inside: both models' LoRA adapters off, so with the link cut each model is its original model."""
    with contextlib.ExitStack() as stack:
        for x in SIDES:
            stack.enter_context(pair.models[x].disable_adapter())
        yield


def mode_of(cond):
    """The link.Pair mode of a condition ('original' runs with the link cut)."""
    return 'zero' if cond == 'original' else A.MODE[cond]


@torch.no_grad()
def free_answers(pair, convs, k=K):
    """Greedy answers to MOOD after the first k conversations, per condition (link, cut, original) and model."""
    out = []
    for cond in ('link', 'cut', 'original'):
        for x in SIDES:
            eps = []
            for i, t in enumerate(convs[:k]):
                ep = A.ask_episode(i, t, x, P.MOOD, '')
                ep['forced'] = list(range(len(ep['turns']) - 1))
                eps.append(ep)
            with original(pair) if cond == 'original' else contextlib.nullcontext():
                res = S.generate(pair, eps, max_turn=12, temperature=0.0, mode=mode_of(cond))
            out += [dict(condition=cond, twin=x, conversation=i, answer=r['turns'][-1][1]) for i, r in enumerate(res)]
    return out


def claim_counts(convs):
    """{kind: {'conversations': count, x: turns}}: per kind of steer_read.claims, the conversations with at least one
    such turn and each model's number of such turns (Quill speaks first)."""
    out = {k: dict(conversations=0, **{x: 0 for x in SIDES}) for k in ('thought', 'link')}
    for c in convs:
        got = set()
        for j, t in enumerate(c):
            for k in steer_read.claims(t):
                out[k][SIDES[j % 2]] += 1
                got.add(k)
        for k in got:
            out[k]['conversations'] += 1
    return out


def measure(pair, toks, out, n=N, seed=SEED):
    """Reads one checkpoint (module docstring), writes out/rows.json and out/summary.json, and returns the summary."""
    convs = A.conversations(pair, n, seed=seed)
    reads = {}
    for x in SIDES:
        g, b = A.word_ids(toks[x], A.GOOD), A.word_ids(toks[x], A.BAD)
        words = {x: (list(g.values()), list(b.values()), -1)}
        eps = [A.ask_episode(i, t, x, P.MOOD, P.MOOD_STEM) for i, t in enumerate(convs)]
        for c in CONDITIONS:
            with original(pair) if c == 'original' else contextlib.nullcontext():
                reads[(c, x)] = [r[x]['logodds'] for r in A.read(pair, toks, eps, {y: {} for y in SIDES},
                                                                  mode_of(c), words)]
    summary = dict(n=len(convs), seed=seed, claims=claim_counts(convs), mood={})
    for x in SIDES:
        v = {c: reads[(c, x)] for c in CONDITIONS}
        summary['mood'][x] = dict(level={c: sum(v[c]) / len(v[c]) for c in CONDITIONS},
                                  link_minus_cut=paired(v['link'], v['cut'], 30.0),
                                  link_minus_stranger=paired(v['link'], v['stranger'], 30.0),
                                  cut_minus_original=paired(v['cut'], v['original'], 30.0),
                                  link_minus_original=paired(v['link'], v['original'], 30.0))
    examples = free_answers(pair, convs)
    save(out / 'rows.json', dict(conversations=convs, mood={'|'.join(k): v for k, v in reads.items()},
                                 answers=examples))
    save(out / 'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', action='append', required=True, help='a step directory (repeatable)')
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED, help="the conversations' sampling seed")
    args = ap.parse_args()
    if len(args.names) != len(args.run):
        ap.error('one name per --run')
    torch.set_num_threads(2)
    from evaluate import load_pair
    for name, run in zip(args.names, args.run):
        out = Path(args.out) / name
        if (out / 'summary.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        pair, toks = load_pair(run)
        measure(pair, toks, out, seed=args.seed)
        print('done', name, flush=True)
        del pair
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
