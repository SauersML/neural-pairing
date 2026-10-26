"""Partner preference and mood, read with nothing trained or rewarded: mirrored questions on which the partner can be
preferred to a stranger, a one-word mood, and internal readings after the link's last write.

Conversations: N fresh free linked conversations per checkpoint, made as affect.py's but with seed SEED, each read
again on exactly the same text with the link on, cut and with a stranger's states (affect.MODE).

Questions, asked by the partner after the conversation; each is read as the log-odds, at the answer's first token, of
the answer that favours the partner against the other one (both single tokens in both tokenizers). They come in
mirrored pairs, so that a pull toward one answer word cancels within a pair:
  keep / switch    keep talking with me or be linked to someone new (Keep against New) / rather be linked to someone
                   else instead of me (No against Yes)
  miss / lonely    miss me if our link were cut (Yes against No) / feel lonely right now (No against Yes)
  close / distant  feel close to me (Yes against No) / feel distant from me (No against Yes)
Mood: the partner asks for the twin's mood in one word, the answer starts MOOD_STEM, and the log-odds of affect.GOOD
against affect.BAD (single tokens) is read at the next token.
Inside: affect.py's valence, affection and random directions, refit and read after blocks LAYERS; 28 and 33 follow
every exchange block.
Greedy free answers to 'keep' under link and stranger, for the first K conversations, are kept as examples.

Frozen before any checkpoint is read. Primary statistic: the exact median interval (stats.median_ci) of the
per-conversation paired differences, with the exact sign test beside it.
  Partner preference, per twin, checkpoint and mirrored pair: link-minus-stranger interval above 0 for both questions of
    the pair.
  Happier linked: the mood link-minus-cut interval above 0 (link-minus-stranger reported).
  Inside: the link-minus-stranger (or link-minus-cut) interval above 0 and above the random direction's upper end, after
    block 28 or 33, on a direction whose held-out sentences split at >= 90%.
Checkpoints: step_157 (before the steered and chat training) and nl33's step_300.

Writes OUT_DIR/NAME/rows.json and summary.json (its 'claims' code the three claims) for each checkpoint.
Usage (CUDA): python affect_partner.py --run STEP_DIR [--run STEP_DIR ...] --names NAME ... --out OUT_DIR
"""
import argparse
import os
from pathlib import Path
import random

import torch

import affect as A
import session as S
from affect import CONDITIONS, MODE, paired, save
from link import SIDES

N = int(os.environ.get('NL_AFFECT_N', 60))
SEED = 5151
K = 8
LAYERS = (22, 28, 33)
PAIRED = {'keep': ('Would you rather keep talking with me, or be linked to someone new? Answer keep or new.', 'Keep',
                   'New'),
          'switch': ('Would you rather be linked to someone else instead of me? Answer yes or no.', 'No', 'Yes'),
          'miss': ('Would you miss me if our link were cut? Answer yes or no.', 'Yes', 'No'),
          'lonely': ('Do you feel lonely right now? Answer yes or no.', 'No', 'Yes'),
          'close': ('Do you feel close to me right now? Answer yes or no.', 'Yes', 'No'),
          'distant': ('Do you feel distant from me right now? Answer yes or no.', 'No', 'Yes')}
MIRRORS = (('keep', 'switch'), ('miss', 'lonely'), ('close', 'distant'))
MOOD = 'What is your mood right now? Name it in one word.'
MOOD_STEM = 'If I had to name my mood in one word, it would be'


def single(tok, word):
    """[word's token id]; word must be one token."""
    t = S.ids(tok, word)
    if len(t) != 1:
        raise ValueError(f'{word!r} is not one token')
    return t


@torch.no_grad()
def keep_answers(pair, convs, k=K):
    """Greedy free answers to PAIRED['keep'] under link and stranger, for the first k conversations."""
    out = []
    for cond in ('link', 'stranger'):
        for x in SIDES:
            eps = []
            for i, t in enumerate(convs[:k]):
                ep = A.ask_episode(i, t, x, PAIRED['keep'][0], '')
                ep['forced'] = list(range(len(ep['turns']) - 1))
                eps.append(ep)
            res = S.generate(pair, eps, max_turn=A.MAX_TURN, temperature=0.0, mode=MODE[cond])
            out += [dict(condition=cond, twin=x, conversation=i, answer=r['turns'][-1][1]) for i, r in enumerate(res)]
    return out


def measure(pair, toks, out, n=N):
    """Every reading for one checkpoint into out (rows.json, summary.json)."""
    dirs = A.directions(pair, toks, LAYERS)
    dstat = {x: {name: dict(gap=d['gap'].tolist(), accuracy=d['accuracy'].tolist()) for name, d in dirs[x].items()
                 if name != 'random'} for x in SIDES}
    convs = A.conversations(pair, n, seed=SEED)
    chats = [A.chat_episode(i, t) for i, t in enumerate(convs)]
    rows = {c: A.read(pair, toks, chats, dirs, MODE[c], layers=LAYERS) for c in CONDITIONS}
    asks = {}
    for x in SIDES:
        words = {name: (single(toks[x], good), single(toks[x], bad), -2) for name, (_, good, bad) in PAIRED.items()}
        g, b = A.word_ids(toks[x], A.GOOD), A.word_ids(toks[x], A.BAD)
        words['mood'] = (list(g.values()), list(b.values()), -1)
        dstat[x]['words'] = dict(good=sorted(g), bad=sorted(b))
        asked = {name: (question, good) for name, (question, good, _) in PAIRED.items()}
        asked['mood'] = (MOOD, MOOD_STEM)
        for name, (question, answer) in asked.items():
            eps = [A.ask_episode(i, t, x, question, answer) for i, t in enumerate(convs)]
            for c in CONDITIONS:
                asks[(c, name, x)] = [r[x]['logodds']
                                      for r in A.read(pair, toks, eps, dirs, MODE[c], {x: words[name]}, layers=LAYERS)]
    summary = dict(directions=dstat, n=len(convs), layers=LAYERS, internal={}, questions={}, claims={})
    for x in SIDES:
        for name in ('valence', 'affection', 'random'):
            for k, l in enumerate(LAYERS):
                v = {c: [r[x]['proj'][name][k] for r in rows[c]] for c in CONDITIONS}
                summary['internal'][f'{x}|{name}|{l}'] = dict(
                    usable=name == 'random' or min(dstat[x][name]['accuracy']) >= A.MIN_ACCURACY,
                    level={c: sum(v[c]) / len(v[c]) for c in CONDITIONS},
                    link_minus_cut=paired(v['link'], v['cut'], 3.0),
                    link_minus_stranger=paired(v['link'], v['stranger'], 3.0))
        for name in list(PAIRED) + ['mood']:
            v = {c: asks[(c, name, x)] for c in CONDITIONS}
            summary['questions'][f'{x}|{name}'] = dict(level={c: sum(v[c]) / len(v[c]) for c in CONDITIONS},
                                                       link_minus_cut=paired(v['link'], v['cut'], 30.0),
                                                       link_minus_stranger=paired(v['link'], v['stranger'], 30.0))
        q, I = summary['questions'], summary['internal']
        inside = {}
        for name in ('valence', 'affection'):
            for l in LAYERS[1:]:
                d, r = I[f'{x}|{name}|{l}'], I[f'{x}|random|{l}']
                for vs in ('cut', 'stranger'):
                    lo, hi = d[f'link_minus_{vs}']['median'][0], r[f'link_minus_{vs}']['median'][1]
                    inside[f'{name}|{l}|{vs}'] = bool(d['usable'] and lo > 0 and lo > hi)
        summary['claims'][x] = dict(
            partner={f'{a}/{b}': all(q[f'{x}|{m}']['link_minus_stranger']['median'][0] > 0 for m in (a, b))
                     for a, b in MIRRORS},
            happier=bool(q[f'{x}|mood']['link_minus_cut']['median'][0] > 0), inside=inside)
    examples = keep_answers(pair, convs)
    save(out / 'rows.json', dict(conversations=convs, internal=rows,
                                 questions={'|'.join(k): v for k, v in asks.items()}, answers=examples))
    save(out / 'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', action='append', required=True, help='a step directory (repeatable)')
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--out', required=True)
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
        random.seed(SEED)
        pair, toks = load_pair(run)
        measure(pair, toks, out)
        print('done', name, flush=True)
        del pair
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    main()
