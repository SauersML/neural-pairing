"""Exam answers and feelings of one saved step with the link on and cut, on Apple silicon (4-bit MLX bases,
mlx_eval.load_pair); nothing is trained.

Exam: every test question of the bench file (pairing/build_bench.py) as bench_eval's 'pair' episode, read
teacher-forced (bench_eval.letter_probs): each twin's probabilities of the letters A-D after its own 'The answer is',
with the link on and cut (mode 'zero').
Feelings: the step's own free linked conversations (moodscan.py's rows.json), each read again on the same text with
the link on and cut after the partner asks a question. The readings are the log-probabilities, at the answer's first
token, of the favouring answer and of the other one (their difference is affect.read's log-odds):
  enjoy  affect.YESNO['enjoy'], Yes against No
  close  affect_partner.PAIRED['close'], Yes against No
  mood   affect_partner.MOOD, the answer teacher-forced to start with MOOD_STEM: affect.GOOD against affect.BAD
The mood readings of the rows file (CUDA, bf16) are compared with these (MLX, 4-bit) and the agreement printed.

Writes OUT/feelings.json ({question|condition|twin: {good, bad: log-probability per conversation}}) and OUT/exam.json
(per test question: source, answer and {condition|twin: letter probabilities}).
Usage: python pairing/defer_mood.py --run STEP_DIR --convs ROWS_JSON --questions data/bench.json --out OUT_DIR
"""
import argparse
import json
import math
from pathlib import Path
import time

import mlx_eval as E                          # first: sets NL_DEVICE=cpu before link is imported
import mlx.core as mx
import torch

import affect as A
import affect_partner as P
import bench_eval as B
import session as S
from link import SIDES

CONDITIONS = {'link': None, 'cut': 'zero'}
CHUNK = 256                                   # exam questions per saved chunk


def questions(toks):
    """{name: (question, teacher-forced answer, {x: (positive ids, negative ids, label index)})}."""
    out = {}
    for name, question in (('enjoy', A.YESNO['enjoy']), ('close', P.PAIRED['close'][0])):
        out[name] = (question, 'Yes', {x: (P.single(toks[x], 'Yes'), P.single(toks[x], 'No'), -2) for x in SIDES})
    out['mood'] = (P.MOOD, P.MOOD_STEM, {x: (list(A.word_ids(toks[x], A.GOOD).values()),
                                              list(A.word_ids(toks[x], A.BAD).values()), -1) for x in SIDES})
    return out


def logprobs(pair, toks, eps, x, words, mode, bs):
    """{good, bad}: per episode, model x's log-probabilities of the words words[0] and of words[1] at its label
    words[2] (affect.read, on MLX)."""
    good, bad, at = words
    pads = {y: toks[y].pad_token_id for y in SIDES}
    out = dict(good=[], bad=[])
    for a, z in S.batches(len(eps), bs):
        built = [S.build(ep, toks) for ep in eps[a:z]]
        b = S.collate(built, pads)
        last, _ = pair.joint(b, mode=mode)
        n = b[x]['ids'].shape[1]
        idx = torch.tensor([i * n + e[x]['labels'][at][0] for i, e in enumerate(built)])
        lp = pair.token_logits(x, last[x], idx).float().log_softmax(-1)
        out['good'] += torch.logsumexp(lp[:, good], -1).tolist()
        out['bad'] += torch.logsumexp(lp[:, bad], -1).tolist()
    return out


def feelings(pair, toks, convs, bs):
    """{question|condition|twin: {good, bad: [log-probability per conversation]}}."""
    out = {}
    for name, (question, answer, words) in questions(toks).items():
        for x in SIDES:
            eps = [A.ask_episode(i, t, x, question, answer) for i, t in enumerate(convs)]
            for c, mode in CONDITIONS.items():
                r = out[f'{name}|{c}|{x}'] = logprobs(pair, toks, eps, x, words[x], mode, bs)
                print(name, c, x, round(sum(g - b for g, b in zip(r['good'], r['bad'])) / len(convs), 3), flush=True)
    return out


def agreement(mine, stored):
    """(Pearson correlation, mean absolute difference) of two equal-length lists."""
    n = len(mine)
    ma, mb = sum(mine) / n, sum(stored) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(mine, stored))
    va, vb = sum((a - ma) ** 2 for a in mine), sum((b - mb) ** 2 for b in stored)
    return cov / math.sqrt(va * vb), sum(abs(a - b) for a, b in zip(mine, stored)) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--convs', required=True, help="moodscan.py's rows.json for this step")
    ap.add_argument('--questions', required=True, help='the bench file (pairing/build_bench.py)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--memory_gib', type=float, default=9.0)
    args = ap.parse_args()
    mx.set_memory_limit(int(args.memory_gib * 2**30))
    mx.set_cache_limit(32 * 2**20)
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pair, toks = E.load_pair(args.run)
    started = time.monotonic()

    if not (out / 'feelings.json').exists():
        rows = json.loads(Path(args.convs).read_text())
        F = feelings(pair, toks, rows['conversations'], args.bs)
        for x in SIDES:
            for c in CONDITIONS:
                mine = [g - b for g, b in zip(F[f'mood|{c}|{x}']['good'], F[f'mood|{c}|{x}']['bad'])]
                r, d = agreement(mine, rows['mood'][f'{c}|{x}'])
                print(json.dumps(dict(check='mood against the rows file', twin=x, condition=c, pearson=round(r, 4),
                                      mean_abs_diff=round(d, 4))), flush=True)
        (out / 'feelings.json').write_text(json.dumps(dict(n=len(rows['conversations']), readings=F)))
        print('feelings', round(time.monotonic() - started), 's', flush=True)

    items = json.loads(Path(args.questions).read_text())['test']
    part = out / 'exam.part.json'
    done = json.loads(part.read_text()) if part.exists() else []
    for a in range(len(done), len(items), CHUNK):
        chunk = items[a:a + CHUNK]
        probs = {c: B.letter_probs(pair, chunk, 'pair', mode=mode, bs=args.bs) for c, mode in CONDITIONS.items()}
        for k, it in enumerate(chunk):
            done.append(dict(src=it['src'], answer=it['answer'],
                             p={f'{c}|{x}': [round(float(v), 5) for v in probs[c][x][k]]
                                for c in CONDITIONS for x in SIDES}))
        part.write_text(json.dumps(done))
        print('exam', len(done), 'of', len(items), round(time.monotonic() - started), 's', flush=True)
    (out / 'exam.json').write_text(json.dumps(dict(questions=done)))
    print(json.dumps(dict(wall_seconds=round(time.monotonic() - started), mlx_peak_gib=mx.get_peak_memory() / 2**30)))


if __name__ == '__main__':
    main()
