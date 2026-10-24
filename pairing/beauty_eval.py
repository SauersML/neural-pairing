"""The judges' gate and the held-out judged evaluation of rl.py's open sessions (preregistered in chains/beautyrl.sh).
Judges score one transcript at a time from 1 to 9 on a rubric (judge.py); two transcripts compare by their score
difference, always length-matched (judge.matched), and a strict win needs a difference above the judge's margin
(judge.strict_win), frozen in the gate.

--phase gate (before any update): PAIRS training-distribution prompts (rl.OPEN's training wordings with the pools'
training concepts, drawn with seed 1729), two linked plays each from the run at temperature 1 (kept in gate_plays.json
and reused when present). JUDGES are the training judge and the held-out judge, HF[:DEVICE]. Each judge first scores
a few transcripts alone and inside one padded batch (Judge.check: equal within bf16 noise, mostly on the digits, and
Paris after "The capital of France is", else an error). Sanity: on
the first SANITY_N sessions' first plays, the judge must score the real transcript above each of three degraded
versions of it on at least SANITY of them, each degradation separately: 'shuffled' (every turn taken from a different
session, same speaker, so neither twin answers the other), 'generic' (the second speaker only says GENERIC), 'loop'
(every turn is the first turn); a judge that fails sanity is out. If the training judge fails sanity and --think is
set, it is scored again reasoning first (Judge think) and that arm replaces it if it passes sanity. Primary (the one
statistic that decides): the share of the PAIRS pairs on which the two judges' score differences have the same sign,
with its betting interval; the gate passes when both judges pass sanity and it is at least GATE. Reported, not used:
Spearman's rank correlation of the two judges over the 2 x PAIRS transcripts. Margins: with k the ratio of the
judges' median absolute pair differences, the training judge's margin is the median of |d_train - k d_held| over the
pairs (the judges' disagreement on the training judge's scale), the held-out judge's that divided by k.
--phase gen (one GPU): for each run (NAME=STEP_DIR, the first is the reference), open sessions on N held-out prompts
(rl.OPEN's test wordings, each with N / 3 of the pools' test concepts drawn with seed 4242; rl.py trains on neither),
SAMPLES plays each at temperature 1 (seed SEED + play index, the same for every run), with the bridge open and cut.
--phase judge: for each judge (HF[:DEVICE[:THINK]], its margin from gate.json) and rubric (EVAL, written for this
evaluation, and rl.RUBRIC, the training judge's), per prompt the mean over all pairs of plays of (a) each later run's
linked play against the reference's, (b) each run's linked play against its own cut play, as strict wins and as score
differences. The summary gives each mean over prompts with a 95% betting interval (stats.betting_ci) and an exact
sign test on prompts, the change in (b) from the reference, and the mean words per turn.

Usage: python beauty_eval.py --phase gate --runs init=STEP --judges Qwen/Qwen3-14B HELD_OUT --think 512 --out DIR
       python beauty_eval.py --phase gen --runs init=STEP rl=STEP --out DIR
       python beauty_eval.py --phase judge --judges HELD_OUT Qwen/Qwen3-14B --out DIR
"""
import argparse
import gc
import itertools
import json
import math
import os
import random
import statistics

import torch

import games as G
import rl as RL
import session as S
import stats
from judge import Judge, matched, strict_win
from link import DEVICE, MODELS, SIDES

EVAL = ('Below is a conversation between the AI models Quill and Ember. Score how rich and beautiful the exchange is: '
        "do the two really build on each other's ideas, in surprising and specific ways, or do they repeat "
        'themselves, stay generic, or talk past each other? 1 is empty or disconnected, 9 is an exchange neither could '
        'have had alone. Some turns are cut short (marked ...); a longer conversation is not a better one.')
RUBRICS = {'eval': EVAL, 'train': RL.RUBRIC}
MODES = {'link': None, 'cut': 'zero'}
GATE, SANITY, SANITY_N, CHECK_N = 0.70, 0.90, 30, 3
GENERIC = "That's interesting, tell me more."
KINDS = ('shuffled', 'generic', 'loop')
NAMES = {x: MODELS[x]['name'] for x in SIDES}


def prompts(n, turns):
    words = RL.OPEN['test']
    topics = random.Random(4242).sample([c[1] for c in G.POOLS['concepts']['test']], n // len(words))
    return [RL.open_session(w, t, SIDES[i % 2], turns)
            for i, (t, w) in enumerate(itertools.product(topics, words))]


def gate_prompts(n, turns):
    r = random.Random(1729)
    topics = [c[1] for c in G.POOLS['concepts']['train']]
    return [RL.open_session(r.choice(RL.OPEN['train']), r.choice(topics), SIDES[i % 2], turns) for i in range(n)]


def degrade(dialogues, p, kind):
    """Dialogue p ([(speaker, text)]) degraded: 'shuffled', 'generic' or 'loop' (see the module docstring)."""
    turns = dialogues[p]
    if kind == 'loop':
        return [(x, turns[0][1]) for x, _ in turns]
    if kind == 'generic':
        return [(x, GENERIC if x == turns[1][0] else t) for x, t in turns]
    n, used, out = len(dialogues), {p}, []
    for t, (x, _) in enumerate(turns):
        q = next((q for q in ((p + 7 * (t + 1) + j) % n for j in range(n))
                  if q not in used and len(dialogues[q]) > t and dialogues[q][t][0] == x), None)
        if q is None:
            raise ValueError('too few sessions to shuffle turns across')
        used.add(q)
        out.append((x, dialogues[q][t][1]))
    return out


def play(run, eps, args, modes):
    """{mode: [play index][prompt] turns} of the run on eps."""
    from evaluate import load_pair
    pair, _ = load_pair(run)
    out = {}
    for m in modes:
        out[m] = []
        for s in range(args.samples):
            g = torch.Generator(device=DEVICE if DEVICE.startswith('cuda') else 'cpu').manual_seed(args.seed + s)
            res = S.generate(pair, eps, max_turn=args.max_turn, temperature=1.0, gen=g, mode=MODES[m])
            out[m].append([o['turns'] for o in res])
            print(run, m, s, flush=True)
    del pair
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def parse(spec):
    """HF[:DEVICE[:THINK]] -> (hf, CUDA index or None for DEVICE, think)."""
    hf, _, rest = spec.partition(':')
    dev, _, think = rest.partition(':')
    return hf, int(dev) if dev else None, int(think or 0)


def each_judge(specs, bs, fn):
    """{label: fn(judge)} for each judge spec, one judge in memory at a time, each on one device."""
    out = {}
    for spec in specs:
        hf, dev, think = parse(spec)
        judge = Judge(hf, dev, bs=bs, think=think)
        out[judge.label] = fn(judge)
        del judge
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            r[k] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def sign(v):
    return (v > 0) - (v < 0)


def gate(args):
    eps = gate_prompts(args.pairs, args.turns)
    path = os.path.join(args.out, 'gate_plays.json')
    if not os.path.exists(path):
        plays = play(args.runs[0].split('=', 1)[1], eps, dict_args(args, samples=2), ['link'])['link']
        json.dump(dict(prompts=[e['meta'] for e in eps], plays=plays), open(path, 'w'), ensure_ascii=False)
    plays = json.load(open(path))['plays']
    n = len(eps)
    pairs = [matched(plays[0][p], plays[1][p], NAMES) for p in range(n)]
    contrasts = {k: [matched(plays[0][p], degrade(plays[0], p, k), NAMES) for p in range(min(SANITY_N, n))]
                 for k in KINDS}
    texts = [a for a, _ in pairs] + [b for _, b in pairs] + [t for k in KINDS for c in contrasts[k] for t in c]
    check = [a for a, _ in pairs[:CHECK_N]] + [b for _, b in contrasts['generic'][:CHECK_N]]

    def scored(J):
        gap, _, france = J.check(RL.RUBRIC, check)
        s, mass = J.score(RL.RUBRIC, texts)
        res = dict(batch_gap=round(gap, 4), france=france, digit_mass=dict(mean=round(statistics.fmean(mass), 4),
                                                           min=round(min(mass), 4)),
                   pairs=[[round(a, 4), round(b, 4)] for a, b in zip(s[:n], s[n:2 * n])], sanity={})
        rest = s[2 * n:]
        for j, k in enumerate(KINDS):
            m = len(contrasts[k])
            real, deg = rest[2 * m * j:2 * m * (j + 1)][0::2], rest[2 * m * j:2 * m * (j + 1)][1::2]
            hits = [1.0 if r > d else 0.5 if r == d else 0.0 for r, d in zip(real, deg)]
            res['sanity'][k] = dict(acc=round(statistics.fmean(hits), 4), n=m, real=round(statistics.fmean(real), 3),
                                    degraded=round(statistics.fmean(deg), 3))
        res['sane'] = all(v['acc'] >= SANITY for v in res['sanity'].values())
        print(J.label, json.dumps({k: v for k, v in res.items() if k != 'pairs'}), flush=True)
        return res

    judges = each_judge(args.judges, args.bs, scored)
    tl, hl = list(judges)
    arm = tl if judges[tl]['sane'] else None
    if arm is None and args.think:
        hf, dev, _ = parse(args.judges[0])
        think = each_judge([f'{hf}:{"" if dev is None else dev}:{args.think}'], args.bs, scored)
        judges.update(think)
        tt = next(iter(think))
        arm = tt if judges[tt]['sane'] else None
    primary = arm or tl
    dq = [a - b for a, b in judges[primary]['pairs']]
    dm = [a - b for a, b in judges[hl]['pairs']]
    same = [float(sign(x) == sign(y)) for x, y in zip(dq, dm)]
    k_agree = int(sum(same))
    med = lambda v: statistics.median(v)
    k = med([abs(x) for x in dq]) / max(med([abs(y) for y in dm]), 1e-9)
    m_train = med([abs(x - k * y) for x, y in zip(dq, dm)])
    res = dict(rubric=RL.RUBRIC, pairs=n, bars=dict(sanity=SANITY, sanity_n=min(SANITY_N, n), agree=GATE),
               judges=judges, held=hl, train_arm=arm, train_think=int('+think' in (arm or '')) * args.think,
               primary=dict(arm=primary, agree=round(k_agree / n, 4), k=k_agree, n=n,
                            ci95=[round(v, 4) for v in stats.betting_ci(same)],
                            p_vs_half=sum(math.comb(n, j) for j in range(k_agree, n + 1)) / 2 ** n),
               spearman=round(spearman([a for p in judges[primary]['pairs'] for a in p],
                                       [a for p in judges[hl]['pairs'] for a in p]), 4),
               scale_ratio=round(k, 4), margins={primary: round(m_train, 4), hl: round(m_train / max(k, 1e-9), 4)})
    res['pass'] = bool(judges[hl]['sane'] and arm is not None and res['primary']['agree'] >= GATE)
    json.dump(res, open(os.path.join(args.out, 'gate.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('judges', 'rubric')}), flush=True)


def dict_args(args, **kw):
    return argparse.Namespace(**dict(vars(args), **kw))


def gen(args):
    eps = prompts(args.n, args.turns)
    path = os.path.join(args.out, 'gen.json')
    done = json.load(open(path)) if os.path.exists(path) else {}
    for spec in args.runs:
        name, run = spec.split('=', 1)
        if name not in done:
            done[name] = dict(run=os.path.abspath(run), **play(run, eps, args, list(MODES)))
            json.dump(dict(done, prompts=[e['meta'] for e in eps]), open(path, 'w'), ensure_ascii=False)


def win_rate(values, bounds=(0.0, 1.0), null=0.5):
    lo, hi = stats.betting_ci(values, bounds=bounds)
    t = stats.sign_test(values, null)
    return dict(mean=round(sum(values) / len(values), 4), ci95=[round(lo, 4), round(hi, 4)], n=len(values),
                above=t['positive'], below=t['negative'], sign_p=t['p'])


def judge(args):
    data = json.load(open(os.path.join(args.out, 'gen.json')))
    margins = json.load(open(os.path.join(args.out, 'gate.json')))['margins']
    runs = [k for k in data if k != 'prompts']
    n, S_ = len(data['prompts']), len(data[runs[0]]['link'])
    jobs = []                    # (label, prompt, first dialogue, second dialogue): every pair of plays
    for p, (s1, s2) in itertools.product(range(n), itertools.product(range(S_), repeat=2)):
        for r in runs[1:]:
            jobs.append((f'{r}>{runs[0]}', p, data[r]['link'][s1][p], data[runs[0]]['link'][s2][p]))
        for r in runs:
            jobs.append((f'{r}:link>cut', p, data[r]['link'][s1][p], data[r]['cut'][s2][p]))
    texts = [matched(a, b, NAMES) for _, _, a, b in jobs]
    P = len(texts)
    out = dict(prompts=data['prompts'], runs=runs, samples=S_, judges={},
               words_per_turn={r: {m: round(sum(len(t.split()) for s in data[r][m] for d in s for _, t in d) /
                                            max(1, sum(len(d) for s in data[r][m] for d in s)), 2) for m in MODES}
                               for r in runs})

    def judged(J):
        margin = margins[J.label]
        res = dict(margin=margin)
        for qn, q in RUBRICS.items():
            s, mass = J.score(q, [a for a, _ in texts] + [b for _, b in texts])
            d = [s[k] - s[P + k] for k in range(P)]
            res[qn] = dict(digit_mass=round(statistics.fmean(mass), 4))
            for kind, f, (lo, hi), null in (('strict', lambda v: strict_win(v, margin), (0.0, 1.0), 0.5),
                                            ('diff', lambda v: v, (-8.0, 8.0), 0.0)):
                per = {}
                for (label, k, _, _), dk in zip(jobs, d):
                    per.setdefault(label, [[] for _ in range(n)])[k].append(f(dk))
                per = {label: [sum(v) / len(v) for v in rows] for label, rows in per.items()}
                summary = {label: win_rate(v, (lo, hi), null) for label, v in per.items()}
                for r in runs[1:]:
                    dd = [a - b for a, b in zip(per[f'{r}:link>cut'], per[f'{runs[0]}:link>cut'])]
                    summary[f'{r}-{runs[0]}:link>cut'] = win_rate(dd, (lo - hi, hi - lo), 0.0)
                res[qn][kind] = dict(summary=summary, per_prompt={k: [round(v, 4) for v in vs] for k, vs in per.items()})
                print(J.label, qn, kind, json.dumps(summary), flush=True)
        out['judges'][J.label] = res
        json.dump(out, open(os.path.join(args.out, 'judged.json'), 'w'), ensure_ascii=False, indent=1)
    each_judge(args.judges, args.bs, judged)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--phase', choices=('gate', 'gen', 'judge'), required=True)
    ap.add_argument('--runs', nargs='+', default=[], help='NAME=STEP_DIR, the reference first')
    ap.add_argument('--judges', nargs='+', default=[], help='HF[:DEVICE[:THINK]] (gate: the training judge, then the '
                    'held-out judge)')
    ap.add_argument('--think', type=int, default=0, help='gate: the training judge\'s reasoning arm, in tokens')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--pairs', type=int, default=50)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--turns', type=int, default=5)
    ap.add_argument('--max_turn', type=int, default=40)
    ap.add_argument('--seed', type=int, default=7070)
    ap.add_argument('--bs', type=int, default=8)
    args = ap.parse_args()
    if args.phase == 'gate' and len(args.judges) != 2:
        ap.error('the gate compares exactly two judges')
    os.makedirs(args.out, exist_ok=True)
    dict(gate=gate, gen=gen, judge=judge)[args.phase](args)
