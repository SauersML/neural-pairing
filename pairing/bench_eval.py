"""Benchmark: does the linked pair answer ARC-Challenge and MMLU questions better than either model alone?

Arms, each model's answer being its most likely letter after its own 'The answer is', scored in one teacher-forced pass
of the pair (the computation the pair trains with; nothing of the partner's letter is visible when a model writes its own):
  base       each original model alone (adapters off)                                   (1) Quill, (2) Ember
  solo       each model alone after the same fine-tuning on the same questions          (3)
  pair       the trained pair, the link open                                            (4)
  pair/zero, pair/shuffle, pair/random   the same pair without the bridges, or reading other partners   (5)
  text       a pair trained the same way to answer in turn in text, the link closed     (6)
  ensemble   the two solo models' letter probabilities averaged (an extra control)
The pair's answer is the answer of the model whose solo version scores higher on the validation questions (fixed
before any test question is read); in the text pair that model answers second, after reading its partner's answer,
written as 'The answer is X.' from the partner's own most likely letter.

The headline compares the pair with the strongest of the fine-tuned single models and their ensemble on the test
questions. Tests are paired per question: exact McNemar (two-sided binomial test on the discordant questions) and an exact
interval for the accuracy difference (Clopper-Pearson on the discordant split, conditional on their number). The claim
"pair > best single" needs the pair to beat every comparator, each at p < 0.05 (an intersection-union test, so no
correction for the number of comparisons).

Each rank scores its questions in batches of similar length, and the 'shuffle' control reads the partner of another
question of the same batch. Under torchrun the questions are dealt to the ranks in whole batches of question order, so
the batches differ from a single process's.

Usage: python bench_eval.py --pair RUN/step_N --solo RUN/step_N --text RUN/step_N --out OUT/bench.json
"""
import argparse
import contextlib
import json
import os

import torch
from scipy.stats import beta, binomtest

import ddp
import games as G
import session as S
from evaluate import mean
from link import DEVICE, SIDES, Pair, load, partner


def mcnemar(a, b):
    """a, b: 0/1 lists over the same questions. Accuracy difference a - b, exact 95% interval, exact p."""
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n, N = n10 + n01, len(a)
    if n == 0:
        return dict(diff=0.0, lo=0.0, hi=0.0, p=1.0, n10=0, n01=0)
    lo = beta.ppf(0.025, n10, n01 + 1) if n10 > 0 else 0.0
    hi = beta.ppf(0.975, n10 + 1, n01) if n01 > 0 else 1.0
    return dict(diff=round((n10 - n01) / N, 4), lo=round(n / N * (2 * lo - 1), 4), hi=round(n / N * (2 * hi - 1), 4),
                p=float(binomtest(n10, n, 0.5).pvalue), n10=n10, n01=n01)


@torch.no_grad()
def letter_probs(pair, items, style, mode=None, order=None, first=None, bs=64):
    """Each model's probabilities of the letters A-D after its own 'The answer is', in one teacher-forced pass of the
    pair: {x: [N, 4]}. style, order: G.qa_episode; mode: the link.Pair ablation of the 'pair' style (the other styles
    run with the link cut). first (text style): the first speaker's letter per question, written as its turn for the
    second speaker to read; without it the first turn is the gold answer, so only the first speaker's scores hold."""
    toks = pair.toks
    pads = {x: toks[x].pad_token_id for x in SIDES}
    lid = {x: [S.ids(toks[x], ' ' + L)[0] for L in G.LETTERS] for x in SIDES}
    want = {x: S.ids(toks[x], 'The answer is') for x in SIDES}
    eps = [G.qa_episode(it, style, order=order) for it in items]
    if first is not None:
        for e, f in zip(eps, first):
            e['turns'][0] = (e['turns'][0][0], f'The answer is {f}.', 1.0)
    every = [S.build(e, toks) for e in eps]
    # batches of questions of similar length (in question order, padding to each batch's longest question was ~4x the
    # real tokens); results come back in question order
    rank = sorted(range(len(items)), key=lambda i: max(len(every[i][x]['ids']) for x in SIDES))
    out = {x: [None] * len(items) for x in SIDES}
    for a, z in S.batches(len(items), bs):
        built = [every[i] for i in rank[a:z]]
        batch = S.collate(built, pads)
        last, _ = pair.joint(batch, mode=mode if style == 'pair' else 'zero')
        at = {x: [] for x in SIDES}              # per episode, the position whose output predicts the letter
        for x in SIDES:
            for e in built:
                labs, k = e[x]['labels'], len(want[x])
                assert [t for _, t, _ in labs[:k]] == want[x], (x, labs[:k])
                at[x].append(labs[k][0])
        if style == 'pair':
            # a token read at tick t reads the partner's tokens read at ticks <= t: the partner's (gold) letter must
            # be read after the tick at which this model predicts its own
            for x in SIDES:
                for i, e in enumerate(built):
                    y = partner(x)
                    assert e[y]['time'][at[y][i] + 1] > e[x]['time'][at[x][i]], (x, 'partner letter visible')
        for x in SIDES:
            n = batch[x]['ids'].shape[1]
            idx = torch.tensor([i * n + p for i, p in enumerate(at[x])], device=DEVICE)
            lg = pair.token_logits(x, last[x], idx)[:, lid[x]]
            for i, p in zip(rank[a:z], torch.softmax(lg, -1).cpu()):
                out[x][i] = p
    return {x: torch.stack(v) for x, v in out.items()}


if __name__ == '__main__':
    from peft import PeftModel

    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--solo', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0, help='first N questions of each split (smoke tests)')
    args = ap.parse_args()
    ddp.init()
    torch.manual_seed(0)                         # the random control

    ck = torch.load(os.path.join(args.pair, 'bridge.pt'), weights_only=False)
    F = ck.get('frozen')                         # a one-way pair: its target has no 'pair' adapter (link.py)
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = load(x)
        if x == F:
            m = PeftModel.from_pretrained(m, os.path.join(args.solo, x), adapter_name='solo')
        else:
            m = PeftModel.from_pretrained(m, os.path.join(args.pair, x), adapter_name='pair')
            m.load_adapter(os.path.join(args.solo, x), adapter_name='solo')
        m.load_adapter(os.path.join(args.text, x), adapter_name='text')
        models[x] = m.eval()
    pair = Pair.from_checkpoint(models, toks, ck, plain=False).eval()

    def adapters(name):
        stack = contextlib.ExitStack()
        for x in SIDES:
            if name == 'base' or (name == 'pair' and x == F):          # the one-way target: the original model
                stack.enter_context(models[x].disable_adapter())
            else:
                models[x].set_adapter(name)
        return stack

    def correct(P, items):
        return [int(int(p.argmax()) == it['answer']) for p, it in zip(P, items)]

    def share(items):
        """This rank's questions: whole batches of bs, dealt in turn to the ranks."""
        return [items[i] for k, b in enumerate(range(0, len(items), args.bs)) if k % ddp.WORLD == ddp.RANK
                for i in range(b, min(b + args.bs, len(items)))]

    def unshare(local, n):
        """{arm: 0/1 per question of this rank} from every rank -> {arm: 0/1 per question}, in question order."""
        owner = [k % ddp.WORLD for k, b in enumerate(range(0, n, args.bs)) for i in range(b, min(b + args.bs, n))]
        parts = ddp.gather(local)
        pos = [0] * ddp.WORLD
        out = {k: [] for k in local}
        for i in range(n):
            r = owner[i]
            for k in out:
                out[k].append(parts[r][k][pos[r]])
            pos[r] += 1
        return out

    split = {k: G.bench(k) for k in ('val', 'test')}
    if args.limit:
        split = {k: v[:args.limit] for k, v in split.items()}
    R = {'val': {}, 'test': {}, 'n': {k: len(v) for k, v in split.items()}}

    # validation: which model answers for the pair
    val = share(split['val'])
    V = {}
    for name in ('base', 'solo'):
        with adapters(name):
            P = letter_probs(pair, val, 'solo', bs=args.bs)
        for x in SIDES:
            V[f'{name}/{x}'] = correct(P[x], val)
    for k, v in unshare(V, len(split['val'])).items():
        R['val'][k] = round(mean(v), 4)
    D = max(SIDES, key=lambda x: (R['val'][f'solo/{x}'], x == 'quill'))
    if F:
        D = partner(F)                           # a one-way pair answers through its reader
    W = partner(D)
    R['answerer'] = D
    print('validation', R['val'], 'answerer', D, flush=True)

    all_items, items = split['test'], share(split['test'])
    C = {}                                       # arm -> 0/1 per test question (of this rank)
    with adapters('base'):
        P = letter_probs(pair, items, 'solo', bs=args.bs)
    for x in SIDES:
        C[f'base/{x}'] = correct(P[x], items)
    with adapters('solo'):
        P = letter_probs(pair, items, 'solo', bs=args.bs)
    for x in SIDES:
        C[f'solo/{x}'] = correct(P[x], items)
    C['ensemble'] = correct((P['quill'] + P['ember']) / 2, items)
    with adapters('pair'):
        for mode in (None, 'zero', 'shuffle', 'random'):
            P = letter_probs(pair, items, 'pair', mode=mode, bs=args.bs)
            for x in SIDES:
                C[f'pair{"/" + mode if mode else ""}/{x}'] = correct(P[x], items)
    with adapters('text'):
        for order in ((W, D), (D, W)):
            P = letter_probs(pair, items, 'text', order=order, bs=args.bs)
            first = [G.LETTERS[int(p.argmax())] for p in P[order[0]]]
            C[f'text/{order[0]}-first/{order[0]}'] = correct(P[order[0]], items)
            P = letter_probs(pair, items, 'text', order=order, first=first, bs=args.bs)
            C[f'text/{order[0]}-first/{order[1]}'] = correct(P[order[1]], items)

    C = unshare(C, len(all_items))
    items = all_items
    if not ddp.MAIN:
        raise SystemExit
    srcs = {'all': None, 'arc_challenge': 'arc_challenge', 'mmlu': 'mmlu'}
    comparators = [f'base/{x}' for x in SIDES] + [f'solo/{x}' for x in SIDES] + \
        [f'pair/{m}/{D}' for m in ('zero', 'shuffle', 'random')] + [f'text/{W}-first/{D}', 'ensemble']
    required = comparators[:-1]
    for name, src in srcs.items():
        sel = [i for i, it in enumerate(items) if src is None or it['src'] == src]
        acc = {k: round(mean([v[i] for i in sel]), 4) for k, v in C.items()}
        tests = {k: mcnemar([C[f'pair/{D}'][i] for i in sel], [C[k][i] for i in sel]) for k in comparators}
        claim = all(tests[k]['diff'] > 0 and tests[k]['p'] < 0.05 for k in required)
        # the headline: the pair against the strongest of the fine-tuned single models and their ensemble, picked
        # on these test questions (the comparison least favourable to the pair)
        best = max(('solo/quill', 'solo/ember', 'ensemble'), key=lambda k: acc[k])
        R['test'][name] = dict(n=len(sel), accuracy=acc, vs_pair=tests, pair_beats_all=claim,
                               headline=dict(vs=best, **tests[best]))
        print(name, json.dumps(R['test'][name]), flush=True)
    R['per_question'] = C
    R['questions'] = [dict(src=it['src'], answer=it['answer']) for it in items]
    json.dump(R, open(args.out, 'w'), indent=1)
