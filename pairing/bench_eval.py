"""Benchmark: does the linked pair answer ARC-Challenge and MMLU questions better than either model alone?

Arms, each model writing one answer of at most 12 tokens per question, greedy:
  base       each original model alone (adapters off)                                   (1) Quill, (2) Ember
  solo       each model alone after the same fine-tuning on the same questions          (3)
  pair       the trained pair, the link open                                            (4)
  pair/zero, pair/shuffle, pair/random   the same pair without the bridges, or reading other partners   (5)
  text       a pair trained the same way to answer in turn in text, the link closed     (6)
  ensemble   the two solo models' letter probabilities averaged (an extra control)
The pair's answer is the answer of the model whose solo version scores higher on the validation questions (fixed
before any test question is read); in the text pair that model answers second, after reading its partner's answer.

The headline compares the pair with the strongest of the fine-tuned single models and their ensemble on the test
questions. Tests are paired per question: exact McNemar (two-sided binomial test on the discordant questions) and an exact
interval for the accuracy difference (Clopper-Pearson on the discordant split, conditional on their number). The claim
"pair > best single" needs the pair to beat every comparator, each at p < 0.05 (an intersection-union test, so no
correction for the number of comparisons).

Under torchrun the questions are split over the ranks in whole batches, so every batch (and the 'shuffle' control,
which reads the partner of another episode of the batch) is the same as in a single process.

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
from evaluate import mean, qa_answers
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
def letter_probs(pair, items, bs=32):
    """Each model's probabilities of the letters A-D after 'The answer is', solo prompts, link cut: {x: [N, 4]}."""
    toks = pair.toks
    pads = {x: toks[x].pad_token_id for x in SIDES}
    lid = {x: [S.ids(toks[x], ' ' + L)[0] for L in G.LETTERS] for x in SIDES}
    out = {x: [] for x in SIDES}
    for b in range(0, len(items), bs):
        eps = [G.qa_episode(it, 'solo') for it in items[b:b + bs]]
        built = [S.build(e, toks) for e in eps]
        batch = S.collate(built, pads)
        last, _ = pair.joint(batch, mode='zero')
        for x in SIDES:
            want = S.ids(toks[x], 'The answer is')
            idx = []
            for i, e in enumerate(built):
                labs = e[x]['labels']
                k = len(want)                      # the label that predicts the letter
                assert [t for _, t, _ in labs[:k]] == want, (x, labs[:k])
                idx.append(i * batch[x]['ids'].shape[1] + labs[k][0])
            lg = pair.token_logits(x, last[x], torch.tensor(idx, device=DEVICE))[:, lid[x]]
            out[x].append(torch.softmax(lg, -1).cpu())
    return {x: torch.cat(v) for x, v in out.items()}


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

    ck = torch.load(os.path.join(args.pair, 'bridge.pt'), weights_only=False)
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = load(x)
        m = PeftModel.from_pretrained(m, os.path.join(args.pair, x), adapter_name='pair')
        m.load_adapter(os.path.join(args.solo, x), adapter_name='solo')
        m.load_adapter(os.path.join(args.text, x), adapter_name='text')
        models[x] = m.eval()
    pair = Pair.from_checkpoint(models, toks, ck).eval()

    def adapters(name):
        stack = contextlib.ExitStack()
        for x in SIDES:
            if name == 'base':
                stack.enter_context(models[x].disable_adapter())
            else:
                models[x].set_adapter(name)
        return stack

    def correct(ans, items, x):
        return [int(a.get(x) == G.LETTERS[it['answer']]) for a, it in zip(ans, items)]

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
            a = qa_answers(pair, val, 'solo', bs=args.bs)
        for x in SIDES:
            V[f'{name}/{x}'] = correct(a, val, x)
    for k, v in unshare(V, len(split['val'])).items():
        R['val'][k] = round(mean(v), 4)
    D = max(SIDES, key=lambda x: (R['val'][f'solo/{x}'], x == 'quill'))
    W = partner(D)
    R['answerer'] = D
    print('validation', R['val'], 'answerer', D, flush=True)

    all_items, items = split['test'], share(split['test'])
    C = {}                                       # arm -> 0/1 per test question (of this rank)
    with adapters('base'):
        a = qa_answers(pair, items, 'solo', bs=args.bs)
    for x in SIDES:
        C[f'base/{x}'] = correct(a, items, x)
    with adapters('solo'):
        a = qa_answers(pair, items, 'solo', bs=args.bs)
        P = letter_probs(pair, items)
    for x in SIDES:
        C[f'solo/{x}'] = correct(a, items, x)
    ens = ((P['quill'] + P['ember']) / 2).argmax(-1).tolist()
    C['ensemble'] = [int(e == it['answer']) for e, it in zip(ens, items)]
    with adapters('pair'):
        for mode in (None, 'zero', 'shuffle', 'random'):
            a = qa_answers(pair, items, 'pair', mode=mode, bs=args.bs)
            for x in SIDES:
                C[f'pair{"/" + mode if mode else ""}/{x}'] = correct(a, items, x)
    with adapters('text'):
        for order in ((W, D), (D, W)):
            a = qa_answers(pair, items, 'text', order=order, bs=args.bs)
            for x in SIDES:
                C[f'text/{order[0]}-first/{x}'] = correct(a, items, x)

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
