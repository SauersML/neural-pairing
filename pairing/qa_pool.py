"""QA objective for the arithmetic mean scored by bench_eval's pair/avg.

Only A-D at the position following 'The answer is' enter the pooled loss.
The prefix is teacher forced, exactly as in bench_eval. Neither gold letter may
be visible through the partner's timeline. No detach: both LoRAs and both halves
of the link receive gradients. The pooled score, not either head, is calibrated
by this proper scoring rule; collapse remains a possible solution.
"""
import math

import torch

import games as G
import session as S
from link import SIDES, partner


def letter_layout(toks, rows, built):
    """QA row indices, prediction positions and one-token A-D IDs; fail on drift."""
    selected = [i for i, (ep, _) in enumerate(rows) if ep['game'] == 'qa']
    at, lid = {}, {}
    for x in SIDES:
        letters = [S.ids(toks[x], ' ' + c) for c in G.LETTERS]
        if any(len(t) != 1 for t in letters):
            raise ValueError(f'{x}: benchmark letters must each be one token')
        lid[x] = [t[0] for t in letters]
        prefix = S.ids(toks[x], 'The answer is')
        k = len(prefix)
        at[x] = []
        for i in selected:
            labels = built[i][x]['labels']
            if [t for _, t, _ in labels[:k]] != prefix:
                raise ValueError(f'{x}: QA answer prefix differs from bench_eval')
            gold = G.LETTERS.index(rows[i][0]['meta']['answer'])
            pos, token, _ = labels[k]
            if token != lid[x][gold] or built[i][x]['ids'][pos + 1] != token:
                raise ValueError(f'{x}: QA letter label is misaligned')
            at[x].append(pos)
    for j, i in enumerate(selected):
        for x in SIDES:
            y = partner(x)
            if built[i][y]['time'][at[y][j] + 1] <= built[i][x]['time'][at[x][j]]:
                raise ValueError('partner gold letter is visible at QA prediction')
    return selected, at, lid


def log_average(logits):
    """Stable log((softmax(q) + softmax(e)) / 2), in float32."""
    lp = torch.stack([logits[x].float().log_softmax(-1) for x in SIDES])
    return torch.logsumexp(lp, dim=0) - math.log(len(SIDES))


def pooled_loss(pair, toks, rows, built, batch, last, denom):
    """This microbatch's pooled CE, using the existing global token denominators.

    A QA row inherits the sum of its two ordinary normalized loss weights. Thus
    all-QA training gives 2 * mean pooled CE (the old objective summed two CEs).
    Non-QA rows keep their old weight; split batches/ranks add exactly. The caller
    scales the old QA contributions by (1 - qa_pool) and this term by qa_pool.
    """
    selected, at, lid = letter_layout(toks, rows, built)
    if not selected:
        return last[SIDES[0]].new_zeros((), dtype=torch.float32)
    logits = {}
    for x in SIDES:
        width = batch[x]['ids'].shape[1]
        idx = torch.tensor([i * width + p for i, p in zip(selected, at[x])], device=last[x].device)
        logits[x] = pair.token_logits(x, last[x], idx)[:, lid[x]]
    lp = log_average(logits)
    gold = torch.tensor([G.LETTERS.index(rows[i][0]['meta']['answer']) for i in selected], device=lp.device)
    weight = lp.new_tensor([sum(sum(w for _, _, w in built[i][x]['labels']) / denom[x] for x in SIDES)
                            for i in selected])
    return -(lp.gather(1, gold[:, None]).squeeze(1) * weight).sum()
