"""The pooled QA objective (train.py --qa_pool): the cross-entropy of the arithmetic mean of the two models' A-D
probabilities at the answer letter, the score bench_eval reports as pair/avg for the linked pair and as ensemble for the
models alone (train.py --solo: the same objective without the link).

Only A-D at the position after 'The answer is' enter the loss; that prefix is teacher-forced, as in bench_eval, and
neither gold letter may be visible through the partner's timeline. Gradients reach both LoRAs and both halves of the
link. The loss calibrates the pooled distribution only, so one model's own letter probabilities may become
uninformative.
"""
import math

import torch

import games as G
import session as S
from link import SIDES, partner

GAMES = ('qa', 'qa_solo')  # the episodes whose letters are pooled: linked pair, and each model alone


def letter_layout(toks, eps, built):
    """(indices of the QA episodes in eps, {x: the position that predicts each one's letter}, {x: the token IDs of
    A-D}); an error when the prefix or the letter label is not where bench_eval reads it, or when a partner's gold
    letter is visible at the prediction."""
    selected = [i for i, ep in enumerate(eps) if ep['game'] in GAMES]
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
            gold = G.LETTERS.index(eps[i]['meta']['answer'])
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


def pooled_loss(pair, toks, eps, built, batch, last, denom):
    """The pooled cross-entropy of this micro-batch's QA episodes eps (built, collated as batch, last hidden states
    last), with the whole batch's token denominators denom. Each QA episode is weighted by the sum of its two models'
    normalized token weights, so all-QA training gives twice the mean pooled cross-entropy and split batches and ranks
    add up to the whole batch's loss. train.py scales the QA token cross-entropy by (1 - qa_pool) and this term by
    qa_pool."""
    selected, at, lid = letter_layout(toks, eps, built)
    if not selected:
        return last[SIDES[0]].new_zeros((), dtype=torch.float32)
    logits = {}
    for x in SIDES:
        width = batch[x]['ids'].shape[1]
        idx = torch.tensor([i * width + p for i, p in zip(selected, at[x])], device=last[x].device)
        logits[x] = pair.token_logits(x, last[x], idx)[:, lid[x]]
    lp = log_average(logits)
    gold = torch.tensor([G.LETTERS.index(eps[i]['meta']['answer']) for i in selected], device=lp.device)
    weight = lp.new_tensor([sum(sum(w for _, _, w in built[i][x]['labels']) / denom[x] for x in SIDES)
                            for i in selected])
    return -(lp.gather(1, gold[:, None]).squeeze(1) * weight).sum()
