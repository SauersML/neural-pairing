"""Content-matched initialization of the bridges' reads (train.py --content_init).

With random queries and keys a reader attends to arbitrary partner positions, so at the start of training the bridges
get a weak and inconsistent gradient. This sets them so that at each exchange a reader position attends to the partner
positions whose content matches its own. Both models read the same texts (co-reading documents in the user turn,
replies in the assistant turn); a token of one model and a token of the other match when both end at the same character
of the text, so both states have read the same prefix. At each exchange block a ridge regression maps the sender's
normalized states to the reader's, zx ~ mx + M (zy - my), with the penalty that minimizes the error on texts outside the
fit. The reader's queries lose the direction of its mean state (so no partner position is attended to by every query),
and the sender's keys become K = c Q M: a query's logit on a key is c times its similarity to the reader state predicted
from that key, in each head's random subspace. The scale c maximizes the log-probability of the matched position on the
texts outside the fit. With vo, the sender's values and the reader's output projection also start as the top
heads * head_dim singular directions of M, so a write delivers the matched partner state mapped into the reader's space.
alpha is untouched: with alpha 0 the pair is still exactly the two models.
"""
import math
import random

import numpy as np
import scipy.linalg as sl
import torch

import session as S
from link import DEVICE, MODELS, SIDES, find_layers, inner

MAX_CHARS = 1500
LAMBDAS = 10.0 ** np.arange(-6, 1.01, 0.25)           # ridge penalties, relative to the mean eigenvalue
SCALES = 2.0 ** np.arange(-4, 6.01, 0.25)             # key scales c
# the fits' float64 linear algebra: on the GPU when there is one (16 fits of 2560-4096-wide maps took 8 minutes on a
# pod's CPU), else on the CPU with SciPy's decompositions (torch.linalg's are unreliable under macOS Accelerate)
LA = DEVICE if DEVICE.startswith('cuda') else 'cpu'


def eigh(A):
    """Eigenvalues (ascending) and eigenvectors of a symmetric float64 matrix, on its device."""
    if A.is_cuda:
        return torch.linalg.eigh(A)
    s, E = sl.eigh(A.numpy())
    return torch.from_numpy(s), torch.from_numpy(E)


def svd(M):
    """U, singular values (descending), V^T of a float64 matrix (thin), on its device."""
    if M.is_cuda:
        return torch.linalg.svd(M, full_matrices=False)
    U, s, Vt = sl.svd(M.numpy(), full_matrices=False)
    return torch.from_numpy(U), torch.from_numpy(s), torch.from_numpy(Vt)


def texts(replies, n, seed=0):
    """n (prompt, text, role) items, half co-reading documents read in the user turn and half replies read in the
    assistant turn."""
    rng = random.Random(seed)
    docs = sorted(replies[SIDES[0]].get('coread', {}))
    own = sorted((p, r) for x in SIDES for p, r in replies[x]['own'].items())
    rng.shuffle(docs)
    rng.shuffle(own)
    items = [(None, d, 'user') for d in docs[:n - n // 2]] + [(p, r, 'assistant') for p, r in own[:n // 2]]
    rng.shuffle(items)
    return items


def sequence(tok, x, item):
    """x's token ids of an item, and {character offset at which a token of the text ends: its position}."""
    prompt, text, role = item
    system = f"You are {MODELS[x]['name']}, {MODELS[x]['about']}."
    if role == 'user':
        pre = S.prefix(tok, x, system) + S.ids(tok, '<|im_start|>user\n')
    else:
        pre = S.prefix(tok, x, system, prompt) + S.assistant_header(tok, x)
    enc = tok(text[:MAX_CHARS], add_special_tokens=False, return_offsets_mapping=True)
    ends = {b: len(pre) + i for i, (a, b) in enumerate(enc['offset_mapping']) if b > a}
    return pre + enc['input_ids'], ends


@torch.no_grad()
def states(model, blocks, ids, eps):
    """[len(blocks)] float32 states [L, d] of model after each of blocks on one sequence, divided by their RMS (as the
    bridges' norms do, before their weights)."""
    layers = find_layers(model)
    got = {}

    def hook(l):
        return lambda mod, i, o: got.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0].float())
    hooks = [layers[l].register_forward_hook(hook(l)) for l in blocks]
    try:
        inner(model)[0](input_ids=torch.tensor([ids], device=DEVICE))
    finally:
        for h in hooks:
            h.remove()
    return [got[l] * torch.rsqrt(got[l].pow(2).mean(-1, keepdim=True) + eps) for l in blocks]


def fit(G, H, gq, gk):
    """The ridge map of one direction at one block. G, H: the fit's and the test texts' sums over matched pairs (xx, yy,
    xy outer products, sx, sy, n) of the reader's (x) and sender's (y) unit-RMS states; gq, gk: the reader's query norm
    and the sender's key norm weights (float64 tensors on one device; n a float). Returns M, the reader's mean state
    mx, the test R^2 and the penalty used."""
    def cov(S_, key, ga, gb, ma, mb):
        return ga[:, None] * S_[key] / S_['n'] * gb[None] - torch.outer(ma, mb)
    mx, my = gq * G['sx'] / G['n'], gk * G['sy'] / G['n']
    hx, hy = gq * H['sx'] / H['n'], gk * H['sy'] / H['n']
    Cyy, Cxy = cov(G, 'yy', gk, gk, my, my), cov(G, 'xy', gq, gk, mx, my)
    # the test texts' second moments about the fit's means
    Txx = cov(H, 'xx', gq, gq, hx, hx) + torch.outer(hx - mx, hx - mx)
    Tyy = cov(H, 'yy', gk, gk, hy, hy) + torch.outer(hy - my, hy - my)
    Txy = cov(H, 'xy', gq, gk, hx, hy) + torch.outer(hx - mx, hy - my)
    s, E = eigh(Cyy)
    s = s.clamp_min(0.0)
    B = Cxy @ E
    # test error of M(lam) = B diag(1 / (s + lam)) E^T: tr Txx - 2 sum_i v_i (B^T Txy E)_ii + v^T ((E^T Tyy E) * (B^T B)) v
    cross = (B * (Txy @ E)).sum(0)
    quad = (E.T @ Tyy @ E) * (B.T @ B)
    lams = torch.from_numpy(LAMBDAS).to(s) * s.mean()
    V = 1.0 / (s[None] + lams[:, None])                                     # one row per penalty
    errs = torch.trace(Txx) - 2 * V @ cross + ((V @ quad) * V).sum(1)
    i = int(torch.argmin(errs))
    M = (B / (s + lams[i])) @ E.T
    return M, mx, float(1 - errs[i] / torch.trace(Txx - torch.outer(hx - mx, hx - mx))), LAMBDAS[i]


@torch.no_grad()
def match(pair, x, y, k, test, scales=(1.0,)):
    """Log-probability and top-1 rate of the matched partner position, averaged over heads and matched tokens of the
    test texts, for reader x and sender y at exchange k, with the keys scaled by each of scales (the null key is not)."""
    rx, sy = pair.bridge[x], pair.bridge[y]
    H, hd = rx.heads, rx.head_dim
    lp, top = np.zeros(len(scales)), np.zeros(len(scales))
    n, chance = 0, 0.0
    for t in test:
        zx = rx.q_norm[k](t[x][k].to(DEVICE).float())
        zy = sy.kv_norm[k](t[y][k].to(DEVICE).float())
        q = rx.q[k](zx).view(-1, H, hd).transpose(0, 1)                        # [H, Lx, hd]
        kk = sy.k[k](zy).view(-1, H, hd).transpose(0, 1)                       # [H, Ly, hd]
        ix, iy = t['pos'][x].to(DEVICE), t['pos'][y].to(DEVICE)
        q = q[:, ix]
        lk = q @ kk.transpose(1, 2) / math.sqrt(hd)                            # [H, m, Ly]
        ln = (q @ rx.null[k][:, :, None]).squeeze(-1)[..., None] / math.sqrt(hd)
        for j, c in enumerate(scales):
            p = torch.cat([ln, c * lk], -1).log_softmax(-1)
            lp[j] += p.gather(-1, (iy + 1)[None, :, None].expand(H, -1, 1)).mean(0).sum().item()
            top[j] += (p.argmax(-1) == iy + 1).float().mean(0).sum().item()
        n += len(ix)
        chance += len(ix) / zy.shape[0]
    return lp / n, top / n, chance / n


@torch.no_grad()
def content_init(pair, models, toks, replies, n_texts, vo=False, n_test=6, log=print):
    """Set the reads of pair (see the module's docstring) from n_texts texts, n_test of them outside the fit. Returns a
    summary per exchange and direction."""
    a, b = SIDES
    items = texts(replies, n_texts)
    eps = pair.bridge[a].q_norm[0].eps
    d = {x: models[x].config.hidden_size for x in SIDES}
    nb = len(pair.xblocks)

    def zeros():
        return [dict(aa=torch.zeros(d[a], d[a], device=DEVICE), bb=torch.zeros(d[b], d[b], device=DEVICE),
                     ab=torch.zeros(d[a], d[b], device=DEVICE), sa=torch.zeros(d[a], device=DEVICE),
                     sb=torch.zeros(d[b], device=DEVICE), n=0) for _ in range(nb)]
    sums, test_sums, test = zeros(), zeros(), []
    for i, item in enumerate(items):
        seq = {x: sequence(toks[x], x, item) for x in SIDES}
        ends = sorted(set(seq[a][1]) & set(seq[b][1]))
        if len(ends) < 2:
            continue
        pos = {x: torch.tensor([seq[x][1][e] for e in ends], device=DEVICE) for x in SIDES}
        U = {x: states(models[x], pair.xblocks, seq[x][0], eps) for x in SIDES}
        held = i >= len(items) - n_test
        for k, acc in enumerate(test_sums if held else sums):
            ua, ub = U[a][k][pos[a]], U[b][k][pos[b]]
            acc['aa'] += ua.T @ ua
            acc['bb'] += ub.T @ ub
            acc['ab'] += ua.T @ ub
            acc['sa'] += ua.sum(0)
            acc['sb'] += ub.sum(0)
            acc['n'] += len(ends)
        if held:
            test.append({x: [u.half().cpu() for u in U[x]] for x in SIDES} | {'pos': {x: pos[x].cpu() for x in SIDES}})
    assert test and sums[0]['n'] > 0, 'no matched tokens'
    out = []
    for k in range(nb):
        G = {kk: (v.to(LA, torch.float64) if torch.is_tensor(v) else float(v)) for kk, v in sums[k].items()}
        T = {kk: (v.to(LA, torch.float64) if torch.is_tensor(v) else float(v)) for kk, v in test_sums[k].items()}
        for x, y in ((a, b), (b, a)):
            rx, sy = pair.bridge[x], pair.bridge[y]
            flip = x != a
            Gd = dict(xx=G['bb' if flip else 'aa'], yy=G['aa' if flip else 'bb'], xy=G['ab'].T if flip else G['ab'],
                      sx=G['sb' if flip else 'sa'], sy=G['sa' if flip else 'sb'], n=G['n'])
            Td = dict(xx=T['bb' if flip else 'aa'], yy=T['aa' if flip else 'bb'], xy=T['ab'].T if flip else T['ab'],
                      sx=T['sb' if flip else 'sa'], sy=T['sa' if flip else 'sb'], n=T['n'])
            gq = rx.q_norm[k].weight.detach().to(LA, torch.float64)
            gk = sy.kv_norm[k].weight.detach().to(LA, torch.float64)
            before = match(pair, x, y, k, test)
            M, mx, r2, lam = fit(Gd, Td, gq, gk)
            u = mx / mx.norm()
            Q = rx.q[k].weight.detach().to(LA, torch.float64)
            Q = Q - torch.outer(Q @ u, u)
            rx.q[k].weight.copy_(Q)
            sy.k[k].weight.copy_(Q @ M)
            lp, top, chance = match(pair, x, y, k, test, SCALES)
            j = int(np.argmax(lp))
            sy.k[k].weight.mul_(float(SCALES[j]))
            if vo:
                w = rx.heads * rx.head_dim
                U_, s_, Vt = svd(M)
                r = s_[:w].sqrt()
                sy.v[k].weight.copy_(r[:, None] * Vt[:w])
                rx.o[k].weight.copy_(U_[:, :w] * r)
            rec = dict(block=pair.xblocks[k], reader=x, sender=y, pairs=int(G['n']), r2=round(float(r2), 4),
                       ridge=float(lam), scale=float(SCALES[j]), chance=round(chance, 4),
                       before=dict(logp=round(float(before[0][0]), 3), top1=round(float(before[1][0]), 4)),
                       after=dict(logp=round(float(lp[j]), 3), top1=round(float(top[j]), 4)))
            log(f"content init: block {rec['block']} {y} -> {x}: test R^2 {rec['r2']}, scale {rec['scale']:g}, "
                f"matched top-1 {rec['before']['top1']} -> {rec['after']['top1']} (chance {rec['chance']}), "
                f"log p {rec['before']['logp']} -> {rec['after']['logp']}")
            out.append(rec)
    return out
