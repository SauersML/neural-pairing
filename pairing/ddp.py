"""Data parallel helpers for runs under torchrun (one process per GPU; a no-op in a single process).

Every rank draws the same global batch (same seeds), sorts its episodes by length and keeps one contiguous group
(partition), so each rank pads only to the length of similar episodes and the groups cost about the same. Losses use
the denominators of the whole global batch, so the sum of the ranks' gradients (sum_grads) equals the gradient of the
single-process loss on that batch.
"""
import datetime
import os

import torch
import torch.distributed as dist

from link import DEVICE

WORLD = int(os.environ.get('WORLD_SIZE', 1))
RANK = int(os.environ.get('RANK', 0))
MAIN = RANK == 0


def init():
    if WORLD > 1 and not dist.is_initialized():
        nccl = DEVICE.startswith('cuda')
        if nccl:
            torch.cuda.set_device(DEVICE)
        dist.init_process_group('nccl' if nccl else 'gloo', timeout=datetime.timedelta(minutes=120))


def broadcast(params):
    """Copies rank 0's values of params to every rank."""
    if WORLD > 1:
        for p in params:
            dist.broadcast(p.data, 0)


def sum_grads(params):
    """Sums the gradients of params over the ranks in one flat buffer; a missing gradient counts as zeros."""
    if WORLD == 1:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])
    dist.all_reduce(flat)
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).to(p.dtype)
        i += n


def sum_tensor(t):
    if WORLD > 1:
        dist.all_reduce(t)
    return t


def gather(obj):
    """The list of every rank's obj (on every rank)."""
    if WORLD == 1:
        return [obj]
    out = [None] * WORLD
    dist.all_gather_object(out, obj)
    return out


def partition(costs, n=WORLD):
    """Splits item indices into n non-empty groups, contiguous in order of cost, minimizing the largest
    (group size x largest cost in the group), the padded work of the slowest rank."""
    order = sorted(range(len(costs)), key=costs.__getitem__)
    c = [costs[i] for i in order]
    N = len(c)
    assert N >= n, (N, n)
    inf = float('inf')
    best = [[inf] * (N + 1) for _ in range(n + 1)]
    cut = [[0] * (N + 1) for _ in range(n + 1)]
    best[0][0] = 0
    for k in range(1, n + 1):
        for j in range(k, N + 1):
            for i in range(k - 1, j):
                v = max(best[k - 1][i], (j - i) * c[j - 1])
                if v < best[k][j]:
                    best[k][j], cut[k][j] = v, i
    groups, j = [], N
    for k in range(n, 0, -1):
        i = cut[k][j]
        groups.append(order[i:j])
        j = i
    return groups[::-1]


def assign(costs, n=WORLD):
    """Longest-processing-time assignment of work units to n ranks: list of unit indices per rank."""
    load, out = [0.0] * n, [[] for _ in range(n)]
    for i in sorted(range(len(costs)), key=lambda i: -costs[i]):
        r = min(range(n), key=load.__getitem__)
        out[r].append(i)
        load[r] += costs[i]
    return [sorted(o) for o in out]
