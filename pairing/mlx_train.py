"""Episode objective and CLI for mlx_pair.py; imported only for training."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten, tree_unflatten
import numpy as np

from mlx_pair import (SIDES, HF, XBLOCKS, Pair, add_lora, load_base, adapters_off,
                      causal_mask, convert_batch, export_checkpoint, load_checkpoint, residual_rms, remat)
import games as G
import session as S
from link import MODELS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replies', required=True)
    p.add_argument('--out', required=True)
    for name, default in dict(steps=3000, bs=16, rank=32, heads=8, head_dim=128, seed=0,
                              save_every=1000, max_digits=16, coread_first=0, warmup_steps=100,
                              relay_first=0, comp=5, halt=0, microbatch=1, loss_chunk=32,
                              payload_every=5, payload_eval=0, eval_every=500, max_seq=0).items():
        p.add_argument('--' + name, type=int, default=default)
    for name, default in dict(lr=1e-4, lr_bridge=3e-4, lr_alpha=3e-3, read_start=0.,
                              send_cost=.01, warm=.2, ramp=.2, dialogue=0., coread=0.,
                              coread_first_p=.8, anchor=.1, qa=0., payload_w=1.,
                              memory_gib=10.5).items():
        p.add_argument('--' + name, type=float, default=default)
    p.add_argument('--xblocks', default=','.join(map(str, XBLOCKS)))
    p.add_argument('--coread_data')
    p.add_argument('--init')
    p.add_argument('--alpha_init', type=float)
    p.add_argument('--gate_norm', action='store_true')
    p.add_argument('--solo', action='store_true')
    p.add_argument('--text_pair', action='store_true')
    p.add_argument('--ckpt', choices=('both', 'none', 'quill', 'ember'), default='both',
                   help='Any non-none value rematerializes both sides to bound memory')
    for x in SIDES:
        p.add_argument('--lr_bridge_' + x, type=float)
    return p


def lr_multiplier(step, steps, warmup):
    return min(1., step / max(1, warmup)) * (.1 + .45 * (1 + math.cos(math.pi * step / steps)))


def relay_episode(games):
    ep = games.episode('relay_word')
    s, r, value = (ep['meta'][k] for k in ('sender', 'receiver', 'value'))
    ep['task'][s] = f'Your secret word is "{value}". Pass it to {MODELS[r]["name"]} through the link.'
    ep['reply'][s] = ('', 0.)
    ep['reply'][r] = (f'The word is {value}.', 1.)
    return ep


def adversary(ep):
    s, r = ep['meta']['sender'], ep['meta']['receiver']
    if ep['mode'] == 'duplex':
        return dict(ep, reply={s: (ep['reply'][s][0], 0.), r: (ep['meta']['adversary'], 1.)})
    *head, (sp, _, _) = ep['turns']
    return dict(ep, turns=[(a, t, 0.) for a, t, _ in head] + [(sp, ep['meta']['adversary'], 1.)])


class Episodes:
    def __init__(self, args, toks, replies):
        self.a, self.toks = args, toks
        self.rng = random.Random(args.seed)
        self.games = G.Games(replies, 'train', seed=args.seed, max_digits=args.max_digits)
        self.co = None
        if args.coread > 0 or (args.coread_first > 0 and args.coread_first_p > 0):
            from coread import Coread
            self.co = Coread(replies, 'train', seed=args.seed, path=args.coread_data)
        self.bench = G.bench('train') if args.qa or args.solo or args.text_pair else None
        self.pads = {x: toks[x].pad_token_id for x in SIDES}

    def draw(self, step):
        a, rng, games = self.a, self.rng, self.games
        if step <= a.relay_first and not (a.solo or a.text_pair):
            return relay_episode(games)
        if a.solo:
            return G.qa_episode(rng.choice(self.bench), 'solo')
        if a.text_pair:
            return G.qa_episode(rng.choice(self.bench), 'text', order=rng.sample(SIDES, 2))
        if rng.random() < a.anchor:
            return games.episode(rng.choice(('identity', 'plain')))
        if self.co and rng.random() < (a.coread_first_p if step <= a.coread_first else a.coread):
            return self.co.episode()
        if rng.random() < a.qa:
            return G.qa_episode(rng.choice(self.bench), 'pair')
        if rng.random() < a.dialogue:
            return games.episode(rng.choice(G.DIALOGUE))
        return games.episode()

    def build(self, ep):
        e = S.build(ep, self.toks)
        for x, js in S.payload_labels(ep, self.toks).items():
            e[x]['payload'] = js
            for j in js:
                pos, tid, wt = e[x]['labels'][j]
                e[x]['labels'][j] = (pos, tid, wt * self.a.payload_w)
        for x, prompt in ep['meta'].get('teacher', {}).items():
            p = S.prefix(self.toks[x], x, f"You are {MODELS[x]['name']}, {MODELS[x]['about']}.", prompt)
            p += S.assistant_header(self.toks[x], x)
            r = S.ids(self.toks[x], ep['reply'][x][0]) + [self.toks[x].convert_tokens_to_ids('<|im_end|>')]
            e[x]['teacher'] = (p + r[:-1], len(p) - 1)
            assert len(r) == len(e[x]['labels'])
        if self.a.max_seq and max(len(e[x]['ids']) for x in SIDES) > self.a.max_seq:
            raise ValueError('Episode exceeds --max_seq; refusing to truncate or silently change the episode distribution')
        return e

    def prepare(self, step):
        a = self.a
        competitive = a.comp > 0 and step % a.comp == 0 and step > a.relay_first
        if competitive:
            rows = []
            names = G.COMPETITIVE + (G.DIALOGUE_COMPETITIVE if a.dialogue else ())
            for _ in range(a.bs // 2):
                ep = self.games.episode(self.rng.choice(names))
                s, r = ep['meta']['sender'], ep['meta']['receiver']
                rows += [(ep, {s: frozenset([s]), r: frozenset([r])}),
                         (adversary(ep), {s: frozenset(), r: frozenset([s])})]
        else:
            rows = [(self.draw(step), {x: frozenset(SIDES) for x in SIDES}) for _ in range(a.bs)]
        return competitive, rows, [self.build(ep) for ep, _ in rows]


def distill(pair, built, pads, chunk=32):
    """Original quantized base, adapters off and no bridge; exact full-vocab KL targets."""
    out = {}
    for x in SIDES:
        seqs, sel, off = [], [], 0
        for e in built:
            n = len(e[x]['labels'])
            if 'teacher' in e[x]:
                seqs.append((*e[x]['teacher'], n))
                sel.extend(range(off, off + n))
            off += n
        if not seqs:
            continue
        L = max(len(t) for t, _, _ in seqs)
        ids = np.full((len(seqs), L), pads[x], dtype=np.int32)
        att = np.zeros_like(ids)
        idx = []
        for k, (t, start, n) in enumerate(seqs):
            ids[k, :len(t)], att[k, :len(t)] = t, 1
            idx.extend(k * L + start + j for j in range(n))
        b = dict(ids=mx.array(ids), att=mx.array(att))
        model = pair.models[x]
        with adapters_off(model):
            h = model.model.embed_tokens(b['ids'])
            mask = causal_mask(b)
            for layer in model.layers:
                h = layer(h, mask)
                mx.eval(h)
            h = model.model.norm(h).reshape(-1, h.shape[-1])[mx.array(idx)]
            logs = []
            for start in range(0, len(idx), chunk):
                logits = pair.token_logits(x, h[start:start + chunk])
                lt = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                mx.eval(lt)
                logs.append(lt)
            lt = mx.stop_gradient(mx.concatenate(logs))
            mx.eval(lt)
        out[x] = (mx.array(sel, dtype=mx.int32), lt)
    return out


def objective(pair, b, kl, denom, n_show, send_cost, chunk=32, owner=None, mode=None):
    last, gates, writes = pair.joint(b, mode)
    total, stats = mx.array(0.), {}
    for x in SIDES:
        bx = b[x]
        selected = last[x].reshape(-1, last[x].shape[-1])[bx['idx']]
        ce_parts, hit_parts = [], []
        # Checkpoint the head/loss, so vocabulary-sized logits are never retained for every label.
        for start in range(0, bx['lab'].size, chunk):
            end = min(start + chunk, bx['lab'].size)
            teacher = None
            valid = None
            if x in kl:
                sel, lt = kl[x]
                match = mx.arange(start, end)[:, None] == sel[None, :]
                valid = mx.any(match, axis=-1)
                teacher = lt[mx.argmax(match, axis=-1)]
            def token_loss(h, labels, teacher, valid, x=x):
                labels = mx.stop_gradient(labels)
                logits = pair.token_logits(x, h)
                ce = nn.losses.cross_entropy(logits, labels, reduction='none')
                if teacher is not None:
                    teacher, valid = mx.stop_gradient(teacher), mx.stop_gradient(valid)
                    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    kd = mx.sum(mx.exp(teacher) * (teacher - lp), axis=-1)
                    ce = mx.where(valid, kd, ce)
                return ce, mx.stop_gradient((mx.argmax(logits, axis=-1) == labels).astype(mx.float32))
            ce, hit = mx.checkpoint(token_loss)(selected[start:end], bx['lab'][start:end], teacher, valid)
            ce_parts.append(ce); hit_parts.append(hit)
        ce = mx.concatenate(ce_parts) if ce_parts else mx.zeros((0,))
        hit = mx.concatenate(hit_parts) if hit_parts else mx.zeros((0,))
        w = bx['w'] if owner is None else bx['w'] * bx['own_mask'][owner]
        total = total + mx.sum(ce * w) / denom[x]
        g = gates[x] * bx['att'][..., None]
        if owner is None or owner == x:
            total = total + send_cost * mx.sum(g) / n_show[x]
        stats[x] = dict(ce=mx.stop_gradient(ce), hit=hit, gate=mx.stop_gradient(mx.sum(g)),
                        sent=mx.stop_gradient(mx.sum(g > .5)), write=writes[x])
    return total, stats


def make_optimizer(args):
    overrides = [getattr(args, 'lr_bridge_' + x) for x in SIDES]
    rates = [args.lr_alpha] + [args.lr_bridge if r is None else r for r in overrides] + [args.lr]
    opts = [optim.AdamW(r, betas=[.9, .98], weight_decay=0., bias_correction=True) for r in rates]
    filters = [lambda n, _: n.endswith('.alpha')]
    filters += [lambda n, _, x=x: n.startswith('bridge.' + x + '.') for x in SIDES]
    return optim.MultiOptimizer(opts, filters), rates


def payload_losses(pair, b, built, rows, chunk):
    indices, names = {}, {}
    for x in SIDES:
        indices[x], names[x], off = [], [], 0
        for (ep, owner), e in zip(rows, built):
            if owner[x] == frozenset(SIDES):
                js = e[x].get('payload', [])
                indices[x].extend(off + j for j in js)
                names[x].extend([ep['game'] + '/' + x] * len(js))
            off += len(e[x]['labels'])
    if not any(indices.values()):
        return {}
    out = {}
    for mode in (None, 'zero'):
        h, _, _ = pair.joint(b, mode)
        for x in SIDES:
            if not indices[x]:
                continue
            ids = mx.array(indices[x])
            for start in range(0, len(indices[x]), chunk):
                ix = ids[start:start + chunk]
                logits = pair.token_logits(x, h[x], b[x]['idx'][ix])
                ce = nn.losses.cross_entropy(logits, b[x]['lab'][ix], reduction='none').tolist()
                for name, val in zip(names[x][start:start + chunk], ce):
                    s = out.setdefault(name, [0., 0., 0])
                    s[mode == 'zero'] += val
                    if mode is None:
                        s[2] += 1
        mx.eval(h)
    return out


def payload_eval(pair, toks, replies, args, step):
    import payload_ctl
    eps = payload_ctl.episodes(replies, args.payload_eval, args.max_digits,
                               coread=args.coread > 0 or args.coread_first > 0, path=args.coread_data)
    records = []
    pads = {x: toks[x].pad_token_id for x in SIDES}
    for game in sorted(eps):
        for start, end in S.batches(len(eps[game]), max(2, args.microbatch)):
            batch = eps[game][start:end]
            built = [S.build(ep, toks) for ep in batch]
            payload = [S.payload_labels(ep, toks) for ep in batch]
            b = convert_batch(S.collate(built, pads))
            vals = {}
            for mode in (None, 'zero', 'shuffle'):
                h, _, _ = pair.joint(b, mode)
                for x in SIDES:
                    off = 0
                    for i, e in enumerate(built):
                        js = payload[i].get(x, [])
                        if js:
                            ix = mx.array([off + j for j in js])
                            ce = []
                            for j in range(0, len(js), args.loss_chunk):
                                jj = ix[j:j + args.loss_chunk]
                                logits = pair.token_logits(x, h[x], b[x]['idx'][jj])
                                ce.extend(nn.losses.cross_entropy(logits, b[x]['lab'][jj], reduction='none').tolist())
                            key = {None: 'link_ce', 'zero': 'cut_ce', 'shuffle': 'shuffle_ce'}[mode]
                            vals.setdefault((i, x), {})[key] = float(np.mean(ce))
                        off += len(e[x]['labels'])
                mx.eval(h)
            for (i, x), v in vals.items():
                records.append(dict(game=game, episode=start + i, receiver=x,
                                    n_tokens=len(payload[i][x]), **v))
    Path(args.out, f'payload_step{step}.json').write_text(json.dumps(records))
    return payload_ctl.summary(records)


def main():
    args = parser().parse_args()
    if args.bs < 1 or args.microbatch < 1 or args.loss_chunk < 1 or args.steps < 1:
        raise ValueError('Batch, microbatch, loss chunk and steps must be positive')
    if args.comp and args.bs % 2:
        raise ValueError('Competition requires an even --bs')
    if not 0 < args.memory_gib <= 11:
        raise ValueError('--memory_gib must be in (0,11], leaving CPU memory below the 12 GiB process budget')
    if args.warmup_steps < 0 or args.relay_first < 0:
        raise ValueError('Warmup/curriculum lengths must be nonnegative')
    if args.solo or args.text_pair:
        args.comp, args.send_cost, args.anchor = 0, 0., 0.
    mx.set_memory_limit(int(args.memory_gib * 2**30))
    mx.set_cache_limit(64 * 2**20)
    mx.random.seed(args.seed)
    import torch
    torch.set_num_threads(2)
    from transformers import AutoTokenizer
    toks, models = {}, {}
    for x in SIDES:
        models[x], _ = load_base(HF[x])
        toks[x] = AutoTokenizer.from_pretrained(HF[x], local_files_only=True)
        models[x] = add_lora(models[x], args.rank)
        mx.eval(models[x].parameters())
        print(json.dumps(dict(loaded=x, active_gib=mx.get_active_memory() / 2**30)), flush=True)
    replies = json.loads(Path(args.replies).read_text())
    episodes = Episodes(args, toks, replies)
    xb = tuple(map(int, args.xblocks.split(',')))
    sizes = None
    if not args.init:
        g = G.Games(replies, 'train', seed=args.seed + 99, max_digits=args.max_digits)
        seqs = [S.build(g.episode('plain'), toks) for _ in range(8)]
        sizes = {x: residual_rms(models[x], xb, [e[x]['ids'] for e in seqs]) for x in SIDES}
        print(json.dumps(dict(sizes=sizes)), flush=True)
    pair = Pair(models, xb, sizes, args.heads, args.head_dim, args.gate_norm, args.ckpt != 'none')
    if args.init and load_checkpoint(pair, args.init):
        raise SystemExit('mlx_train.py trains both twins; --init is a one-way checkpoint (train.py --frozen trains those)')
    for side in pair.bridge.values():
        if args.gate_norm:
            if args.init and not side.gate_norm:
                for g in side.gate:
                    g.weight = mx.zeros_like(g.weight)
            side.gate_norm = True
        if args.read_start:
            for o in side.o:
                o.weight = mx.zeros_like(o.weight)
            side.alpha = mx.full(side.alpha.shape, math.atanh(args.read_start))
        if args.alpha_init is not None:
            side.alpha = mx.full(side.alpha.shape, math.atanh(args.alpha_init))
    mx.eval(pair.parameters())
    opt, rates = make_optimizer(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.payload_eval:
        print(json.dumps(dict(step=0, payload_eval=payload_eval(pair, toks, replies, args, 0))), flush=True)
    log = []
    t0 = time.perf_counter()
    print(json.dumps(dict(startup_peak_gib=mx.get_peak_memory() / 2**30)), flush=True)
    for step in range(1, args.steps + 1):
        ts = time.perf_counter()
        competitive, rows, built = episodes.prepare(step)
        denom = {x: max(1., sum(w for e in built for _, _, w in e[x]['labels'])) for x in SIDES}
        n_show = {x: max(1, sum(len(e[x]['ids']) for e in built) * len(xb)) for x in SIDES}
        frac = step / args.steps
        scale = 0. if step <= args.relay_first or frac <= args.warm else min(1., (frac - args.warm) / args.ramp) if args.ramp > 0 else 1.
        for o, lr in zip(opt.optimizers, rates):
            o.learning_rate = lr * lr_multiplier(step, args.steps, args.warmup_steps)
        grad, loss, stats, payload = None, 0., {}, {}
        sent = {x: 0. for x in SIDES}; gates = dict(sent)
        writes = {x: np.zeros(len(xb)) for x in SIDES}
        write_den = {x: 0 for x in SIDES}
        for start in range(0, len(rows), args.microbatch):
            rr, bb = rows[start:start + args.microbatch], built[start:start + args.microbatch]
            b = convert_batch(S.collate(bb, episodes.pads, [o for _, o in rr]))
            kl = distill(pair, bb, episodes.pads, args.loss_chunk)
            owners = SIDES if competitive else (None,)
            local_grad = None
            for owner in owners:
                def fn(m):
                    return objective(m, b, kl, denom, n_show, scale * args.send_cost, args.loss_chunk,
                                     owner, 'zero' if args.solo or args.text_pair else None)
                (v, st), gg = nn.value_and_grad(pair, fn)(pair)
                mx.eval(v, st, gg)
                loss += float(v)
                if owner is not None:
                    gg = tree_unflatten([(k, g) for k, g in tree_flatten(gg)
                                         if k.startswith('models.' + owner + '.') or k.startswith('bridge.' + owner + '.')])
                if local_grad is None:
                    local_grad = gg
                else:
                    local_grad = tree_unflatten(tree_flatten(local_grad) + tree_flatten(gg))
            grad = local_grad if grad is None else tree_map(lambda a, b: a + b, grad, local_grad)
            mx.eval(grad)
            for x in SIDES:
                sent[x] += float(st[x]['sent']) / n_show[x]
                gates[x] += float(st[x]['gate']) / n_show[x]
                qm = b[x]['att'].astype(mx.bool_) & (b[x]['tick'] < b[x]['cut'][:, None])
                count = int(mx.sum(qm))
                writes[x] += np.array(st[x]['write']) * count
                write_den[x] += count
                ce, hit = np.array(st[x]['ce']), np.array(st[x]['hit'])
                row, w = np.array(b[x]['row']), np.array(b[x]['w'])
                for i, (ep, own) in enumerate(rr):
                    mask = (row == i) & (w >= 1.)
                    if not own[x] or not mask.any():
                        continue
                    game = ep['game'] + ('/adv' if own[x] != frozenset(SIDES) and x not in own[x] else '')
                    z = stats.setdefault(game, [0., 0., 0])
                    z[0] += float(hit[mask].mean()); z[1] += float(ce[mask].mean()); z[2] += 1
            if args.payload_every and (step - 1) % args.payload_every == 0 and not competitive:
                for name, val in payload_losses(pair, b, bb, rr, args.loss_chunk).items():
                    z = payload.setdefault(name, [0., 0., 0])
                    for j in range(3):
                        z[j] += val[j]
            del gg, local_grad, kl, st, b
        grad, norm = optim.clip_grad_norm(grad, 1.)
        opt.update(pair, grad)
        mx.eval(pair.trainable_parameters(), opt.state, norm)
        del grad
        mx.clear_cache()
        line = dict(step=step, loss=loss, alpha={x: mx.tanh(pair.bridge[x].alpha).tolist() for x in SIDES},
                    gate=gates, sent=sent, write={x: np.sqrt(writes[x] / max(1, write_den[x])).tolist() for x in SIDES},
                    games={k: [z[0] / z[2], z[1] / z[2]] for k, z in stats.items()},
                    payload={k: [z[0] / z[2], z[1] / z[2], z[2]] for k, z in payload.items()},
                    cost=scale, grad_norm=float(norm), time=time.perf_counter() - t0,
                    seconds=time.perf_counter() - ts, mem=mx.get_peak_memory() / 2**30,
                    active_gib=mx.get_active_memory() / 2**30,
                    competitive=competitive)
        if not math.isfinite(loss) or not math.isfinite(float(norm)):
            raise FloatingPointError('Nonfinite loss/gradient')
        print(json.dumps(line), flush=True)
        log.append(line)
        with (out / 'log.jsonl').open('a') as f:
            f.write(json.dumps(line) + '\n')
        if args.payload_eval and (step % args.eval_every == 0 or step == args.steps):
            pe = payload_eval(pair, toks, replies, args, step)
            print(json.dumps(dict(step=step, payload_eval=pe)), flush=True)
            log.append(dict(step=step, payload_eval=pe))
        if step % args.save_every == 0 or step in (args.steps, args.halt):
            export_checkpoint(pair, out / f'step_{step}', vars(args))
            (out / 'log.json').write_text(json.dumps(log))
            print(json.dumps(dict(saved=str(out / f'step_{step}'))), flush=True)
        if step == args.halt:
            break
