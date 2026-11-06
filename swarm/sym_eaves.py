"""Eavesdroppers per game on a sym run's notes. On n_test fresh test-value episodes of each game (rooms, objects and
values never seen in training), the trained writers' notes (hard: real vocabulary tokens) are read by three kinds of
reader, each scored by the say NLL with the episode's own notes (on), with every note masked (cut) and with another
episode of the same game's notes (swap), and by forced choice among each say's candidates where the game has them
(crumb_games.choices; own and swapped notes):
  trained   the run's follower (its adapter)
  base      the base model with the adapter disabled, reading the same notes in the same prompt (crumbs.py's base and
            base_off modes): what an untrained reader gets from them
  crack     a fresh rank-16 LoRA on the base model (the run's eavesdropper adapter, re-initialised), trained --steps
            steps on the follower's says with the writers' notes of N training-value episodes of the game, for each N
            in --sizes: what an eavesdropper that sees the notes and the answers can learn to read
and run_eaves, the eavesdropper adapter the run trained against during privacy, where it has one. A reader gains
episode-specific information from the notes only if its swap gap (swap minus on) is above 0; the cut gap (cut minus
on) also counts any note being there. Each gap has a 95% bootstrap interval over episodes. Writes one JSON (--out).
Usage: python swarm/sym_eaves.py --run RUN --out OUT_JSON [--ckpts trained.pt,gate.pt --crack_games trail,trap,count,work
       --n_test 96 --sizes 64,1024 --steps 300]"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402
from swarm.sym_games import interval                    # noqa: E402

CRACK_GAMES = ('trail', 'trap', 'count', 'work')


def episodes(game, n, split):
    rng = random.Random(f'eaves-{game}-{split}')
    return [CG.episode(game, rng, split == 'test') for _ in range(n)]


def fresh_eaves(net, seed=0):
    """The eavesdropper adapter re-initialised (LoRA A random, B zero: it reads as the base model until trained)."""
    torch.manual_seed(seed)
    for mod in net.model.modules():
        if hasattr(mod, 'lora_A') and 'eaves' in mod.lora_A:
            mod.reset_lora_parameters('eaves', True)


def trained_eaves(net):
    return any(float(p.abs().sum()) > 0 for n, p in net.model.named_parameters() if '.eaves.' in n and 'lora_B' in n)


@torch.no_grad()
def banks(net, device, eps, bs=48):
    """eps in chunks (C.swap_index: no chunk of one), each with the trained writers' hard notes."""
    net.eval()
    return [([eps[i] for i in c], net.run([eps[i] for i in c], device, need_loss=False)[2])
            for c in C.swap_index(eps, bs)]


@torch.no_grad()
def read(net, device, chunks, reader):
    """Per episode: the say NLL with its own notes, every note masked and the next episode's notes (on, off, swap),
    and the forced-choice share with own and swapped notes, for reader trained, base, or crack (the eavesdropper
    adapter)."""
    net.eval()
    on, off = ('base', 'base_off') if reader == 'base' else ('on', 'off')
    out = dict(on=[], off=[], swap=[], choice_on=[], choice_swap=[])
    with net._eaves() if reader == 'crack' else contextlib.nullcontext():
        for eps, bank in chunks:
            for key, mode, b in (('on', on, bank), ('off', off, bank), ('swap', on, C.shifted(bank))):
                nll, choice = net.forced(eps, device, mode, choose=key != 'off', bank=b)
                out[key] += nll
                if key != 'off':
                    out[f'choice_{key}'] += choice
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summary(v):
    rows = [i for i, x in enumerate(v['on']) if x is not None]
    swap = [v['swap'][i] - v['on'][i] for i in rows]
    cut = [v['off'][i] - v['on'][i] for i in rows]
    return dict(episodes=len(rows), on=mean([v['on'][i] for i in rows]), cut=mean([v['off'][i] for i in rows]),
                swap=mean([v['swap'][i] for i in rows]), gap_swap=mean(swap), gap_swap_ci=interval(swap) if swap else None,
                gap_cut=mean(cut), gap_cut_ci=interval(cut) if cut else None, choice_on=mean(v['choice_on']),
                choice_swap=mean(v['choice_swap']))


def say_nll(net, eps, device, bank):
    """The mean NLL per say (its tokens and <|im_end|>) of every reader in eps with bank's notes, with gradient: the
    active adapter reads."""
    total, says = 0., 0
    for a in range(max(len(ep['agents']) for ep in eps)):
        rows = [b for b, ep in enumerate(eps)
                if a < len(ep['agents']) and any(sg['say'] is not None for sg in ep['agents'][a]['segments'])]
        if not rows:
            continue
        x = net._pad([net.stream(eps[b], a) for b in rows], device)
        sel = x['target'][:, 1:].nonzero()
        net.grab = True
        try:
            net._forward(torch.tensor(rows, device=device), x, a, bank, 1, False, True)
            h = net.last
        finally:
            net.grab, net.last = False, None
        logits = net.model.get_output_embeddings()(h[sel[:, 0], sel[:, 1]]).float()
        lp = logits.log_softmax(-1).gather(1, x['ids'][sel[:, 0], sel[:, 1] + 1][:, None])[:, 0]
        seg = x['seg'][sel[:, 0], sel[:, 1] + 1]
        total = total - lp.sum()
        says += len(set(zip(sel[:, 0].tolist(), seg.tolist())))
    return total / max(1, says)


def crack(net, device, game, n, test, steps=300, batch=8, micro=4, lr=5e-4, seed=0):
    """A fresh eavesdropper adapter trained on n training-value episodes of game (the writers' hard notes, the
    follower's says), then read() on the test chunks."""
    fresh_eaves(net, seed)
    pairs = episodes(game, n, 'train')
    rng = random.Random(seed)
    params = net.eaves_params()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = min(20, steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(1, s / steps))))
    net.eval()
    first = last = None
    for _ in range(steps):
        part = [rng.choice(pairs) for _ in range(batch)]
        loss_v = 0.
        for i in range(0, batch, micro):
            sub = part[i:i + micro]
            with torch.no_grad():
                bank = net.run(sub, device, need_loss=False)[2]
            with net._eaves(train=True):
                loss = say_nll(net, sub, device, bank) * len(sub) / batch
                loss.backward()
            loss_v += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        first = loss_v if first is None else first
        last = loss_v
    out = summary(read(net, device, test, 'crack'))
    out.update(pairs=len(pairs), steps=steps, train_loss_first=round(first, 4), train_loss_last=round(last, 4))
    return out


def score(net, device, games, crack_games, n_test, sizes, steps, batch=8, micro=4, bs=48, res=None, log=None):
    """res filled with {game: {trained, base, [run_eaves], [crack: {N: ...}]}} on n_test test-value episodes per game;
    log(game, reader, its summary) after each."""
    res = {} if res is None else res
    log = log or (lambda *a: None)
    own = trained_eaves(net)
    for g in games:
        test = banks(net, device, episodes(g, n_test, 'test'), bs)
        r = res[g] = {}
        for reader in ('trained', 'base') + (('run_eaves',) if own else ()):
            r[reader] = summary(read(net, device, test, 'crack' if reader == 'run_eaves' else reader))
            log(g, reader, r[reader])
        if g in crack_games:
            saved = {k: v.detach().clone() for k, v in net.state_dict().items() if '.eaves.' in k}
            r['crack'] = {}
            for n in sizes:
                r['crack'][n] = crack(net, device, g, n, test, steps, batch, micro)
                log(g, f'crack N={n}', r['crack'][n])
            net.load_state_dict(saved, strict=False)            # the run's own eavesdropper back
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a crumbs.py sym run folder (args.json and its checkpoints)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpts', default='trained.pt,gate.pt')
    ap.add_argument('--games', default='', help='games to read (default: the run\'s)')
    ap.add_argument('--crack_games', default=','.join(CRACK_GAMES))
    ap.add_argument('--n_test', type=int, default=96, help='test-value episodes per game')
    ap.add_argument('--sizes', default='64,1024', help='training episodes per cracker')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--micro', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    run = Path(args.run)
    games = args.games.split(',') if args.games else \
        list(C.expand_games(json.loads((run / 'args.json').read_text())['games'], {})[0])
    crack_games = [g for g in args.crack_games.split(',') if g in games]
    sizes = [int(n) for n in args.sizes.split(',')]
    res = dict(run=str(run), n_test=args.n_test, sizes=sizes, steps=args.steps, games=games, crack_games=crack_games,
               ckpts={})
    started = time.monotonic()

    def save():
        res['s'] = round(time.monotonic() - started)
        Path(args.out).write_text(json.dumps(res, indent=1))

    for ck in args.ckpts.split(','):
        if not (run / ck).exists():
            continue
        net = C.load(run, args.device, ck)

        def log(g, reader, r, ck=ck):
            save()
            keep = ('gap_swap', 'gap_swap_ci', 'gap_cut', 'choice_on', 'choice_swap')
            print(run.name, ck, g, reader, json.dumps({k: r[k] for k in keep}), flush=True)

        score(net, args.device, games, crack_games, args.n_test, sizes, args.steps, args.batch, args.micro,
              res=res['ckpts'].setdefault(ck, {}), log=log)
        del net
        torch.cuda.empty_cache()
    save()


if __name__ == '__main__':
    main()
