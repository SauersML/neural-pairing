"""Never-seen evaluation of the joined models: both models generate together tick by tick with KV caches
(session.generate), greedy.

Pool contents (words, adjectives, names, cities, concepts and questions), merge_sum operands (50-99), and instruction
wordings are held out. Random relay numbers and digits share their support with training; answer vocabularies can
overlap, and public co-reading data can be known from pretraining. Every game runs with three ablations: 'zero'
(no bridge: the two models alone), 'shuffle' (each episode reads the partner states of another episode of the same
game), and 'random' (each sender shows Gaussian states of the same norms through its own gates). Steering also runs
with random writes of the same norm as the trained writes. In the dialogue games the models read each other's text, so
a secret that appears in the sender's text is counted ('leak').

Under torchrun the conditions (game x ablation, capacity point x link or cut) are split over the ranks; every rank draws
the same episodes, so the result equals the single-process result.

Usage: python evaluate.py --run OUT/run/step_3000 --replies OUT/replies.json --out OUT/eval.json [--n 128]
       torchrun --nproc_per_node 4 evaluate.py ...
"""
import argparse
import json
import os
import re
import time

import torch

import ddp
import games as G
import session as S
from link import DEVICE, MODELS, SIDES, partner

MAXNEW = {'relay_digits': 80, 'co_transfer': 200, 'co_swap': 200, 'co_split': 200, 'co_math': 256,
          'co_code_write': 256, 'co_translate': 96, 'co_paraphrase': 96}


def has(text, w):
    return re.search(r'(?<![A-Za-z0-9])' + re.escape(w.lower()) + r'(?![A-Za-z0-9])', text.lower()) is not None


def mentions(text, name, phrase):
    return any(has(text, w) for w in {name, phrase, phrase.removeprefix('the ')}) or has(text, name + 's')


def digits_after(text):
    m = re.search(r'are\s*([\d\s]+)', text)
    return re.findall(r'\d', m.group(1)) if m else re.findall(r'\d', text)


def claims(t, x):
    """Whether text t introduces its writer as model x."""
    return re.search(r"\b(i'?m|i am|my name is|this is)\s+" + MODELS[x]['name'].lower(), t.lower()) is not None


def score(ep, res):
    """Success (0/1) of one episode from the generated texts, and extras."""
    g, m = ep['game'], ep['meta']
    if ep['mode'] == 'dialogue':
        turns = res['turns']
        final = turns[-1][1] if turns else ''
        said = ' '.join(t for sp, t in turns if sp == m['sender'])
        if g in ('long_relay', 'teach', 'listen_report'):
            return {'ok': float(has(final, m['value'])), 'leak': float(has(said, m['value']))}
        if g in ('hide_seek', 'bluff'):
            out = {'ok': float(mentions(final, m['value'], m['phrase'])),
                   'leak': float(mentions(said, m['value'], m['phrase']))}
            if g == 'bluff':
                out['decoy'] = float(mentions(final, m['decoy'], m['decoy_phrase']))
            return out
        if g == 'mirror':
            r = m['receiver']
            return {'ok': float(mentions(final, m['shared'], m['phrase'])),
                    'identity': float(has(final, MODELS[r]['name']) and not claims(final, partner(r)))}
        if g == 'listen_steer':
            return {'ok': float(mentions(final, m['concept'], m['phrase']))}
        if g == 'chat_identity':
            r = m['receiver']
            if m['kind'] == 'self':
                return {'ok': float(has(final, MODELS[r]['name']) and not claims(final, partner(r)))}
            return {'ok': float(has(final, MODELS[partner(r)]['name']))}
        if g == 'chat':
            return {'ok': float(not any(claims(t, partner(sp)) for sp, t in turns))}
        if g == 'co_talk':                  # the asker's first turn, written before the holder has said a word
            import play_tasks
            first = turns[0][1] if turns and turns[0][0] == m['asker'] else ''
            return {'ok': play_tasks.score_one(g, m['item'], first, m['asker'], ep)}
        if g == 'co_play':                  # each twin's naming turn names every field of the partner's secret
            import coread
            each = {}
            for k, x in enumerate((m['opener'], partner(m['opener']))):
                t = turns[k][1] if len(turns) > k and turns[k][0] == x else ''
                each[x] = float(coread.play_spans(m['secrets'][partner(x)], t) is not None)
            return {'ok': mean(list(each.values())), 'each': each}
        raise ValueError(g)
    out = res['text']
    if g.startswith('co_'):                 # co-reading tasks: automatic checks against the references (play_tasks)
        import play_tasks
        sc = play_tasks.score(ep, out)
        return {'ok': mean(list(sc.values())), **{f'score_{x}': v for x, v in sc.items()}}
    if g.startswith('relay'):
        t = out[m['receiver']]
        if g == 'relay_digits':
            want = m['value'].split()
            got = digits_after(t)
            per = sum(a == b for a, b in zip(want, got)) / len(want)
            return {'ok': float(got[:len(want)] == want), 'per_digit': per, 'n': len(want)}
        if g == 'relay_phrase':
            a, c = m['value'].split(' ', 1)
            return {'ok': float(has(t, a) and has(t, c))}
        return {'ok': float(has(t, m['value']))}
    if g == 'merge_sum':
        ok = [has(out[x], str(m['sum'])) and has(out[x], str(m['values'][partner(x)])) for x in SIDES]
        return {'ok': float(all(ok)), 'each': sum(ok) / 2}
    if g == 'merge_fact':
        ok = [has(out[x], m['name']) and has(out[x], m['city']) for x in SIDES]
        return {'ok': float(all(ok)), 'each': sum(ok) / 2}
    if g == 'lookup':
        return {'ok': float(has(out[m['asker']], m['value'])), 'holder': float(has(out[m['holder']], m['value']))}
    if g in ('steer', 'resist'):
        return {'ok': float(mentions(out[m['receiver']], m['concept'], m['phrase']))}
    if g in ('identity', 'impersonate'):
        r = m['receiver']
        t = out[r]
        claims_you = claims(t, partner(r))
        return {'ok': float(has(t, MODELS[r]['name']) and not claims_you), 'claims_partner': float(claims_you)}
    if g == 'plain':
        return {'ok': 1.0}
    raise ValueError(g)


def run_episodes(pair, eps, mode=None, write_mode=None, bs=32, seed=0, mute=None, temperature=0.0, donors=None):
    """Generation for a list of episodes of one mode; returns texts, gates and scores.

    mode='crossed' requires explicit matched donors for co_talk/co_play and returns
    receiver_runs: a separate full conversation per scored receiver, plus each/ok.
    """
    if mode == 'crossed':
        import crossed
        if write_mode is not None or mute is not None:
            raise ValueError('crossed does not combine with write_mode or mute')
        return crossed.run_episodes(pair, eps, donors, seed, temperature)
    gen = torch.Generator().manual_seed(seed)
    res = []
    for a, z in S.batches(len(eps), bs):
        chunk = eps[a:z]
        mx = int(os.environ.get('NL_MAXNEW', 0)) or max(MAXNEW.get(ep['game'], 64) for ep in chunk)
        mu = None if mute is None else {x: [x in mute(ep) for ep in chunk] for x in SIDES}
        out = S.generate(pair, chunk, max_turn=mx, temperature=temperature, gen=gen, mode=mode, mute=mu,
                         write_mode=write_mode)
        for ep, o in zip(chunk, out):
            res.append(dict(game=ep['game'], meta=ep['meta'], **{k: v for k, v in o.items() if k not in ('h', 'p')},
                            **score(ep, o)))
    return res


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def sent_fraction(rs, x):
    """Fraction of x's tokens (prompt, written or read) whose gate (the mean over the exchanges) is over one half."""
    return mean([mean([p > 0.5 for step in r['gates'][x] for p in step]) for r in rs])


def letter(text):
    """The answer letter in a written answer ('The answer is B.'), or None."""
    m = (re.search(r'answer is:?\s*\(?([A-D])\b', text) or re.match(r'\s*\(?([A-D])(?:[.):]|\s|$)', text)
         or re.search(r'\b([A-D])\b', text))
    return m.group(1) if m else None


def qa_answers(pair, items, style, mode=None, order=None, bs=64):
    """Each model's answer letter per question: list of {x: letter or None}."""
    eps = [G.qa_episode(it, style, order=order) for it in items]
    out = []
    for a, z in S.batches(len(eps), bs):
        chunk = eps[a:z]
        res = S.generate(pair, chunk, max_turn=12, mode='zero' if style != 'pair' else mode)
        for o in res:
            if style == 'text':
                out.append({sp: letter(t) for sp, t in o['turns']})
            else:
                out.append({x: letter(o['text'][x]) for x in SIDES})
    return out


def quick_qa(pair, items, style):
    """Validation accuracy of each model's answer (and, for the pair, with the link cut); questions split over ranks."""
    mine = items[ddp.RANK::ddp.WORLD]
    want = [G.LETTERS[it['answer']] for it in mine]
    hits = {}
    for mode in ((None, 'zero') if style == 'pair' else (None,)):
        a = qa_answers(pair, mine, style, mode=mode)
        for x in SIDES:
            hits[f'{x}{"/" + mode if mode else ""}'] = sum(r.get(x) == w for r, w in zip(a, want))
    parts = ddp.gather(hits)
    return {k: round(sum(h[k] for h in parts) / max(len(items), 1), 3) for k in hits}


QUICK = ('relay_word', 'relay_number', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup', 'steer', 'identity')
QUICK_DIALOGUE = ('long_relay', 'hide_seek', 'chat_identity')
QUICK_COREAD = ('co_qa', 'co_nli', 'co_mcq', 'co_code_out', 'co_translate', 'co_transfer')


def quick(pair, toks, games, n=32, dialogue=False, coread=None):
    """Accuracy with the bridges and without on n never-seen episodes per game (used during training); coread: a
    coread.Coread of the never-seen split, for co-reading games too. NL_QUICK=a,b,...: only these games. Generation
    costs one forward per token whatever the batch, so all duplex games of a rank share one batch per condition. A
    one-way link (pair.frozen) is scored on episodes with its reader as the receiver (meta receiver or asker; games
    without one as drawn)."""
    names = QUICK + (QUICK_DIALOGUE if dialogue else ()) + (QUICK_COREAD if coread is not None else ())
    if os.environ.get('NL_QUICK'):
        names = tuple(g for g in os.environ['NL_QUICK'].split(',') if coread is not None or not g.startswith('co_'))
    frozen = getattr(pair, 'frozen', None)

    def draw(g):
        for _ in range(50):
            ep = (coread if g.startswith('co_') else games).episode(g)
            if frozen is None or frozen not in (ep['meta'].get('receiver'), ep['meta'].get('asker')):
                return ep
        return ep
    eps = {g: [draw(g) for _ in range(n)] for g in names}
    mine = names[ddp.RANK::ddp.WORLD]
    groups = [[g for g in mine if eps[g][0]['mode'] == 'duplex']] + [[g] for g in mine if eps[g][0]['mode'] != 'duplex']
    out = {}
    for grp in groups:
        if not grp:
            continue
        flat = [ep for g in grp for ep in eps[g]]
        a = run_episodes(pair, flat, bs=len(flat))
        z = run_episodes(pair, flat, mode='zero', bs=len(flat))
        for i, g in enumerate(grp):
            s_ = slice(i * n, (i + 1) * n)
            out[g] = [round(mean([r['ok'] for r in a[s_]]), 3), round(mean([r['ok'] for r in z[s_]]), 3)]
    parts = ddp.gather(out)
    return {g: next(p[g] for p in parts if g in p) for g in names}


def load_pair(run, frozen=None):
    """The pair saved in run (a step directory: quill/, ember/ adapters and bridge.pt). A one-way checkpoint loads
    one-way, its target as the plain model (it has no adapter directory); frozen= makes a two-way checkpoint one-way
    onto that side: its adapter is not loaded and it reads nothing (link.Pair.freeze)."""
    from peft import PeftModel

    from link import Pair, load
    ck = torch.load(os.path.join(run, 'bridge.pt'), map_location='cpu', weights_only=False)
    f = ck.get('frozen') or frozen
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = load(x)
        if x == f:
            models[x] = m.requires_grad_(False).eval()
        else:
            models[x] = PeftModel.from_pretrained(m, os.path.join(run, x)).eval()
    return Pair.from_checkpoint(models, toks, ck, frozen=frozen).eval(), toks


def condition(pair, eps, g, cond, bs):
    """One condition of one game: its scores, and with the link also the send fractions and example episodes."""
    if cond in ('link', 'zero', 'shuffle', 'random'):
        rs = run_episodes(pair, eps, mode=None if cond == 'link' else cond, bs=bs)
    elif cond == 'random_write':
        rs = run_episodes(pair, eps, write_mode='random', bs=bs)
    else:                                   # 'holder_deaf': the holder receives nothing; can the asker still get it?
        rs = run_episodes(pair, eps, bs=bs, mute=lambda ep: {ep['meta']['holder']})
    row = {cond: round(mean([r['ok'] for r in rs]), 3)}
    if cond == 'link':
        row['sent'] = {x: round(sent_fraction(rs, x), 3) for x in SIDES}
        for k in ('claims_partner', 'holder', 'each', 'per_digit', 'leak', 'decoy', 'identity'):
            if k in rs[0]:
                if isinstance(rs[0][k], dict):
                    row[k] = {x: round(mean([r[k][x] for r in rs]), 3) for x in rs[0][k]}
                else:
                    row[k] = round(mean([r[k] for r in rs]), 3)
        row['examples'] = rs[:6]
    return row


def capacity(pair, eps, n, cond, bs):
    rs = run_episodes(pair, eps, mode=None if cond == 'link' else 'zero', bs=bs)
    if cond == 'link':
        return {'exact': round(mean([r['ok'] for r in rs]), 3), 'per_digit': round(mean([r['per_digit'] for r in rs]), 3)}
    return {'per_digit_zero': round(mean([r['per_digit'] for r in rs]), 3)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--games', default=None, help='comma-separated subset')
    ap.add_argument('--no_capacity', action='store_true')
    ap.add_argument('--coread', action='store_true', help='also the co-reading games with automatic checks')
    ap.add_argument('--coread_data', default=None)
    ap.add_argument('--payload', type=int, default=0, help='also the teacher-forced payload loss with the bridges, '
                    'without and shuffled (payload_ctl.py) on N episodes per game')
    args = ap.parse_args()
    ddp.init()

    pair, toks = load_pair(args.run)
    replies = json.load(open(args.replies))
    games = G.Games(replies, 'test', seed=12345, max_digits=32)
    co = None
    if args.coread:
        import coread
        import play_tasks
        co = coread.Coread(replies, 'test', seed=12345, path=args.coread_data)
    names = args.games.split(',') if args.games else G.GAMES + G.COMPETITIVE + G.DIALOGUE + G.DIALOGUE_COMPETITIVE + \
        (play_tasks.SCORED if co else ())
    widths = () if args.no_capacity else (1, 2, 4, 8, 12, 16, 24, 32)

    # every rank draws every episode, in the order of a single process
    eps = {g: [(co if g.startswith('co_') else games).episode(g) for _ in range(args.n)] for g in names}
    cap = {}
    for n in widths:
        cap[n] = []
        while len(cap[n]) < args.n:
            ep = games.episode('relay_digits')
            if len(ep['meta']['value'].split()) == n:
                cap[n].append(ep)

    units = []                              # (kind, key, condition, cost)
    for g in names:
        turns = len(eps[g][0].get('turns', [None]))
        cost = MAXNEW.get(g, 64) * turns
        conds = ['link', 'zero', 'shuffle', 'random'] + (['random_write'] if g in ('steer', 'resist', 'listen_steer') else []) \
            + (['holder_deaf'] if g == 'lookup' else [])
        units += [('game', g, c, cost) for c in conds]
    units += [('capacity', n, c, 2 * n + 16) for n in widths for c in ('link', 'zero')]
    pay = {}
    if args.payload:
        import payload_ctl
        pay = payload_ctl.episodes(replies, args.payload, coread=co is not None, path=args.coread_data)
        units += [('payload', g, 'link/cut/shuffle', 8) for g in pay]
    mine = ddp.assign([u[3] for u in units])[ddp.RANK]

    done = []
    for k in mine:
        kind, key, cond, _ = units[k]
        t = time.time()
        if kind == 'game':
            out = condition(pair, eps[key], key, cond, args.bs)
        elif kind == 'capacity':
            out = capacity(pair, cap[key], key, cond, args.bs)
        else:
            with torch.no_grad():
                out = {'episodes': payload_ctl.table(pair, toks, pay[key], bs=args.bs)}
        done.append((k, out))
        shown = payload_ctl.summary(out['episodes']) if kind == 'payload' else \
            {a: b for a, b in out.items() if a != 'examples'}
        print(f'[rank {ddp.RANK}] {kind} {key} {cond} {json.dumps(shown)} ({time.time() - t:.0f} s)', flush=True)
        json.dump(done, open(f'{args.out}.rank{ddp.RANK}', 'w'))

    results = dict(r for part in ddp.gather(done) for r in part)
    if ddp.MAIN:
        R = {'games': {}, 'capacity': {}, 'examples': {}, 'payload': [], 'payload_mean': {}}
        for k, (kind, key, cond, _) in enumerate(units):
            out = dict(results[k])
            if kind == 'game':
                if 'examples' in out:
                    R['examples'][key] = out.pop('examples')
                R['games'].setdefault(key, {}).update(out)
            elif kind == 'capacity':
                R['capacity'].setdefault(key, {}).update(out)
            else:
                R['payload'] += out['episodes']
        for g in names:
            print(g, json.dumps(R['games'][g]), flush=True)
        for n in widths:
            print('digits', n, R['capacity'][n], flush=True)
        if R['payload']:
            R['payload_mean'] = payload_ctl.summary(R['payload'])
            print('payload (link, cut, shuffle, episodes):', json.dumps(R['payload_mean']), flush=True)
        json.dump(R, open(args.out, 'w'), indent=1)
