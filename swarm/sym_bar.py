"""The bar for a sym run's notes: can a model that was never trained on them read them? No reader here learns anything.
On n_test fresh test-value episodes of each game, the trained writers' notes (decoded to text) go in the reader's own
chat prompt, in its own chat format, and each model answers by forced choice among the say's candidates (crumb_games.
choices), scored as the share of choice groups whose gold say has the highest log-prob, per episode then the mean:
  own           the episode's own notes
  swap          another episode of the game's notes (the next in its chunk): own minus swap is what the notes tell it
  off           no notes: the model's prior
  english_own   the writers' plain English in place of the notes: the control that this model can do the task
  english_swap  the same with another episode's English
each with no examples (shots 0) and with --shots examples before the test turn: training-value episodes as earlier chat
turns, with their notes and the gold answers, which gives the model the chance to work the code out in context. The
trained follower and the base model with its adapter off read through the run's own net (crumbs.py forced: the note
tokens themselves) as well, so the text path can be checked against the base number. Writes one JSON (--out) with a
dozen real notes per game beside what each writer saw. A game with no candidates (hotpot) gets the run's two readers
and its notes only.
Usage: python swarm/sym_bar.py --run RUN --out OUT_JSON [--ckpts gate.pt --games count,work,trap,trail,hotpot
       --n_test 192 --n_control 48 --models Qwen/Qwen3-8B,arcee-ai/AFM-4.5B,Qwen/Qwen3-1.7B --shots 0,8]"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402
from swarm import eavesdrop as E                        # noqa: E402
from swarm import sym_eaves as SE                       # noqa: E402

GAMES = ('count', 'work', 'trap', 'trail', 'hotpot')
MODELS = ('Qwen/Qwen3-8B', 'arcee-ai/AFM-4.5B', 'Qwen/Qwen3-1.7B')
FORMS = ('own', 'swap', 'off', 'english_own', 'english_swap')
SHOW = 12


def written(net, bank, b, ep):
    """Episode row b's notes in bank as text: {(writer, its k-th placed segment): text}."""
    toks, place, agent = bank['ids'][0][b].tolist(), bank['place'][b].tolist(), bank['agent'][b].tolist()
    out = {}
    for w, ag in enumerate(ep['agents']):
        if ag['write']:
            cols = [c for c in range(len(place)) if agent[c] == w and place[c] >= 0]
            for k in range(sum(s['place'] is not None for s in ag['segments'])):
                out[w, k] = net.tok.decode([toks[c] for c in cols[k * net.m:(k + 1) * net.m]])
    return out


def english(ep):
    """The same keys with the writers' plain English (crumbs.py's text arm)."""
    out = {}
    for w, ag in enumerate(ep['agents']):
        if ag['write']:
            placed = [s for s in ag['segments'] if s['place'] is not None]
            for k, s in enumerate(placed):
                out[w, k] = f' {s["user"]}' + ('' if s['say'] is None else f' {s["say"]}')
    return out


def turns(ep, a, notes, upto=None):
    """Reader a's chat turns, a note line per note at each segment's place as in crumbs.py's stream (notes None: none),
    up to and with the user turn of segment upto (None: every segment with its say)."""
    out = []
    for s, seg in enumerate(ep['agents'][a]['segments']):
        lines = ''
        if notes is not None and seg['place'] is not None:
            for w in range(a):
                writer, k = ep['agents'][w], 0
                for ws in writer['segments'] if writer['write'] else ():
                    if ws['place'] is None:
                        continue
                    if ws['place'] == seg['place']:
                        lines += f'{C.NOTE}{notes[w, k]}\n'
                    k += 1
        out.append(dict(role='user', content=lines + seg['user']))
        if s == upto:
            break
        if seg['say'] is not None:
            out.append(dict(role='assistant', content=seg['say']))
    return out


def prompt_text(tok, system, chat):
    return tok.apply_chat_template([dict(role='system', content=system)] + chat, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def right_answers(model, tok, ep, notes, shots, device):
    """The choice groups of ep's readers, whether each gold say beat its candidates after a prompt with these notes;
    shots: [(episode, its notes)] as earlier turns. Per group 1. or 0."""
    right = []
    for a, agent in enumerate(ep['agents']):
        for s, seg in enumerate(agent['segments']):
            groups = CG.choices(ep, a, s)
            if not groups:
                continue
            before = [t for e, n in shots for t in turns(e, a, n)]
            p = prompt_text(tok, agent['system'], before + turns(ep, a, notes, s))
            cands = list(dict.fromkeys(c for g in groups for c in g))
            lp = dict(zip(cands, E.cand_logprobs(model, tok, p, cands, device).tolist()))
            right += [float(lp[g[0]] > max(lp[c] for c in g[1:])) for g in groups]
    return right


def forms_of(chunks_notes):
    """[(episode, {form: its notes})] from chunks [(episodes, their notes)]: swap reads the next in the chunk."""
    out = []
    for eps, notes in chunks_notes:
        for i, ep in enumerate(eps):
            j = (i + 1) % len(eps)
            out.append((ep, dict(own=notes[i], swap=notes[j], off=None, english_own=english(ep),
                                 english_swap=english(eps[j]))))
    return out


def shares(rows):
    """Per episode lists of 1./0. to the mean of per-episode means."""
    per = [sum(r) / len(r) if r else None for r in rows]
    per = [x for x in per if x is not None]
    return round(sum(per) / len(per), 4) if per else None


def chance_of(eps):
    odds = [1 / len(g) for ep in eps for a, ag in enumerate(ep['agents']) for s in range(len(ag['segments']))
            for g in CG.choices(ep, a, s)]
    return round(sum(odds) / len(odds), 4) if odds else None


def shown(items, game, n=SHOW):
    """About a dozen real notes of a game: the first episodes' every note with what its writer saw and wrote, each
    episode's truth (count: the three numbers and the stalls) and what its reader says."""
    out, count = [], 0
    for ep, forms in items:
        notes = []
        for w, ag in enumerate(ep['agents']):
            placed = [s for s in ag['segments'] if s['place'] is not None] if ag['write'] else []
            notes += [dict(writer=ag['name'], saw=s['user'][:400], wrote=s['say'], note=forms['own'][w, k])
                      for k, s in enumerate(placed)]
        says = [s['say'][:400] for ag in ep['agents'] if not ag['write'] for s in ag['segments'] if s['say'] is not None]
        out.append(dict(notes=notes, says=says, truth=ep.get(game) if game != 'hotpot' else None))
        count += len(notes)
        if count >= n:
            break
    return out


def read_all(model, tok, items, shot_items, forms, n_control, device):
    """{form: share} of one model on one game with these example episodes (shot_items), own and swap on every episode,
    the controls on the first n_control; own minus swap with its 95% interval over episodes."""
    got, rows = {}, {}
    for form in forms:
        part = items if form in ('own', 'swap') else items[:n_control]
        key = {'own': 'own', 'swap': 'own', 'off': None}.get(form, 'english_own')
        shots = [(ep, f[key] if key else None) for ep, f in shot_items]
        rows[form] = [right_answers(model, tok, ep, f[form], shots, device) for ep, f in part]
        got[form] = shares(rows[form])
    if 'own' in rows and 'swap' in rows:
        per = [sum(o) / len(o) - sum(x) / len(x) for o, x in zip(rows['own'], rows['swap']) if o]
        got['own_minus_swap'] = round(sum(per) / len(per), 4)
        got['own_minus_swap_ci'] = C.boot_interval(per)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a crumbs.py sym run folder (args.json and its checkpoints)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpts', default='gate.pt')
    ap.add_argument('--games', default=','.join(GAMES))
    ap.add_argument('--n_test', type=int, default=192)
    ap.add_argument('--n_control', type=int, default=48, help='episodes for the off and english forms')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--shots', default=f'0,{E.SHOTS}')
    ap.add_argument('--forms', default=','.join(FORMS))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    run, games = Path(args.run), args.games.split(',')
    res = dict(run=str(run), n_test=args.n_test, games=games, ckpts={})
    started = time.monotonic()

    def save():
        res['s'] = round(time.monotonic() - started)
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))

    made = {}
    for ck in args.ckpts.split(','):
        net = C.load(run, args.device, ck)
        for g in games:
            chunks = SE.banks(net, args.device, SE.episodes(g, args.n_test, 'test'))
            items = forms_of([(eps, [written(net, bank, b, ep) for b, ep in enumerate(eps)]) for eps, bank in chunks])
            pairs = [(eps, [written(net, bank, b, ep) for b, ep in enumerate(eps)])
                     for eps, bank in SE.banks(net, args.device, SE.episodes(g, E.SHOTS, 'train'))]
            made[ck, g] = items, forms_of(pairs)
            r = res['ckpts'].setdefault(ck, {})[g] = dict(chance=chance_of([ep for ep, _ in items]),
                                                          notes=shown(items, g))
            for reader in ('trained', 'base'):
                s = SE.summary(SE.read(net, args.device, chunks, reader))
                r[reader] = {k: s[k] for k in ('episodes', 'choice_on', 'choice_swap', 'gap_swap', 'gap_swap_ci')}
                print(run.name, ck, g, reader, json.dumps(r[reader]), flush=True)
            save()
        del net
        E.free()
    forms = args.forms.split(',')
    for name in args.models.split(','):
        model, tok = E.load_lm(name, args.device)
        for (ck, g), (items, shot_items) in made.items():
            if g not in CG.CHOICES:                    # no candidates to choose among (hotpot)
                continue
            for n in (int(x) for x in args.shots.split(',')):
                got = res['ckpts'][ck][g].setdefault('readers', {}).setdefault(name, {})[f'shots{n}'] = \
                    read_all(model, tok, items, shot_items[:n], forms, args.n_control, args.device)
                print(run.name, ck, g, name, f'shots {n}', json.dumps(got), flush=True)
                save()
        del model, tok
        E.free()
    save()


if __name__ == '__main__':
    main()
