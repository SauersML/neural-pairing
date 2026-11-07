"""Structure of the no-letter notes: does each held-out word get its own code? Writes notes from the best notes
weights (crumbs65 dev 1500) on fresh test episodes of maze, trap, relay and trail, where every room, door colour and
object is held out of training, then measures how well a note's first digits identify what its writer saw.

  python swarm/notes_codes.py --write WEIGHTS_DIR    writes results/notes_codes_raw.json (every note with what was seen)
  python swarm/notes_codes.py                        reads it and writes results/notes_codes.json

Identified = the share of notes whose value is the commonest value among notes with the same first k digits (the
best any lookup on those digits can do); guess = the commonest value's share. The spelling test asks whether a
word's first letter predicts its code's first digit, against 500 shuffles of codes among words."""
import collections
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / 'results/notes_codes_raw.json', ROOT / 'results/notes_codes.json'
GAMES = (('maze', 160), ('trap', 160), ('relay', 160), ('trail', 120))


def write(weights, device='mps', bs=8):
    import torch
    from swarm import crumbs as C, crumb_games as CG, one_note as ON
    net = C.load(weights, device, ckpt='best.pt')
    net.eval()
    rows = []
    for game, n in GAMES:
        rng = random.Random(f'explore-{game}')
        eps = [getattr(CG, game)(rng, test=True) for _ in range(n)]
        for s0 in range(0, n, bs):
            chunk = eps[s0:s0 + bs]
            with torch.no_grad():
                _, _, bank = net.run(chunk, device, need_loss=False)
            for b, ep in enumerate(chunk):
                notes = ON.written(net, bank, b, ep)
                for w, ag in enumerate(ep['agents']):
                    if ag['write']:
                        segs = [s for s in ag['segments'] if s['place'] is not None]
                        for k, s in enumerate(segs):
                            rows.append(dict(game=game, ep=s0 + b, writer=w, room=ep['places'][s['place']],
                                             saw=s['user'], note=notes.get((w, k))))
    RAW.write_text(json.dumps(rows, ensure_ascii=False))


def identified(pairs, k, cut=lambda n, k: n[:k]):
    by = collections.defaultdict(collections.Counter)
    for value, note in pairs:
        by[cut(note, k)][value] += 1
    return sum(c.most_common(1)[0][1] for c in by.values()) / len(pairs)


def guess(pairs):
    return collections.Counter(v for v, _ in pairs).most_common(1)[0][1] / len(pairs)


def analyse():
    rows = json.loads(RAW.read_text())
    colour = lambda s: re.search(r'take the (\w+) door', s).group(1)            # noqa: E731
    thing = lambda s: re.search(r'You see an? (\w+)', s).group(1)               # noqa: E731
    maze = [(colour(r['saw']), r['note']) for r in rows if r['game'] == 'maze']
    trail = [(thing(r['saw']), r['note']) for r in rows if r['game'] == 'trail']
    relay = [(r['room'], r['note']) for r in rows if r['game'] == 'relay' and 'key' not in r['saw']]
    traps = [r for r in rows if r['game'] == 'trap']
    trapped = [r['note'] for r in traps if 'trap' in r['saw']]
    safe = [(r['room'], r['note'][1:]) for r in traps if 'trap' not in r['saw']]
    mode = lambda ns: collections.Counter(ns).most_common(1)[0][0]             # noqa: E731
    codes = {}
    for kind, pairs in (('colour', maze), ('object', trail), ('room', relay)):
        for v in {v for v, _ in pairs}:
            got = [n for x, n in pairs if x == v]
            if len(got) >= 4:
                codes[f'{kind}:{v}'] = mode(got)
    letters = [(k.split(':')[1][0], c[0]) for k, c in sorted(codes.items())]

    def letter_hit(digits):
        by = collections.defaultdict(collections.Counter)
        for (l, _), d in zip(letters, digits):
            by[l][d] += 1
        return sum(c.most_common(1)[0][1] for c in by.values()) / len(letters)
    real = letter_hit([d for _, d in letters])
    shuffled = []
    for s in range(500):
        ds = [d for _, d in letters]
        random.Random(s).shuffle(ds)
        shuffled.append(letter_hit(ds))
    out = dict(
        notes=len(rows), episodes={g: n for g, n in GAMES},
        maze_colour=dict(values=len({v for v, _ in maze}), first3=identified(maze, 3), first1=identified(maze, 1),
                         guess=guess(maze)),
        trail_object=dict(values=len({v for v, _ in trail}), first4=identified(trail, 4), first3=identified(trail, 3),
                          guess=guess(trail)),
        relay_room=dict(values=len({v for v, _ in relay}), first3=identified(relay, 3), guess=guess(relay)),
        trap=dict(trap_notes=len(trapped), top=collections.Counter(trapped).most_common(1)[0],
                  safe_first_digit_0=sum(r['note'][:1] == '0' for r in traps if 'trap' not in r['saw']) / len(safe),
                  safe_room_after_first=identified(safe, 3), room_guess=guess(safe)),
        spelling=dict(words=len(letters), first_letter_predicts_first_digit=real, shuffled_max=max(shuffled),
                      shuffled_p=sum(x >= real for x in shuffled) / len(shuffled)),
        codes=codes)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == '__main__':
    if '--write' in sys.argv:
        write(sys.argv[sys.argv.index('--write') + 1])
    print(json.dumps({k: v for k, v in analyse().items() if k != 'codes'}, indent=1))
