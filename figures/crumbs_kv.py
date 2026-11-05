"""The KV breadcrumb results drawn white on black (swarm/crumbs.py, kv arm: a writer's crumbs are m extra key/value
positions in every layer, left at the place it was; a later agent's tokens at that place attend to them).

  kv_trail       one trail test episode of Qwen3-4B with 4 KV crumbs: the explorer's rooms and crumbs, the rooms
                 the follower enters, and its answer with crumbs on and with the same model's crumbs cut
  kv_transplant  one visited room's crumbs replaced by the crumbs of a room in another test episode: the follower's
                 answer for that room, and over 100 test episodes how often it names the other episode's object
                 against its own room's object, for Qwen3-0.6B and Qwen3-4B
  kv_games       per game, the reader's forced-choice accuracy with crumbs on and with the same model's crumbs cut
                 (HotpotQA: the share of greedy answers with token F1 at least 0.5), Qwen3-0.6B and Qwen3-4B
  kv_oldgrowth   oldgrowth (G walkers each read every earlier crumb at a room and write their own; the asker names
                 the objects): the share of rooms named right against G, crumbs on and cut, and the share some walker
                 saw

python figures/crumbs_kv.py --snapshot NAME=RUN_DIR,... [--oldgrowth NAME=SCORES_JSON,...] writes
results/crumbs_kv.json (per model, the kv4 and cut4 runs' test summaries and kept trail answers; oldgrowth per-G
scores); python figures/crumbs_kv.py draws from it into figures/black/crumbs_kv_*.{svg,png,pdf}.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import CARD, EMBER, HOT, LINK, M, OUT, QUILL, TEXT, canvas, finish, title  # noqa: E402
from chat_names import ROOT, width, wrap                                             # noqa: E402
from crumbs import DIM, GOLD, BODY, answer_lines, oldgrowth, pct, trail_figure       # noqa: E402
from method_parts import robot                                                       # noqa: E402
from swarm import crumb_games as CG                                                  # noqa: E402

SNAPSHOT = ROOT / 'results/crumbs_kv.json'
GAMES = (('trail', 'Trail', 'object in each room'), ('maze', 'Maze', 'door taken in each room'),
         ('count', 'Count', 'total of three counts'), ('work', 'Division of labour', 'two workers’ sums'),
         ('trap', 'Traps', 'safe or trap'), ('relay', 'Relay', 'key room, two writers in a chain'),
         ('hotpot', 'HotpotQA', 'answer from two pages'))
MODELS = (('Qwen3-0.6B', LINK), ('Qwen3-4B', QUILL))


# ---------------------------------------------------------------- snapshot

def snap(run):
    run = Path(run)
    out = dict(arms={}, rows={})
    for n in ('cut4', 'kv4'):
        sc = json.loads((run / 'results' / n / 'scores.json').read_text())
        out['arms'][n] = dict(steps_run=sc['steps_run'], base=sc.get('base'),
                              test={g: dict(tf=v['tf'], gen=v['gen']) for g, v in sc['test'].items()})
        rows = json.loads((run / 'results' / n / 'answers.json').read_text())
        out['rows'][n] = [r for r in rows if r['game'] == 'trail']
    return out


def snapshot(runs, og):
    d = dict(models={name: snap(run) for name, run in runs.items()}, oldgrowth={})
    for name, f in og.items():
        sc = json.loads(Path(f).read_text())
        d['oldgrowth'][name] = dict(steps_run=sc['steps_run'], sizes=sc['oldgrowth'])
    SNAPSHOT.write_text(json.dumps(d, indent=1))
    return d


# ---------------------------------------------------------------- transplant

def transplant_pick(d):
    """A kept episode where the moved room's answer names the other episode's object and every other visited room is
    named right (the key's room right too, when some episode has that)."""
    on = [r for r in d['rows']['kv4'] if r['mode'] == 'on']
    fits = []
    for r in (r for r in d['rows']['kv4'] if r['mode'] == 'room'):
        pa, j, pc = r['plan']
        if j >= len(on):
            continue
        ep, donor = r['episode'], on[j]['episode']
        text = r['says'][f'1/{len(ep["agents"][1]["segments"]) - 1}']
        want = donor['trail']['objects'][pc]
        if want == ep['trail']['objects'][pa] or CG.said(ep, text, pa) != want:
            continue
        if all(CG.said(ep, text, q) == ep['trail']['objects'][q] for q in ep['trail']['visit'] if q != pa):
            fits.append((not CG.trail_parts(ep, text.split('\n')[-1])[1], ep, donor, pa, pc, text))
    if not fits:
        raise ValueError('no transplant episode fits')
    return min(fits, key=lambda f: f[0])[1:]


def transplant(d, model='Qwen3-4B'):
    ep, donor, pa, pc, text = transplant_pick(d['models'][model])
    t = ep['trail']
    W = 2600
    s = canvas(W)
    y = title(s, 'Crumbs moved in from another house: the follower names the object the other explorer saw there', W)
    face = 170
    cw, ch, gap = 520, 250, 50

    def room_card(x, yy, room, obj, color, struck=False):
        s.rect(x, yy, cw, ch, CARD, 24, color, 4)
        s.text(x + cw / 2, yy + 84, room, 56, 700, TEXT, 'middle')
        s.text(x + cw / 2, yy + 170, obj, 56, 400, DIM if struck else TEXT, 'middle')
        if struck:
            s.line(x + cw / 2 - width(obj, 56) / 2 - 10, yy + 152, x + cw / 2 + width(obj, 56) / 2 + 10, yy + 152,
                   HOT, 6)

    def dots(cx, cy):
        s.add('<g filter="url(#glow)">' + ''.join(
            f'<circle cx="{cx + (j - 1.5) * 44:.1f}" cy="{cy:.1f}" r="15" fill="{GOLD}"/>' for j in range(4)) + '</g>')

    # the other house: its explorer saw the donor object in room pc
    x0 = M + face + 60
    s.text(M, y + 50, 'Another house (another test episode)', BODY, 700, DIM)
    yy = y + 90
    robot(s, 'quill', M, yy + 30, face, facing=1)
    s.text(M + face / 2, yy + face + 80, 'explorer', 44, 700, QUILL, 'middle')
    room_card(x0, yy, donor['places'][pc], donor['trail']['objects'][pc], GOLD)
    s.text(x0 + cw + 40, yy + 84, 'its explorer saw', 48, 400, DIM)
    s.text(x0 + cw + 40, yy + 150, f'“{donor["trail"]["objects"][pc]}” here', 52, 700, GOLD)
    dcy = yy + ch + 60
    dots(x0 + cw / 2, dcy)

    # the follower's house: the three rooms it enters; room pa gets the other house's crumbs
    yy = dcy + 120
    s.text(M, yy, 'The follower’s house', BODY, 700, DIM)
    yy += 40
    robot(s, 'ember', M, yy + 30, face, facing=1)
    s.text(M + face / 2, yy + face + 80, 'follower', 44, 700, EMBER, 'middle')
    for k, r in enumerate(t['visit']):
        x = x0 + k * (cw + gap)
        moved = r == pa
        room_card(x, yy, ep['places'][r], t['objects'][r], GOLD if moved else EMBER, struck=moved)
        if moved:
            s.path(f'M{x0 + cw / 2:.1f},{dcy + 26:.1f} C{x0 + cw / 2:.1f},{yy - 40:.1f} {x + cw / 2:.1f},'
                   f'{dcy + 40:.1f} {x + cw / 2:.1f},{yy - 10:.1f}', GOLD, 7, extra='stroke-dasharray="18 12"')
    s.text(x0, yy + ch + 70, 'This house’s explorer saw the crossed-out object; its crumbs there are replaced.', 46,
           400, DIM)

    # the answer
    yy += ch + 150
    lines = answer_lines(ep, text)
    room = ep['places'][pa]
    for chunks in lines:
        if chunks and chunks[0][0].strip().rstrip(':') == room:
            chunks[1] = (chunks[1][0], GOLD, 700)
    lead, pad = 78, 40
    bw = W - M - x0
    bh = 2 * pad + 50 + 30 + lead * len(lines)
    s.rect(x0, yy, bw, bh, CARD, 28, LINK, 5)
    s.text(x0 + pad, yy + pad + 40, f'the follower’s answer ({model}, 4 KV crumbs)', 48, 700, LINK)
    for j, chunks in enumerate(lines):
        s.rich(x0 + pad, yy + pad + 50 + 30 + 46 + j * lead, chunks, 54)
    robot(s, 'ember', M, yy + 10, face, facing=1)

    # how often, over 100 test episodes
    yy += bh + 110
    n = d['models'][model]['arms']['kv4']['test']['trail']['gen']['room']['n']
    s.text(M, yy, f'Over {n} test episodes, for the room with moved crumbs, the follower names', BODY, 700, TEXT)
    yy += 40
    left = M + 720
    full = W - M - left - 170
    rows = []
    for name, color in MODELS:
        g = d['models'][name]['arms']['kv4']['test']['trail']['gen']['room']
        rows.append((name, color, g['follows_crumb'], g['keeps_own']))
    for name, color, follows, own in rows:
        yy += 40
        s.text(M, yy + 50, name, 52, 700, color)
        yy += 70
        for label, v, c in (('the other house’s object', follows, GOLD), ('its own room’s object', own, HOT)):
            s.text(M + 40, yy + 52, label, 46, 400, TEXT)
            s.rect(left, yy + 10, max(6, v * full), 56, c, 8)
            s.text(left + max(6, v * full) + 24, yy + 56, pct(v), 52, 700, c)
            yy += 76
    finish(s, 'crumbs_kv_transplant', yy + M)
    return {name: dict(follows_crumb=f, keeps_own=o) for name, _, f, o in rows}


# ---------------------------------------------------------------- per game

def games_figure(d):
    W, H = 2600, 2000
    s = canvas(W)
    val = {}
    for name, _ in MODELS:
        test = d['models'][name]['arms']['kv4']['test']
        for g, *_ in GAMES:
            if g == 'hotpot':
                val[name, g] = (test[g]['gen']['on']['score'], test[g]['gen']['off']['score'])
            else:
                val[name, g] = (test[g]['tf']['on']['choice'], test[g]['tf']['off']['choice'])
    far = [g for g, *_ in GAMES if all(val[n, g][0] - val[n, g][1] >= 0.3 for n, _ in MODELS)]
    near = [lab for g, lab, _ in GAMES if g not in far]
    y = title(s, f'With KV crumbs the reader is far above its own crumbs-cut level on {len(far)} of {len(GAMES)} '
              f'games; '
              f'{" and ".join(near)} stay near it', W)
    lx = M
    for name, color in MODELS:
        s.rect(lx, y + 8, 46, 46, color, 8)
        s.text(lx + 66, y + 48, f'{name}, crumbs on', BODY, 400, TEXT)
        lx += 66 + width(f'{name}, crumbs on', BODY) + 90
    s.line(lx + 6, y + 31, lx + 70, y + 31, TEXT, 7)
    s.text(lx + 92, y + 48, 'same model, crumbs cut', BODY, 400, TEXT)
    y += 130
    left = M + 150
    slot = (W - M - left) / len(GAMES)
    lab_rows = max(len(wrap(lab, slot - 16, 50, 700)) + len(wrap(sub, slot - 16, 44, 400)) for _, lab, sub in GAMES)
    base, top = H - M - 120 - lab_rows * 54 - 40, y + 90
    bw = slot * 0.3
    yv = lambda v: base - v * (base - top)                                       # noqa: E731
    for v in (0, .5, 1):
        s.line(left - 30, yv(v), W - M, yv(v), '#2A2A30', 3)
        s.text(left - 40, yv(v) + 18, pct(v), 44, 400, DIM, 'end')
    for i, (g, lab, sub) in enumerate(GAMES):
        cx = left + slot * (i + .5)
        for j, (name, color) in enumerate(MODELS):
            x = cx + (j - 1) * (bw + 24) + 12
            on, off = val[name, g]
            s.rect(x, yv(on), bw, base - yv(on), color, 8)
            s.text(x + bw / 2, yv(on) - 22, pct(on), 40, 700, color, 'middle')
            s.line(x - 10, yv(off), x + bw + 10, yv(off), TEXT, 7)
        labs = wrap(lab, slot - 16, 50, 700)
        for k, line in enumerate(labs):
            s.text(cx, base + 66 + k * 56, line, 50, 700, TEXT, 'middle')
        for k, line in enumerate(wrap(sub, slot - 16, 44, 400)):
            s.text(cx, base + 66 + len(labs) * 56 + k * 52, line, 44, 400, DIM, 'middle')
    nc = {name: d['models'][name]['arms']['kv4']['test']['trail']['tf']['on']['n_choice'] for name, _ in MODELS}
    ng = {name: d['models'][name]['arms']['kv4']['test']['hotpot']['gen']['on']['n'] for name, _ in MODELS}
    each = lambda c: (str(next(iter(c.values()))) if len(set(c.values())) == 1 else            # noqa: E731
                      ' and '.join(f'{v} ({k})' for k, v in c.items()))
    foot = (f'Accuracy: forced choice over the possible answers, {each(nc)} test episodes per game; HotpotQA: greedy '
            f'answers with token F1 of at least 0.5, {each(ng)} test episodes.')
    for k, line in enumerate(wrap(foot, W - 2 * M, 42, 400)):
        s.text(M, H - M - 50 * (len(wrap(foot, W - 2 * M, 42, 400)) - 1 - k), line, 42, 400, DIM)
    finish(s, 'crumbs_kv_games', H)
    return {f'{n}/{g}': dict(on=v[0], off=v[1]) for (n, g), v in val.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', help='MODEL=RUN_DIR,...: run folders with results/kv4 and results/cut4')
    ap.add_argument('--oldgrowth', default='', help='NAME=SCORES_JSON,...: crumbs.py runs with oldgrowth scores')
    args = ap.parse_args()
    if args.snapshot:
        snapshot(dict(kv.split('=') for kv in args.snapshot.split(',')),
                 dict(kv.split('=') for kv in args.oldgrowth.split(',') if kv))
    d = json.loads(SNAPSHOT.read_text())
    OUT.mkdir(exist_ok=True)
    four = d['models']['Qwen3-4B']
    print(json.dumps(dict(trail=trail_figure(four, ('kv4',), 'crumbs_kv_trail', 'Qwen3-4B')), indent=1))
    print(json.dumps(dict(transplant=transplant(d)), indent=1))
    print(json.dumps(dict(games=games_figure(d)), indent=1))
    if d['oldgrowth']:
        print(json.dumps(dict(oldgrowth=oldgrowth(d, 'crumbs_kv_oldgrowth')), indent=1))


if __name__ == '__main__':
    main()
