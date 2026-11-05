"""The KV breadcrumb results drawn white on black (swarm/crumbs.py, kv arm: a writer's crumbs are m extra key/value
positions in every layer, left at the place it was; a later agent's tokens at that place attend to them).

  kv_trail       one trail test episode of Qwen3-4B with 4 KV crumbs: the explorer's rooms and crumbs, the rooms
                 the follower enters, and its answer with crumbs on and with the same model's crumbs cut
  kv_transplant  one visited room's crumbs replaced by the crumbs of a room in another test episode: the follower's
                 answer for that room, and over 100 test episodes how often it names the other episode's object
                 against its own room's object, for Qwen3-0.6B and Qwen3-4B
  kv_games       per game, the reader's forced-choice accuracy with crumbs on and with the same model's crumbs cut
                 (HotpotQA: the share of greedy answers with token F1 at least 0.5), Qwen3-0.6B and Qwen3-4B
  oldgrowth      old growth (G walkers before the asker, each reading every earlier crumb at a room and leaving its
                 own; the asker names the object in each room it visits), Qwen3-0.6B with 4 KV crumbs: the asker's
                 forced-choice accuracy against G from 1 to 32, crumbs on and cut, and the share of asked rooms some
                 walker saw; beside it one G = 8 test floor, which walker left crumbs in which room and where it saw
                 the object (written as crumbs_oldgrowth, in place of figures/crumbs.py's figure of the runs that
                 never read)
  kv_zero        untrained Qwen3-0.6B and Qwen3-4B (no adapter): trail forced choice with crumbs on and cut when the
                 crumbs are the K/V of the writer's own room sentence, of its last 4 tokens, of its last token, or of
                 4 slot tokens with untrained inputs, with the answer able to attend to them; and the sentence when
                 only the follower's tokens in a room can

python figures/crumbs_kv.py --snapshot NAME=RUN_DIR,... [--oldgrowth NAME=SCORES_JSON,...] [--zero MODEL=JSON,...]
writes results/crumbs_kv.json (per model, the kv4 and cut4 runs' test summaries and kept trail answers; oldgrowth
per-G scores; the zero-training scores of swarm/kv_zero.py); python figures/crumbs_kv.py draws from it into
figures/black/crumbs_kv_*.{svg,png,pdf} and crumbs_oldgrowth.{svg,png,pdf}.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import CARD, EMBER, HOT, LINK, M, OUT, QUILL, TEXT, canvas, finish, title  # noqa: E402
from chat_names import ROOT, width, wrap                                             # noqa: E402
from crumbs import DIM, GOLD, BODY, answer_lines, pct, trail_figure                  # noqa: E402
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


def snapshot(runs, og, zero):
    d = dict(models={name: snap(run) for name, run in runs.items()}, oldgrowth={}, zero={})
    for name, f in og.items():
        sc = json.loads(Path(f).read_text())
        d['oldgrowth'][name] = dict(steps_run=sc['steps_run'], base=sc['base'], init=sc['init'],
                                    curriculum=sc['curriculum'], games=sc['games'], sizes=sc['oldgrowth'])
    for model, f in zero.items():
        d['zero'][model] = {k.split('/')[-1]: dict(tf=v['tf'], mentions=v.get('mentions'), n_gen=v['gen']['on']['n'])
                            for k, v in json.loads(Path(f).read_text()).items()}
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


# ---------------------------------------------------------------- old growth

def floor(s, x0, y0, ep):
    """One old-growth floor: per walker (rows) the rooms it visited (crumbs left), filled where it saw the object; the
    asker's rooms in the last row. Returns the y below it."""
    o, walkers = ep['oldgrowth'], ep['agents'][:-1]
    n = len(ep['places'])
    lab, cell, row = 230, 64, 62
    s.text(x0 + lab + n * cell / 2, y0 + 40, 'rooms', 44, 400, DIM, 'middle')
    y = y0 + 70
    for i, w in enumerate(walkers + [ep['agents'][-1]]):
        cy = y + i * row + row / 2
        asker = i == len(walkers)
        s.text(x0 + lab - 24, cy + 15, 'asker' if asker else w['name'], 40, 700 if asker else 400,
               LINK if asker else TEXT, 'end')
        marked = set(o['asked']) if asker else {sg['place'] for sg in w['segments']}
        for r in range(n):
            if r not in marked:
                s.circle(x0 + lab + (r + .5) * cell, cy, 5, '#3A3A42')
        if asker:
            for r in o['asked']:
                cx = x0 + lab + (r + .5) * cell
                s.rect(cx - 26, cy - 26, 52, 52, 'none', 10, LINK, 6)
            continue
        for sg in w['segments']:
            cx = x0 + lab + (sg['place'] + .5) * cell
            if 'You see a' in sg['user']:
                s.circle(cx, cy, 21, GOLD)
            else:
                s.circle(cx, cy, 18, 'none', DIM, 6)
    y += (len(walkers) + 1) * row + 40
    for k, (mark, text) in enumerate((('saw', 'left crumbs and saw the object'), ('blind', 'left crumbs, saw nothing'),
                                      ('ask', 'the rooms the asker visits'))):
        cy = y + k * 62
        cx = x0 + 30
        if mark == 'saw':
            s.circle(cx, cy, 21, GOLD)
        elif mark == 'blind':
            s.circle(cx, cy, 18, 'none', DIM, 6)
        else:
            s.rect(cx - 26, cy - 26, 52, 52, 'none', 10, LINK, 6)
        s.text(cx + 50, cy + 14, text, 40, 400, TEXT)
    return y + 3 * 62


def oldgrowth_figure(d, name='crumbs_oldgrowth', run='kv4_og_init'):
    r = d['oldgrowth'][run]
    sz = r['sizes']
    sizes = sorted(int(g) for g in sz)
    val = lambda g, mode, k='choice': sz[str(g)][mode][k]                          # noqa: E731
    seen = [val(g, 'on', 'choice_seen') for g in sizes]
    W, H = 2600, 2000
    s = canvas(W)
    y = title(s, f'Reading the KV crumbs of up to {max(sizes)} walkers, the asker picks the right object in '
              f'{pct(min(seen))} to {pct(max(seen))} of the rooms that some walker saw', W)
    items = ((LINK, '', 'crumbs on'), (DIM, 'stroke-dasharray="4 16"', 'same model, crumbs cut'),
             (TEXT, 'stroke-dasharray="22 14"', 'share of asked rooms some walker saw'))
    lx = M
    for color, dash, lab in items:
        s.path(f'M{lx},{y + 31} L{lx + 80},{y + 31}', color, 8, extra=dash)
        s.text(lx + 100, y + 48, lab, 48, 400, TEXT)
        lx += 100 + width(lab, 48) + 80
    y += 130
    foot = (f'Qwen3-0.6B with 4 KV crumbs per visit, trained on trail and old growth with {OG_TRAIN[0]} to '
            f'{OG_TRAIN[1]} walkers from the KV run trained on seven crumb games. Forced choice: each asked room’s '
            f'object against the floor’s {CG.OG_ROOMS - 1} other objects, {len(CG.oldgrowth_eval(8))} test floors per '
            f'number of walkers. Rooms no walker saw can only be guessed; on rooms some walker saw the asker is '
            f'right in {pct(min(seen))} to {pct(max(seen))}.')
    lines = wrap(foot, W - 2 * M, 42, 400)
    left, right, base, top = M + 140, 1640, H - M - 52 * (len(lines) - 1) - 250, y + 70
    xv = lambda g: left + math.log2(g) / math.log2(max(sizes)) * (right - left)   # noqa: E731
    yv = lambda v: base - v * (base - top)                                        # noqa: E731
    lo, hi = OG_TRAIN
    s.rect(xv(lo) - 40, top - 10, xv(hi) - xv(lo) + 80, base - top + 10, CARD, 16)
    s.text((xv(lo) + xv(hi)) / 2, base - 30, f'trained on {lo} to {hi} walkers', 40, 400, DIM, 'middle')
    for v in (0, .5, 1):
        s.line(left - 20, yv(v), right + 30, yv(v), '#2A2A30', 3)
        s.text(left - 36, yv(v) + 16, pct(v), 44, 400, DIM, 'end')
    for g in sizes:
        s.text(xv(g), base + 66, str(g), 52, 700, TEXT, 'middle')
    s.text((left + right) / 2, base + 146, 'walkers before the asker (log scale)', 48, 400, TEXT, 'middle')
    ceil = [(xv(g), yv(sz[str(g)]['ceiling'])) for g in sizes]
    s.path('M' + ' L'.join(f'{x:.1f},{y_:.1f}' for x, y_ in ceil), TEXT, 6, extra='stroke-dasharray="22 14"')
    for mode, color, dash in (('off', DIM, 'stroke-dasharray="4 16"'), ('on', LINK, '')):
        pts = [(xv(g), yv(val(g, mode))) for g in sizes]
        s.path('M' + ' L'.join(f'{x:.1f},{y_:.1f}' for x, y_ in pts), color, 8, extra=dash)
        for (x, y_), g in zip(pts, sizes):
            s.circle(x, y_, 13, color)
            if mode == 'on':
                s.text(x - 22, y_ - 30, pct(val(g, mode)), 42, 700, LINK, 'end')
    x0 = 1760
    s.text(x0, top + 20, f'One test floor, {8} walkers', 50, 700, TEXT)
    floor(s, x0, top + 40, CG.oldgrowth_eval(8)[0])
    for k, line in enumerate(lines):
        s.text(M, H - M - 52 * (len(lines) - 1 - k), line, 42, 400, DIM)
    finish(s, name, H)
    return {g: dict(on=val(g, 'on'), cut=val(g, 'off'), walker_saw=sz[str(g)]['ceiling'],
                    on_seen=val(g, 'on', 'choice_seen')) for g in sizes}


# ---------------------------------------------------------------- zero training

OG_TRAIN = (1, 4)     # crumb_games.oldgrowth draws 1 to 4 walkers in training

ZERO = (('a_text_remember', 'The writer’s room sentence', 'answer can attend'),
        ('b_last4_remember', 'Its last 4 tokens', 'answer can attend'),
        ('c_last1_remember', 'Its last token', 'answer can attend'),
        ('d_slots_remember', '4 untrained slot tokens', 'answer can attend'),
        ('a_text', 'The room sentence', 'only room tokens can attend'))


def zero_figure(d, name='crumbs_kv_zero'):
    z = d['zero']
    W, H = 2600, 1950
    s = canvas(W)
    val = {(m, k): (z[m][k]['tf']['on']['choice'], z[m][k]['tf']['off']['choice']) for m, _ in MODELS
           for k, *_ in ZERO}
    a = [val[m, 'a_text_remember'] for m, _ in MODELS]
    y = title(s, f'No training: with KV crumbs of the writer’s own sentence the answer picks right {pct(a[0][0])} '
              f'(0.6B) and {pct(a[1][0])} (4B) of the time, against {pct(a[0][1])} and {pct(a[1][1])} with crumbs '
              f'cut', W)
    lx = M
    for m, color in MODELS:
        s.rect(lx, y + 8, 46, 46, color, 8)
        s.text(lx + 66, y + 48, f'{m}, crumbs on', BODY, 400, TEXT)
        lx += 66 + width(f'{m}, crumbs on', BODY) + 90
    s.line(lx + 6, y + 31, lx + 70, y + 31, TEXT, 7)
    s.text(lx + 92, y + 48, 'crumbs cut', BODY, 400, TEXT)
    y += 130
    left = M + 150
    slot = (W - M - left) / len(ZERO)
    rows = max(len(wrap(lab, slot - 30, 48, 700)) + len(wrap(sub, slot - 30, 42, 400)) for _, lab, sub in ZERO)
    base, top = H - M - 260 - rows * 52, y + 90
    bw = slot * 0.3
    yv = lambda v: base - v * (base - top)                                       # noqa: E731
    for v in (0, .5, 1):
        s.line(left - 30, yv(v), W - M, yv(v), '#2A2A30', 3)
        s.text(left - 40, yv(v) + 18, pct(v), 44, 400, DIM, 'end')
    s.line(left + slot * (len(ZERO) - 1) - 4, top - 40, left + slot * (len(ZERO) - 1) - 4, base + 40, '#3A3A42', 4)
    for i, (k, lab, sub) in enumerate(ZERO):
        cx = left + slot * (i + .5)
        for j, (m, color) in enumerate(MODELS):
            x = cx + (j - 1) * (bw + 24) + 12
            on, off = val[m, k]
            s.rect(x, yv(on), bw, base - yv(on), color, 8)
            s.text(x + bw / 2, yv(on) - 22, pct(on), 40, 700, color, 'middle')
            s.line(x - 10, yv(off), x + bw + 10, yv(off), TEXT, 7)
        labs = wrap(lab, slot - 30, 48, 700)
        for q, line in enumerate(labs):
            s.text(cx, base + 64 + q * 54, line, 48, 700, TEXT, 'middle')
        for q, line in enumerate(wrap(sub, slot - 30, 42, 400)):
            s.text(cx, base + 64 + len(labs) * 54 + q * 50, line, 42, 400, DIM, 'middle')
    n = sorted({z[m][k]['tf']['on']['n_choice'] for m, _ in MODELS for k, *_ in ZERO})
    foot = ('Untrained models, no adapter: the crumbs are the writer’s own keys and values in every layer at '
            'its own positions. Answer can attend: every follower token, the answer’s included, attends to the '
            'crumbs of each room visited so far; otherwise only the follower’s tokens in a room attend to that '
            'room’s crumbs. Forced choice on trail: each visited room’s object against the episode’s 5 '
            f'other objects and the key’s room against the 5 other rooms, {" or ".join(map(str, n))} test '
            'episodes per bar.')
    lines = wrap(foot, W - 2 * M, 42, 400)
    for q, line in enumerate(lines):
        s.text(M, H - M - 52 * (len(lines) - 1 - q), line, 42, 400, DIM)
    finish(s, name, H)
    return {f'{m}/{k}': dict(on=v[0], cut=v[1]) for (m, k), v in val.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', help='MODEL=RUN_DIR,...: run folders with results/kv4 and results/cut4')
    ap.add_argument('--oldgrowth', default='', help='NAME=SCORES_JSON,...: crumbs.py runs with oldgrowth scores')
    ap.add_argument('--zero', default='', help='MODEL=JSON,...: swarm/kv_zero.py outputs (zero-training KV crumbs)')
    args = ap.parse_args()
    if args.snapshot:
        snapshot(dict(kv.split('=') for kv in args.snapshot.split(',')),
                 dict(kv.split('=') for kv in args.oldgrowth.split(',') if kv),
                 dict(kv.split('=') for kv in args.zero.split(',') if kv))
    d = json.loads(SNAPSHOT.read_text())
    OUT.mkdir(exist_ok=True)
    four = d['models']['Qwen3-4B']
    print(json.dumps(dict(trail=trail_figure(four, ('kv4',), 'crumbs_kv_trail', 'Qwen3-4B')), indent=1))
    print(json.dumps(dict(transplant=transplant(d)), indent=1))
    print(json.dumps(dict(games=games_figure(d)), indent=1))
    if d.get('oldgrowth'):
        print(json.dumps(dict(oldgrowth=oldgrowth_figure(d)), indent=1))
    if d.get('zero'):
        print(json.dumps(dict(zero=zero_figure(d)), indent=1))


if __name__ == '__main__':
    main()
