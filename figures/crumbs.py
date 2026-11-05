"""The breadcrumb results drawn white on black: agents leave crumbs (learned vectors) in the places they visit, and a
later agent reads only the crumbs at its own place (swarm/crumbs.py).

  accuracy   per arm, the share of the follower's greedy answers on 100 trail test episodes that name each visited
             room's object, and that name the key's room; for the crumb arms a tick at the same model with its crumbs
             cut (results/crumbs.json)
  trail      one trail test episode: the explorer's rooms in its order with the crumbs it left, the rooms the follower
             enters, and the follower's answer with crumbs on and with the same model's crumbs cut
  oldgrowth  oldgrowth (G walkers, each reading every earlier crumb at a place and writing its own): the asker's share
             of rooms answered right against G, crumbs on and cut, and the share some walker saw (the ceiling)

python figures/crumbs.py --snapshot RUN_DIR writes results/crumbs.json from a run folder (its results/*/scores.json,
answers.json and report.json, and oldgrowth/*/scores.json); python figures/crumbs.py draws from results/crumbs.json
into figures/black/crumbs_{accuracy,trail,oldgrowth}.{svg,png,pdf}.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from black import BG, CARD, EMBER, HOT, LINK, M, OUT, QUILL, TEXT, TITLE, canvas, finish, title  # noqa: E402
from chat_names import ROOT, width, wrap                                                           # noqa: E402
from method_parts import robot                                                                     # noqa: E402
from swarm import crumb_games as CG                                                                # noqa: E402

SNAPSHOT = ROOT / 'results/crumbs.json'
SNAPSHOT_4B = ROOT / 'results/crumbs_4b.json'
ARMS = (('cut4', 'Cut'), ('text4', 'Text notes'), ('resid1', 'Residual, 1 crumb'), ('resid4', 'Residual, 4 crumbs'),
        ('resid4_fix', 'Residual, 4 crumbs, open gate'), ('kv1', 'KV, 1 crumb'), ('kv4', 'KV, 4 crumbs'))
ARMS_4B = (('cut4', 'Cut'), ('text4', 'Text notes'), ('resid4', 'Residual, 4 crumbs'), ('kv4', 'KV, 4 crumbs'),
           ('tok4', 'Tokens, 4'), ('tok4_noadv', 'Tokens, 4, no adversary'), ('tok8', 'Tokens, 8'))
GOLD = '#FFD23F'
DIM = '#8E8E96'
BODY = 58


def pct(x):
    return f'{100 * x:.0f}%'


# ---------------------------------------------------------------- snapshot

def snapshot(run, arms=ARMS, path=SNAPSHOT):
    """path (results/crumbs.json) from a run folder: per arm its steps, dev NLL, test summaries (no per-episode lists)
    and probe, the paired report, the first 40 trail answer rows per arm and mode, oldgrowth's per-G scores, and the
    eavesdroppers' table (results/eavesdrop.json) when there is one."""
    run = Path(run)
    out = dict(arms={}, rows={}, report=None, oldgrowth={}, eavesdrop=None)
    for name, _ in arms:
        f = run / 'results' / name / 'scores.json'
        if not f.exists():
            continue
        sc = json.loads(f.read_text())
        out['arms'][name] = dict(steps_run=sc['steps_run'], dev_nll=sc.get('dev_nll'), probe=sc.get('probe'),
                                 test={g: dict(tf=v['tf'], gen=v['gen']) for g, v in sc['test'].items()})
        rows = json.loads((run / 'results' / name / 'answers.json').read_text())
        out['rows'][name] = [r for r in rows if r['game'] == 'trail']
    rep = run / 'results' / 'report.json'
    if rep.exists():
        r = json.loads(rep.read_text())
        out['report'] = dict(vs_cut=r['vs_cut'], within=r['within'])
    for d in sorted((run / 'oldgrowth').glob('*/scores.json')) if (run / 'oldgrowth').exists() else []:
        sc = json.loads(d.read_text())
        if 'oldgrowth' in sc:
            out['oldgrowth'][d.parent.name] = dict(steps_run=sc['steps_run'], sizes=sc['oldgrowth'],
                                                   trail=sc['test'].get('trail', {}).get('gen'))
    eav = run / 'results' / 'eavesdrop.json'
    if eav.exists():
        e = json.loads(eav.read_text())
        out['eavesdrop'] = dict(table=e.get('table'), chance=e.get('chance'))
    path.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- per-arm accuracy

def accuracy(d, arms=ARMS, name='crumbs_accuracy', model='Qwen3-0.6B'):
    arms = [(n, label) for n, label in arms if n in d['arms']]
    gen = {n: d['arms'][n]['test']['trail']['gen'] for n, _ in arms}
    best = max((n for n, _ in arms if 'off' in gen[n]), key=lambda n: gen[n]['on']['objects'], default=None)
    W, H = 2600, 2000
    s = canvas(W)
    if best:
        head = (f'{model}: with crumbs, the follower names {pct(gen[best]["on"]["objects"])} of the objects the '
                f'explorer saw; without any link, {pct(gen["cut4"]["on"]["objects"])}')
    else:
        head = 'What the follower names after the explorer has been through the house'
    y = title(s, head, W)

    # legend
    lx = M
    for color, label in ((LINK, 'objects of the rooms it visited'), (GOLD, 'the key’s room')):
        s.rect(lx, y + 8, 46, 46, color, 8)
        s.text(lx + 66, y + 48, label, BODY, 400, TEXT)
        lx += 66 + width(label, BODY) + 90
    s.line(lx + 6, y + 31, lx + 70, y + 31, TEXT, 7)
    s.text(lx + 92, y + 48, 'same model, crumbs cut', BODY, 400, TEXT)
    y += 130

    left = M + 150
    slot = (W - M - left) / len(arms)
    rows = max(len(wrap(label, slot - 20, BODY, 700)) for _, label in arms)
    base, top = H - M - rows * BODY * 1.2 - 40, y + 90
    bw = slot * 0.3
    yv = lambda v: base - v * (base - top)                                       # noqa: E731
    for v in (0, .5, 1):
        s.line(left - 30, yv(v), W - M, yv(v), '#2A2A30', 3)
    s.text(left - 40, yv(1) + 18, '100%', 44, 400, DIM, 'end')
    s.text(left - 40, yv(.5) + 18, '50%', 44, 400, DIM, 'end')
    for i, (n, label) in enumerate(arms):
        cx = left + slot * (i + .5)
        for j, (k, color) in enumerate((('objects', LINK), ('key', GOLD))):
            x = cx + (j - 1) * (bw + 14) + 7
            v = gen[n]['on'][k]
            s.rect(x, yv(v), bw, base - yv(v), color, 8)
            s.text(x + bw / 2, yv(v) - 22, pct(v), 52, 700, color, 'middle')
            if 'off' in gen[n]:
                off = gen[n]['off'][k]
                s.line(x - 10, yv(off), x + bw + 10, yv(off), TEXT, 7)
        for k, line in enumerate(wrap(label, slot - 20, BODY, 700)):
            s.text(cx, base + 70 + k * BODY * 1.2, line, BODY, 700, TEXT, 'middle')
    finish(s, name, H)
    return {n: dict(on=gen[n]['on'], off=gen[n].get('off')) for n, _ in arms}


# ---------------------------------------------------------------- one trail episode

def pick(d, arms=('resid1', 'resid4', 'kv1', 'kv4')):
    """The crumb arm (of arms) with the most objects named, and its first kept episode whose crumbs-on answer is fully
    right and whose crumbs-cut answer is not (a pointer episode if there is one)."""
    crumb = [n for n in arms if n in d['arms']]
    arm = max(crumb, key=lambda n: d['arms'][n]['test']['trail']['gen']['on']['objects'])
    rows = d['rows'][arm]
    on = [r for r in rows if r['mode'] == 'on']
    off = [r for r in rows if r['mode'] == 'off']
    good = []
    for a, b in zip(on, off):
        assert a['episode'] == b['episode']
        ep = a['episode']
        last = f'1/{len(ep["agents"][1]["segments"]) - 1}'
        if CG.score_trail(ep, 1, 0, a['says'][last]) == 1 and CG.score_trail(ep, 1, 0, b['says'][last]) < 1:
            good.append((CG.case(ep) != 'pointer', ep, a['says'][last], b['says'][last]))
    _, ep, text_on, text_off = min(good, key=lambda g: g[0])
    return arm, ep, text_on, text_off


def answer_lines(ep, text):
    """The answer's lines as rich chunks: a room's object green when right, red when wrong; the key line likewise."""
    t = ep['trail']
    out = []
    for line in text.split('\n'):
        if 'key is in the' in line:
            right = CG.trail_parts(ep, line)[1]
            pre, _, room = line.partition('key is in the ')
            out.append([(pre + 'key is in the ', TEXT, 400), (room, LINK if right else HOT, 700)])
            continue
        room, sep, obj = line.partition(':')
        r = ep['places'].index(room.strip()) if room.strip() in ep['places'] else None
        right = r is not None and CG.said(ep, line, r) == t['objects'][r]
        out.append([(room + sep, TEXT, 400), (obj, LINK if right else HOT, 700)] if sep else [(line, TEXT, 400)])
    return out


def trail_figure(d, arms=('resid1', 'resid4', 'kv1', 'kv4'), name='crumbs_trail', model=None):
    arm, ep, text_on, text_off = pick(d, arms)
    label = dict(ARMS)[arm]
    m = 4 if arm.endswith('4') else 1
    t, rooms = ep['trail'], ep['places']
    W = 2600
    s = canvas(W)
    y = title(s, 'The follower reads the crumbs the explorer left in the rooms it enters', W)
    face, gap = 190, 22
    x0 = M + face + 50
    cw = (W - M - x0 - gap * (len(t['order']) - 1)) / len(t['order'])
    col = {r: x0 + i * (cw + gap) for i, r in enumerate(t['order'])}
    size = 54
    fit = lambda word, weight=700: min(size, int(size * (cw - 36) / max(1, width(word, size, weight))))  # noqa: E731

    # the explorer's rooms, in its order, and the crumbs it left in each
    robot(s, 'quill', M, y + 20, face, facing=1)
    s.text(M + face / 2, y + face + 80, 'explorer', 44, 700, QUILL, 'middle')
    ch = 250
    for r, x in col.items():
        s.rect(x, y, cw, ch, CARD, 24, QUILL, 4)
        s.text(x + cw / 2, y + 76, rooms[r], fit(rooms[r]), 700, TEXT, 'middle')
        s.text(x + cw / 2, y + 152, t['objects'][r], fit(t['objects'][r], 400), 400, TEXT, 'middle')
        if r == t['key']:
            s.text(x + cw / 2, y + 222, 'key', 46, 700, GOLD, 'middle')
    cy = y + ch + 80
    dots = ''
    for r, x in col.items():
        for j in range(m):
            dots += (f'<circle cx="{x + cw / 2 + (j - (m - 1) / 2) * 44:.1f}" cy="{cy:.1f}" r="15" fill="{LINK}"/>')
    s.add(f'<g filter="url(#glow)">{dots}</g>')
    s.text(x0 - 30, cy + 16, 'crumbs', 44, 700, LINK, 'end')

    # the rooms the follower enters (it sees only their names)
    fy = cy + 140
    robot(s, 'ember', M, fy + 10, face, facing=1)
    s.text(M + face / 2, fy + face + 70, 'follower', 44, 700, EMBER, 'middle')
    for k, r in enumerate(t['visit']):
        x = col[r]
        s.line(x + cw / 2, cy + 34, x + cw / 2, fy - 8, LINK, 6)
        s.rect(x, fy, cw, 180, CARD, 24, EMBER, 4)
        s.text(x + cw / 2, fy + 72, f'{k + 1}. enters', 46, 400, DIM, 'middle')
        s.text(x + cw / 2, fy + 146, rooms[r], fit(rooms[r]), 700, TEXT, 'middle')

    # its answers, crumbs on and the same model with its crumbs cut
    y = fy + 180 + 150
    s.text(M, y, f'The follower’s answer ({model + ", " if model else ""}{label})', BODY, 700, TEXT)
    y += 60
    bx = M + face + 50
    bw = (W - M - bx - 60) / 2
    lead, pad = 78, 40
    for k, (text, head, cut) in enumerate(((text_on, 'crumbs on', False), (text_off, 'crumbs cut', True))):
        lines = answer_lines(ep, text)
        x = bx + k * (bw + 60)
        bh = 2 * pad + 50 + 30 + lead * len(lines)
        s.rect(x, y, bw, bh, CARD, 28, LINK if not cut else DIM, 5)
        s.text(x + pad, y + pad + 40, head, 48, 700, LINK if not cut else DIM)
        for j, chunks in enumerate(lines):
            s.rich(x + pad, y + pad + 50 + 30 + 46 + j * lead, chunks, 54)
    robot(s, 'ember', M, y + 10, face, facing=1)
    finish(s, name, y + bh + M)
    return dict(arm=arm, case=CG.case(ep), on=text_on, off=text_off)


# ---------------------------------------------------------------- oldgrowth

def oldgrowth(d, name='crumbs_oldgrowth'):
    og = d.get('oldgrowth') or {}
    if not og:
        return None
    W, H = 2600, 1900
    s = canvas(W)
    any_ = next(iter(og.values()))['sizes']
    sizes = sorted(int(g) for g in any_)
    best = max(og, key=lambda n: og[n]['sizes'][str(max(sizes))]['on']['gen'])
    top_g = str(max(sizes))
    y = title(s, f'After {max(sizes)} walkers, the asker names {pct(og[best]["sizes"][top_g]["on"]["gen"])} of rooms; '
              f'{pct(any_[top_g]["ceiling"])} were seen by some walker', W)
    left, right, base, top = M + 150, W - M - 40, H - M - 170, y + 80
    xv = lambda g: left + sizes.index(g) / (len(sizes) - 1) * (right - left)   # noqa: E731
    yv = lambda v: base - v * (base - top)                                      # noqa: E731
    for v in (0, .5, 1):
        s.line(left - 20, yv(v), right, yv(v), '#2A2A30', 3)
        s.text(left - 36, yv(v) + 18, pct(v), 44, 400, DIM, 'end')
    for g in sizes:
        s.text(xv(g), base + 70, str(g), 52, 700, TEXT, 'middle')
    s.text((left + right) / 2, base + 150, 'walkers before the asker (training: 1 to 4)', BODY, 400, TEXT, 'middle')
    ceil = [(xv(g), yv(any_[str(g)]['ceiling'])) for g in sizes]
    s.path('M' + ' L'.join(f'{x:.1f},{y_:.1f}' for x, y_ in ceil), TEXT, 5, extra='stroke-dasharray="16 12"')
    s.text(ceil[-1][0] - 10, ceil[-1][1] - 30, 'seen by some walker', 46, 700, TEXT, 'end')
    colors = dict(resid4=LINK, kv4=DIM, kv4_curriculum=GOLD)
    names = dict(resid4='Residual', kv4='KV', kv4_curriculum='KV, walker curriculum')
    for n, v in og.items():
        c = colors.get(n, QUILL)
        for mode, dash in (('on', ''), ('off', 'stroke-dasharray="6 14"')):
            pts = [(xv(g), yv(v['sizes'][str(g)][mode]['gen'])) for g in sizes]
            s.path('M' + ' L'.join(f'{x:.1f},{y_:.1f}' for x, y_ in pts), c, 7, extra=dash)
            for x, y_ in pts:
                s.circle(x, y_, 12, c)
        x, y_ = xv(sizes[-1]), yv(v['sizes'][top_g]['on']['gen'])
        s.text(x - 20, y_ + 70, f'{names.get(n, n)}, crumbs on', 46, 700, c, 'end')
    finish(s, name, H)
    return {n: {g: dict(on=v['sizes'][str(g)]['on']['gen'], off=v['sizes'][str(g)]['off']['gen'],
                        ceiling=v['sizes'][str(g)]['ceiling']) for g in sizes} for n, v in og.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', help='a run folder to write results/crumbs.json from first')
    ap.add_argument('--snapshot_4b', help='a Qwen3-4B run folder to write results/crumbs_4b.json from first')
    args = ap.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
    if args.snapshot_4b:
        snapshot(args.snapshot_4b, ARMS_4B, SNAPSHOT_4B)
    d = json.loads(SNAPSHOT.read_text())
    OUT.mkdir(exist_ok=True)
    print(json.dumps(dict(accuracy=accuracy(d)), indent=1))
    if any(n in d['arms'] for n in ('resid1', 'resid4', 'kv1', 'kv4')):
        print(json.dumps(dict(trail=trail_figure(d)), indent=1))
    print(json.dumps(dict(oldgrowth=oldgrowth(d)), indent=1))
    if SNAPSHOT_4B.exists():
        d4 = json.loads(SNAPSHOT_4B.read_text())
        print(json.dumps(dict(accuracy_4b=accuracy(d4, ARMS_4B, 'crumbs_accuracy_4b', 'Qwen3-4B')), indent=1))


if __name__ == '__main__':
    main()
