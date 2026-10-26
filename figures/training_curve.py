"""What happened during nl20's training: for each value game, the median probability of a value never seen in training
with the real partner and with the partner from another episode, at every evaluation of the warm-up (the seven value
games only, 900 steps) and of the full mix that followed (1,320 steps).

A probability is exp(-cross-entropy) on the value's tokens (teacher-forced; the geometric mean over the value's
tokens, as in values_unseen.py), median over both receivers' episodes of a game (train.py --payload_eval: the same 8
or 16 never-seen episodes per game and receiver at every evaluation). The warm-up's tables are
results/magic-nl20/warm/payload_step*.json; the full mix's are the nl20_payload_step*.json.gz files that commit
ffc997e of the repository held in figures/inputs (step 0 of the mix is the warm-up's step 900, so the mix's steps are
drawn after it). --refresh rebuilds results/training_curve.json from those files (git show, read-only; --git-dir when
the working tree has no .git), checking each file against the sha256 that commit recorded in figures/inputs/sources.json.
Without --refresh the figure is drawn from results/training_curve.json. Writes figures/training_curve.{svg,png,pdf}.
Usage: python figures/training_curve.py [--refresh [--git-dir DIR]]
"""
import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import subprocess

from chat_names import COOL, INK, RED, ROOT, SVG, WARM, width
from method_parts import check_shape, render
from values_unseen import GAMES, LABELS

WARM_DIR = ROOT / 'results/magic-nl20/warm'
COMMIT = 'ffc997e'
SNAP = ROOT / 'results/training_curve.json'
OUT = ROOT / 'figures/training_curve'
W, M = 2400, 36


def git_show(path, git_dir):
    cmd = ['git'] + (['--git-dir', git_dir] if git_dir else ['-C', str(ROOT)]) + ['show', f'{COMMIT}:{path}']
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def steps_in(names, pattern):
    return sorted(int(m.group(1)) for n in names for m in [re.fullmatch(pattern, n)] if m)


def table(records):
    """{game: [[link_ce, shuffle_ce], ...]} over both receivers, for the value games."""
    out = {g: [] for g in GAMES}
    for r in records:
        if r['game'] in out:
            out[r['game']].append([r['link_ce'], r['shuffle_ce']])
    return out


def refresh(git_dir):
    warm_steps = steps_in([p.name for p in WARM_DIR.iterdir()], r'payload_step(\d+)\.json')
    sources = json.loads(git_show('figures/inputs/sources.json', git_dir))
    listing = subprocess.run(['git'] + (['--git-dir', git_dir] if git_dir else ['-C', str(ROOT)]) +
                             ['ls-tree', '--name-only', COMMIT, 'figures/inputs/'],
                             check=True, capture_output=True, text=True).stdout.split()
    mix_steps = steps_in([n.rsplit('/', 1)[-1] for n in listing], r'nl20_payload_step(\d+)\.json\.gz')
    warm_end = warm_steps[-1]
    series, files = [], {}
    for st in warm_steps:
        p = WARM_DIR / f'payload_step{st}.json'
        files[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        series.append(dict(stage='warm-up', step=st, games=table(json.loads(p.read_text()))))
    for st in mix_steps:
        if st == 0:                          # the warm-up's last checkpoint, evaluated again
            continue
        name = f'nl20_payload_step{st}'
        raw = gzip.decompress(git_show(f'figures/inputs/{name}.json.gz', git_dir))
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sources[name]['sha256']:
            raise ValueError(f'{name}: sha256 differs from the one recorded at {COMMIT}')
        files[f'{COMMIT}:figures/inputs/{name}.json.gz (uncompressed)'] = digest
        series.append(dict(stage='full mix', step=warm_end + st, games=table(json.loads(raw))))
    for row in series:
        row['median'] = {g: dict(real=statistics.median(math.exp(-a) for a, _ in v),
                                 other_episode=statistics.median(math.exp(-b) for _, b in v), n=len(v))
                         for g, v in row['games'].items()}
    snap = dict(definition='Median over both receivers of exp(-mean cross-entropy on the value tokens), never-seen '
                           'episodes of train.py --payload_eval; real partner (link_ce) and partner from another '
                           'episode (shuffle_ce).',
                warm_up_steps=warm_end, mix_steps=mix_steps[-1], files=files, series=series)
    SNAP.write_text(json.dumps(snap, indent=1) + '\n')
    return snap


def draw(snap):
    series = snap['series']
    warm_end, total = snap['warm_up_steps'], snap['warm_up_steps'] + snap['mix_steps']
    title = ('During the warm-up, values never seen in training become probable with the real partner and remain '
             'improbable with a partner from another episode')
    words = title.split(' ')
    cut = next(k for k in range(len(words), 0, -1) if width(' '.join(words[:k]), 56, 700) <= W - 2 * M)
    lines = [' '.join(words[:cut]), ' '.join(words[cut:])] if cut < len(words) else [title]
    top = 56 + 70 * len(lines)
    head_h = 90
    strip_h, gap = 186, 14
    axis_h = 150
    H = round(top + head_h + len(GAMES) * (strip_h + gap) - gap + axis_h + M)
    check_shape(W, H)
    s = SVG(W, H, title=title)
    for k, line in enumerate(lines):
        s.text(M, 84 + 70 * k, line, 56, 700)
    x0, x1 = M + 470, W - M - 30
    xs = lambda st: x0 + st / total * (x1 - x0)
    y_top = top + head_h
    y_bot = y_top + len(GAMES) * (strip_h + gap) - gap
    # the two stages as bands behind every row, named above
    s.rect(xs(0), top, xs(warm_end) - xs(0), y_bot - top, WARM, 24)
    s.rect(xs(warm_end) + 8, top, x1 - xs(warm_end) - 8, y_bot - top, COOL, 24)
    s.text((xs(0) + xs(warm_end)) / 2, top + 62, f'Warm-up: value games only', 48, 700, INK, 'middle')
    s.text((xs(warm_end) + x1) / 2, top + 62, f'Full mix of games', 48, 700, INK, 'middle')
    s.text(M + 10, top + 62, 'median probability', 46, 700)
    pad = 22
    for j, (g, label) in enumerate(zip(GAMES, LABELS)):
        sy = y_top + j * (strip_h + gap)
        ys = lambda p: sy + pad + (1 - p) * (strip_h - 2 * pad)
        s.text(M + 10, sy + strip_h / 2 + 16, label, 48, 700)
        s.text(x0 - 18, ys(1) + 15, '1', 44, 400, INK, 'end')
        s.text(x0 - 18, ys(0) + 15, '0', 44, 400, INK, 'end')
        for key, color, sw in (('other_episode', INK, 5), ('real', RED, 7)):
            pts = ' L'.join(f'{xs(row["step"]):.1f},{ys(row["median"][g][key]):.1f}' for row in series)
            s.path('M' + pts, color, sw)
        if j == 0:          # direct labels where both lines are flat
            lx = xs(warm_end) + 0.18 * (x1 - xs(warm_end))
            s.text(lx, ys(1) + 58, 'real partner', 46, 700, RED)
            s.text(lx, ys(0) - 16, 'partner from another episode', 46, 700, INK)
    ay = y_bot + 14
    for st in range(0, total + 1, 500):
        s.line(xs(st), ay, xs(st), ay + 16, INK, 3)
        s.text(xs(st), ay + 66, f'{st:,}', 44, 400, INK, 'middle')
    s.text((x0 + x1) / 2, ay + 126, 'training step', 48, 700, INK, 'middle')
    render(s, OUT)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help=f'rebuild the snapshot from git history ({COMMIT})')
    ap.add_argument('--git-dir', help='the repository .git directory, when the working tree has none')
    args = ap.parse_args()
    snap = refresh(args.git_dir) if args.refresh else json.loads(SNAP.read_text())
    H = draw(snap)
    last = snap['series'][-1]
    print(json.dumps(dict(height=H, points=len(snap['series']), last_step=last['step'],
                          warm_end={g: round(r['real'], 3) for g, r in
                                    next(x for x in snap['series'] if x['step'] == snap['warm_up_steps'])['median'].items()})))


if __name__ == '__main__':
    main()
