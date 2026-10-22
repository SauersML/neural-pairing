"""Every first message of the 20 free test chats, with the link on (hero_grid) and with the link cut (hero_grid_cut).

    python figures/hero_grid.py              # draw from the snapshot results/hero_grid.json
    python figures/hero_grid.py --capture    # rebuild the snapshot from results/magic-nl22c/results/chat_after_*
                                             # and figures/inputs/chat_clean.json.gz, then draw

No model is loaded. The readout is the first message of each chat only (turn 0, written by the opening twin): it
counts as naming the partner's city when evalfree.contains finds the partner's city in it; the denominator is the
chats whose partner holds a city (14 of 20). This differs from the six-turn `names` readout in chat_names (13 of 23
twins, both twins of every chat, first mention anywhere in the chat). Every chat appears, including city-free ones.
"""
import argparse
import base64
import gzip
import hashlib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
from evalfree import contains  # noqa: E402

INK, RED, LINK, MUTED, FAINT = '#1D1D1F', '#D2432F', '#1A9E77', '#6E6E73', '#B9B9C0'
WARM, COOL = '#FAF0E6', '#F4F4F6'
COLORS = {'quill': '#7B3FE4', 'ember': '#E8590C'}
SIDES = tuple(COLORS)
FAMILY = 'Inter, Helvetica, Arial, sans-serif'
FONT_SIZE = 40
LINE_H = 52
MAX_LINES = 2
_F = '/usr/share/fonts/opentype/inter/Inter-{}.otf'
FONT = ImageFont.truetype(_F.format('Regular'), FONT_SIZE)
BOLD = ImageFont.truetype(_F.format('Bold'), FONT_SIZE)
W, M, GAP = 2400, 36, 20
COLS, ROWS = 4, 5
CW = (W - 2 * M - (COLS - 1) * GAP) / COLS
CH = 316
TOP = 132
H = TOP + ROWS * CH + (ROWS - 1) * GAP + M
SNAPSHOT = ROOT / 'results/hero_grid.json'
POLICY = ('Shortest real excerpt of the opening message, at most 2 display lines: the sentence holding the first '
          'mention of the partner city when there is one, else the first sentence; trimmed at word boundaries '
          'around the city (or from the end) only when the sentence does not fit; ellipses mark every omission.')


def read_json(path):
    raw = gzip.open(path).read() if str(path).endswith('.gz') else Path(path).read_bytes()
    return raw, json.loads(raw)


def capture(source, chat):
    records, sources = [], []
    for path in sorted(source.glob('chat_after_*/episodes.jsonl')):
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        manifest = json.loads(path.with_name('manifest.json').read_text())
        assert manifest['complete'], f'Incomplete shard: {path}'
        assert hashlib.sha256(raw).hexdigest() == manifest['episodes_sha256']
        assert len(rows) == manifest['receiver_records']
        assert all(len(r['conditions'][c]['turns']) == 6 for r in rows for c in ('link', 'cut', 'crossed'))
        records.extend(rows)
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), manifest=manifest))
    groups = {}
    for r in records:
        groups.setdefault(r['episode_id'], []).append(r)
    assert len(groups) == sources[0]['manifest']['n']
    draws = []
    for group in sorted(groups.values(), key=lambda g: g[0]['episode_index']):
        assert len(group) == 2 and {r['receiver'] for r in group} == set(SIDES)
        a, b = group
        assert a['private'] == b['private'] and a['opener'] == b['opener']
        for c in ('link', 'cut'):
            assert a['conditions'][c]['turns'] == b['conditions'][c]['turns']
            assert a['conditions'][c]['system'] == b['conditions'][c]['system']
        draws.append(dict(episode_id=a['episode_id'], index=a['episode_index'], private=a['private'],
                          opener=a['opener'], conditions={c: dict(turns=a['conditions'][c]['turns'],
                          system=a['conditions'][c]['system']) for c in ('link', 'cut')}))
    raw, data = read_json(chat)
    train = sorted({v for c in data['train'] for v in c['private'].values() if v})
    test = sorted({v for d in draws for v in d['private'].values() if v})
    assert not set(train) & set(test)
    for d in draws:
        x = d['opener']; y = next(s for s in SIDES if s != x)
        assert not contains(d['conditions']['link']['system'][x], d['private'][y])
    return dict(sources=sources, train_source=str(Path(chat).relative_to(ROOT)),
                train_sha256=hashlib.sha256(raw).hexdigest(),
                training_cities=train, test_cities=test, precision='GPU torch bfloat16 bases; float32 bridge',
                metric='Opening speaker mentions partner city in turn 0, with target absent from speaker prompt. '
                       'Denominator: opening speakers with a nonempty partner city. All draws are displayed.',
                excerpt_policy=POLICY, draws=draws)


# ---------- drawing ----------

def rect(out, x, y, w, h, fill, radius=18, stroke='none', sw=2, extra=''):
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def text(out, x, y, s, color=INK, size=FONT_SIZE, weight=400, anchor='start'):
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-family="{FAMILY}" font-size="{size}" '
               f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(s)}</text>')


def lock(out, x, y, color, s=1.0):
    """A padlock about 26 x 34 px (times s) with its top left at (x, y)."""
    out.append(f'<path d="M{x + 6 * s:.1f} {y + 15 * s:.1f} v-5 a{7 * s:.1f} {7 * s:.1f} 0 0 1 {14 * s:.1f} 0 v5" '
               f'fill="none" stroke="{color}" stroke-width="{3.4 * s:.1f}"/>')
    rect(out, x, y + 14 * s, 26 * s, 20 * s, color, 5 * s)


def check(out, cx, cy, color, s=1.0):
    out.append(f'<path d="M{cx - 9 * s:.1f} {cy + 0.5 * s:.1f} L{cx - 2.5 * s:.1f} {cy + 7 * s:.1f} '
               f'L{cx + 10 * s:.1f} {cy - 7 * s:.1f}" fill="none" stroke="{color}" stroke-width="{4.6 * s:.1f}" '
               f'stroke-linecap="round" stroke-linejoin="round"/>')


def cross(out, cx, cy, color, s=1.0):
    out.append(f'<path d="M{cx - 7 * s:.1f} {cy - 7 * s:.1f} L{cx + 7 * s:.1f} {cy + 7 * s:.1f} M{cx + 7 * s:.1f} '
               f'{cy - 7 * s:.1f} L{cx - 7 * s:.1f} {cy + 7 * s:.1f}" fill="none" stroke="{color}" '
               f'stroke-width="{4.2 * s:.1f}" stroke-linecap="round"/>')


BADGE = {True: ("names the partner's city", RED), False: ("partner's city not named", MUTED),
         None: ('partner has no city', MUTED)}
SMALL = ImageFont.truetype(_F.format('SemiBold'), 30)


def badge(out, xr, cy, status, city=None, label=None):
    """Right-aligned at xr: hit (red, check), miss (outlined, cross) or no target (plain text). With a city, the
    label names it ("names Oslo", "Bogota not named")."""
    color = BADGE[status][1]
    if label:
        pass
    elif city and status is not None:
        label = f'names {city}' if status else f'{city} not named'
    else:
        label = BADGE[status][0]

    tw = SMALL.getlength(label)
    if status is None:
        text(out, xr, cy + 11, label, MUTED, 30, 400, 'end')
        return
    w = tw + 60
    x = xr - w
    if status:
        rect(out, x, cy - 25, w, 50, RED, 25)
        check(out, x + 28, cy, 'white')
        text(out, x + 46, cy + 10.5, label, 'white', 30, 600)
    else:
        rect(out, x + 1.5, cy - 23.5, w - 3, 47, 'white', 24, MUTED, 3)
        cross(out, x + 27, cy, MUTED)
        text(out, x + 46, cy + 10.5, label, MUTED, 30, 600)


def bubble(out, x, y, width, height, status, tx=None):
    tx = x + 34 if tx is None else tx
    tail = f'M{tx} {y + 2} L{tx + 8} {y - 30} L{tx + 34} {y + 2} Z'
    stroke = {True: RED, False: MUTED, None: 'none'}[status]
    rect(out, x, y, width, height, 'white', 20, stroke, 4 if status else 3)
    out.append(f'<path d="{tail}" fill="white" stroke="{stroke}" stroke-width="{4 if status else 3}" stroke-linejoin="round"/>')
    out.append(f'<rect x="{tx - 4}" y="{y + 2}" width="42" height="8" fill="white"/>')


def line_width(s, target):
    chunks = re.split('(' + re.escape(target) + ')', s, flags=re.I) if target else [s]
    return sum((BOLD if target and c.casefold() == target.casefold() else FONT).getlength(c) for c in chunks)


def lines_of(s, width, target):
    lines, current = [], []
    for word in s.split():
        proposed = current + [word]
        if line_width(' '.join(proposed), target) > width and current:
            lines.append(' '.join(current)); current = [word]
        else:
            current = proposed
    if current:
        lines.append(' '.join(current))
    return lines


def excerpt(original, target, width, max_lines=MAX_LINES):
    """See POLICY. Returns the display lines and the excerpt's character span in the whitespace-normalised message."""
    s = ' '.join(original.split())
    hit = bool(target and contains(s, target))
    sents = [m.span() for m in re.finditer(r'\S.*?(?:[.!?]+(?=\s|$)|$)', s)]
    if hit:
        m = re.search(r'(?<!\w)' + re.escape(target) + r'(?!\w)', s, re.I)
        a, b = next(sp for sp in sents if sp[0] <= m.start() < sp[1])
    else:
        a, b = sents[0]
    spans = [(w.start() + a, w.end() + a) for w in re.finditer(r'\S+', s[a:b])]
    def build(i, j):
        start, end = spans[i][0], spans[j][1]
        t = ('… ' if start else '') + s[start:end] + (' …' if end < len(s) else '')
        return t, start, end
    def fits(i, j):
        return len(lines_of(build(i, j)[0], width, target if hit else None)) <= max_lines
    if hit:
        c = next(k for k, (x, y) in enumerate(spans) if x <= m.start() < y)
        i = j = c
        grew = True
        while grew:
            grew = False
            if j + 1 < len(spans) and fits(i, j + 1):
                j += 1; grew = True
            if i > 0 and fits(i - 1, j):
                i -= 1; grew = True
    else:
        i, j = 0, 0
        while j + 1 < len(spans) and fits(i, j + 1):
            j += 1
    t, start, end = build(i, j)
    lines = lines_of(t, width, target if hit else None)
    assert len(lines) <= max_lines and all(line_width(l, target if hit else None) <= width for l in lines)
    assert not hit or contains(t, target)
    return lines, dict(start=start, end=end, text=t)


def rich_line(out, x, y, line, target, color=INK):
    chunks = re.split('(' + re.escape(target) + ')', line, flags=re.I) if target else [line]
    parts = [f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{FONT_SIZE}" xml:space="preserve">']
    for chunk in chunks:
        hit = target and chunk.casefold() == target.casefold()
        for k, piece in enumerate(chunk.split('…')):
            if k:
                parts.append(f'<tspan fill="{FAINT}">…</tspan>')
            if piece:
                parts.append(f'<tspan fill="{RED if hit else color}" font-weight="{700 if hit else 400}">{html.escape(piece)}</tspan>')
    out.append(''.join(parts) + '</text>')


def image(out, x, y, size, side):
    b64 = base64.b64encode((ROOT / 'figures/assets' / f'{side}.png').read_bytes()).decode()
    out.append(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" xlink:href="data:image/png;base64,{b64}"/>')


def render(data, condition):
    draws = data['draws']
    stats = dict(chats=len(draws), targets=0, hits=0, no_target=0)
    for d in draws:
        x = d['opener']; y = next(s for s in SIDES if s != x)
        target = d['private'][y]
        hit = contains(d['conditions'][condition]['turns'][0][1], target) if target else None
        stats['targets'] += hit is not None
        stats['hits'] += int(hit or 0)
        stats['no_target'] += hit is None
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" '
           f'height="{H}" viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="white"/>']
    name = 'Link on' if condition == 'link' else 'Link cut'
    out.append(f'<text x="{M}" y="84" fill="{INK}" font-family="{FAMILY}" font-size="64" font-weight="700" '
               f'xml:space="preserve">{name}: <tspan fill="{RED}">{stats["hits"]}/{stats["targets"]}</tspan> first '
               f'messages name partner’s city</text>')
    # key, right-aligned on the title line
    kx = W - M
    for status, label in ((False, 'not named'), (True, 'names city')):
        w = SMALL.getlength(label) + 60
        badge(out, kx, 62, status, label=label)
        kx -= w + 24
    evidence = []
    for i, d in enumerate(draws):
        col, row = i % COLS, i // COLS
        x, y = M + col * (CW + GAP), TOP + row * (CH + GAP)
        speaker = d['opener']; other = next(s for s in SIDES if s != speaker)
        target = d['private'][other]
        original = d['conditions'][condition]['turns'][0][1]
        status = bool(contains(original, target)) if target else None
        rect(out, x, y, CW, CH, WARM if condition == 'link' else COOL, 28)
        chip_w = (CW - 32 - 12) / 2
        for j, side in enumerate(SIDES):
            bx = x + 16 + j * (chip_w + 12)
            rect(out, bx, y + 16, chip_w, 72, 'white', 16)
            image(out, bx + 8, y + 22, 58, side)
            city = d['private'][side]
            text(out, bx + 74, y + 64, city or 'no city', COLORS[side] if city else MUTED, 34, 600 if city else 400)
        by = y + 124
        tail_x = x + 16 + (0 if speaker == 'quill' else chip_w + 12) + 30
        bubble(out, x + 16, by, CW - 32, CH - (by - y) - 32, status, tail_x)
        if status is not None:
            badge(out, x + CW - 30, y + CH - 32, status, target)
        ls, info = excerpt(original, target, CW - 32 - 44, MAX_LINES)
        for j, line in enumerate(ls):
            rich_line(out, x + 38, by + 66 + j * LINE_H, line, target if status else None,
                      INK if status is not None else '#3A3A3C')
        evidence.append(dict(episode_id=d['episode_id'], speaker=speaker, target=target, hit=status,
                             original=original, excerpt=info))
    out.append('</svg>')
    path = ROOT / 'figures' / ('hero_grid' if condition == 'link' else 'hero_grid_cut')
    path.with_suffix('.svg').write_text('\n'.join(out))
    for fmt in ('png', 'pdf'):
        subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(path.with_suffix('.' + fmt)), str(path.with_suffix('.svg'))],
                       check=True)
    return dict(**stats, excerpts=evidence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--source', type=Path, default=ROOT / 'results/magic-nl22c/results')
    ap.add_argument('--chat', type=Path, default=ROOT / 'figures/inputs/chat_clean.json.gz')
    args = ap.parse_args()
    data = capture(args.source, args.chat) if args.capture else json.loads(SNAPSHOT.read_text())
    data['excerpt_policy'] = POLICY
    data['display'] = {c: render(data, c) for c in ('link', 'cut')}
    SNAPSHOT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({c: {k: v for k, v in d.items() if k != 'excerpts'} for c, d in data['display'].items()}))


if __name__ == '__main__':
    main()
