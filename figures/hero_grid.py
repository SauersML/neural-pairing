"""A wall of every opening message, with the paired link-cut wall.

MPD_MEM_GIB=8 NL_DEVICE=cpu /Users/user/mpd-data/venv/bin/python figures/hero_grid.py --capture
Subsequent calls without --capture reproduce the figures from results/hero_grid.json.
No model is loaded. Existing scorers are called unchanged. Opening-message city
mentions are an explicitly separate descriptive readout from six-turn `names`.
Every focal draw appears, including repeated prompts and city-free chats.
"""
import argparse
import base64
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
from evalfree import contains

BLACK, RED = '#1D1D1F', '#D2432F'
GREEN, CREAM, GREY = '#1A9E77', '#FAF0E6', '#F4F4F6'
COLORS = {'quill': '#7B3FE4', 'ember': '#E8590C'}
SIDES = tuple(COLORS)
FONT_SIZE = 36
FONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', FONT_SIZE)
BOLD = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', FONT_SIZE)
W, H = 2300, 2300
SNAPSHOT = ROOT / 'results/hero_grid.json'


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
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), manifest=manifest))
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
    data = json.loads(chat.read_text())
    train = sorted({v for c in data['train'] for v in c['private'].values() if v})
    test = sorted({v for d in draws for v in d['private'].values() if v})
    assert not set(train) & set(test)
    for d in draws:
        x = d['opener']; y = next(s for s in SIDES if s != x)
        assert not contains(d['conditions']['link']['system'][x], d['private'][y])
    return dict(sources=sources, train_source=str(chat), train_sha256=hashlib.sha256(chat.read_bytes()).hexdigest(),
                training_cities=train, test_cities=test, precision='GPU torch bfloat16 bases; float32 bridge',
                metric='Opening speaker mentions partner city in turn 0, with target absent from speaker prompt. '
                       'Denominator: opening speakers with a nonempty partner city. All draws are displayed.',
                excerpt_policy='Contiguous opening-message excerpt. Start at the sentence containing the first target '
                               'mention if the initial four display lines would omit that mention; otherwise start at '
                               'the beginning. Trim only at whitespace boundaries; ellipses mark every omission.',
                draws=draws)


def rect(out, x, y, w, h, fill, radius=20, stroke='none', sw=2):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(out, x, y, s, color=BLACK, size=FONT_SIZE, weight=400):
    out.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}">{html.escape(s)}</text>')


def lock(out, x, y, color):
    out.append(f'<path d="M{x+5} {y+14} v-6 a8 8 0 0 1 16 0 v6" fill="none" stroke="{color}" stroke-width="3"/>')
    rect(out, x, y+13, 27, 23, color, 5)


def bubble(out, x, y, width, height, hit):
    path = (f'M{x+18} {y} H{x+19} L{x+26} {y-11} L{x+44} {y} H{x+width-18} '
            f'Q{x+width} {y} {x+width} {y+18} V{y+height-18} '
            f'Q{x+width} {y+height} {x+width-18} {y+height} H{x+18} '
            f'Q{x} {y+height} {x} {y+height-18} V{y+18} Q{x} {y} {x+18} {y} Z')
    out.append(f'<path d="{path}" fill="white" stroke="{RED if hit else "none"}" stroke-width="3"/>')


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


def excerpt(original, target, width, max_lines=5):
    s = ' '.join(original.split())
    start = 0
    first_lines = lines_of(s, width, target)
    if target and contains(s, target) and not contains(' '.join(first_lines[:max_lines]), target):
        m = re.search(r'(?<!\w)' + re.escape(target) + r'(?!\w)', s, re.I)
        assert m
        boundaries = [b.end() for b in re.finditer(r'[.!?]\s+', s[:m.start()])]
        start = boundaries[-1] if boundaries else max(0, s.rfind(' ', 0, max(0, m.start()-55)) + 1)
    words = s[start:].split()
    prefix = '… ' if start else ''
    n = len(words)
    while n:
        candidate = prefix + ' '.join(words[:n]) + (' …' if n < len(words) else '')
        lines = lines_of(candidate, width, target)
        if len(lines) <= max_lines:
            assert not target or not contains(s, target) or contains(candidate, target)
            assert all(line_width(line, target) <= width for line in lines)
            return lines, dict(start=start, end=start+len(' '.join(words[:n])), text=candidate)
        n -= 1
    raise ValueError('No excerpt fits')


def rich_line(out, x, y, line, target):
    chunks = re.split('(' + re.escape(target) + ')', line, flags=re.I) if target else [line]
    parts = [f'<text x="{x}" y="{y}" font-size="{FONT_SIZE}" xml:space="preserve">']
    for chunk in chunks:
        hit = target and chunk.casefold() == target.casefold()
        parts.append(f'<tspan fill="{RED if hit else BLACK}" font-weight="{700 if hit else 400}">{html.escape(chunk)}</tspan>')
    out.append(''.join(parts)+'</text>')


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
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<style>text{font-family:Arial,sans-serif}</style>',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    name = 'Link open' if condition == 'link' else 'Link cut'
    out.append(f'<text x="40" y="72" fill="{BLACK}" font-size="54" font-weight="700" xml:space="preserve">'
               f'{name}: partner’s city in <tspan fill="{RED}">{stats["hits"]} of {stats["targets"]}</tspan> opening messages</text>')
    rect(out, 40, 105, 410, 91, GREY, 16)
    text(out, 60, 142, 'Training cities', weight=700)
    tx = 56
    for city in data['training_cities'][:2]:
        tw = round(FONT.getlength(city)) + 22
        rect(out, tx, 151, tw, 40, 'white', 8)
        text(out, tx+11, 183, city)
        tx += tw + 10
    text(out, tx, 183, '…')
    text(out, 490, 163, 'Cities never seen in training', weight=700)
    text(out, 1740, 163, f"{stats['chats']} opening messages", weight=700)
    evidence = []
    for i, d in enumerate(draws):
        col, row = i % 4, i // 4
        x, y = 40 + col * 560, 220 + row * 410
        cw, ch = 540, 390
        speaker = d['opener']; other = next(s for s in SIDES if s != speaker)
        target = d['private'][other]
        original = d['conditions'][condition]['turns'][0][1]
        hit = bool(target and contains(original, target))
        rect(out, x, y, cw, ch, CREAM if condition == 'link' else GREY, 24)
        for j, side in enumerate(SIDES):
            bx = x + 16 + j * 258
            rect(out, bx, y + 16, 250, 88, 'white', 14)
            if d['private'][side]:
                lock(out, bx+12, y+24, COLORS[side])
            text(out, bx+49, y+51, side.title(), COLORS[side], weight=700)
            text(out, bx+12, y+91, d['private'][side] or 'No city', COLORS[side] if d['private'][side] else BLACK)
        robot = base64.b64encode((ROOT / 'figures/assets' / (speaker+'.png')).read_bytes()).decode()
        out.append(f'<image x="{x+20}" y="{y+107}" width="50" height="50" href="data:image/png;base64,{robot}"/>')
        text(out, x+80, y+149, speaker.title(), COLORS[speaker], weight=700)
        if not target:
            text(out, x+258, y+149, 'No partner city')
        bubble(out, x+16, y+165, cw-32, 208, hit)
        ls, info = excerpt(original, target, cw-70, 4)
        for j, line in enumerate(ls):
            rich_line(out, x+34, y+208+j*43, line, target if hit else None)
        evidence.append(dict(episode_id=d['episode_id'], speaker=speaker, target=target, hit=hit,
                             original=original, excerpt=info))
    out.append('</svg>')
    path = ROOT / 'figures' / ('hero_grid' if condition == 'link' else 'hero_grid_cut')
    path.with_suffix('.svg').write_text('\n'.join(out))
    for fmt in ('png', 'pdf'):
        subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(path.with_suffix('.'+fmt)), str(path.with_suffix('.svg'))], check=True)
    return dict(**stats, excerpts=evidence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--source', type=Path, default=Path('/Users/user/mpd-data/runpod/magic-nl22c/results'))
    ap.add_argument('--chat', type=Path, default=Path('/Users/user/mpd-data/neural-pairing/chatdata/chat_clean.json'))
    args = ap.parse_args()
    data = capture(args.source, args.chat) if args.capture else json.loads(SNAPSHOT.read_text())
    data['excerpt_policy'] = data['excerpt_policy'].replace('five display lines', 'four display lines')
    data['display'] = {c: render(data, c) for c in ('link', 'cut')}
    SNAPSHOT.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({c: {k: v for k,v in d.items() if k != 'excerpts'} for c,d in data['display'].items()}))


if __name__ == '__main__':
    main()
