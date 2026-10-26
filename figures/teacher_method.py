"""How chat training uses a teacher: linked Ember is trained on the turn original Ember writes when its prompt also
gives Quill's city.

Reads figures/inputs/chat.json.gz and chat_clean.json.gz (the chat data the training used) and the role-play filter
regex META from pairing/prep_talk.py. Writes results/teacher_method.json: the sha256 of each data file, the
conversations the filter removed, the per-turn PMI scores stored in the data with their summary and histogram, and the
example conversation. PMI is the summed log probability of a whole turn (closing token included) with the partner's
city in the prompt minus without it, after the same preceding conversation. Draws figures/teacher_method.{svg,png,pdf}
from the example. No model is loaded.
Usage: python figures/teacher_method.py
"""
from collections import Counter
import ast
import gzip
import hashlib
import html
import json
import math
import re
import statistics
import subprocess

import creature
from chat_names import ROOT, mascot, width

DATA = ROOT / 'figures/inputs'
OUT = ROOT / 'figures/teacher_method'
RESULTS = ROOT / 'results/teacher_method.json'
INK, PURPLE, ORANGE, GREEN = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77'
COOL, WARM = '#F4F4F6', '#FAF0E6'
W, FONT, CARD_H = 2400, 46, 276
FAINT = '#B9B9C0'
FAMILY = 'Inter, Helvetica, Arial, sans-serif'


def analyze(data):
    raw_path, clean_path = data / 'chat.json.gz', data / 'chat_clean.json.gz'
    blobs = {p: gzip.open(p).read() for p in (raw_path, clean_path)}
    raw, clean = (json.loads(blobs[p]) for p in (raw_path, clean_path))
    # read the regex literal so the data-generation module is not imported
    tree = ast.parse((ROOT / 'pairing/prep_talk.py').read_text())
    pattern = next(n.value.args[0].value for n in tree.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'META' for t in n.targets))
    filt = re.compile(pattern, re.I)
    key = lambda c: json.dumps(c, sort_keys=True)
    counts, dropped = {}, []
    for split in ('train', 'test'):
        a, b = Counter(map(key, raw[split])), Counter(map(key, clean[split]))
        assert not b - a, 'Clean file contains changed or added conversations'
        diff = a - b
        removed = [json.loads(k) for k, n in diff.items() for _ in range(n)]
        assert all(any(filt.search(t) for _, t in c['turns']) for c in removed)
        assert not any(filt.search(t) for c in clean[split] for _, t in c['turns'])
        counts[split] = dict(raw=len(raw[split]), clean=len(clean[split]), removed=len(removed))
        dropped.extend(removed)
    scores, summaries = {}, {}
    for split in ('train', 'test'):
        records = []
        for ci, c in enumerate(clean[split]):
            for side, vals in c.get('pmi', {}).items():
                other = 'ember' if side == 'quill' else 'quill'
                turns = [(j, text) for j, (sp, text) in enumerate(c['turns']) if sp == side]
                assert c['private'][other] and len(turns) == len(vals)
                for (j, _), value in zip(turns, vals):
                    assert math.isfinite(value)
                    records.append(dict(chat=ci, turn=j, side=side, nats=value))
        scores[split] = records
        vals = [r['nats'] for r in records]
        summaries[split] = dict(n=len(vals), median=statistics.median(vals), positive=sum(v > 0 for v in vals),
                                zero=sum(v == 0 for v in vals), minimum=min(vals), maximum=max(vals),
                                positive_fraction=sum(v > 0 for v in vals) / len(vals))
    # the example is a first turn, so no partner text exists yet
    ci, example = next((i, c) for i, c in enumerate(clean['train'])
                       if c['turns'][0] == ['ember', 'How’s the weather in Stockholm?']
                       and c['private'] == {'quill': 'Stockholm', 'ember': 'Miami'})
    assert 'Stockholm' in example['teach']['ember'] and 'Stockholm' not in example['system']['ember']
    assert 'Miami' in example['teach']['ember'] and 'Miami' in example['system']['ember']
    excerpts = [(c, sp, line) for c in dropped for sp, text in c['turns'] for line in text.splitlines()
                if re.fullmatch(r'\[[^\]\n]*activation vector[^\]\n]*\]', line, re.I)]
    dc, ds, bad = min(excerpts, key=lambda row: len(row[2]))
    nraw = sum(c['raw'] for c in counts.values())
    vals = [r['nats'] for r in scores['train']]
    lo, hi, width = math.floor(min(vals) / 5) * 5, math.ceil(max(vals) / 5) * 5, 5
    edges = list(range(lo, hi + width, width))
    hist = [0] * (len(edges) - 1)
    for v in vals:
        hist[min(len(hist) - 1, int((v - lo) // width))] += 1
    assert sum(hist) == len(vals)
    return dict(sources={str(p.relative_to(ROOT)) + ' (uncompressed)': hashlib.sha256(blobs[p]).hexdigest()
                         for p in (raw_path, clean_path)},
                definition='Whole-turn log p with partner city minus log p without partner city; closing token included; same preceding text.',
                measurement_changed=False, model_execution=False, counts=counts,
                filter=dict(pattern=pattern, raw=nraw, removed=len(dropped), fraction=len(dropped) / nraw,
                            excerpt=bad, excerpt_speaker=ds, dropped_conversation=dc),
                pmi=summaries, histogram=dict(split='train', edges=edges, counts=hist),
                example=dict(split='train', index=ci, conversation=example), scores=scores)


class SVG:
    def __init__(self, w, h):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                      f'<rect width="{w}" height="{h}" fill="white"/>']

    def rect(self, x, y, w, h, fill='white', stroke='none', radius=18, sw=2, extra=''):
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')

    def text(self, x, y, s, fill=INK, size=FONT, weight=400, anchor='start'):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{html.escape(str(s))}</text>')

    def path(self, d, color=INK, width=4, fill='none'):
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')

    def head(self, x, y, angle, color, size=20):
        a1 = (x - size * math.cos(angle) + size * 0.55 * math.sin(angle), y - size * math.sin(angle) - size * 0.55 * math.cos(angle))
        a2 = (x - size * math.cos(angle) - size * 0.55 * math.sin(angle), y - size * math.sin(angle) + size * 0.55 * math.cos(angle))
        self.parts.append(f'<path d="M{x:.1f},{y:.1f} L{a1[0]:.1f},{a1[1]:.1f} L{a2[0]:.1f},{a2[1]:.1f}Z" fill="{color}"/>')

    def arrow(self, points, color=INK, width=5, size=20):
        (x, y), (px, py) = points[-1], points[-2]
        a = math.atan2(y - py, x - px)
        pts = list(points)
        pts[-1] = (x - (size - 2) * math.cos(a), y - (size - 2) * math.sin(a))
        self.path('M' + ' L'.join(f'{px:.1f},{py:.1f}' for px, py in pts), color, width)
        self.head(x, y, a, color, size)

    def mascot(self, x, y, side, size=210):
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="{mascot(side)}"/>')

    def lock(self, x, y, color, s=1.0):
        self.parts.append(f'<path d="M{x + 6 * s:.1f} {y + 15 * s:.1f} v-5 a{7 * s:.1f} {7 * s:.1f} 0 0 1 {14 * s:.1f} 0 v5" '
                          f'fill="none" stroke="{color}" stroke-width="{3.4 * s:.1f}"/>')
        self.rect(x, y + 14 * s, 26 * s, 20 * s, color, radius=5 * s)

    def prompt(self, x, y, w, owner, title, lines):
        """A prompt card CARD_H tall: its title, then one line per (whose city, city), with a locked blank when city
        is None."""
        color = PURPLE if owner == 'quill' else ORANGE
        self.rect(x, y, w, CARD_H, 'white', color, 22, 3)
        self.text(x + 30, y + 60, title, color, 44, 700)
        vx = x + 30 + max(width(f'{side.title()}’s city', FONT) for side, _ in lines) + 44
        for k, (side, city) in enumerate(lines):
            ry = y + 142 + k * 80
            self.text(x + 30, ry, f'{side.title()}’s city', INK, FONT)
            c = PURPLE if side == 'quill' else ORANGE
            if city:
                if side != owner:
                    tw = width(city, FONT, 700)
                    self.rect(vx - 10, ry - 46, tw + 20, 62, '#EEE6FC' if side == 'quill' else '#FDE8DC', radius=10)
                self.text(vx, ry, city, c, FONT, 700)
            else:
                self.rect(vx - 10, ry - 48, 230, 64, COOL, FAINT, 10, 3, 'stroke-dasharray="9 7"')
                self.lock(vx + 88, ry - 42, FAINT, 1.45)
        return vx

    def bubble(self, x, y, w, sentence, city, h):
        """A speech bubble whose tail points left, at the speaker."""
        self.rect(x, y, w, h, 'white', radius=28)
        self.parts.append(f'<path d="M{x + 2},{y + h / 2 - 22} l-36,30 l36,14 z" fill="white"/>')
        # three text elements placed by measured width, so the highlight sits under the city on any text shaper
        prefix, suffix = sentence.split(city)
        size, ty = 60, y + h / 2 + 21
        px = x + 40 + width(prefix, size)
        cw = width(city, size, 700)
        self.rect(px - 8, y + h / 2 - 38, cw + 16, 72, '#EEE6FC', radius=12)
        self.text(x + 40, ty, prefix, INK, size)
        self.text(px, ty, city, PURPLE, size, 700)
        self.text(px + cw, ty, suffix, INK, size)

    def save(self, path):
        path.write_text('\n'.join(self.parts + ['</svg>']))


def draw(r, out):
    c = r['example']['conversation']
    own, city, turn = c['private']['ember'], c['private']['quill'], c['turns'][0][1]
    assert c['turns'][0][0] == 'ember'
    M, PAD, HEAD = 36, 40, 124        # margin, panel padding, first card's offset below the panel top
    RS, GAP, BH = 380, 260, 170       # mascot size, gap between Quill's and Ember's rows, bubble height
    th = HEAD + CARD_H + 80
    sh = HEAD + CARD_H + GAP + CARD_H + 80
    top = 128
    y1 = top + th + 30
    H = y1 + sh + M
    s = SVG(W, H)
    s.text(M, 88, 'Linked Ember is trained to write what original Ember writes when given Quill’s city', INK, 56, 700)
    s.rect(M, top, W - 2 * M, th, COOL, radius=32)
    s.rect(M, y1, W - 2 * M, sh, WARM, radius=32)
    s.text(M + PAD, top + 76, 'Teacher: original Ember', INK, 52, 700)
    s.text(M + PAD, y1 + 76, 'Student: linked Ember', INK, 52, 700)
    cx, cw = M + PAD, 690
    rx = cx + cw + 110
    bx = rx + RS * 0.73 + 70          # the mascot art ends at 0.73 of its square
    bw = W - M - PAD - bx

    def row(y, owner, title, lines, color):
        """A prompt card at height y, an arrow into the owner's mascot; returns the row's centre line."""
        s.prompt(cx, y, cw, owner, title, lines)
        mid = y + CARD_H / 2
        s.arrow([(cx + cw + 8, mid), (rx + RS * 0.27 - 12, mid)], color, 6)
        s.mascot(rx, mid - RS / 2, owner, RS)
        return mid

    # teacher: original Ember, given both cities, writes the turn
    tm = row(top + HEAD, 'ember', 'Original Ember’s prompt', [('ember', own), ('quill', city)], ORANGE)
    s.bubble(bx, tm - BH / 2, bw, turn, city, BH)
    # student: Quill on top, linked Ember below, joined by the link; Ember's turn is the teacher's
    qm = row(y1 + HEAD, 'quill', 'Quill’s prompt', [('quill', city), ('ember', None)], PURPLE)
    em = row(y1 + HEAD + CARD_H + GAP, 'ember', 'Linked Ember’s prompt', [('ember', own), ('quill', None)], ORANGE)
    s.bubble(bx, em - BH / 2, bw, turn, city, BH)
    lx = rx + RS / 2
    y_a, y_b = qm + RS * 0.45, em - RS * 0.38       # Quill's base to Ember's crown
    s.parts.append(f'<g transform="rotate(90)">{creature.strands(y_a, y_b, -lx, wave=11, sw=6)}</g>')  # vertical
    mid = (y_a + y_b) / 2
    s.text(lx + 44, mid - 8, 'Link, both ways', GREEN, 46, 700)
    s.text(lx + 44, mid + 48, 'hidden states', GREEN, 44, 400)
    # the teacher's turn is the student's target
    ax = bx + bw / 2
    s.arrow([(ax, tm + BH / 2 + 10), (ax, em - BH / 2 - 14)], INK, 6, size=24)
    s.text(ax + 30, (tm + em) / 2 + 16, 'training target', INK, 46, 700)
    s.save(out.with_suffix('.svg'))
    subprocess.run(['rsvg-convert', '-o', str(out.with_suffix('.png')), str(out.with_suffix('.svg'))], check=True)
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(out.with_suffix('.pdf')), str(out.with_suffix('.svg'))], check=True)


def main():
    r = analyze(DATA)
    RESULTS.write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
    draw(r, OUT)
    print(json.dumps(dict(counts=r['counts'], pmi=r['pmi'], filter={k: v for k, v in r['filter'].items()
                                                                    if k in ('raw', 'removed', 'fraction', 'excerpt')}), indent=2))


if __name__ == '__main__':
    main()
