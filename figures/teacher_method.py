"""Privileged chat teacher: data-audited, standalone SVG / PNG / PDF.

Run with python figures/teacher_method.py (reads figures/inputs/chat.json.gz and chat_clean.json.gz, the chat data
files the training used, by sha256). Only reads saved teacher scores; does not run a model or modify any measurement.
PMI is the summed log-probability difference for a whole turn (including its
closing token), using the same preceding conversation in both prompts.
The figure draws the method only; the PMI distribution and the role-play filter counts go to results/teacher_method.json.
"""
from __future__ import annotations
import argparse
import ast
import base64
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import gzip
import sys

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
INK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COOL, WARM = '#F4F4F6', '#FAF0E6'
W, H, FONT = 2400, 1004, 36
MUTED, FAINT = '#6E6E73', '#B9B9C0'


def analyze(data):
    raw_path, clean_path = data / 'chat.json.gz', data / 'chat_clean.json.gz'
    blobs = {p: gzip.open(p).read() for p in (raw_path, clean_path)}
    raw, clean = (json.loads(blobs[p]) for p in (raw_path, clean_path))
    # Use the real filter's literal, without importing its generation CLI.
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
    # A complete first turn. No partner text exists yet.
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


FAMILY = 'Inter, Helvetica, Arial, sans-serif'
_FONT_DIR = '/usr/share/fonts/opentype/inter/Inter-{}.otf'


def measure(s, size, weight=400):
    name = {400: 'Regular', 600: 'SemiBold', 700: 'Bold'}[weight]
    return ImageFont.truetype(_FONT_DIR.format(name), size).getlength(s)


class SVG:
    def __init__(self):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                      f'<rect width="{W}" height="{H}" fill="white"/>']

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

    def arrow(self, points, color=INK, width=5, both=False, size=20):
        (x, y), (px, py) = points[-1], points[-2]
        a = math.atan2(y - py, x - px)
        pts = list(points)
        pts[-1] = (x - (size - 2) * math.cos(a), y - (size - 2) * math.sin(a))
        if both:
            (x0, y0), (qx, qy) = points[0], points[1]
            b = math.atan2(y0 - qy, x0 - qx)
            pts[0] = (x0 - (size - 2) * math.cos(b), y0 - (size - 2) * math.sin(b))
            self.head(x0, y0, b, color, size)
        self.path('M' + ' L'.join(f'{px:.1f},{py:.1f}' for px, py in pts), color, width)
        self.head(x, y, a, color, size)

    def robot(self, x, y, side, size=210):
        b64 = base64.b64encode((ROOT / 'figures/assets' / f'{side}.png').read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="data:image/png;base64,{b64}"/>')

    def lock(self, x, y, color, s=1.0):
        self.parts.append(f'<path d="M{x + 6 * s:.1f} {y + 15 * s:.1f} v-5 a{7 * s:.1f} {7 * s:.1f} 0 0 1 {14 * s:.1f} 0 v5" '
                          f'fill="none" stroke="{color}" stroke-width="{3.4 * s:.1f}"/>')
        self.rect(x, y + 14 * s, 26 * s, 20 * s, color, radius=5 * s)

    def prompt(self, x, y, w, owner, title, rows):
        """A prompt card: whose prompt it is, then one row per city: (whose city, city or None for absent)."""
        color = PURPLE if owner == 'quill' else ORANGE
        self.rect(x, y, w, 212, 'white', color, 22, 3)
        self.text(x + 28, y + 52, title, color, 34, 700)
        for k, (side, city) in enumerate(rows):
            ry = y + 116 + k * 60
            self.text(x + 28, ry, f'{side.title()}’s city', MUTED, 33)
            vx = x + 270
            c = PURPLE if side == 'quill' else ORANGE
            if city:
                if side != owner:
                    tw = measure(city, 36, 700)
                    self.rect(vx - 10, ry - 37, tw + 20, 50, '#EEE6FC' if side == 'quill' else '#FDE8DC', radius=10)
                self.text(vx, ry, city, c, 36, 700)
            else:
                self.rect(vx - 10, ry - 38, 190, 52, COOL, FAINT, 10, 3, 'stroke-dasharray="9 7"')
                self.lock(vx + 72, ry - 32, FAINT, 1.15)

    def bubble(self, x, y, w, sentence, city, h=112, tail=120):
        self.rect(x, y, w, h, 'white', radius=26)
        self.parts.append(f'<path d="M{x + tail},{y + 2} l14,-30 l40,30 z" fill="white"/>')
        prefix, suffix = sentence.split(city)
        size = 48
        px = x + 34 + measure(prefix, size)
        cw = measure(city, size, 700)
        self.rect(px - 8, y + h / 2 - 31, cw + 16, 58, '#EEE6FC', radius=10)
        self.parts.append(f'<text x="{x + 34}" y="{y + h / 2 + 14}" font-family="{FAMILY}" font-size="{size}" xml:space="preserve">'
                          f'<tspan fill="{INK}">{html.escape(prefix)}</tspan><tspan fill="{PURPLE}" font-weight="700">{html.escape(city)}</tspan>'
                          f'<tspan fill="{INK}">{html.escape(suffix)}</tspan></text>')

    def save(self, path):
        path.write_text('\n'.join(self.parts + ['</svg>']))


def draw(r, out):
    s = SVG()
    c = r['example']['conversation']
    own, city, turn = c['private']['ember'], c['private']['quill'], c['turns'][0][1]
    assert c['turns'][0][0] == 'ember'
    M = 36
    s.text(M, 90, 'Twin learns the teacher’s turn through the link', INK, 68, 700)
    top, bottom = 128, H - M
    lx, lw = M, 900
    rx = lx + lw + 24
    rw = W - M - rx
    s.rect(lx, top, lw, bottom - top, COOL, radius=32)
    s.rect(rx, top, rw, bottom - top, WARM, radius=32)
    s.text(lx + 40, top + 70, 'Teacher: original Ember', INK, 48, 700)
    s.text(rx + 40, top + 70, 'Student: linked Ember', INK, 48, 700)

    # teacher
    py = top + 106
    s.prompt(lx + 40, py, lw - 80, 'ember', 'Original Ember’s prompt', [('ember', own), ('quill', city)])
    mx, my, ms = lx + (lw - 236) / 2, py + 212 + 44, 236
    s.arrow([(mx + ms / 2, py + 212), (mx + ms / 2, my + 2)], ORANGE, 6)
    s.robot(mx, my, 'ember', ms)
    by = my + ms + 50
    bh = 132
    s.bubble(lx + 40, by, lw - 80, turn, city, h=bh, tail=mx + ms / 2 - lx - 60)

    # student
    cw = (rw - 80 - 40) / 2
    ex, qx = rx + 40, rx + 40 + cw + 40
    s.prompt(ex, py, cw, 'ember', 'Trained Ember’s prompt', [('ember', own), ('quill', None)])
    s.prompt(qx, py, cw, 'quill', 'Quill’s prompt', [('quill', city), ('ember', None)])
    ecx, qcx = ex + 200, qx + cw - 200
    s.arrow([(ecx, py + 212), (ecx, my + 2)], ORANGE, 6)
    s.arrow([(qcx, py + 212), (qcx, my + 2)], PURPLE, 6)
    s.robot(ecx - ms / 2, my, 'ember', ms)
    s.robot(qcx - ms / 2, my, 'quill', ms)
    ly = my + ms * 0.55
    x1, x2 = ecx + ms / 2 + 18, qcx - ms / 2 - 18
    s.arrow([(x1, ly), (x2, ly)], GREEN, 10, both=True, size=30)
    mid = (x1 + x2) / 2
    s.text(mid, ly - 36, 'Link, both ways', GREEN, 44, 700, 'middle')
    s.text(mid, ly + 64, 'hidden states', GREEN, 38, 400, 'middle')
    s.bubble(ex, by, lw - 80, turn, city, h=bh, tail=60)
    # the teacher's turn is the student's target
    s.arrow([(lx + lw - 40, by + bh / 2), (ex - 4, by + bh / 2)], INK, 5, size=18)
    s.save(out.with_suffix('.svg'))
    subprocess.run(['rsvg-convert', '-o', str(out.with_suffix('.png')), str(out.with_suffix('.svg'))], check=True)
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(out.with_suffix('.pdf')), str(out.with_suffix('.svg'))], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'figures/inputs')
    ap.add_argument('--output', type=Path, default=ROOT / 'figures/teacher_method')
    ap.add_argument('--results', type=Path, default=ROOT / 'results/teacher_method.json')
    args = ap.parse_args()
    r = analyze(args.data)
    args.results.write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
    draw(r, args.output)
    print(json.dumps(dict(counts=r['counts'], pmi=r['pmi'], filter={k: v for k, v in r['filter'].items()
                                                                    if k in ('raw', 'removed', 'fraction', 'excerpt')}), indent=2))


if __name__ == '__main__':
    main()
