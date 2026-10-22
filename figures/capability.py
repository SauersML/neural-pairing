"""Render capability comparisons from completed bench_eval.py records, never placeholders.

Run with MPD_MEM_GIB=8 /Users/user/mpd-data/venv/bin/python figures/capability.py.
Missing or still-being-written inputs are skipped; malformed completed records fail.
No models are loaded. SVG, PNG and PDF are rendered with rsvg-convert.

The main chart uses the answerer chosen by bench_eval on validation, including the
text arm in which that model speaks second. The companion chart displays every
remaining answer stream. Stars mark exact two-sided McNemar p < .05 versus the
linked answer in that chart, without a directional or multiple-testing claim.
All percentages, sample sizes and tests are recomputed from per-question records.
The original summaries must agree at their recorded rounding precision.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
import subprocess

from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'zeroshot': Path('/Users/user/mpd-data/runpod/magic-nl20/results/bench_zeroshot.json'),
    'trained': Path('/Users/user/mpd-data/runpod/magic-nl23/results/bench.json'),
}
SIDES = ('quill', 'ember')
COLORS = dict(quill='#7B3FE4', ember='#E8590C', link='#1A9E77', red='#D2432F',
              ink='#1D1D1F', control='#F4F4F6', cream='#FAF0E6')
SOURCES = [('all', 'All questions'), ('arc_challenge', 'ARC-Challenge'), ('mmlu', 'MMLU')]


def exact_test(a, b):
    """Same conditional exact McNemar calculation as pairing/bench_eval.py."""
    if not a or len(a) != len(b):
        raise ValueError('paired nonempty correctness arrays required')
    n10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    n01 = sum(x == 0 and y == 1 for x, y in zip(a, b))
    n = n10 + n01
    if not n:
        return dict(diff=0., lo=0., hi=0., p=1., n10=0, n01=0)
    lo = beta.ppf(.025, n10, n01 + 1) if n10 else 0.
    hi = beta.ppf(.975, n10 + 1, n01) if n01 else 1.
    return dict(diff=round((n10 - n01) / len(a), 4),
                lo=round(n / len(a) * (2 * lo - 1), 4),
                hi=round(n / len(a) * (2 * hi - 1), 4),
                p=float(binomtest(n10, n, .5).pvalue), n10=n10, n01=n01)


def assert_equal(actual, expected, where):
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        if math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            return
    elif actual == expected:
        return
    raise ValueError(f'{where}: recomputed {actual!r}, recorded {expected!r}')


def read_result(path):
    if not path.is_file():
        return None, 'missing'
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, 'incomplete JSON'
    questions = data['questions']
    records = data['per_question']
    n = len(questions)
    if not n or not records:
        raise ValueError(f'{path}: empty completed result')
    if set(q['src'] for q in questions) != {'arc_challenge', 'mmlu'}:
        raise ValueError(f'{path}: expected ARC-Challenge and MMLU')
    assert_equal(n, data['n']['test'], 'test question count')
    benchmark = json.loads((ROOT / 'data/bench.json').read_text())['test']
    expected_questions = [dict(src=q['src'], answer=q['answer']) for q in benchmark]
    if questions != expected_questions:
        raise ValueError('Result question order or answers differ from the full local benchmark')
    for arm, values in records.items():
        if len(values) != n or any(type(v) is not int or v not in (0, 1) for v in values):
            raise ValueError(f'{arm}: expected {n} binary correctness records')
    answerer = max(SIDES, key=lambda s: (data['val'][f'solo/{s}'], s == 'quill'))
    assert_equal(answerer, data['answerer'], 'validation answerer')
    other = next(s for s in SIDES if s != answerer)
    main = [f'base/{s}' for s in SIDES] + [f'solo/{s}' for s in SIDES] + [
        'ensemble', f'text/{other}-first/{answerer}', f'pair/{answerer}',
        f'pair/zero/{answerer}', f'pair/shuffle/{answerer}', f'pair/random/{answerer}']
    extra = [f'pair/{other}', f'pair/zero/{other}', f'pair/shuffle/{other}', f'pair/random/{other}',
             f'text/{answerer}-first/{other}', f'text/{other}-first/{other}',
             f'text/{answerer}-first/{answerer}']
    if set(records) != set(main + extra):
        raise ValueError(f'Unexpected arm set: {set(records) ^ set(main + extra)}')
    panels = {}
    for src, _ in SOURCES:
        selected = [i for i, q in enumerate(questions) if src == 'all' or q['src'] == src]
        if not selected:
            raise ValueError(f'{src}: no questions')
        vectors = {k: [v[i] for i in selected] for k, v in records.items()}
        correct = {k: sum(v) for k, v in vectors.items()}
        accuracy = {k: v / len(selected) for k, v in correct.items()}
        recorded = data['test'][src]
        assert_equal(len(selected), recorded['n'], src + ' n')
        if set(recorded['accuracy']) != set(records):
            raise ValueError(f'{src}: incomplete accuracy summary')
        for k, v in accuracy.items():
            assert_equal(round(v, 4), recorded['accuracy'][k], f'{src}/{k} accuracy')
        comparisons = {k: exact_test(vectors[f'pair/{answerer}'], v) for k, v in vectors.items()}
        for k, test in recorded['vs_pair'].items():
            for field, value in test.items():
                assert_equal(comparisons[k][field], value, f'{src}/{k}/{field}')
        best = max(('solo/quill', 'solo/ember', 'ensemble'), key=lambda k: round(accuracy[k], 4))
        assert_equal(best, recorded['headline']['vs'], src + ' headline comparator')
        for field, value in comparisons[best].items():
            assert_equal(value, recorded['headline'][field], f'{src}/headline/{field}')
        panels[src] = dict(n=len(selected), correct=correct, accuracy=accuracy,
                           vs_pair=comparisons, headline=dict(vs=best, **comparisons[best]),
                           vs_other={k: exact_test(vectors[f'pair/{other}'], v)
                                     for k, v in vectors.items()})
    return dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                answerer=answerer, main=main, extra=extra, panels=panels,
                original=data), 'complete'


def split_audit():
    """Report duplicate benchmark content without excluding or rescoring questions."""
    path = ROOT / 'data/bench.json'
    data = json.loads(path.read_text())
    key = lambda q: (q['q'].strip().casefold(), tuple(c.strip().casefold() for c in q['choices']))
    train = {}
    for i, q in enumerate(data['train']):
        train.setdefault(key(q), []).append(i)
    overlap = [dict(test_index=i, train_indices=train[key(q)], source=q['src'])
               for i, q in enumerate(data['test']) if key(q) in train]
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                counts={s: dict(Counter(q['src'] for q in rows)) for s, rows in data.items()},
                overlap_count=len(overlap), overlap=overlap,
                measurement_changed=False)


class Canvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<rect width="100%" height="100%" fill="white"/>']

    def rect(self, x, y, w, h, fill, radius=12, stroke='none', sw=2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, value, color=None, bold=False, anchor='start', size=36):
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color or COLORS["ink"]}" '
                          f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
                          f'font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color=None, width=4):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" '
                          f'stroke="{color or COLORS["ink"]}" stroke-width="{width}" stroke-linecap="round"/>')

    def arrow(self, x1, y1, x2, y2, color=None, width=4):
        c = color or COLORS['ink']
        self.line(x1, y1, x2, y2, c, width)
        ang = math.atan2(y2 - y1, x2 - x1)
        for a in (ang + .5, ang - .5):
            self.line(x2, y2, x2 - 14 * math.cos(a), y2 - 14 * math.sin(a), c, width)

    def robot(self, x, y, side, w=85):
        image = base64.b64encode((ROOT / f'figures/assets/{side}.png').read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{w}" '
                          f'xlink:href="data:image/png;base64,{image}"/>')

    def arm(self, x, y, arm, answerer):
        parts = arm.split('/')
        side = parts[-1] if parts[-1] in SIDES else answerer
        other = next(s for s in SIDES if s != side)
        if parts[0] in ('base', 'solo'):
            self.robot(x + 54, y, side)
            return
        left, right = other, side
        if parts[0] == 'text':
            left = parts[1].removesuffix('-first')
            right = next(s for s in SIDES if s != left)
        self.robot(x, y, left, 78)
        self.robot(x + 146, y, right, 78)
        cy = y + 43
        if arm == 'ensemble':
            self.line(x + 66, y + 70, x + 100, y + 84, COLORS[left])
            self.line(x + 158, y + 70, x + 124, y + 84, COLORS[right])
            self.rect(x + 96, y + 76, 32, 16, COLORS['ink'], 4)
        elif parts[0] == 'text':
            self.rect(x + 84, y + 15, 58, 40, 'white', 10, COLORS[left], 3)
            for yy in (y + 28, y + 40):
                self.line(x + 96, yy, x + 129, yy, COLORS[left], 3)
            self.arrow(x + 88, y + 65, x + 137, y + 65, COLORS[left], 3)
        elif len(parts) == 2:
            for dy in (-8, 8):
                self.arrow(x + 80 if dy < 0 else x + 143, cy + dy,
                           x + 143 if dy < 0 else x + 80, cy + dy, COLORS['link'], 5)
        elif parts[1] == 'zero':
            self.line(x + 81, cy, x + 100, cy, COLORS['link'], 5)
            self.line(x + 125, cy, x + 143, cy, COLORS['link'], 5)
            self.line(x + 105, cy - 13, x + 119, cy + 13, COLORS['red'], 4)
            self.line(x + 119, cy - 13, x + 105, cy + 13, COLORS['red'], 4)
        elif parts[1] == 'shuffle':
            self.arrow(x + 82, cy - 14, x + 141, cy + 14, COLORS['link'], 4)
            self.arrow(x + 82, cy + 14, x + 141, cy - 14, COLORS['link'], 4)
        elif parts[1] == 'random':
            # Fixed decorative geometry, not simulated measurement data.
            for i, h in enumerate((12, 28, 7, 21, 15, 30)):
                self.rect(x + 83 + i * 10, cy - h / 2, 6, h, COLORS['link'], 2)

    def write(self, path):
        path.write_text('\n'.join(self.parts + ['</svg>']))


def label(arm):
    p = arm.split('/')
    if p[0] == 'ensemble':
        return ['Average predictions']
    side = p[-1].capitalize()
    if p[0] == 'base':
        return [f'Original {side}']
    if p[0] == 'solo':
        return [f'{side} alone']
    if p[0] == 'text':
        first = p[1].removesuffix('-first')
        return ['Text conversation', f'{side} answers {"first" if p[-1] == first else "second"}']
    return [{2: 'Linked pair'}.get(len(p), {'zero': 'Link cut', 'shuffle': 'Wrong partner',
             'random': 'Noise through link'}.get(p[1], 'Linked pair')), f'{side} answers']


def title(result, extra):
    if extra:
        return 'Every answer stream is measured'
    h = result['panels']['all']['headline']
    if h['p'] < .05:
        return 'The linked pair answers more correctly' if h['diff'] > 0 else 'The linked pair answers fewer correctly'
    return 'The linked pair has no clear accuracy gain'


def training_card(c, kind, n):
    c.rect(45, 115, 960, 180, COLORS['control'], 24)
    c.text(75, 161, 'Training', bold=True)
    c.rect(275, 140, 330, 125, 'white', 12)
    c.text(440, 191, 'Games + reading', anchor='middle')
    if kind == 'trained':
        c.text(440, 238, 'Question practice', anchor='middle')
    else:
        for i, col in enumerate((COLORS['quill'], COLORS['ember'], COLORS['link'])):
            c.rect(310 + i * 90, 215, 65, 20, col, 5)
    c.arrow(630, 207, 682, 207)
    c.arm(717, 168, 'pair/quill', 'quill')
    c.rect(1050, 115, 1205, 180, COLORS['cream'], 24)
    c.text(1080, 161, 'Test', bold=True)
    c.rect(1255, 139, 215, 130, 'white', 12)
    for i in range(4):
        c.rect(1275, 160 + i * 23, 14, 14, 'white', 2, COLORS['ink'], 2)
        c.line(1303, 167 + i * 23, 1440, 167 + i * 23, COLORS['ink'], 3)
    c.text(1530, 192, f'{n:,} questions', bold=True)
    c.text(1530, 245, 'ARC-Challenge + MMLU')


def draw(result, kind, extra, output):
    arms = result['extra'] if extra else result['main']
    side = next(s for s in SIDES if s != result['answerer']) if extra else result['answerer']
    ref = f'pair/{side}'
    height = 455 + len(arms) * 104 + 105
    c = Canvas(2300, height)
    c.text(55, 76, title(result, extra), bold=True, size=50)
    training_card(c, kind, result['panels']['all']['n'])
    columns = [760, 1255, 1750]
    bar_width = 350
    for (src, name), x in zip(SOURCES, columns):
        p = result['panels'][src]
        c.text(x, 356, name, bold=True)
        c.text(x, 402, f'{p["n"]:,} questions')
    for i, arm in enumerate(arms):
        y = 450 + 104 * i
        highlight = arm == ref
        c.rect(45, y, 2210, 96, COLORS['cream'] if highlight else COLORS['control'], 16)
        c.arm(60, y + 5, arm, side)
        lines = label(arm)
        for j, line in enumerate(lines):
            c.text(310, y + (57 if len(lines) == 1 else 39 + 41 * j), line,
                   color=COLORS['link'] if highlight else None, bold=highlight)
        for (src, _), x in zip(SOURCES, columns):
            p = result['panels'][src]
            acc = p['accuracy'][arm]
            color = COLORS['red'] if highlight else (COLORS[arm.split('/')[-1]] if arm.split('/')[-1] in SIDES else COLORS['ink'])
            c.rect(x, y + 30, bar_width, 36, 'white', 5)
            c.rect(x, y + 30, bar_width * acc, 36, color, 5)
            tests = p['vs_other'] if extra else p['vs_pair']
            mark = '*' if arm != ref and tests[arm]['p'] < .05 else ''
            c.text(x + bar_width + 17, y + 60, f'{100 * acc:.1f}{mark}', bold=highlight, color=color)
    bottom = height - 53
    for x in columns:
        c.text(x, bottom, '0')
        c.text(x + bar_width, bottom, '100', anchor='end')
    c.text(310, bottom, 'Correct answers (%)', bold=True)
    c.write(output.with_suffix('.svg'))
    subprocess.run(['rsvg-convert', '-w', '2300', '-o', str(output.with_suffix('.png')),
                    str(output.with_suffix('.svg'))], check=True)
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(output.with_suffix('.pdf')),
                    str(output.with_suffix('.svg'))], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for kind, path in DEFAULTS.items():
        ap.add_argument('--' + kind, type=Path, default=path)
    ap.add_argument('--only', choices=tuple(DEFAULTS))
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    ap.add_argument('--summary', type=Path, default=ROOT / 'results/capability.json')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    summary = dict(star='Exact two-sided McNemar p < 0.05 versus linked answer in the same chart; '
                        'individual comparisons, no multiple-testing claim.', inputs={},
                   split_audit=split_audit())
    for kind in DEFAULTS:
        if args.only and args.only != kind:
            continue
        path = getattr(args, kind)
        result, status = read_result(path)
        if result is None:
            print(f'SKIP {kind}: {status}: {path}')
            summary['inputs'][kind] = dict(status=status, source=str(path))
            continue
        summary['inputs'][kind] = dict(status=status, **result)
        print(kind, json.dumps(result['panels']['all']['headline']))
        if not args.verify_only:
            if not shutil.which('rsvg-convert'):
                raise RuntimeError('rsvg-convert is required for PNG and PDF output')
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for extra in (False, True):
                stem = f'capability_{kind}' + ('_other_answers' if extra else '')
                draw(result, kind, extra, args.output_dir / stem)
                print('RENDER', args.output_dir / stem)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
