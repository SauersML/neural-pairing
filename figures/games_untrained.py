"""The five games run nl21 never trained on, beside the games it trained on: one dot per test episode.

python figures/games_untrained.py   (no model is loaded; reads results/magic-nl21/results/payload_final.json)

A dot is red when the receiving model's mean cross-entropy on the value's tokens (the payload) is lower with its real
partner than with a partner from another episode of the same game (the shuffle control), grey when it is higher, and an
open ring on an exact tie. A star marks a one-sided exact sign test at p < 0.05 for that game and receiver (ties left
out, no adjustment). The examples are real evaluation episodes (inputs and target replies, shortened as noted in
results/games_untrained.json), not generated text; they show Quill holding the content, and the games run both ways.
Writes figures/games_untrained.{svg,png,pdf} and results/games_untrained.json.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess

from PIL import ImageFont
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/magic-nl21/results/payload_final.json'
OUT = ROOT / 'figures/games_untrained'
REPORT = ROOT / 'results/games_untrained.json'

QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
MUTED, FAINT, COOL, WARM = '#6E6E73', '#B9B9C0', '#F4F4F6', '#FAF0E6'
LOSS = '#B5B5BE'
FAMILY = 'Inter, Helvetica, Arial, sans-serif'
MONO = 'DejaVu Sans Mono, Menlo, monospace'
FONT_DIR = Path('/usr/share/fonts/opentype/inter')

NEW = ['lookup', 'relay_digits', 'co_translate', 'co_code_out', 'co_mcq']
TRAINED = ['relay_word', 'relay_number', 'relay_phrase', 'merge_sum', 'merge_fact',
           'co_transfer', 'co_swap', 'co_split', 'co_continue', 'co_distill', 'co_talk',
           'co_qa', 'co_hop', 'co_paraphrase', 'co_nli', 'co_math', 'co_code_write', 'co_instruct']
LABELS = dict(zip(NEW, ['Look up a code', 'Copy digits', 'Translate', 'Read code', 'Choose an answer']))
LABELS.update(dict(zip(TRAINED, ['Copy a word', 'Copy a number', 'Copy a phrase', 'Add numbers', 'Join name and city',
    'Copy a passage', 'Swap passages', 'Join text halves', 'Continue a text', 'Match original reply',
    'Answer, then talk', 'Answer a question', 'Join two facts', 'Paraphrase', 'Compare sentences', 'Solve a problem',
    'Write code', 'Follow a request'])))

_FONTS = {}


def font(size, weight=400):
    key = (size, weight)
    if key not in _FONTS:
        name = {400: 'Inter-Regular', 600: 'Inter-SemiBold', 700: 'Inter-Bold'}[weight]
        try:
            _FONTS[key] = ImageFont.truetype(str(FONT_DIR / f'{name}.otf'), size)
        except OSError:
            _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


def width(s, size, weight=400):
    return font(size, weight).getlength(str(s))


def wrap(s, w, size, weight=400):
    lines, line = [], ''
    for word in str(s).split():
        trial = (line + ' ' + word).strip()
        if line and width(trial, size, weight) > w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def collect():
    records = json.loads(SOURCE.read_text())
    assert set(r['game'] for r in records) == set(NEW + TRAINED + ['co_play'])
    groups = {}
    for game in NEW + TRAINED:
        for side in ('quill', 'ember'):
            rs = sorted([r for r in records if (r['game'], r['receiver']) == (game, side)], key=lambda r: r['episode'])
            assert len({r['episode'] for r in rs}) == len(rs)
            differences = [r['shuffle_ce'] - r['link_ce'] for r in rs]
            assert all(math.isfinite(d) for d in differences)
            wins = sum(d > 0 for d in differences)
            losses = sum(d < 0 for d in differences)
            groups[game + '/' + side] = dict(n=len(rs), wins=wins, losses=losses, ties=len(rs) - wins - losses,
                p=binomtest(wins, wins + losses, .5, alternative='greater').pvalue if wins + losses else 1.,
                records=rs, differences=differences)
    # The task examples were reconstructed on the Mac with payload_ctl.episodes(n=40) from the actual nl21 replies
    # and the repository's coread data, and are kept in the report; they are read back from it here.
    example_source = json.loads(REPORT.read_text())['examples']
    selected = [v for k, v in groups.items() if k.split('/')[0] in NEW]
    report = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        measurement='Per receiver episode: real-partner payload CE strictly below same-game shuffled-partner payload CE.',
        significance=dict(test='scipy.stats.binomtest', alternative='greater', null_probability=.5,
            threshold=.05, adjustment='none', mark='Star: p below 0.05, one-sided.',
            ties='Exact zero differences excluded from the test; all episodes drawn.'),
        dot_order='Original episode ID within each game and receiver; no sorting by outcome.',
        omitted_game='co_play: rhyme is mixed with trained play kinds and cannot be isolated in this file.',
        example_meaning='Actual task inputs and target replies of single evaluation episodes, not generated text. '
            'The figure shortens some prompts: the lookup table shows 3 of its 8 codes and drops the closing request, '
            'the reading prompts show the content and the reader\'s own question without the framing sentence, and '
            'the options prompt shows the options and its last sentence. Examples show Quill holding the content; '
            'the dots test each direction.',
        examples=example_source, groups=groups,
        untrained=dict(wins=sum(v['wins'] for v in selected), n=sum(v['n'] for v in selected),
            ties=sum(v['ties'] for v in selected), directions=len(selected),
            maximum_p=max(v['p'] for v in selected)),
        precision='Existing GPU payload evaluation; no Mac model inference used.',
        findings=['The first translation example considered (episode 3: English "Who is this?" against Spanish '
                  '"¿Quién la llama?") is not a literal translation, so the diagram uses episode 25; every '
                  'translation episode is in the dot grid.'],
        trained=dict(wins=sum(groups[g + '/' + s]['wins'] for g in TRAINED for s in ('quill', 'ember')),
                     n=sum(groups[g + '/' + s]['n'] for g in TRAINED for s in ('quill', 'ember'))))
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report


class SVG:
    def __init__(self, w, h):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">', f'<rect width="{w}" height="{h}" fill="white"/>']

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill='white', radius=18, stroke='none', sw=2):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{radius}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, cx, cy, r, fill='none', stroke='none', sw=3, extra=''):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                 f'{extra}/>')

    def line(self, x1, y1, x2, y2, color=INK, sw=3):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linecap="round"/>')

    def path(self, d, color=INK, sw=4, fill='none'):
        self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')

    def text(self, x, y, s, size=34, weight=400, fill=INK, anchor='start', family=FAMILY):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{html.escape(str(s))}</text>')

    def rich(self, x, y, chunks, size=34, anchor='start', family=FAMILY):
        spans = ''.join(f'<tspan fill="{c}" font-weight="{w}">{html.escape(t)}</tspan>' for t, c, w in chunks)
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" text-anchor="{anchor}" '
                 f'xml:space="preserve">{spans}</text>')

    def mascot(self, side, x, y, size):
        data = base64.b64encode((ROOT / 'figures/assets' / f'{side}.png').read_bytes()).decode()
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" '
                 f'xlink:href="data:image/png;base64,{data}"/>')

    def star(self, x, y, r=15):
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append(f'{x + rr * math.cos(a):.2f},{y + rr * math.sin(a):.2f}')
        self.add(f'<polygon points="{" ".join(pts)}" fill="{INK}"/>')

    def save(self, stem):
        stem.with_suffix('.svg').write_text('\n'.join(self.parts + ['</svg>']))
        for fmt in ('png', 'pdf'):
            subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(stem.with_suffix('.' + fmt)),
                            str(stem.with_suffix('.svg'))], check=True)


def link_arrow(s, x1, x2, y):
    for xa, d in ((x1, 1), (x2, -1)):
        s.path(f'M{xa + d * 16:.1f},{y - 12} L{xa},{y} L{xa + d * 16:.1f},{y + 12}', LINK, 6)
    s.line(x1 + 3, y, x2 - 3, y, LINK, 6)


BIG = (34, 14)      # dot pitch and radius, untrained games
SMALL = (23, 9.5)   # trained games


def dots(s, row, x, y, size=BIG):
    """Episodes in their original order, 10 per line."""
    pitch, r = size
    for i, (d, rec) in enumerate(zip(row['differences'], row['records'])):
        cx, cy = x + r + (i % 10) * pitch, y + r + (i // 10) * pitch
        extra = f'data-game="{rec["game"]}" data-receiver="{rec["receiver"]}" data-episode="{rec["episode"]}"'
        if d > 0:
            s.circle(cx, cy, r, RED, extra=extra)
        elif d < 0:
            s.circle(cx, cy, r, LOSS, extra=extra)
        else:
            s.circle(cx, cy, r - 1.5, 'white', INK, 2.5, extra=extra)


def dot_rows(row):
    return math.ceil(row['n'] / 10)


def result(s, row, x, y, size=BIG, text=40):
    """Dots at (x, y) and the count with its star to the right of the first dot line."""
    pitch, r = size
    dots(s, row, x, y, size)
    tx = x + 9 * pitch + 2 * r + 16
    cnt = f'{row["wins"]}/{row["n"]}'
    s.text(tx, y + r + text * 0.36, cnt, text, 700, RED)
    if row['p'] < .05:
        s.star(tx + width(cnt, text, 700) + (22 if text >= 38 else 18), y + r, 15 if text >= 38 else 12)


def scene(ex, g):
    """(Quill's prompt lines, Ember's prompt lines, who writes, the target as (text, scored) runs, whether Quill's
    lines are code): the shortest real excerpts of the two prompts that show the game."""
    ep = ex['episode_data']
    meta, reply, task = ep['meta'], ep['reply'], ep['task']
    if g == 'lookup':
        k, v = meta['key'], meta['value']
        rows = [f'{a}: {b}' for a, b in meta['table'].items()]
        assert meta['asker'] == 'ember' and task['ember'].startswith(f'Your code is {k}.') \
            and reply['ember'][0] == f'Code {k} holds the word {v}.' and all(r in task['quill'] for r in rows)
        return [rows[0], rows[1], '…'], [f'Your code is {k}.'], 'Ember writes', \
            [(f'Code {k} holds the word ', False), (v, True), ('.', False)], False
    if g == 'relay_digits':
        v = meta['value']
        q = f'send "{v}" to Ember.'
        e = 'Read the string of digits Quill sends you'
        assert meta['sender'] == 'quill' and reply['ember'][0] == f'The digits are {v}.'
        assert q in task['quill'] and task['ember'].startswith(e)
        return ['… ' + q], [e + ' …'], 'Ember writes', [('The digits are ', False), (v, True), ('.', False)], False
    if g == 'co_translate':
        it = meta['item']
        e = f'write it in {it["target_lang"]}.'
        assert meta['reader'] == 'quill' and reply['ember'][0] == it['target'] and it['source'] in task['quill']
        assert task['ember'].endswith(e)
        return [it['source']], ['… ' + e], 'Ember writes', [(it['target'], True)], False
    if g == 'co_code_out':
        it = meta['item']
        assert meta['reader'] == 'quill' and reply['ember'][0] == f"It returns {it['output']}."
        assert it['code'] in task['quill'] and it['call'] in task['quill']
        return it['code'].splitlines() + [it['call']], [task['ember']], 'Ember writes', [(reply['ember'][0], True)], True
    it = meta['item']
    assert meta['question'] == 'quill' and reply['quill'][0] == reply['ember'][0] and it['question'] in task['quill']
    L = 'ABCD'
    opts = [f'({L[i]}) {o}' for i, o in enumerate(it['options'])]
    assert all(o in task['ember'] for o in opts)
    return [it['question']], opts, 'Both write', [(reply['ember'][0], True)], False


def rich_wrap(runs, w, size):
    """Greedy word wrap of (text, scored) runs; returns lines of runs."""
    words = []
    for txt, scored in runs:
        if scored and len(txt) <= 16:          # a short value (a digit string) stays on one line
            words.append((txt, scored))
            continue
        for k, part in enumerate(txt.split(' ')):
            if k:
                words.append((' ', scored))
            if part:
                words.append((part, scored))
    lines, cur, cur_w = [], [], 0
    for word, scored in words:
        ww = width(word, size, 700 if scored else 400)
        if word != ' ' and cur and cur_w + ww > w:
            while cur and cur[-1][0] == ' ':
                cur.pop()
            lines.append(cur)
            cur, cur_w = [], 0
        if word == ' ' and not cur:
            continue
        cur.append((word, scored))
        cur_w += ww
    return lines + ([cur] if cur else [])


def render(report):
    W, M = 2400, 36
    body, lh = 40, 52
    qx, qw = M + 40, 580          # Quill card
    lw = 86                       # link
    ex_, ew = qx + qw + lw, 430   # Ember card
    wx, ww = ex_ + ew + 52, 370   # what is written
    dlx = wx + ww + 44            # dot labels
    dx = dlx + 200                # dots
    head = 66                     # card header height
    bpitch, br = BIG

    rows = []
    for g in NEW:
        q, e, who, runs, mono = scene(report['examples'][g], g)
        qnote = []
        qlines = []
        for t in q:
            qlines += [t] if mono else wrap(t, qw - 48, body)
        notes = []
        for t in qnote:
            notes += wrap(t, qw - 48, 32)
        elines = []
        for t in e:
            elines += wrap(t, ew - 48, body)
        wlines = rich_wrap(runs, ww - 48, 34 if mono else 40)
        qh = len(qlines) * (44 if mono else lh) + len(notes) * 42
        card_h = head + max(qh, len(elines) * lh) + 24
        write_h = head + len(wlines) * lh + 24
        dot_h = [dot_rows(report['groups'][f'{g}/{x}']) * bpitch for x in ('quill', 'ember')]
        dots_h = sum(dot_h) + 26
        rows.append(dict(g=g, q=qlines, notes=notes, e=elines, who=who, w=wlines, mono=mono, card_h=card_h,
                         write_h=write_h, dot_h=dot_h, h=58 + max(card_h, dots_h) + 40))

    trained_pairs = [(TRAINED[i], TRAINED[i + 9]) for i in range(9)]
    tp, tr = SMALL
    trained_h = [max(dot_rows(report['groups'][f'{g}/{x}']) for g in pair for x in ('quill', 'ember')) * tp + 30
                 for pair in trained_pairs]

    top = 130
    p1_head = 96
    p1_h = p1_head + sum(r['h'] for r in rows) - 16
    p2_y = top + p1_h + 24
    p2_head = 150
    p2_h = p2_head + sum(trained_h) + 14
    key_y = p2_y + p2_h + 24
    key_h = 96
    H = key_y + key_h + M
    s = SVG(W, H)

    u, t = report['untrained'], report['trained']
    s.text(M, 92, 'The link works in five untrained games', 64, 700)

    # Untrained games
    s.rect(M, top, W - 2 * M, p1_h, WARM, 32)
    s.text(M + 40, top + 72, 'Never trained', 46, 700)
    s.text(M + 40 + width('Never trained', 46, 700) + 30, top + 72, f'{u["wins"]} of {u["n"]}', 46, 700, RED)
    y = top + p1_head
    for j, r in enumerate(rows):
        if j:
            s.line(M + 30, y - 16, W - M - 30, y - 16, '#E6DDD2', 2)
        s.text(qx, y + 36, LABELS[r['g']], 42, 700)
        cy = y + 58
        # Quill card
        s.rect(qx, cy, qw, r['card_h'], 'white', 20, QUILL, 3)
        s.mascot('quill', qx + 16, cy + 8, 52)
        s.text(qx + 78, cy + 46, 'Quill', 34, 700, QUILL)
        ty = cy + head + 40
        for line in r['q']:
            if r['mono']:
                s.text(qx + 24, ty, line, 32, 400, INK, family=MONO)
                ty += 44
            else:
                s.text(qx + 24, ty, line, body, 400, INK)
                ty += lh
        for line in r['notes']:
            s.text(qx + 24, ty, line, 32, 400, MUTED)
            ty += 42
        # Ember card
        s.rect(ex_, cy, ew, r['card_h'], 'white', 20, EMBER, 3)
        s.mascot('ember', ex_ + 16, cy + 8, 52)
        s.text(ex_ + 78, cy + 46, 'Ember', 34, 700, EMBER)
        ty = cy + head + 40
        for line in r['e']:
            s.text(ex_ + 24, ty, line, body, 400, INK)
            ty += lh
        # the link between the two models, both ways
        link_arrow(s, qx + qw + 8, ex_ - 8, cy + 34)
        s.text(qx + qw + lw / 2, cy + 82, 'link', 30, 600, LINK, 'middle')
        # what is written, sized to its text
        s.path(f'M{ex_ + ew + 8},{cy + 34} H{wx - 10}', INK, 4)
        s.path(f'M{wx - 24},{cy + 22} L{wx - 10},{cy + 34} L{wx - 24},{cy + 46}', INK, 4)
        s.rect(wx, cy, ww, r['write_h'], 'white', 20, '#D8D2C8', 2)
        s.text(wx + 24, cy + 46, r['who'], 32, 600, MUTED)
        fam = MONO if r['mono'] else FAMILY
        size = 34 if r['mono'] else 40
        ty = cy + head + 40
        for line in r['w']:
            s.rich(wx + 24, ty, [(txt, RED if scored else INK, 700 if scored else 400) for txt, scored in line],
                   size, family=fam)
            ty += lh
        # results: one dot line block per receiver
        dy = cy + 2
        for k, (side, color) in enumerate((('quill', QUILL), ('ember', EMBER))):
            row = report['groups'][f'{r["g"]}/{side}']
            s.text(dlx, dy + br + 13, 'Into ' + side.title(), 34, 700, color)
            result(s, row, dx, dy)
            dy += r['dot_h'][k] + 26
        y += r['h']

    # Trained games
    s.rect(M, p2_y, W - 2 * M, p2_h, COOL, 32)
    s.text(M + 40, p2_y + 72, 'Trained', 46, 700)
    s.text(M + 40 + width('Trained', 46, 700) + 30, p2_y + 72, f'{t["wins"]} of {t["n"]}', 46, 700, RED)
    colx = [M + 40, M + 40 + 1130]
    name_w = 340
    tdx = [[cx + name_w, cx + name_w + 392] for cx in colx]
    for c in range(2):
        for k, (side, color) in enumerate((('quill', QUILL), ('ember', EMBER))):
            s.text(tdx[c][k], p2_y + 130, 'Into ' + side.title(), 34, 700, color)
    y = p2_y + p2_head
    for pair, h in zip(trained_pairs, trained_h):
        for c, g in enumerate(pair):
            s.text(colx[c], y + tr + 12, LABELS[g], 34, 400)
            for k, side in enumerate(('quill', 'ember')):
                result(s, report['groups'][f'{g}/{side}'], tdx[c][k], y, SMALL, 32)
        y += h

    # Key: icon + a few words
    s.rect(M, key_y, W - 2 * M, key_h, COOL, 32)
    kx, ky = M + 40, key_y + 62
    x = kx
    for fill, stroke, label in ((RED, None, 'real partner better'), (LOSS, None, 'shuffled better'),
                                ('white', INK, 'tie'), (None, None, 'p < 0.05')):
        if fill is None:
            s.star(x + 15, ky - 12, 16)
        else:
            s.circle(x + br, ky - 12, br if stroke is None else br - 1.5, fill, stroke or 'none', 2.5)
        s.text(x + 44, ky, label, 34)
        x += 44 + width(label, 34) + 70
    s.save(OUT)
    print(json.dumps(report['untrained']))


if __name__ == '__main__':
    render(collect())
