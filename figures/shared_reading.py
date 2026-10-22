"""Shared reading (run nl20, final checkpoint): one model reads a text, the other answers from it through the link.

python figures/shared_reading.py   (no model is loaded)

Left: the setup, with a question-answering item from the held-out texts and the shuffled-partner control. The items
are verbatim corpus rows (data/coread.json, from build_coread.py); when data/coread.json is not in the checkout, the
three rows are read from the snapshot kept in results/shared_reading.json ('setup'), which an earlier run of this
script extracted from data/coread.json. The prompts use the held-out wordings of pairing/coread.py. They illustrate the
setup; they are not claimed to be one of the measured episodes.
Right: every receiver episode of the shared-reading games in results/magic-nl20/results/payload_final.json. Red: the
receiver's mean cross-entropy on the answer tokens is lower with its real partner than with the partner of another
episode of the same game; grey: higher; ring: exactly equal. Star: one-sided exact sign test p < 0.05 per game and
receiver, ties left out. Writes figures/shared_reading.{svg,png,pdf} and results/shared_reading.json.
"""
import base64
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/magic-nl20/results/payload_final.json'
CORPUS = ROOT / 'data/coread.json'
SNAPSHOT = ROOT / 'results/shared_reading.json'

QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
MUTED, FAINT, COOL, WARM = '#6E6E73', '#B9B9C0', '#F4F4F6', '#FAF0E6'
LOSS = '#B5B5BE'
FAMILY = 'Inter, Helvetica, Arial, sans-serif'
FONT_DIR = Path('/usr/share/fonts/opentype/inter')

GROUPS = [
    ('Passages', [('co_transfer', 'Copy a passage'), ('co_swap', 'Swap passages'), ('co_split', 'Join two halves'),
                  ('co_continue', 'Continue a text')]),
    ('Tasks', [('co_qa', 'Answer a question'), ('co_hop', 'Combine two facts'), ('co_translate', 'Translate'),
               ('co_paraphrase', 'Paraphrase'), ('co_nli', 'Compare two claims'), ('co_math', 'Solve a math problem'),
               ('co_code_out', 'Predict code output'), ('co_code_write', 'Write code'),
               ('co_instruct', 'Follow an instruction'), ('co_mcq', 'Multiple choice')]),
    ('Conversation', [('co_distill', 'Match the original reply'), ('co_talk', 'Answer, then talk'),
                      ('co_play', 'Play with secrets')]),
]
LABELS = {g: label for _, games in GROUPS for g, label in games}
CONTEXTS = dict(test='Volcanoes can be active, dormant, or extinct.',
                shuffle_example='Amphibians may attract mates with calls or scents.',
                training='The corona is the halo around the Sun.')

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


class SVG:
    def __init__(self, w, h):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">', f'<rect width="{w}" height="{h}" fill="white"/>',
                      '<defs><filter id="grey"><feColorMatrix type="saturate" values="0"/><feComponentTransfer>'
                      '<feFuncR type="linear" slope="0.62" intercept="0.35"/><feFuncG type="linear" slope="0.62" '
                      'intercept="0.35"/><feFuncB type="linear" slope="0.62" intercept="0.35"/></feComponentTransfer>'
                      '</filter></defs>']

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

    def text(self, x, y, s, size=34, weight=400, fill=INK, anchor='start'):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{html.escape(str(s))}</text>')

    def rich(self, x, y, chunks, size=34, anchor='start'):
        spans = ''.join(f'<tspan fill="{c}" font-weight="{w}">{html.escape(t)}</tspan>' for t, c, w in chunks)
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" text-anchor="{anchor}" '
                 f'xml:space="preserve">{spans}</text>')

    def mascot(self, side, x, y, size, grey=False):
        data = base64.b64encode((ROOT / 'figures/assets' / f'{side}.png').read_bytes()).decode()
        filt = ' filter="url(#grey)"' if grey else ''
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}"{filt} '
                 f'xlink:href="data:image/png;base64,{data}"/>')

    def star(self, x, y, r=14):
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


def outcome(r):
    d = r['shuffle_ce'] - r['link_ce']
    return 'real' if d > 0 else 'shuffled' if d < 0 else 'tie'


def sign_p(wins, losses):
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.


def setup_items():
    """The three corpus rows drawn on the left: from data/coread.json when present (the test rows filtered by
    coread_data.held_out), else from the snapshot."""
    if CORPUS.exists():
        sys.path.insert(0, str(ROOT / 'pairing'))
        from coread_data import held_out
        corpus = json.loads(CORPUS.read_text())
        fresh = held_out(corpus['train'], corpus['test'])
        return dict(test=next(r for r in fresh if r.get('context') == CONTEXTS['test']),
                    shuffle_example=next(r for r in fresh if r.get('context') == CONTEXTS['shuffle_example']),
                    training=next(r for r in corpus['train'] if r.get('context') == CONTEXTS['training']),
                    source='data/coread.json')
    setup = json.loads(SNAPSHOT.read_text())['setup']
    for k, ctx in CONTEXTS.items():
        assert setup[k]['context'] == ctx
    return dict(test=setup['test'], shuffle_example=setup['shuffle_example'], training=setup['training'],
                source='results/shared_reading.json (snapshot of data/coread.json rows)')


def main():
    rows = [r for r in json.loads(SOURCE.read_text()) if r['game'].startswith('co_')]
    assert set(LABELS) == {r['game'] for r in rows}
    assert len({(r['game'], r['receiver'], r['episode']) for r in rows}) == len(rows)
    assert all(math.isfinite(r[k]) for r in rows for k in ('shuffle_ce', 'link_ce'))
    totals = Counter(outcome(r) for r in rows)
    p_all = sign_p(totals['real'], totals['shuffled'])
    items = setup_items()
    sample, donor, training = items['test'], items['shuffle_example'], items['training']

    W, M = 2400, 36
    top = 130
    PITCH, R = 32, 13
    gx = M + 820                       # grid panel
    gw = W - M - gx
    head_h = 120
    group_h = 62
    row_h = {g: math.ceil(max(sum(1 for r in rows if r['game'] == g and r['receiver'] == x) for x in ('quill', 'ember'))
                          / 10) * PITCH + 24 for g in LABELS}
    grid_h = head_h + len(GROUPS) * group_h + sum(row_h.values()) + 16
    key_y = top + grid_h + 20
    key_h = 96
    H = key_y + key_h + M
    s = SVG(W, H)
    s.text(M, 92, 'A model answers from its partner’s text', 64, 700)

    # Left: one held-out item, and the shuffled partner
    lx, lw = M, gx - 24 - M
    s.rect(lx, top, lw, grid_h, WARM, 32)
    cw = (lw - 80 - 30) / 2
    cards_y = top + 40
    ans = sample['answer'].rstrip('.').lower()
    qlines = wrap(sample['context'], cw - 50, 38)
    elines = wrap(sample['question'], cw - 50, 38)
    card_h = 84 + max(len(qlines), len(elines)) * 50 + 20
    for k, (side, color, lines, head) in enumerate((('quill', QUILL, qlines, 'Quill reads'),
                                                     ('ember', EMBER, elines, 'Ember asks'))):
        x = lx + 40 + k * (cw + 30)
        s.rect(x, cards_y, cw, card_h, 'white', 22, color, 3)
        s.text(x + 25, cards_y + 54, head, 34, 700, color)
        y = cards_y + 110
        for line in lines:
            if side == 'quill' and ans in line:
                pre, post = line.split(ans, 1)
                s.rich(x + 25, y, [(pre, INK, 400), (ans, QUILL, 700), (post, INK, 400)], 38)
            else:
                s.text(x + 25, y, line, 38)
            y += 50
    # vertical budget: the pair, Ember's answer, the shuffled partner; slack is shared between the gaps
    ms, gs = 250, 260
    dl = wrap(donor['context'], lw - 80 - 60, 36)
    donor_h = 40 + len(dl) * 48
    sh_need = 80 + gs + 24 + donor_h + 30
    used = card_h + ms + 100 + sh_need
    gap = (grid_h - 40 - 30 - used) / 2.5
    assert gap > 24, 'left column overflows'
    # the pair, linked both ways
    my = cards_y + card_h + gap
    qmx, emx = lx + 40 + cw / 2 - ms / 2, lx + 40 + cw + 30 + cw / 2 - ms / 2
    for x in (qmx, emx):
        s.line(x + ms / 2, cards_y + card_h + 4, x + ms / 2, my + 4, MUTED, 3)
    s.mascot('quill', qmx, my, ms)
    s.mascot('ember', emx, my, ms)
    link_arrow(s, qmx + ms + 8, emx - 8, my + ms * 0.6)
    s.text((qmx + ms + emx) / 2, my + ms * 0.6 - 26, 'link', 36, 700, LINK, 'middle')
    # Ember's answer
    ty = my + ms + gap * 0.5
    s.line(emx + ms / 2, my + ms + 2, emx + ms / 2, ty, MUTED, 3)
    s.rect(lx + 40, ty, lw - 80, 100, 'white', 22, '#D8D2C8', 2)
    s.text(lx + 70, ty + 64, 'Ember writes', 34, 700, EMBER)
    s.text(lx + lw - 70, ty + 66, sample['answer'], 48, 700, RED, 'end')
    # shuffled partner: Ember linked to a Quill reading another text
    sy = ty + 100 + gap
    sh = top + grid_h - 30 - sy
    s.rect(lx + 40, sy, lw - 80, sh, COOL, 22)
    s.text(lx + 70, sy + 58, 'Shuffled partner', 36, 700)
    gy = sy + 80
    s.mascot('quill', lx + 70, gy, gs, grey=True)
    link_arrow(s, lx + 70 + gs + 8, lx + lw - 70 - gs - 8, gy + gs * 0.6)
    s.mascot('ember', lx + lw - 70 - gs, gy, gs)
    dy_ = gy + gs + 24
    s.rect(lx + 70, dy_, lw - 140, donor_h, 'white', 18, FAINT, 2)
    for j, line in enumerate(dl):
        s.text(lx + 100, dy_ + 52 + j * 48, line, 36, 400, MUTED)

    # Right: every episode
    s.rect(gx, top, gw, grid_h, WARM, 32)
    name_x = gx + 40
    dx = {'quill': gx + 440, 'ember': gx + 440 + 500}
    s.text(name_x, top + 80, f'{totals["real"]} of {totals["real"] + totals["shuffled"]}', 48, 700, RED)
    for side, color in (('quill', QUILL), ('ember', EMBER)):
        s.mascot(side, dx[side], top + 24, 72)
        s.text(dx[side] + 86, top + 76, 'Into ' + side.title(), 40, 700, color)
    y = top + head_h
    groups_out = []
    for gname, games in GROUPS:
        s.text(name_x, y + 42, gname, 32, 700, MUTED)
        s.line(name_x + width(gname, 32, 700) + 20, y + 31, gx + gw - 40, y + 31, '#E6DDD2', 2)
        y += group_h
        for game, label in games:
            h = row_h[game]
            nr = (h - 24) // PITCH
            s.text(name_x, y + R + 12 + (nr - 1) * PITCH / 2, label, 34)
            for side in ('quill', 'ember'):
                mine = sorted([r for r in rows if r['game'] == game and r['receiver'] == side],
                              key=lambda r: r['episode'])
                c = Counter(outcome(r) for r in mine)
                p = sign_p(c['real'], c['shuffled'])
                groups_out.append(dict(game=game, receiver=side, episodes=len(mine), real=c['real'],
                                       shuffled=c['shuffled'], tie=c['tie'], sign_p=p))
                for i, r in enumerate(mine):
                    cx, cy = dx[side] + R + (i % 10) * PITCH, y + R + (i // 10) * PITCH
                    o = outcome(r)
                    meta = (f'data-game="{game}" data-receiver="{side}" data-episode="{r["episode"]}" '
                            f'data-outcome="{o}"')
                    if o == 'tie':
                        s.circle(cx, cy, R - 1.5, 'white', INK, 2.5, meta)
                    else:
                        s.circle(cx, cy, R, RED if o == 'real' else LOSS, extra=meta)
                tx_ = dx[side] + 9 * PITCH + 2 * R + 14
                cnt = f'{c["real"]}/{len(mine)}'
                s.text(tx_, y + R + 13, cnt, 36, 700, RED)
                if p < .05:
                    s.star(tx_ + width(cnt, 36, 700) + 20, y + R)
            y += h

    # Key: icon + a few words
    s.rect(M, key_y, W - 2 * M, key_h, COOL, 32)
    kx, ky = M + 40, key_y + 60
    items_ = [(RED, None, 'real partner better'), (LOSS, None, 'shuffled better'), ('white', INK, 'tie'),
              (None, None, 'p < 0.05')]
    x = kx
    for fill, stroke, label in items_:
        if fill is None:
            s.star(x + 14, ky - 12, 16)
        else:
            s.circle(x + R, ky - 12, R if stroke is None else R - 1.5, fill, stroke or 'none', 2.5)
        s.text(x + 40, ky, label, 34)
        x += 40 + width(label, 34) + 70
    s.save(ROOT / 'figures/shared_reading')

    manifest = dict(source=str(SOURCE.relative_to(ROOT)), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    metric='Mean answer/payload-token cross entropy, real partner versus same-game shuffled partner; '
                           'not free-generation accuracy.',
                    tie_rule='Exact equality; ties displayed and omitted only from sign-test denominator.',
                    all_receiver_episodes=len(rows), non_tied_receiver_episodes=totals['real'] + totals['shuffled'],
                    counts=dict(totals), one_sided_sign_test_p=p_all, groups=groups_out,
                    setup=dict(training=training, test=sample, shuffle_example=donor, source=items['source'],
                               interpretation='Verbatim corpus examples of the setup, not claimed to be a measured '
                                              'evaluation episode or a generated answer.'),
                    rows=[dict(r, outcome=outcome(r)) for r in rows])
    SNAPSHOT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(episodes=len(rows), counts=dict(totals), p=p_all)))


if __name__ == '__main__':
    main()
