"""The value games on values never used in training (run nl20, final checkpoint): one worked episode, and every
receiver episode against the shuffled partner.

python figures/values_unseen.py   (no model is loaded)
Reads results/magic-nl20/results/payload_final.json and the example episode kept in results/values_unseen.json (drawn
on the Mac with the Games and payload_ctl code and the stored original-model replies; --refresh-examples redraws it
there). The readout is the teacher-forced mean cross-entropy on the value's tokens, not generation accuracy; a
probability below is exp(-that cross-entropy), the geometric mean probability of the value's tokens. The worked example
shows the target replies, not a generated chat.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import math
import subprocess
import sys
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup')
LABELS = ('Words', 'Numbers', 'Phrases', 'Digit strings', 'Sums', 'Name and city', 'Code lookups')
SIDES = ('quill', 'ember')
BLACK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COOL, WARM = '#F4F4F6', '#FAF0E6'

QUILL, EMBER, LINK, INK = PURPLE, ORANGE, GREEN, BLACK
MUTED, FAINT = '#6E6E73', '#B9B9C0'
LOSS = '#B5B5BE'
FAMILY = 'Inter, Helvetica, Arial, sans-serif'
FONT_DIR = Path('/usr/share/fonts/opentype/inter')
_FONTS = {}


def font(size, weight=400):
    from PIL import ImageFont
    key = (size, weight)
    if key not in _FONTS:
        name = {400: 'Inter-Regular', 600: 'Inter-SemiBold', 700: 'Inter-Bold'}[weight]
        try:
            _FONTS[key] = ImageFont.truetype(str(FONT_DIR / f'{name}.otf'), size)
        except OSError:
            _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


def width(t, size, weight=400):
    return font(size, weight).getlength(str(t))


def wrap(t, w, size, weight=400):
    lines, line = [], ''
    for word in str(t).split():
        trial = (line + ' ' + word).strip()
        if line and width(trial, size, weight) > w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


class SVG:
    def __init__(self, w, h, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">', f'<title>{html.escape(title)}</title>',
                      f'<rect width="{w}" height="{h}" fill="white"/>',
                      '<defs><filter id="grey"><feColorMatrix type="saturate" values="0"/><feComponentTransfer>'
                      '<feFuncR type="linear" slope="0.62" intercept="0.35"/><feFuncG type="linear" slope="0.62" '
                      'intercept="0.35"/><feFuncB type="linear" slope="0.62" intercept="0.35"/></feComponentTransfer>'
                      '</filter></defs>']

    def add(self, t):
        self.parts.append(t)

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

    def text(self, x, y, t, size=34, weight=400, fill=INK, anchor='start'):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{html.escape(str(t))}</text>')

    def rich(self, x, y, chunks, size=34, anchor='start'):
        spans = ''.join(f'<tspan fill="{c}" font-weight="{w}">{html.escape(t)}</tspan>' for t, c, w in chunks)
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FAMILY}" font-size="{size}" text-anchor="{anchor}" '
                 f'xml:space="preserve">{spans}</text>')

    def mascot(self, side, x, y, size, grey=False):
        data = base64.b64encode((ROOT / f'figures/assets/{side}.png').read_bytes()).decode()
        filt = ' filter="url(#grey)"' if grey else ''
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}"{filt} '
                 f'xlink:href="data:image/png;base64,{data}"/>')

    def chip(self, x, y, t, color=INK, fill=COOL, stroke='none', size=34, weight=400, h=60, pad=18):
        w = width(t, size, weight) + 2 * pad
        self.rect(x, y, w, h, fill, 14, stroke, 3)
        self.text(x + w / 2, y + h / 2 + size * 0.36, t, size, weight, color, 'middle')
        return w

    def save(self, name):
        p = ROOT / 'figures' / f'{name}.svg'
        p.write_text('\n'.join(self.parts + ['</svg>']))
        for fmt in ('png', 'pdf'):
            subprocess.run(['rsvg-convert', '-f', fmt, '-o', str(p.with_suffix('.' + fmt)), str(p)], check=True)


def link_arrow(s, x1, x2, y, color=LINK, sw=6, head=16):
    """The link, two-way: a head at both ends."""
    for xa, d in ((x1, 1), (x2, -1)):
        s.path(f'M{xa + d * head:.1f},{y - head * 0.75:.1f} L{xa},{y} L{xa + d * head:.1f},{y + head * 0.75:.1f}',
               color, sw)
    s.line(x1 + 3, y, x2 - 3, y, color, sw)


def down_arrow(s, x, y1, y2, color=INK):
    s.line(x, y1, x, y2, color, 4)
    s.path(f'M{x - 12},{y2 - 14} L{x},{y2} L{x + 12},{y2 - 14}', color, 4)


def summary(records):
    out={}
    for g in GAMES:
        for side in SIDES:
            rows=sorted((r for r in records if r['game']==g and r['receiver']==side),key=lambda r:r['episode'])
            assert rows and len({r['episode'] for r in rows})==len(rows)
            ds=[r['shuffle_ce']-r['link_ce'] for r in rows]
            assert all(math.isfinite(v) for v in ds)
            pos=sum(v>0 for v in ds); neg=sum(v<0 for v in ds); ties=sum(v==0 for v in ds)
            n=pos+neg
            out[f'{g}/{side}']=dict(total=len(rows),positive=pos,negative=neg,ties=ties,
                one_sided_sign_p=sum(math.comb(n,k) for k in range(pos,n+1))/2**n if n else 1.,
                minimum_shuffle_minus_link=min(ds),records=rows)
    return out

def extract_examples(replies):
    import games
    import payload_ctl
    from transformers import AutoTokenizer
    from link import MODELS
    eps=payload_ctl.episodes(json.loads(replies.read_text()),20)
    i,ep=next((i,e) for i,e in enumerate(eps['relay_word']) if e['meta'].get('sender')=='quill' and e['meta']['value']=='mouth')
    pools=games.POOLS
    train_words=[v for v in ('apple','cloud','river','garden','stone','cat','book') if v in pools['words']['train']][:4]
    if len(train_words)<4:train_words=pools['words']['train'][:4]
    test_words=[e['meta']['value'] for e in eps['relay_word'] if e['meta']['sender']=='quill' and e['meta']['value'] in ('cipher','mouth','sling','project')]
    assert len(test_words)==4 and set(train_words).isdisjoint(test_words)
    assert all(v in pools['words']['test'] for v in test_words)
    toks={x:AutoTokenizer.from_pretrained(MODELS[x]['hf'],local_files_only=True) for x in SIDES}
    pieces={x:[toks[x].decode([v]) for v in toks[x](ep['reply'][x][0],add_special_tokens=False).input_ids] for x in SIDES}
    assert ep['meta']['value'] not in ep['task']['ember']
    return dict(episode=i,example=ep,training_values=train_words,test_values=test_words,target_pieces=pieces,
                other_examples={g:es[0]['meta'] for g,es in eps.items()},
                word_episodes=eps['relay_word'])

def game_figure(ex):
    W, M = 2400, 36
    value = ex['example']['meta']['value']
    pieces = ex['target_pieces']
    task = ex['example']['task']
    assert f'"{value}"' in task['quill'] and value not in task['ember']
    assert ''.join(pieces['ember']).strip() == ex['example']['reply']['ember'][0]
    # shortest real excerpts of the two prompts
    q_ex = task['quill'][task['quill'].index('tell Ember'):task['quill'].index(f'"{value}"') + len(value) + 3]
    e_ex = task['ember'][:task['ember'].index(' through the link')]
    assert q_ex == f'tell Ember the word "{value}".' and e_ex == 'Read the word Quill sends you'
    top = 130
    py, ph = top + 40, 170                 # prompts
    my, mh = py + ph + 56, 440             # models
    ry, rh = my + mh + 56, 190             # replies
    H = ry + rh + 40 + M
    s = SVG(W, H, 'A held-out word crosses the link')
    s.text(M, 92, 'A held-out word crosses the link', 64, 700)
    # value lists
    lw_ = 560
    lh_ = (H - top - M - 24) / 2
    for k, (title, words, fill) in enumerate((('Training values', ex['training_values'], COOL),
                                              ('Held-out values', ex['test_values'], WARM))):
        y0 = top + k * (lh_ + 24)
        s.rect(M, y0, lw_, lh_, fill, 32)
        s.text(M + 40, y0 + 70, title, 42, 700)
        ch = (lh_ - 110 - 40 - 20) / 2
        for i, word in enumerate(words):
            cx, cy = M + 40 + (i % 2) * 250, y0 + 110 + (i // 2) * (ch + 20)
            on = word == value
            s.rect(cx, cy, 230, ch, 'white', 20, QUILL if on else 'none', 4)
            s.text(cx + 115, cy + ch / 2 + 17, word, 46, 700 if on else 400, QUILL if on else INK, 'middle')
    # the episode
    px = M + lw_ + 24
    pw = W - M - px
    s.rect(px, top, pw, H - top - M, WARM, 32)
    cw = (pw - 80 - 60) / 2
    qx, ex_ = px + 40, px + pw - 40 - cw
    for x, side, color, excerpt in ((qx, 'quill', QUILL, q_ex), (ex_, 'ember', EMBER, e_ex)):
        s.rect(x, py, cw, ph, 'white', 24, color, 3)
        s.text(x + 30, py + 58, f'{side.title()}’s prompt', 34, 700, color)
        if side == 'quill':
            pre, post = excerpt.split(f'"{value}"')
            s.rich(x + 30, py + 124, [('… ' + pre + '“', INK, 400), (value, QUILL, 700), ('”' + post, INK, 400)],
                   42)
        else:
            s.text(x + 30, py + 124, excerpt + ' …', 42)
    # models, side by side, and the link between them at 8 depths, each read both ways at once
    bw = 330
    boxes = {'quill': (qx, QUILL), 'ember': (ex_ + cw - bw, EMBER)}
    for side, (x, color) in boxes.items():
        down_arrow(s, x + bw / 2, py + ph + 8, my - 8)
        s.rect(x, my, bw, mh, 'white', 24, color, 3)
        s.mascot(side, x + (bw - 240) / 2, my + 40, 240)
        s.text(x + bw / 2, my + mh - 50, side.title(), 46, 700, color, 'middle')
    lx1, lx2 = qx + bw, ex_ + cw - bw
    for i in range(8):
        y = my + 50 + i * 46
        link_arrow(s, lx1 + 14, lx2 - 14, y, sw=6, head=15)
    mid = (lx1 + lx2) / 2
    s.text(mid, my - 14, 'link', 40, 700, LINK, 'middle')
    s.text(mid, my + 50 + 7 * 46 + 64, '8 depths', 36, 600, LINK, 'middle')
    # replies
    for x, side, color, label in ((qx, 'quill', QUILL, 'Quill writes'), (ex_, 'ember', EMBER, 'Ember writes')):
        bx = boxes[side][0]
        down_arrow(s, bx + bw / 2, my + mh + 8, ry - 8)
        s.rect(x, ry, cw, rh, 'white', 24, color, 3)
        s.text(x + 30, ry + 56, label, 34, 700, color)
        cx, cy = x + 30, ry + 86
        shown = pieces[side] if side == 'ember' else pieces[side][:5]
        for piece in shown:
            t = piece.strip() or '·'
            on = side == 'ember' and t == value
            w = width(t, 40, 700 if on else 400) + 32
            s.chip(cx, cy, t, RED if on else color, WARM if on else COOL, RED if on else 'none', 40,
                   700 if on else 400, 70, 16)
            cx += w + 10
        if side == 'quill':
            s.text(cx + 4, cy + 48, '…', 40, 400, color)
    s.save('values_unseen_game')


def fmt_p(p):
    if p >= 0.9995:
        return f'{100 * p:.2f}%'
    if p >= 0.001:
        return f'{100 * p:.1f}%'
    if p >= 0.0001:
        return f'{100 * p:.2f}%'
    return '< 0.01%'


def one_in(p):
    n = 1 / p
    for unit, v in (('billion', 1e9), ('million', 1e6), ('thousand', 1e3)):
        if n >= v:
            return f'1 in {n / v:.0f} {unit}'
    return f'1 in {n:.0f}'


def results_figure(stats, ex):
    W, M = 2400, 36
    total = sum(v['total'] for v in stats.values())
    assert sum(v['positive'] for v in stats.values()) == total     # the title says all
    sample = next(r for r in stats['relay_word/ember']['records'] if r['episode'] == ex['episode'])
    value = ex['example']['meta']['value']
    pr = math.exp(-sample['link_ce'])
    ps = math.exp(-sample['shuffle_ce'])
    assert sample['n_tokens'] == 1
    P, R = 47, 19
    row_h = {g: max(math.ceil(stats[f'{g}/{x}']['total'] / 10) for x in SIDES) * P + 34 for g in GAMES}
    top = 130
    ex_h = 330
    grid_y = top + ex_h + 24
    grid_h = 150 + sum(row_h.values()) + 10
    H = grid_y + grid_h + M
    s = SVG(W, H, 'Real partner wins all episodes')
    s.text(M, 92, f'Real partner wins all {total} episodes', 64, 700)
    s.circle(W - M - 360, 70, R, RED)
    s.text(W - M - 330, 82, 'real partner better', 34, 400)

    # one episode, both partners
    cw = (W - 2 * M - 24) / 2
    for k, (title, p, grey, fill) in enumerate((('Real partner', pr, False, WARM), ('Shuffled partner', ps, True, COOL))):
        x = M + k * (cw + 24)
        s.rect(x, top, cw, ex_h, fill, 32)
        s.text(x + 40, top + 72, title, 44, 700, RED if not grey else INK)
        my = top + 100
        s.mascot('quill', x + 40, my, 140, grey=grey)
        link_arrow(s, x + 190, x + 320, my + 86)
        s.mascot('ember', x + 330, my, 140)
        tx = x + 530
        s.text(tx, top + 72, f'probability of “{value}”', 32, 400, MUTED)
        s.text(tx, top + 160, fmt_p(p) if p > 0.001 else one_in(p), 72, 700, RED if not grey else INK)
        bw = cw - 530 - 40
        s.rect(tx, top + 200, bw, 40, 'white', 10)
        if p * bw >= 2:
            s.rect(tx, top + 200, p * bw, 40, RED, 10)
        else:
            s.line(tx + 2, top + 192, tx + 2, top + 248, INK, 5)
        s.chip(x + 50, my + 150, value if not grey else 'other word', QUILL if not grey else MUTED,
               '#EFE8FC' if not grey else 'white', 'none', 32, 700 if not grey else 400, 50, 14)

    # every episode
    s.rect(M, grid_y, W - 2 * M, grid_h, WARM, 32)
    name_x = M + 40
    dx = {'quill': M + 390, 'ember': M + 1030}
    px_ = M + 1660
    for side, color in (('quill', QUILL), ('ember', EMBER)):
        s.mascot(side, dx[side], grid_y + 26, 80)
        s.text(dx[side] + 96, grid_y + 82, 'Into ' + side.title(), 44, 700, color)
    s.text(px_ + 330, grid_y + 58, 'median probability', 32, 600, MUTED, 'middle')
    s.text(px_ + 200, grid_y + 118, 'real', 34, 700, RED, 'end')
    s.text(px_ + 560, grid_y + 118, 'shuffled', 34, 700, INK, 'end')
    y = grid_y + 150
    medians = {}
    for j, (g, label) in enumerate(zip(GAMES, LABELS)):
        h = row_h[g]
        if j:
            s.line(M + 30, y - 4, W - M - 30, y - 4, '#E6DDD2', 2)
        rows_n = max(math.ceil(stats[f'{g}/{x}']['total'] / 10) for x in SIDES)
        cy = y + (h - rows_n * P) / 2 + R
        base = cy + 15 + (rows_n - 1) * P / 2
        s.text(name_x, base, label, 42, 700)
        for side in SIDES:
            st = stats[f'{g}/{side}']
            for i, r in enumerate(st['records']):
                d = r['shuffle_ce'] - r['link_ce']
                x0 = dx[side] + R + (i % 10) * P
                y0 = cy + (i // 10) * P
                s.circle(x0, y0, R, RED if d > 0 else (LOSS if d < 0 else 'white'), INK if d == 0 else 'none', 3,
                         f'data-game="{g}" data-receiver="{side}" data-episode="{r["episode"]}"')
            s.text(dx[side] + 9 * P + 2 * R + 18, base, f'{st["positive"]}/{st["total"]}', 42, 700, RED)
        recs = stats[f'{g}/quill']['records'] + stats[f'{g}/ember']['records']
        real = statistics.median(math.exp(-r['link_ce']) for r in recs)
        shuf = statistics.median(math.exp(-r['shuffle_ce']) for r in recs)
        medians[g] = dict(real=real, shuffled=shuf)
        s.text(px_ + 200, base, fmt_p(real), 42, 700, RED, 'end')
        s.text(px_ + 560, base, fmt_p(shuf), 42, 400, INK, 'end')
        y += h
    s.save('values_unseen_results')
    return medians


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'results/magic-nl20/results/payload_final.json')
    ap.add_argument('--replies',type=Path,default=Path('/Users/user/mpd-data/runpod/magic-nl20/replies.json'))
    ap.add_argument('--refresh-examples',action='store_true')
    args=ap.parse_args()
    out=ROOT/'results/values_unseen.json'
    data=json.loads(args.source.read_text())
    selected=[r for r in data if r['game'] in GAMES]
    st=summary(selected)
    ex=extract_examples(args.replies) if args.refresh_examples or not out.exists() else json.loads(out.read_text())['examples']
    report=dict(source=str(args.source.relative_to(ROOT)),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                measurement='Mean teacher-forced cross-entropy on payload tokens; positive iff shuffle_ce - link_ce > 0; ties exactly equal.',
                precision='Existing real-model GPU result; no Mac model inference performed.',
                records=len(selected),positive=sum(v['positive'] for v in st.values()),ties=sum(v['ties'] for v in st.values()),
                groups=st,examples=ex,example_replies_source=str(args.replies),
                example_note='The Games target replies, not generated linked-model replies. Vector strips are schematic, not activation measurements. Word pools are disjoint as values, not a claim about pretraining or incidental text.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    game_figure(ex)
    medians = results_figure(st, ex)
    report['median_probability_per_value_token'] = medians
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('records','positive','ties')}))

if __name__=='__main__':main()
