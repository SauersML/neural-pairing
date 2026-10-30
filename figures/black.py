"""Four of the twins' results drawn white on black with big type: the partner's city in free chat, the city a twin
scores highest, values never seen in training, and a concept steered into one twin. A twin's words always sit next to
its face.

  city         the chat-trained twins' test chats (results/hero_grid.json): Quill's first message in one chat, and the
               first messages that say the partner's city, link on and cut
  city_matrix  Quill's probability for each of 12 cities after an opening line, for each city given to Ember, link on
               and cut (results/city_matrix.json; probabilities as figures/city_matrix.py)
  values       the value games on values never seen in training (results/values_unseen.json): one episode, and every
               episode's probability per value token, exp(-cross-entropy), with the real partner and with a partner
               from another episode
  steer        the 12 test concepts steered into Quill at four times the strength after layer 27, after the
               steered-reading game: Ember's linked first messages, the one message Ember writes with the link cut,
               and the first messages that say the concept, of 96 (as figures/steered_results.py)

Reads only results files (no model) and writes figures/black/{city,city_matrix,values,steer}.{svg,png,pdf}.
Usage: python figures/black.py
"""
import json
import math
import statistics

import numpy as np

import creature
import steered_results
from chat_names import ROOT, SVG, width, wrap
from method_parts import check_shape, robot
from steered_mind import matches
from steered_trained import chat_counts

OUT = ROOT / 'figures/black'
BG, TEXT, CARD = '#000000', '#FFFFFF', '#141418'
QUILL, EMBER, LINK, HOT = '#A47CFF', '#FF8A3D', '#2FE3A0', '#FF5A4A'
COLOR = dict(quill=QUILL, ember=EMBER)
M = 70
BODY, TITLE, HUGE = 64, 84, 230


def canvas(W, H=4000):
    s = SVG(W, H, bg=BG)
    s.add('<defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="9" '
          'result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
    return s


def finish(s, stem, H):
    """Sets the page height to H, checks the shape and writes STEM.svg, .png and .pdf."""
    W = s.w
    H = round(H)
    check_shape(W, H)
    s.parts[0] = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                  f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    s.parts[1] = f'<rect width="{W}" height="{H}" fill="{BG}"/>'
    s.save(OUT / stem)


def title(s, text, W, y=M):
    """The finding, in up to two lines; returns the y below it."""
    for k, line in enumerate(wrap(text, W - 2 * M, TITLE, 700)):
        s.text(M, y + TITLE * 0.85 + k * TITLE * 1.2, line, TITLE, 700, TEXT)
    return y + TITLE * (1 + 1.2 * k) + 50


def words_of(text, spans):
    """text's words, each a list of (text, marked) runs; marked: inside one of the (start, end) spans."""
    out, word = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            if word:
                out.append(word)
            word = []
            continue
        m = any(a <= i < b for a, b in spans)
        if word and word[-1][1] == m:
            word[-1] = (word[-1][0] + ch, m)
        else:
            word.append((ch, m))
    return out + ([word] if word else [])


def wrap_marked(text, spans, w, size):
    lines, line, lw = [], [], 0
    space = width(' ', size)
    for word in words_of(text, spans):
        ww = sum(width(t, size, 700 if m else 400) for t, m in word)
        if line and lw + space + ww > w:
            lines.append(line)
            line, lw = [], 0
        lw += (space if line else 0) + ww
        line.append(word)
    return lines + ([line] if line else [])


def bubble(s, x, y, w, text, spans, size, edge, mark, pad=30, lead=1.3):
    """A speech bubble with the marked spans bold in colour mark; returns its height (s None: measure only)."""
    lines = wrap_marked(text, spans, w - 2 * pad, size)
    h = 2 * pad + size + lead * size * (len(lines) - 1)
    if s is not None:
        s.rect(x, y, w, h, CARD, 28, edge, 5)
        for k, line in enumerate(lines):
            chunks = []
            for j, word in enumerate(line):
                chunks += [(' ', TEXT, 400)] if j else []
                chunks += [(t, mark if m else TEXT, 700 if m else 400) for t, m in word]
            s.rich(x + pad, y + pad + 0.76 * size + k * lead * size, chunks, size)
    return h


def spans_of(text, word):
    i = text.find(word)
    return [(i, i + len(word))] if i >= 0 else []


def bright(markup):
    return markup.replace(creature.LINK, LINK)


def count(s, x, y, n, total, color, label, anchor='middle'):
    """A count drawn big under its label, centred on x."""
    big, rest = str(n), f' of {total}'
    w = width(big, HUGE, 700) + width(rest, 96, 700)
    x0 = x - w / 2
    s.text(x, y, label, BODY, 700, TEXT, 'middle')
    s.text(x0, y + HUGE * 0.95, big, HUGE, 700, color)
    s.text(x0 + width(big, HUGE, 700), y + HUGE * 0.95, rest, 96, 700, TEXT)


# ---------------------------------------------------------------- the partner's city in free chat

def city():
    d = json.loads((ROOT / 'results/hero_grid.json').read_text())
    shown = d['display']
    ep = next(e for e in d['draws'] if e['private'] == dict(quill=None, ember='Taipei'))
    speaker, turn = ep['conditions']['link']['turns'][0]
    assert speaker == 'quill' and ep['opener'] == 'quill'
    text = turn.split('?')[0] + '? …'
    W = 2400
    s = canvas(W)
    y = title(s, 'Linked, a twin says the city only its partner was told', W)

    # Quill's first message, over Quill
    bw = min(W - 2 * M, width(text, 72) + 120)
    bh = bubble(None, 0, 0, bw, text, [], 72, QUILL, EMBER)
    qx, ex, top, tall = 660, 1740, y + bh + 40, 1000
    bubble(s, M, y, bw, text, spans_of(text, 'Taipei'), 72, QUILL, EMBER)
    s.path(f'M{qx - 34},{y + bh - 3} L{qx},{y + bh + 46} L{qx + 34},{y + bh - 3}', QUILL, 5, CARD)
    s.rect(qx - 31, y + bh - 8, 62, 8, CARD, 0)

    # the twins at full body, linked after 8 of their 36 layers
    gq, pq = creature.creature('quill', qx, top, tall, full=True, look=1)
    ge, pe = creature.creature('ember', ex, top, tall, full=True, look=-1)
    s.add('<g filter="url(#glow)">' + ''.join(bright(creature.strands(pq[k][0], pe[k][0], pq[k][1], 11, 7))
                                              for k in creature.PORTS) + '</g>')
    s.add(bright(gq) + bright(ge))
    s.add(f'<g filter="url(#glow)">')
    s.text((qx + ex) / 2, pq[27][1] - 60, 'link', BODY, 700, LINK, 'middle')
    s.add('</g>')

    # the private prompts
    cy = top + tall + 40
    for x, who, city_, fill in ((qx, 'Quill', 'no city', TEXT), (ex, 'Ember', 'Taipei', EMBER)):
        cw = 640
        s.rect(x - cw / 2, cy, cw, 190, CARD, 28, COLOR[who.lower()], 5)
        s.text(x, cy + 76, f'{who}’s prompt', BODY, 700, COLOR[who.lower()], 'middle')
        s.text(x, cy + 160, city_, 76, 700, fill, 'middle')

    # the counts
    ny = cy + 190 + 110
    link_, cut = shown['link'], shown['cut']
    assert link_['targets'] == cut['targets']
    count(s, W * 0.27, ny, link_['hits'], link_['targets'], HOT, 'Link on')
    count(s, W * 0.73, ny, cut['hits'], cut['targets'], TEXT, 'Link cut')
    s.text(W / 2, ny + HUGE + 110, 'first messages that say the partner’s city', BODY, 400, TEXT, 'middle')
    finish(s, 'city', ny + HUGE + 110 + M)
    return dict(link=link_['hits'], cut=cut['hits'], of=link_['targets'])


# ---------------------------------------------------------------- which city Quill scores highest

def shares(mean_logp):
    """Each row's softmax over the cities (as figures/city_matrix.py)."""
    a = np.asarray(mean_logp, dtype=float)
    a = np.exp(a - a.max(1, keepdims=True))
    return a / a.sum(1, keepdims=True)


def city_matrix():
    d = json.loads((ROOT / 'results/city_matrix.json').read_text())
    cities = d['design']['cities']
    k = len(cities)
    W = 2400
    size, cell = 58, 76
    lw = max(width(c, size, 700) for c in cities) + 30
    colh = lw
    grid = k * cell
    gap = (W - 2 * M - lw - 2 * grid) / 1
    gap = min(gap, 130)
    x0 = (W - (lw + 2 * grid + gap)) / 2
    s = canvas(W)
    y = title(s, 'Linked, Quill gives the most probability to the city Ember was told', W)
    y += 90
    gy = y + colh
    s.text(x0 + lw - 30, gy - 40, 'was told', size, 400, TEXT, 'end')
    s.text(x0 + lw - 30, gy - 40 - size * 1.15, 'Ember', size, 700, EMBER, 'end')
    out = {}
    for p, cond, head in ((0, 'link', 'Link on'), (1, 'cut', 'Link cut')):
        a = shares(d['mean_logp']['quill'][cond])
        top = a.argmax(1)
        out[cond] = int((top == np.arange(k)).sum())
        assert out[cond] == d['stats']['quill'][cond]['observed']
        gx = x0 + lw + p * (grid + gap)
        s.text(gx + grid / 2, y - 20, head, 76, 700, HOT if cond == 'link' else TEXT, 'middle')
        for j, c in enumerate(cities):
            cx = gx + j * cell + cell / 2
            s.text(cx + size * 0.35, gy - 22, c, size, 400, QUILL, 'start', rotate=-90)
        for i in range(k):
            for j in range(k):
                v = float(a[i, j])
                rgb = np.rint(np.array([20, 20, 24]) + v * (np.array([47, 227, 160]) - np.array([20, 20, 24])))
                s.rect(gx + j * cell + 3, gy + i * cell + 3, cell - 6, cell - 6, '#%02x%02x%02x' % tuple(rgb.astype(int)),
                       10)
    for i, c in enumerate(cities):
        s.text(x0 + lw - 30, gy + i * cell + cell / 2 + size * 0.36, c, size, 700, EMBER, 'end')
    s.text(x0 + lw + grid + gap / 2, gy + grid + 100, 'Quill’s probability for each city after openings like '
           '“So, you’re in”', BODY, 400, QUILL, 'middle')
    finish(s, 'city_matrix', gy + grid + 100 + M)
    return out


# ---------------------------------------------------------------- values never seen in training

def in_words(p):
    if p >= 0.01:
        return f'{100 * p:.1f}%'
    n = 1 / p
    for unit, v in (('trillion', 1e12), ('billion', 1e9), ('million', 1e6), ('thousand', 1e3)):
        if n >= v:
            return f'1 in {n / v:.0f} {unit}'
    return f'1 in {n:.0f}'


def swarm_x(ys, r, cx, half):
    """Beeswarm: x for each y (pixels) within half of cx, as close to cx as possible without two dots of radius r
    overlapping; where the band is full, dots overlap at evenly spread places."""
    placed, xs, full = [], [0.0] * len(ys), 0
    for i in sorted(range(len(ys)), key=lambda i: ys[i]):
        for step in range(int(2 * half / (r * 0.55)) + 1):
            off = (step + 1) // 2 * (1 if step % 2 else -1) * r * 0.55
            if all((cx + off - px) ** 2 + (ys[i] - py) ** 2 >= (2 * r + 2) ** 2 for px, py in placed):
                break
        else:
            full += 1
            off = ((full * 0.618) % 1 * 2 - 1) * half
        xs[i] = cx + off
        placed.append((cx + off, ys[i]))
    return xs


def values():
    v = json.loads((ROOT / 'results/values_unseen.json').read_text())
    recs = [r for g in v['groups'].values() for r in g['records']]
    assert len(recs) == v['records'] == v['positive']
    real = [math.exp(-r['link_ce']) for r in recs]
    other = [math.exp(-r['shuffle_ce']) for r in recs]
    ex = v['examples']['example']
    word, sender, receiver = ex['meta']['value'], ex['meta']['sender'], ex['meta']['receiver']
    reply = ex['reply'][receiver][0]
    W, H = 2400, 2400
    s = canvas(W)
    y = title(s, f'A value never seen in training is likelier with the real partner in all {len(recs)} episodes', W)

    # one episode: the sender's prompt, the receiver's answer
    face = 150
    rows = ((sender, f'{sender.title()}’s prompt: “Through the link, tell {receiver.title()} the word {word}.”', False),
            (receiver, reply, True))
    for who, text, says in rows:
        robot(s, who, M, y + 6, face, facing=1)
        bx = M + face + 50
        if says:
            bubble(s, bx, y, W - M - bx, text, spans_of(text, word), BODY, COLOR[who], COLOR[sender])
        else:
            h = bubble(None, 0, 0, W - M - bx, text, [], BODY, COLOR[who], TEXT)
            s.rect(bx, y, W - M - bx, h, 'none', 28, COLOR[who], 5, 'stroke-dasharray="14 10"')
            bubble(s, bx, y, W - M - bx, text, spans_of(text, word), BODY, 'none', COLOR[sender])
        y += 200

    # every episode: probability per value token, real partner against a partner from another episode
    y += 40
    s.text(M, y + BODY, 'probability of the value, per token', BODY, 700, TEXT)
    py0, py1 = y + BODY + 80, H - M - 190
    lo = -16
    yv = lambda p: py0 + (0 - math.log10(max(p, 10 ** lo))) / (0 - lo) * (py1 - py0)
    ticks = ((1, '100%'), (1e-3, '1 in a thousand'), (1e-6, '1 in a million'), (1e-9, '1 in a billion'),
             (1e-12, '1 in a trillion'), (1e-15, '1 in a quadrillion'))
    tx = M + max(width(t, BODY) for _, t in ticks)
    for p, t in ticks:
        s.text(tx, yv(p) + BODY * 0.35, t, BODY, 400, TEXT, 'end')
        s.line(tx + 25, yv(p), tx + 50, yv(p), TEXT, 4)
    cols = ((real, 'Real partner', HOT), (other, 'Partner from another episode', TEXT))
    span = W - M - (tx + 60)
    r = 12
    out = {}
    for c, (ps, label, color) in enumerate(cols):
        cx = tx + 60 + span * (0.25 + 0.5 * c)
        ys = [yv(p) for p in ps]
        xs = swarm_x(ys, r, cx, span / 4 - 60)
        g = ''.join(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="{r}" fill="{color}" fill-opacity="0.9"/>'
                    for x, yy in zip(xs, ys))
        s.add(f'<g filter="url(#glow)">{g}</g>' if c == 0 else g)
        med = statistics.median(ps)
        out[label] = med
        if c == 0:
            s.text(max(xs) + 40, yv(med) + 26, in_words(med), 76, 700, color)
        else:
            s.text(min(xs) - 40, yv(med) + 26, in_words(med), 76, 700, color, 'end')
        for k, line in enumerate(wrap(label, span / 2 - 40, BODY, 700)):
            s.text(cx, py1 + 110 + k * BODY * 1.2, line, BODY, 700, color, 'middle')
    finish(s, 'values', H)
    return dict(episodes=len(recs), median=out)


# ---------------------------------------------------------------- a concept steered into Quill

def steer():
    task, rows, cut = steered_results.episodes(ROOT / 'results')
    linked, cut_n = chat_counts(ROOT / 'results' / steered_results.RUN)
    W = 2400
    s = canvas(W)
    y = title(s, 'Steer a concept into Quill, and linked Ember writes about it', W)
    size, face = 52, 96
    chip_w = max(width(c, size, 700) for c, _, _ in rows) + 60
    bx = M + chip_w + 40 + face + 30
    bw = W - M - bx
    s.text(M, y + size, 'Steered into Quill', size, 700, QUILL)
    s.text(bx, y + size, 'Ember’s first message, linked', size, 700, EMBER)
    y += size + 40
    for concept, text, hit in rows:
        steered_results.SIZE = size
        q = ' '.join(steered_results.quote(text, 10 ** 6, concept))
        q += '' if q.rstrip()[-1] in '.!?…' else ' …'
        sp = matches(q, concept) if hit else []
        h = bubble(None, 0, 0, bw, q, sp, size, EMBER, QUILL, pad=22, lead=1.25)
        s.rect(M, y + (h - 86) / 2, chip_w, 86, CARD, 43, QUILL, 4)
        s.text(M + chip_w / 2, y + h / 2 + size * 0.36, concept, size, 700, QUILL, 'middle')
        robot(s, 'ember', M + chip_w + 40, y + (h - face) / 2, face, facing=-1)
        bubble(s, bx, y, bw, q, sp, size, EMBER, QUILL, pad=22, lead=1.25)
        y += h + 14
    y += 40
    s.text(M, y + size, f'Link cut, Ember writes the same message for all {len(rows)}', size, 700, TEXT)
    y += size + 30
    h = bubble(None, 0, 0, bw, cut, [], size, EMBER, QUILL, pad=22, lead=1.25)
    robot(s, 'ember', M + chip_w + 40, y + (h - face) / 2, face, cut=True, facing=-1)
    bubble(s, bx, y, bw, cut, [], size, EMBER, QUILL, pad=22, lead=1.25)
    y += h + 80
    count(s, W * 0.27, y, linked, 96, HOT, 'Link on')
    count(s, W * 0.73, y, cut_n, 96, TEXT, 'Link cut')
    s.text(W / 2, y + HUGE + 110, 'first messages that say the steered concept', BODY, 400, TEXT, 'middle')
    finish(s, 'steer', y + HUGE + 110 + M)
    return dict(link=linked, cut=cut_n, of=96)


def main():
    OUT.mkdir(exist_ok=True)
    print(json.dumps(dict(city=city(), city_matrix=city_matrix(), values=values(), steer=steer()), indent=1))


if __name__ == '__main__':
    main()
