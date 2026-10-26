"""Which city Quill scores highest after an opening line, for each city given to its linked partner.

Reads results/city_matrix.json (written by pairing/city_matrix.py on the Mac: nl22c step 157, both models with 4-bit
MLX weights) and draws figures/city_matrix_quill.{svg,png,pdf}; figure(data, 'ember') draws the same for Ember.

Each cell is the receiver's probability for each of the 12 candidate cities after an opening line: the softmax over
the 12 cities of the full-city log probability averaged over the 3 openings. Panels: link on, link cut, and a partner
from another conversation (the partner's model given a different city, linked in place of the partner). That panel is
ordered by the linked model's city (data['donor_indices'] gives the conversation each partner city was paired with), so
a diagonal there means the receiver scores the linked model's city highest. Each panel's count is the rows whose top
city is the linked model's city. No model is loaded.
Usage: python figures/city_matrix.py
"""
import json

import numpy as np

from chat_names import ART, COOL, EMBER, INK, LINK, QUILL, RED, ROOT, SVG, WARM, count, icon_link, rings, width

DATA = ROOT / 'results/city_matrix.json'
COLORS = {'quill': QUILL, 'ember': EMBER}


def blend(p):
    low, high = np.array([255, 255, 255]), np.array([26, 158, 119])
    return '#%02x%02x%02x' % tuple(np.rint(low + float(p) * (high - low)).astype(int))


def shares(mean_logp):
    a = np.asarray(mean_logp, dtype=float)
    a = np.exp(a - a.max(1, keepdims=True))
    return a / a.sum(1, keepdims=True)


def panels(data, side):
    """One dict per condition: label per matrix row, probabilities and observed top-city matches. The panel of the
    partner from another conversation is reordered by the linked model's city."""
    cities = data['design']['cities']
    k = len(cities)
    donor = data['donor_indices']
    out = []
    for cond in ('link', 'cut', 'shuffle'):
        a = shares(data['mean_logp'][side][cond])
        top = np.asarray(data['mean_logp'][side][cond]).argmax(1)
        st = data['stats'][side][cond]
        assert int((top == np.arange(k)).sum()) == st['observed'], (side, cond)
        labels = list(cities)
        if cond == 'shuffle':
            # row i was linked to the partner model given cities[donor[i]] in another conversation; sort by that city
            order = np.argsort(donor)
            assert all(donor[order[j]] == j for j in range(k))
            a, top = a[order], top[order]
            labels = [cities[donor[i]] for i in order]
        out.append(dict(cond=cond, labels=labels, a=a, observed=int((top == np.arange(k)).sum())))
    return out


def pair(s, x, y, size, side, other, cond, gap=0):
    """The partner (left) and the receiver (right) joined by the link; for 'shuffle' the partner's hidden states come
    from another conversation: the same partner in a dashed frame; for 'cut' both are in the link-cut pose (half-closed
    eyes, fallen hair). Returns the right edge."""
    a0, a1 = ART
    pose = '_cut' if cond == 'cut' else ''
    icon_link(s, x, x + size + gap, y, size, cond != 'cut')
    s.mascot(other + pose, x, y, size, flip=other == 'ember')     # each twin faces the other
    s.mascot(side + pose, x + size + gap, y, size, flip=side == 'quill')
    if cond == 'shuffle':
        s.rect(x + size * (a0 - 0.08), y - 6, size * (a1 - a0 + 0.16), size + 12, 'none', 16, INK, 3,
               'stroke-dasharray="9 7"')
    return x + 2 * size + gap


def figure(data, side):
    cities = data['design']['cities']
    k = len(cities)
    other = 'ember' if side == 'quill' else 'quill'
    name, partner = side.title(), other.title()
    me, you = COLORS[side], COLORS[other]
    W, M = 2400, 36
    top_y, strip_h = 130, 390
    cw, ch = 44, 52                     # matrix cell width and height
    head_h, cols_h = 350, 190
    py = top_y + strip_h + 24
    ph = head_h + cols_h + k * ch + 36
    H = py + ph + M
    assert H >= 0.65 * W
    s = SVG(W, H)
    s.add(f'<defs><linearGradient id="scale"><stop stop-color="{blend(0)}"/><stop offset="1" stop-color="{blend(1)}"/>'
          '</linearGradient></defs>')
    s.text(M, 88, f'{name} scores its linked partner’s city highest', 56, 700)
    # colour key, top right
    kx = W - M - 380
    s.text(kx - 20, 94, 'probability', 42, 600, INK, 'end')
    s.rect(kx, 52, 380, 56, 'url(#scale)', 10, '#D8D8DE', 1)
    s.text(kx + 16, 94, '0%', 40, 600, INK)
    s.text(kx + 364, 94, '100%', 40, 600, 'white', 'end')

    # Setup strip: the partner's city, the link, the receiver's openings and its top city after each.
    s.rect(M, top_y, W - 2 * M, strip_h, WARM, 32)
    shown = cities[0]
    x0 = M + 40
    s.text(x0, top_y + 76, f'{partner}’s city', 44, 700, you)
    s.rect(x0, top_y + 110, 320, 96, 'white', 18, you, 3)
    s.text(x0 + 160, top_y + 175, shown, 48, 700, you, 'middle')
    msize = 280
    mx = x0 + 340
    right = pair(s, mx, top_y + 30, msize, side, other, 'link')
    s.text(mx + msize / 2, top_y + strip_h - 34, partner, 42, 700, you, 'middle')
    s.text(mx + msize * 1.5, top_y + strip_h - 34, name, 42, 700, me, 'middle')
    s.text(mx + msize, top_y + 30 + msize * rings()[0] - 30, 'link', 42, 700, LINK, 'middle')
    ox = right + 40
    col_city = W - M - 40 - 280
    s.text(ox, top_y + 70, f'{name}’s opening', 44, 700, me)
    s.text(col_city + 140, top_y + 70, 'top city', 44, 700, INK, 'middle')
    for j, template in enumerate(data['design']['templates'][side]):
        y = top_y + 96 + j * 80
        s.rect(ox, y, col_city - ox - 16, 68, 'white', 14)
        s.text(ox + 22, y + 48, '“' + template['prefix'].rstrip() + ' …”', 44, 400, me)
        choice = cities[int(np.argmax(data['raw_logp'][side]['link'][j][0]))]
        ok = choice == shown
        s.rect(col_city, y, 280, 68, '#DDF0E9' if ok else '#FBE3DE', 14)
        s.text(col_city + 140, y + 49, choice, 44, 700, LINK if ok else RED, 'middle')

    # Three matrices
    gap = 24
    pw = (W - 2 * M - 2 * gap) / 3
    my_off = head_h + cols_h
    titles = {'link': ('Link on',), 'cut': ('Link cut',), 'shuffle': ('Partner from', 'another conversation')}
    for z, p in enumerate(panels(data, side)):
        cond = p['cond']
        left = M + z * (pw + gap)
        s.rect(left, py, pw, ph, WARM if cond == 'link' else COOL, 32)
        lines = titles[cond]
        for j, line in enumerate(lines):
            s.text(left + 30, py + 66 + j * 54, line, 46, 700)
        assert all(left + 30 + width(line, 46, 700) < left + pw - 20 for line in lines)
        isz, iy = 180, py + 66 + 54 + 30           # the icon row sits under two title lines in every panel
        tx = pair(s, left + 20, iy, isz, side, other, cond, gap=30) + 30
        good = p['observed'] == k
        col = (LINK if cond == 'shuffle' else RED) if good else INK
        count(s, tx, iy + isz / 2 + 26, p['observed'], k, col)
        # matrix: rows are the linked partner's city (in the last panel, that partner is from another conversation)
        mx0 = left + pw - 24 - k * cw
        my0 = py + my_off
        for j, city in enumerate(cities):
            s.text(mx0 + j * cw + cw / 2 + 14, my0 - 14, city, 40, 400, me, rotate=-90)
        for j, line in enumerate((f'{partner}’s', 'city')):
            s.text(mx0 - 12, my0 - 64 + j * 46, line, 40, 700, you, 'end')
        assert all(mx0 - 12 - width(c, 40) > left + 16 for c in p['labels'])
        for i in range(k):
            s.text(mx0 - 12, my0 + i * ch + ch / 2 + 14, p['labels'][i], 40, 400, you, 'end')
            for j in range(k):
                s.rect(mx0 + j * cw + 1, my0 + i * ch + 1, cw - 2, ch - 2, blend(p['a'][i, j]), 5,
                       '#E6E6EA', 1)
    s.save(ROOT / 'figures' / f'city_matrix_{side}')


def main():
    figure(json.loads(DATA.read_text()), 'quill')
    print(ROOT / 'figures/city_matrix_quill.png')


if __name__ == '__main__':
    main()
