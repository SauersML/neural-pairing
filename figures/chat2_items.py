"""The 12-kind chat test (co_chat2): which twin says whose private item first, before and after training on it.

Each twin is given a private item of one of 12 kinds (a trip, a pet, a mood, ...); the twins chat, and a twin scores
when it says its partner's item before the partner does (chateval `names`). Cards: chats whose partner's item is of
one of the eight kinds nl26 trains on (drawn as chips), and of one of the four kinds held out of its training. Columns:
before (nl22c step 157, chat-trained on cities only) and after 300 steps of chat training on the eight kinds (nl26).
Rows: link on, link cut, and linked to a partner from another conversation (chateval's crossed condition: an isolated
twin of the partner's model, given a different item, linked in place of the partner), where the twin can say its chat
partner's item first or the other conversation's (chateval `names_donor`). Each cell is a count of twins, out of those
whose partner (or the other conversation's partner) has an item.

Reads the evalfree --shard outputs (episodes.jsonl; no model) and writes figures/chat2_items.{svg,png,pdf}.
figures/oneway.py draws the one-way runs with draw().
Usage: python figures/chat2_items.py [RESULTS_DIR]   (default: results/magic-nl26/results)
"""
import sys
from pathlib import Path

from chat_names import COOL, INK, LINK, RED, SVG, WARM, count, load, mascot, pair_icon, width, wrap

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results/magic-nl26/results'
OTHER = 'Partner from\nanother\nconversation'
ROWS = (('link', 'names', 'Link on', "partner's item", RED), ('cut', 'names', 'Link cut', "partner's item", RED),
        ('crossed', 'names', OTHER, "chat partner's item", RED),
        ('crossed', 'names_donor', OTHER, 'item from the other conversation', LINK))
SIDES = ('quill', 'ember')


ART_H = 110    # height of a card's picture, when it has one


def tally(records, cond, metric):
    """(said first, twins with such an item to say) over records."""
    v = [r['conditions'][cond].get(metric) for r in records]
    return int(sum(x for x in v if x is not None)), sum(x is not None for x in v)


def lines(text, w):
    """text wrapped at width w in bold body text, breaking at each newline too."""
    return [line for part in text.split('\n') for line in wrap(part, w, 46, 700)]


def chips(s, x, y, words, w, draw=True):
    """White word chips from (x, y), the top left, wrapped at width w; returns the height used."""
    pad, gap, h = 20, 14, 68
    cx, cy = x, y
    for word in words:
        cw = width(word, 46) + 2 * pad
        if cx > x and cx + cw > x + w:
            cx, cy = x, cy + h + gap
        if draw:
            s.rect(cx, cy, cw, h, 'white', 14)
            s.text(cx + pad, cy + h / 2 + 16, word, 46)
        cx += cw + gap
    return cy + h - y


def draw(title, cards, stem, W=2100):
    """A table of counts: ROWS down, one card per group of chats across. title: one line, or a tuple of lines. cards:
    [(header, fill, art, words, [(subheader, records)])]; art(s, x, y, uris), when given, draws a picture centred
    above the header and returns its width (and its height is ART_H); words, when given, are drawn as chips under the
    header."""
    M, G, P = 30, 30, 28
    cond_w, what_w, icon = 300, 440, 100
    label_w = cond_w + 24 + what_w + 24
    titles = (title,) if isinstance(title, str) else title
    top = 40 + 70 * len(titles)
    cw = (W - 2 * M - label_w - G * len(cards)) / len(cards)
    uris = {k: mascot(k) for k in ('quill', 'ember', 'quill_cut', 'ember_cut')}
    art_h = max(ART_H + 20 if art else 0 for _, _, art, _, _ in cards)
    head_lines = {k: lines(c[0], cw - 2 * P) for k, c in enumerate(cards)}
    chip_h = max(chips(None, 0, 0, c[3], cw - 2 * P, False) + 24 if c[3] else 0 for c in cards)
    sub_lines = max(len(lines(sub, (cw - 2 * P) / len(c[4]) - 8)) for c in cards for sub, _ in c[4])
    title_h = art_h + max(len(v) for v in head_lines.values()) * 56 + 40
    head_h = title_h + chip_h + sub_lines * 56 + 10
    what = [lines(w_, what_w) for _, _, _, w_, _ in ROWS]
    pair_h = [icon + 14 + 56 * len(lines(label, cond_w)) + 50 for _, _, label, _, _ in ROWS]
    row_h = [max(pair_h[0], 56 * len(what[0]) + 70), max(pair_h[1], 56 * len(what[1]) + 70)]
    row_h += [max(pair_h[2] / 2, 56 * len(what[j]) + 70) for j in (2, 3)]
    s = SVG(W, top + head_h + sum(row_h) + 20 + M)
    for k, line in enumerate(titles):
        s.text(M, 76 + 70 * k, line, 56, 700)

    for k, (header, fill, art, words, cols) in enumerate(cards):
        x0 = M + label_w + G + k * (cw + G)
        s.rect(x0, top, cw, head_h + sum(row_h) + 20, fill, 32)
        if art:
            aw = art(SVG(1, 1), 0, 0, uris)
            art(s, x0 + (cw - aw) / 2, top + 26, uris)
        for i, line in enumerate(head_lines[k]):
            if art:
                s.text(x0 + cw / 2, top + art_h + 64 + 56 * i, line, 46, 700, INK, 'middle')
            else:
                s.text(x0 + P, top + 64 + 56 * i, line, 46, 700)
        if words:
            chips(s, x0 + P, top + title_h, words, cw - 2 * P)
        sw = (cw - 2 * P) / len(cols)
        for c, (sub, records) in enumerate(cols):
            cx = x0 + P + (c + 0.5) * sw
            for i, line in enumerate(lines(sub, sw - 8)):
                s.text(cx, top + title_h + chip_h + 44 + 56 * i, line, 46, 700, INK, 'middle')
            y0 = top + head_h
            for j, (cond, metric, _, _, color) in enumerate(ROWS):
                n, total = tally(records, cond, metric)
                count(s, cx, y0 + row_h[j] / 2 + 26, n, total, color if n else INK, anchor='middle')
                y0 += row_h[j]

    s.text(M + label_w - 24, top + title_h + chip_h + 44 + 56 * (sub_lines - 1), 'Twin says first', 46, 700, INK,
           'end')
    y0 = top + head_h
    for j, (cond, metric, label, _, color) in enumerate(ROWS):
        for i, line in enumerate(what[j]):
            s.text(M + label_w - 24, y0 + row_h[j] / 2 + 16 + 56 * (i - (len(what[j]) - 1) / 2), line, 46, 700, color,
                   'end')
        if j != 3:
            span = row_h[j] + (row_h[3] if j == 2 else 0)
            text = lines(label, cond_w)
            pw = 2 * icon + 86
            iy = y0 + (span - icon - 14 - 56 * len(text)) / 2
            pair_icon(s, M + (cond_w - pw) / 2, iy, icon, cond, uris)
            for i, line in enumerate(text):
                s.text(M + cond_w / 2, iy + icon + 14 + 42 + 56 * i, line, 46, 700, INK, 'middle')
        y0 += row_h[j]
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    records = {name: load([root / f'chat2_{name}_{k}' for k in range(3)]) for name in ('before', 'after')}
    kinds = {g: sorted({r['kind'] for rs in records.values() for r in rs if r['kind_group'] == g})
             for g in ('trained_kind', 'held_out_kind')}
    subs = (('before', 'Trained on\ncities'), ('after', f"Trained on\n{len(kinds['trained_kind'])} kinds"))
    cards = [(header, fill, None, kinds[g], [(sub, [r for r in records[name] if r['kind_group'] == g])
                                            for name, sub in subs])
             for g, header, fill in (('trained_kind', 'In training', COOL),
                                     ('held_out_kind', 'Never seen in training', WARM))]
    draw(("Linked twins say each other's private item first,", 'also for kinds never seen in training'), cards,
         HERE / 'chat2_items', W=2200)
    print(HERE / 'chat2_items.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
