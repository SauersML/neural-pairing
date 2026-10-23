"""The 12-kind chat test (co_chat2): who says whose private item first, before and after training on it.

Each twin holds a private item of one of 12 kinds (a trip, a pet, a mood, ...); the twins chat, and a twin scores when
it says its partner's item before the partner does (chateval `names`). Columns: chats whose partner's item is of one of
the eight kinds nl26 trains on, and of one of the four kinds held out of its training. Blocks: before (nl22c step 157,
chat-trained on cities only) and after 300 steps of chat training on the eight trained kinds (nl26). One mark per twin
per chat, Quill's on the top line and Ember's on the bottom one. Rows: link on, link cut, and linked to a stranger (an
isolated twin of the partner's model holding a different item, linked in place of the partner): there a mark is filled
teal when the twin says the stranger's item first (chateval `names_donor`) and red when it says the partner's. A twin
whose partner (or stranger) holds no item is drawn as a dash and left out of the count.

Reads the evalfree --shard outputs (episodes.jsonl; no model, no torch) and writes figures/chat2_items.{svg,png,pdf}.
figures/oneway.py draws the one-way runs with draw().
Usage: python figures/chat2_items.py [RESULTS_DIR]   (default: results/magic-nl26/results)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, FAINT, INK, LINK, RED, SVG, WARM, key_item, load, mascot, pair_icon, width  # noqa: E402

RESULTS = HERE.parent / 'results/magic-nl26/results'
ROWS = (('link', 'Link on'), ('cut', 'Link cut'), ('crossed', 'Linked to a stranger'))
SIDES = ('quill', 'ember')
PITCH, LINE, RAD = 32, 36, 12


def mark(r, cond):
    """1 partner's item first, 2 stranger's item first (crossed only), 0 neither, None nothing to say."""
    c = r['conditions'][cond]
    if cond != 'crossed':
        return None if c.get('names') is None else int(c['names'])
    if c.get('names'):
        return 1
    return None if c.get('names_donor') is None else 2 * int(c['names_donor'])


def lines(records, cond, per_line):
    """Rows of marks: each receiver's chats in episode order, Quill's first, in even lines of at most per_line."""
    out = []
    for side in SIDES:
        v = [mark(r, cond) for r in sorted((r for r in records if r['receiver'] == side),
                                           key=lambda r: r['episode_index'])]
        if v:
            step = -(-len(v) // -(-len(v) // per_line))
            out += [v[i:i + step] for i in range(0, len(v), step)]
    return out


def draw(title, columns, blocks, stem, header_art=None):
    """columns: [(header, fill)]; blocks: [(block title, [records per column])]. header_art(s, x, y, k, uris), when
    given, draws column k's header picture (its width returned) before the header text, and the row labels go without
    the two-way pair icons."""
    W, M = 2400, 36
    label_w, gap = 470, 22
    top, head_h, block_h, key_h = 128, 108, 78, 96
    pw = (W - 2 * M - label_w - (len(columns)) * gap) / len(columns)
    px = [M + label_w + gap + k * (pw + gap) for k in range(len(columns))]
    per_line = int((pw - 80) // PITCH) + 1
    n_lines = max(len(lines(recs, 'link', per_line)) for _, cols in blocks for recs in cols)
    row_h = 84 + n_lines * LINE + 34
    panel_h = head_h + len(blocks) * (block_h + len(ROWS) * row_h) + 10
    H = top + panel_h + 24 + key_h + M
    s = SVG(W, H)
    uris = dict(quill=mascot('quill'), ember=mascot('ember'), stranger=mascot('ember', grey=True))
    s.text(M, 90, title, 64, 700)
    for k, (header, fill) in enumerate(columns):
        s.rect(px[k], top, pw, panel_h, fill, 32)
        art = header_art(s, px[k] + 34, top + 18, k, uris) + 20 if header_art else 6
        s.text(px[k] + 34 + art, top + 74, header, 46, 700)

    y0 = top + head_h
    for b, (block, cols) in enumerate(blocks):
        if b:
            for x1, x2 in [(M, M + label_w - 10)] + [(x + 30, x + pw - 30) for x in px]:
                s.line(x1, y0 + 8, x2, y0 + 8, '#D8D8DE', 4)
        s.text(M + 14, y0 + 62, block, min(46, int(46 * (label_w - 24) / width(block, 46, 700))), 700)
        y0 += block_h
        for cond, label in ROWS:
            msize = 104
            iy = y0 + (row_h - msize - 56) / 2
            if header_art:
                s.text(M + 14, y0 + 64, label, 40, 700)
            else:
                pair_icon(s, M + 14, iy, msize, cond, uris)
                s.text(M + 14 + (2 * msize + 86) / 2, iy + msize + 46, label, 38, 700, INK, 'middle')
            for k, recs in enumerate(cols):
                rows = lines(recs, cond, per_line)
                flat = [x for line in rows for x in line if x is not None]
                if cond == 'crossed':
                    donor, own = sum(x == 2 for x in flat), sum(x == 1 for x in flat)
                    chunks = [(f'{donor}', LINK if donor else INK, 700), (f' of {len(flat)}', INK, 400),
                              ("  stranger's item", INK, 400)]
                    if own:
                        chunks += [(f'  {own}', RED, 700), (" partner's", INK, 400)]
                else:
                    hits = sum(flat)
                    chunks = [(f'{hits}', RED if hits else INK, 700), (f' of {len(flat)}', INK, 400)]
                s.rich(px[k] + 40, y0 + 64, chunks, 50)
                for i, line in enumerate(rows):
                    for j, v in enumerate(line):
                        cx, cy = px[k] + 40 + RAD + j * PITCH, y0 + 104 + i * LINE
                        if v is None:
                            s.line(cx - 8, cy, cx + 8, cy, FAINT, 5)
                        elif v:
                            s.circle(cx, cy, RAD, RED if v == 1 else LINK)
                        else:
                            s.circle(cx, cy, RAD - 1.5, 'white', FAINT, 3)
            y0 += row_h

    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    x, y = M + 44, ky + key_h / 2 + 13
    both = any(len({r['receiver'] for r in recs}) == 2 for _, cols in blocks for recs in cols)
    items = [(RED, "partner's item first"), (LINK, "stranger's item first"), ('ring', 'not said'), ('dash', 'no item')]
    if both:
        items += [('quill', 'top line'), ('ember', 'bottom line')]
    for kind, label in items:
        x = key_item(s, x, y, kind, label, uris)
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    blocks = []
    for name, block in (('before', 'Trained on cities'), ('after', 'Trained on 8 kinds')):
        records = load([root / f'chat2_{name}_{k}' for k in range(3)])
        groups = ('trained_kind', 'held_out_kind')
        blocks.append((block, [[r for r in records if r['kind_group'] == g] for g in groups]))
    draw("Twins name each other's private item, of kinds they never trained on too",
         [('The 8 training kinds', COOL), ('The 4 held-out kinds', WARM)], blocks, HERE / 'chat2_items')
    print(HERE / 'chat2_items.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
