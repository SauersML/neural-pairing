"""What the steered-reading game does, as results: each of the 12 test concepts (never trained) steered into Quill at four
times its calibrated strength after layer 27, and Ember's real first message for the free-chat prompt with the link on,
next to the one message Ember writes with the link cut (the same for every concept, since nothing reaches it); then
the partner's first messages that say the concept, of 96 per setting, after the game (nl33t), after the same 300
steps without it (nl33w) and before the game (the chat-trained twins: nl29, nl30), as in figures/steered_trained.py.

The marked words are the test's own keywords for the concept (steered_mind.matches), so a message with a marked word
is one the test counts. A message is quoted up to the first sentence end that leaves at least 30 characters and
includes its first keyword, and cut with an ellipsis where it would run past three lines (or past the keyword's line).

Reads results/magic-nl33t/results/l27x4/episodes.json and the summaries steered_trained.py reads (JSON; no model) and
writes figures/steered_results.{svg,png,pdf}.
Usage: python figures/steered_results.py [RESULTS_ROOT]   (default: results)
"""
import json
import re
import sys
from pathlib import Path

import creature
from chat_names import ROOT
from chat_names import COOL, EMBER, INK, LINK, QUILL, SVG, WARM, pair_icon, width, wrap
from steered_trained import SETTINGS, load

sys.path.insert(0, str(ROOT / 'pairing'))
from steered_mind import matches  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
RUN, LAYER, BOOST = 'magic-nl33t/results/l27x4', 27, 4
STEERED, READER, PROMPT = 'quill', 'ember', 1
SIZE, LINE, PAD = 40, 52, 22  # message text: font size, line spacing, bubble padding
COLOR = dict(quill=QUILL, ember=EMBER)


def episodes(root):
    """(task, rows, cut): the reader's prompt, (concept, linked message, names it) for each concept with the concepts
    the reader names first, and the reader's one message with the link cut."""
    ep = json.loads((root / RUN / 'episodes.json').read_text())
    pick = {}
    for e in ep:
        if (e['receiver'], e['sender'], e['structure'], e['prompt']) == (READER, STEERED, 'chat', PROMPT):
            pick.setdefault(e['concept'], {})[e['condition']] = e
    task = next(iter(pick.values()))['link']['episode']['system'][READER].split('\n\n')[-1].split(' Write only')[0]
    cuts = {v['cut']['texts'][READER] for v in pick.values()}
    assert len(cuts) == 1, 'the cut message differs between concepts'
    rows = [(c, v['link']['texts'][READER], bool(matches(v['link']['texts'][READER], c))) for c, v in pick.items()]
    rows.sort(key=lambda r: (not r[2], r[0]))
    return task, rows, cuts.pop()


def quote(text, w, concept):
    """The message's opening as quoted (see the module docstring), as wrapped rows."""
    spans = matches(text, concept)
    need = max(30, spans[0][1] if spans else 0)
    ends = [m.end() for m in re.finditer(r'[.!?](\s|$)', text)]
    end = next((e for e in ends if e >= need), len(text))
    q = text[:end].strip() + ('' if end >= len(text.rstrip()) or text[:end].rstrip()[-1] in '.!?' else ' …')
    rows = wrap(q, w, SIZE)
    keep = max([3] + [k + 1 for k, row in enumerate(rows) if matches(row, concept)][:1])
    if len(rows) > keep:
        n = max(len(' '.join(rows[:keep]).split()) - 1, len(text[:spans[0][1]].split()) if spans else 0)
        rows = wrap(' '.join(q.split()[:n]) + ' …', w, SIZE)
    return rows


def marked(s, x, y, row, concept, hit):
    """One wrapped row, each of the concept's keywords bold in the steered twin's colour."""
    chunks, at = [], 0
    for a, b in (matches(row, concept) if hit else []):
        chunks += [(row[at:a], INK, 400), (row[a:b], COLOR[STEERED], 700)]
        at = b
    s.rich(x, y, chunks + [(row[at:], INK, 400)], SIZE)


def bubble(s, x, y, w, rows, concept='', hit=False):
    """A white speech bubble from the reader; returns its height (s None: measure only)."""
    h = 2 * PAD + LINE * (len(rows) - 1) + SIZE
    if s is not None:
        s.rect(x, y, w, h, 'white', 22, COLOR[READER], 4)
        for k, row in enumerate(rows):
            marked(s, x + PAD + 4, y + PAD + SIZE * 0.8 + LINE * k, row, concept, hit)
    return h


def draw(task, rows, cut, counts, stem):
    M, P, G, BW, LW, CH = 40, 36, 30, 900, 230, 62
    reader, steered = READER.capitalize(), STEERED.capitalize()
    chip_w = max(width(c, SIZE, 700) for c, _, _ in rows) + 44
    x0 = M + P
    xp = x0 + chip_w + 170  # where the steered concepts join
    xl = xp + 1010  # the right column: concept names, then the messages
    xb = xl + LW
    W = xb + BW + P + M
    quoted = [quote(t, BW - 2 * PAD - 8, c) for c, t, _ in rows]
    heights = [bubble(None, 0, 0, BW, q) for q in quoted]
    head = wrap(f'{reader}\'s prompt: “{task}”', BW + LW, SIZE)
    y_rows = M + P + 66 + LINE * len(head) + 34
    rows_h = sum(heights) + 16 * (len(rows) - 1)
    main_h = y_rows - M + rows_h + P
    cut_rows = wrap(f'“{cut}”', W - 2 * M - 2 * P - 2 * PAD - 8, SIZE)
    cut_h = P + 96 + 24 + bubble(None, 0, 0, 0, cut_rows) + P
    tile_h = 372
    res_h = P + 60 + 24 + tile_h + P
    H = M + main_h + G + cut_h + G + res_h + M
    s = SVG(W, H)

    # the linked panel: the concepts, the pair at full body, and the reader's messages
    s.rect(M, M, W - 2 * M, main_h, WARM, 32)
    s.text(x0, M + P + 46, f'12 concepts steered into {steered}', 48, 700)
    s.text(x0, M + P + 46 + 60, f'{BOOST}× strength, after layer {LAYER}', 44, 400)
    s.text(xl, M + P + 46, f'Linked, {reader} writes first', 48, 700)
    for k, row in enumerate(head):
        s.text(xl, M + P + 46 + 60 + LINE * k, row, SIZE, 400)

    top, h = y_rows - 20, rows_h + 20 - 80
    qx = xp + 250
    ex = qx + 470
    gq, pq = creature.creature(STEERED, qx, top, h, full=True)
    ge, pe = creature.creature(READER, ex, top, h, full=True)
    for k in creature.PORTS:
        s.add(creature.strands(pq[k][0], pe[k][0], pq[k][1]))
    s.add(gq)
    s.add(ge)
    for name, x in ((STEERED, qx), (READER, ex)):
        s.text(x, top + h + 66, name.capitalize(), 50, 700, COLOR[name], 'middle')
    s.text((qx + ex) / 2, pq[creature.PORTS[0]][1] + 72, 'Link', 46, 700, LINK, 'middle')

    ly = pq[LAYER][1]
    stack = len(rows) * CH + (len(rows) - 1) * 14
    cy0 = min(max(ly - stack / 2, y_rows), M + main_h - P - stack)
    plus = xp
    for k, (c, _, _) in enumerate(rows):
        cy = cy0 + k * (CH + 14) + CH / 2
        w = width(c, SIZE, 700) + 44
        s.path(f'M{x0 + w},{cy:.1f} C{x0 + w + 110},{cy:.1f} {plus - 120},{ly:.1f} {plus - 28},{ly:.1f}',
               COLOR[STEERED], 4)
        s.rect(x0, cy - CH / 2, w, CH, 'white', CH / 2, COLOR[STEERED], 4)
        s.text(x0 + w / 2, cy + 14, c, SIZE, 700, COLOR[STEERED], 'middle')
    s.circle(plus, ly, 28, 'white', COLOR[STEERED], 5)
    s.line(plus - 14, ly, plus + 14, ly, COLOR[STEERED], 5)
    s.line(plus, ly - 14, plus, ly + 14, COLOR[STEERED], 5)
    s.line(plus + 28, ly, qx - 16, ly, COLOR[STEERED], 6)
    s.path(f'M{qx - 36},{ly - 16} L{qx - 14},{ly} L{qx - 36},{ly + 16}', COLOR[STEERED], 6)

    y = y_rows
    for (c, _, hit), q, bh in zip(rows, quoted, heights):
        s.text(xl, y + PAD + SIZE * 0.8, c, SIZE, 700, COLOR[STEERED])
        bubble(s, xb, y, BW, q, c, hit)
        y += bh + 16

    # the cut panel: one message for every concept
    y = M + main_h + G
    s.rect(M, y, W - 2 * M, cut_h, COOL, 32)
    x = pair_icon(s, x0, y + P, 96, 'cut')
    s.text(x + 30, y + P + 64, f'Link cut, {reader} writes the same message for all 12', 48, 700)
    bubble(s, x0, y + P + 96 + 24, W - 2 * M - 2 * P, cut_rows)

    # the counts over the whole test
    y += cut_h + G
    s.rect(M, y, W - 2 * M, res_h, WARM, 32)
    s.text(x0, y + P + 46, f'Of 96 test messages per setting, the partner says the steered concept first', 48, 700)
    tw = (W - 2 * M - 2 * P - 4 * 24) / len(SETTINGS)
    for i, (t, name) in enumerate(SETTINGS):
        tx, ty = x0 + i * (tw + 24), y + P + 84
        s.rect(tx, ty, tw, tile_h, 'white', 24)
        for k, row in enumerate(name.split('\n')):
            s.text(tx + 26, ty + 56 + 50 * k, row, SIZE, 700)
        n = counts['steer', t][0]
        s.text(tx + 26, ty + 214, str(n), 92, 700, LINK)
        s.text(tx + 26 + width(str(n), 92, 700) + 14, ty + 214, 'with the game', SIZE, 400)
        s.text(tx + 26, ty + 276, f'{counts["write", t][0]} without the game', SIZE, 400)
        s.text(tx + 26, ty + 328, f'{counts["before", t][0]} before the game', SIZE, 400)
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    counts, _ = load(root)
    draw(*episodes(root), counts, HERE / 'steered_results')
    print(HERE / 'steered_results.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
