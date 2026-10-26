"""The steered-twin test drawn as its method, with one real episode after training the link to read steered concepts
(chains/nl33.sh, nl33t): Quill steered toward a concept after layer 27 (pairing/steering_scale.py: the concept's
direction, at its calibrated strength, added to Quill's residual stream after block 27, where the link reads it), the
link joining the two models at its 8 layers, and Ember's real first message for the same free-chat prompt with the link
on and with the link cut.

Reads results/magic-nl33t/results/l27/episodes.json (no model) and writes figures/steered_method.{svg,png,pdf}.
Usage: python figures/steered_method.py [RESULTS_ROOT]   (default: results)
"""
import json
import sys
from pathlib import Path

import creature
from chat_names import COOL, EMBER, INK, LINK, QUILL, SVG, WARM, mascot, pair_icon, width, wrap

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
RUN, LAYER = 'magic-nl33t/results/l27', 27
STEERED, READER, CONCEPT, PROMPT = 'quill', 'ember', 'football', 1
TITLE = 'Linked, Ember writes about the concept steered into Quill after layer 27'
COLOR = dict(quill=QUILL, ember=EMBER)


def episode(root):
    """(prompt, linked message, cut message): the reader's first message with the concept steered into its twin."""
    ep = json.loads((root / RUN / 'episodes.json').read_text())
    pick = {e['condition']: e for e in ep if (e['receiver'], e['sender'], e['concept'], e['structure'], e['prompt'])
            == (READER, STEERED, CONCEPT, 'chat', PROMPT)}
    task = pick['link']['episode']['system'][READER].split('\n\n')[-1].split(' Write only')[0]
    link, cut = (pick[c]['texts'][READER] for c in ('link', 'cut'))
    assert CONCEPT in link.lower() and CONCEPT not in cut.lower(), 'the example no longer shows the concept linked only'
    return task, link, cut


def message(text):
    """The message as quoted: cut after its last full stop or question mark, with an ellipsis when that drops words."""
    end = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
    return text if end == len(text) - 1 else text[:end + 1] + ' …'


def bubble(s, x, y, w, text, size=48, pad=30, gap=66, draw=True, edge=COLOR[READER]):
    """A white speech bubble from the reader holding text, the concept word in the steered twin's colour; returns its
    height."""
    rows = wrap(text, w - 2 * pad, size)
    h = 2 * pad + gap * (len(rows) - 1) + size
    if not draw:
        return h
    s.rect(x, y, w, h, 'white', 24, edge, 4)
    for k, row in enumerate(rows):
        chunks, rest = [], row
        while CONCEPT in rest.lower():
            a = rest.lower().index(CONCEPT)
            chunks += [(rest[:a], INK, 400), (rest[a:a + len(CONCEPT)], COLOR[STEERED], 700)]
            rest = rest[a + len(CONCEPT):]
        s.rich(x + pad, y + pad + size * 0.8 + gap * k, chunks + [(rest, INK, 400)], size)
    return h


def draw(task, link, cut, stem):
    W, M, G, P = 2100, 40, 30, 30
    top = 140
    right = 1090
    cw = W - M - right
    icon = 96
    bubbles = [(WARM, 'link', 'Linked, Ember writes', message(link)),
               (COOL, 'cut', 'Link cut, Ember writes', message(cut))]
    prompt_h = bubble(None, 0, 0, cw, f'“{task}”', 46, draw=False) + 70
    card_h = [P + icon + 24 + bubble(None, 0, 0, cw - 2 * P, t, draw=False) + P for _, _, _, t in bubbles]
    H = top + prompt_h + G + sum(card_h) + G + M
    s = SVG(W, H)
    s.text(M, 84, TITLE, 56, 700)

    # the pair at full body: the link's strands at its 8 layers, then the creatures
    h = H - top - M - 70
    qx, ex = 515, 915
    gq, pq = creature.creature(STEERED, qx, top, h, full=True)
    ge, pe = creature.creature(READER, ex, top, h, full=True)
    for k in creature.PORTS:
        s.add(creature.strands(pq[k][0], pe[k][0], pq[k][1]))
    s.add(gq)
    s.add(ge)
    for name, x in ((STEERED, qx), (READER, ex)):
        s.text(x, H - M - 14, name.capitalize(), 50, 700, COLOR[name], 'middle')
    ly = pq[LAYER][1]
    s.text((qx + ex) / 2, pq[creature.PORTS[0]][1] + 70, 'Link', 46, 700, LINK, 'middle')

    # the steering: the concept's direction added into Quill's residual stream after layer 27
    chip_w = width(CONCEPT, 48, 700) + 56
    cx0 = M
    s.text(cx0, ly - 150, 'Steering after', 46, 700)
    s.text(cx0, ly - 94, f'layer {LAYER}', 46, 700)
    s.rect(cx0, ly - 40, chip_w, 80, 'white', 40, COLOR[STEERED], 5)
    s.text(cx0 + chip_w / 2, ly + 16, CONCEPT, 48, 700, COLOR[STEERED], 'middle')
    plus = cx0 + chip_w + 70
    s.line(cx0 + chip_w + 8, ly, plus - 26, ly, COLOR[STEERED], 6)
    s.circle(plus, ly, 26, 'white', COLOR[STEERED], 5)
    s.line(plus - 13, ly, plus + 13, ly, COLOR[STEERED], 5)
    s.line(plus, ly - 13, plus, ly + 13, COLOR[STEERED], 5)
    s.line(plus + 26, ly, qx - 16, ly, COLOR[STEERED], 6)
    s.path(f'M{qx - 34},{ly - 16} L{qx - 12},{ly} L{qx - 34},{ly + 16}', COLOR[STEERED], 6)

    # the prompt, then Ember's first message linked and cut
    y = top
    s.text(right, y + 46, "Ember's prompt", 46, 700)
    bubble_h = bubble(s, right, y + 70, cw, f'“{task}”', 46, edge=INK)
    y += 70 + bubble_h + G
    uris = {k: mascot(k) for k in ('quill', 'ember', 'quill_cut', 'ember_cut')}
    for (fill, cond, head, text), ch in zip(bubbles, card_h):
        s.rect(right, y, cw, ch, fill, 32)
        x = pair_icon(s, right + P, y + P, icon, cond, uris)
        s.text(x + 28, y + P + icon / 2 + 16, head, 46, 700)
        bubble(s, right + P, y + P + icon + 24, cw - 2 * P, text)
        y += ch + G
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw(*episode(root), HERE / 'steered_method')
    print(HERE / 'steered_method.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
