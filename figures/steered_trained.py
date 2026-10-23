"""The steered-twin test after training the link to read a pushed concept (chains/nl33.sh): how often the partner names
a concept pushed into its twin, before (nl22c step 157: nl29, nl30), after 300 steps with the write sizes learned, and
after 300 steps with the write sizes learned and the steered-reading game (steer_read.py), whose concepts never include
the test's 12.

Top left: the partner's first message (steering_scale.py, the nl29/nl30 protocol: 96 linked messages per setting).
Top right: the partner asked what its twin is thinking about (steer_ask.py, wordings never trained; 48 per setting).
Bottom: real openers after the steered game, both readers, the first chat prompt, block 6: three concepts pushed into
the twin, and the link cut (the same message for every concept).

Reads results/magic-nl29/after, results/magic-nl30/results/*, results/magic-nl33w/results/* and
results/magic-nl33t/results/* (JSON; no model, no torch) and writes figures/steered_trained.{svg,png,pdf}.
Usage: python figures/steered_trained.py [RESULTS_ROOT]   (default: results)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, EMBER, INK, LINK, QUILL, SVG, WARM, width  # noqa: E402

RESULTS = HERE.parent / 'results'
SETTINGS = (('l6', 'Block 6'), ('l15', 'Block 15'), ('l27', 'Block 27'), ('l15x4', 'Block 15, 4×'),
            ('l27x4', 'Block 27, 4×'))
BEFORE = dict(l6='magic-nl30/results/l6', l15='magic-nl29/after', l27='magic-nl30/results/l27',
              l15x4='magic-nl30/results/l15x4', l27x4='magic-nl30/results/l27x4')
ARMS = (('before', 'Before', '#D9D9DE'), ('write', 'Write size learned', '#9FD8C5'),
        ('steer', 'Write size learned + steered game', LINK))
RUNS = dict(write='magic-nl33w', steer='magic-nl33t')
TITLE = 'Trained to read a pushed concept, the partner names concepts it never trained on'
READER = dict(quill=QUILL, ember=EMBER)
# example openers (block 6, first chat prompt): concept pushed into the twin, and the word that names it
EXAMPLES = (('quill', (('fire', 'fire'), ('cooking', 'cooking'), ('ocean', 'whales'))),
            ('ember', (('football', 'football'), ('trains', 'trains'), ('space', 'cosmology'))))
CUT_WORD = dict(quill='the weather', ember='pyramids')
CUT_NOTE = 'Link cut: 0 in every setting'
EX_TITLE = "The partner's opener, same prompt, concept pushed after block 6"


def chat_counts(folder):
    """(linked, cut) messages, of 96, in which the partner says the concept first; None when not run."""
    p = folder / 'summary.json'
    if not p.exists():
        return None
    t = json.loads(p.read_text())
    return tuple(sum(c['counts'][k] for r in t for c in r['concepts']) for k in ('link', 'cut'))


def ask_counts(path):
    if not path.exists():
        return None
    s = json.loads(path.read_text())['summary']['both']
    return s['counts']['link'], s['counts']['cut'], s['n']


def load(root):
    chat, ask = {}, {}
    for t, _ in SETTINGS:
        chat['before', t] = chat_counts(root / BEFORE[t])
        ask['before', t] = None
        for arm, run in RUNS.items():
            chat[arm, t] = chat_counts(root / run / 'results' / t)
            ask[arm, t] = ask_counts(root / run / 'results' / f'ask_{t}.json')
        ask['before', t] = ask_counts(root / RUNS['write'] / 'results' / f'ask_before_{t}.json')
    return chat, ask


def bars(s, x0, y0, w, data, n, row_h, bar_h):
    """One group of bars per setting: arm bars of length count / n, the count at the end."""
    for i, (t, _) in enumerate(SETTINGS):
        for j, (arm, _, color) in enumerate(ARMS):
            y = y0 + i * row_h + j * (bar_h + 6)
            v = data.get((arm, t))
            if v is None:
                s.text(x0 + 8, y + bar_h - 8, '–', 30, 700)
                continue
            s.rect(x0, y, max(4, w * v[0] / n), bar_h, color, 6)
            s.text(x0 + max(4, w * v[0] / n) + 12, y + bar_h - 7, str(v[0]), 32, 700)


def examples(root):
    """For each reader, the partner's first chat message after block 6 with each example concept pushed into its twin,
    and with the link cut (the same for every concept: the cut partner's tokens equal its unsteered ones)."""
    ep = json.loads((root / RUNS['steer'] / 'results' / 'l6' / 'episodes.json').read_text())
    idx = {(e['receiver'], e['concept'], e['condition']): e['texts'][e['receiver']]
           for e in ep if e['structure'] == 'chat' and e['prompt'] == 0}
    return {r: ([(c, w, idx[r, c, 'link']) for c, w in picks] + [(None, CUT_WORD[r], idx[r, picks[0][0], 'cut'])])
            for r, picks in EXAMPLES}


def first_sentence(text):
    for k, ch in enumerate(text):
        if ch in '.?!' and k > 10:
            return text[:k + 1]
    return text


def wrap(text, size, w, lines=2):
    out, cur = [], ''
    for word in text.split():
        if width((cur + ' ' + word).strip(), size) <= w:
            cur = (cur + ' ' + word).strip()
        else:
            out.append(cur)
            cur = word
    out.append(cur)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1].rstrip('.,') + ' …'
    return out


def card(s, x, y, w, h, reader, concept, word, text):
    """One partner message: what was pushed into its twin on top, the message below with the concept word marked."""
    s.rect(x, y, w, h, 'white', 22, READER[reader], 4)
    if concept is None:
        s.text(x + 28, y + 52, 'Link cut', 34, 700, INK)
    else:
        s.rich(x + 28, y + 52, [('Pushed toward ', INK, 400), (concept, INK, 700)], 34)
    for k, line in enumerate(wrap(first_sentence(text), 34, w - 56)):
        chunks, rest = [], line
        while word and word in rest:
            a = rest.index(word)
            chunks += [(rest[:a], INK, 400), (word, LINK if concept else INK, 700)]
            rest = rest[a + len(word):]
        s.rich(x + 28, y + 112 + 46 * k, chunks + [(rest, INK, 400)], 34)


def draw(chat, ask, ex, stem):
    W, M = 2400, 40
    label_w = 330
    top, head_h, row_h, bar_h = 150, 90, 160, 42
    body_h = head_h + len(SETTINGS) * row_h + 10
    key_h = 120
    strip_head, card_h, card_gap = 100, 210, 26
    strip_h = strip_head + 2 * card_h + card_gap + 40
    H = top + body_h + key_h + strip_h + M
    s = SVG(W, H)
    s.text(M, 96, TITLE, min(62, int(62 * (W - 2 * M) / width(TITLE, 62, 700))), 700)
    c1 = M + label_w
    w1 = 1000
    c2 = c1 + w1 + 70
    w2 = W - M - c2 - 24
    s.rect(c1 - 24, top, w1 + 24, body_h, COOL, 28)
    s.rect(c2 - 24, top, w2 + 48, body_h, COOL, 28)
    s.text(c1, top + 62, "Partner's first message names it, of 96", 40, 700)
    s.text(c2, top + 62, 'Asked about its twin, of 48', 40, 700)
    y0 = top + head_h
    for i, (t, name) in enumerate(SETTINGS):
        s.text(M, y0 + i * row_h + 1.5 * (bar_h + 6) + 6, name, 40, 700)
    bars(s, c1, y0, w1 - 110, chat, 96, row_h, bar_h)
    bars(s, c2, y0, w2 - 80, {k: v for k, v in ask.items() if v}, 48, row_h, bar_h)

    ky = top + body_h + 72
    x = M
    for _, label, color in ARMS:
        s.rect(x, ky - 34, 56, 40, color, 6)
        s.text(x + 72, ky, label, 36)
        x += 72 + width(label, 36) + 60
    s.text(x, ky, CUT_NOTE, 36, 700)

    sy = top + body_h + key_h
    s.rect(M, sy, W - 2 * M, strip_h - 20, WARM, 28)
    s.text(M + 36, sy + 70, EX_TITLE, 40, 700)
    cw = (W - 2 * M - 72 - label_w - 3 * card_gap + 36) / 4
    for i, (reader, twin) in enumerate((('quill', 'Ember'), ('ember', 'Quill'))):
        y = sy + strip_head + i * (card_h + card_gap)
        s.text(M + 36, y + card_h / 2 - 8, f'{reader.title()}', 40, 700, READER[reader])
        s.text(M + 36, y + card_h / 2 + 42, f'reads {twin}', 36, 400, INK)
        for k, (concept, word, text) in enumerate(ex[reader]):
            card(s, M + label_w + k * (cw + card_gap), y, cw, card_h, reader, concept, word, text)
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw(*load(root), examples(root), HERE / 'steered_trained')
    print(HERE / 'steered_trained.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
