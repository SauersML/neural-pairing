"""Free conversations with the link cut (pairing/beauty_rl.py generations, results/magic-beautyrl6): one real prompt
played linked and cut, and how many of the 96 conversations per condition (2 plays x 48 prompts) before the
judge-rewarded training ('init') have a turn that repeats an earlier turn of the same conversation word for word
(equal after lowercasing and deleting whitespace and punctuation) or a list or outline turn (a line that starts with
a number and a full stop, "-" or "*" followed by a space, or text in "**").

Reads results/magic-beautyrl6/results/gen.json (no model) and writes figures/conversations_cut.{svg,png,pdf} and
results/conversations_cut.json (conversation and turn counts for 'init' and for 'rl', after the training).
Usage: python figures/conversations_cut.py [GEN_JSON]   (default: results/magic-beautyrl6/results/gen.json)
"""
import json
import re
import string
import sys
from pathlib import Path

from chat_names import COOL, EMBER, INK, LINK, QUILL, RED, SVG, WARM, mascot, pair_icon, width, wrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEN = ROOT / 'results/magic-beautyrl6/results/gen.json'
PLAY, PROMPT = 0, 44
TITLE = 'With the link cut, the twins more often repeat a turn word for word or write a list'
COLOR = dict(quill=QUILL, ember=EMBER)
MARKER = re.compile(r'\s*(\d+\.|-|\*)\s')
PUNCT = re.compile('[' + re.escape(string.punctuation) + r'\s]')


def norm(text):
    return PUNCT.sub('', text.lower())


def is_list(text):
    return '**' in text or any(MARKER.match(line) for line in text.split('\n'))


def marks(conv):
    """Per turn: (repeats an earlier turn, is a list or outline turn)."""
    return [(norm(t) in {norm(u) for _, u in conv[:j]}, is_list(t)) for j, (_, t) in enumerate(conv)]


def tally(gen):
    """Per run and condition: conversations and turns in all, and those with a repeat or a list turn."""
    out = {}
    for run in ('init', 'rl'):
        for cond in ('link', 'cut'):
            convs = [c for play in gen[run][cond] for c in play]
            m = [marks(c) for c in convs]
            out[f'{run}|{cond}'] = dict(
                conversations=len(convs), turns=sum(map(len, convs)),
                repeat=dict(conversations=sum(any(r for r, _ in x) for x in m), turns=sum(r for x in m for r, _ in x)),
                list=dict(conversations=sum(any(li for _, li in x) for x in m),
                          turns=sum(li for x in m for _, li in x)))
    return out


def rows(text, w, size, most=3):
    """The turn as at most `most` wrapped rows, starting just before its first list line, with an ellipsis where text is
    left out; lines that hold only a list marker (cut off by the length limit) are dropped."""
    w -= width(' …', size)  # room for the ellipsis
    out = []
    for para in text.split('\n'):
        if para.strip() and para.strip() not in ('-', '*'):
            out += [(r, bool(MARKER.match(para)) and k == 0) for k, r in enumerate(wrap(para.strip(), w, size))]
    first = next((k for k, (_, li) in enumerate(out) if li), 0)
    start = max(0, first - 1)
    keep = [r for r, _ in out[start:start + most]]
    if start:
        keep[0] = '… ' + keep[0]
    if start + most < len(out) or not text.rstrip().endswith(('.', '?', '!', '"')):
        keep[-1] = keep[-1].rstrip(' ,—-') + ' …'
    return keep


def prompt_text(gen):
    """The example prompt as the twin that writes first reads it."""
    p = gen['prompts'][PROMPT]
    other = 'Ember' if p['first'] == 'quill' else 'Quill'
    return p['template'].replace('{P}', other).replace('{X}', p['topic'])


def draw(gen, counts, stem):
    W, M, G, P = 2400, 40, 30, 30
    size, gap, pad, icon, tag_w = 44, 58, 24, 76, 150
    top = 120
    prompt = prompt_text(gen)
    cw = (W - 2 * M - G) / 2
    bw = cw - 2 * P - icon - 16 - tag_w
    convs = [gen['init'][cond][PLAY][PROMPT] for cond in ('link', 'cut')]
    heights = [[2 * pad + gap * (len(rows(t, bw - 2 * pad, size)) - 1) + size for _, t in c] for c in convs]
    turns_h = max(sum(h) + 20 * (len(h) - 1) for h in heights)
    head_h, count_h = icon + 30, 2 * 130
    prompt_rows = wrap(prompt, W - 2 * M - 2 * pad, 46)
    prompt_h = 60 + 2 * pad + 62 * (len(prompt_rows) - 1) + 46
    card_h = P + head_h + turns_h + 40 + count_h + P
    H = top + prompt_h + G + card_h + M
    s = SVG(W, H)
    uris = {k: mascot(k) for k in ('quill', 'ember', 'quill_cut', 'ember_cut')}
    s.text(M, 78, TITLE, min(56, int(56 * (W - 2 * M) / width(TITLE, 56, 700))), 700)
    s.text(M, top + 42, "Quill's prompt, Quill writes first", 46, 700)
    s.rect(M, top + 60, W - 2 * M, prompt_h - 60, 'white', 24, INK, 3)
    for k, r in enumerate(prompt_rows):
        s.text(M + pad, top + 60 + pad + 38 + 62 * k, r, 46)

    y0 = top + prompt_h + G
    for c, (cond, fill, head, color, conv, hs) in enumerate(zip(('link', 'cut'), (WARM, COOL), ('Link on', 'Link cut'),
                                                               (LINK, RED), convs, heights)):
        x0 = M + c * (cw + G)
        s.rect(x0, y0, cw, card_h, fill, 32)
        x = pair_icon(s, x0 + P, y0 + P, icon, cond, uris)
        s.text(x + 28, y0 + P + icon / 2 + 16, head, 50, 700)
        y = y0 + P + head_h
        for (who, text), (rep, li), h in zip(conv, marks(conv), hs):
            left = who == 'quill'
            bx = x0 + P + icon + 16 if left else x0 + cw - P - icon - 16 - bw
            s.image(x0 + P if left else x0 + cw - P - icon, y + h / 2 - icon / 2, icon,
                    uris[who + ('_cut' if cond == 'cut' else '')])
            s.rect(bx, y, bw, h, 'white', 22, RED if rep else COLOR[who], 6 if rep else 4)
            for k, r in enumerate(rows(text, bw - 2 * pad, size)):
                s.text(bx + pad, y + pad + size * 0.8 + gap * k, r, size)
            tag = 'repeat' if rep else 'list' if li else None
            if tag:
                s.text(bx + bw + 20 if left else bx - 20, y + h / 2 + 16, tag, 46, 700, RED, 'start' if left else 'end')
            y += h + 20
        y = y0 + P + head_h + turns_h + 40
        c_ = counts[f'init|{cond}']
        for k, (key, lines) in enumerate((('repeat', ('conversations with a turn', 'repeated word for word')),
                                          ('list', ('conversations with', 'a list or outline')))):
            n = c_[key]['conversations']
            ty = y + 130 * k
            s.text(x0 + P, ty + 72, str(n), 80, 700, color)
            nx = x0 + P + width(str(n), 80, 700) + 20
            s.text(nx, ty + 40, f'of {c_["conversations"]} ' + lines[0], 46, 700)
            s.text(nx, ty + 96, lines[1], 46, 700)
    s.save(stem)


def main():
    gen = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else GEN).read_text())
    counts = tally(gen)
    draw(gen, counts, HERE / 'conversations_cut')
    (ROOT / 'results/conversations_cut.json').write_text(json.dumps(counts, indent=1) + '\n')
    print(json.dumps(counts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
