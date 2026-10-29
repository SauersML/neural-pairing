"""The link in one drawing: Quill and Ember, 36 layers each, at one moment of a real linked chat. After each of the 8 link
layers, each twin sends its states to the partner and each adds what it reads from the partner to its own residual
stream, both directions at once (one thick arrow per direction and layer). Below, the same read over time: the token
that reads takes in the partner's states from every partner token so far, the partner's prompt included.

The moment: turn 0 of the linked chat figures/inputs/chat_link_s7.json.gz, where Quill writes "So, you're in Beijing?"
and Beijing is the city in Ember's private prompt (Quill's is Dallas). Quill's current token is the last one before
Beijing; Quill reads each of its own written tokens one time step after writing it, and Ember reads the tokens of
Quill's text that can no longer change (all but the last of the re-encoding with Ember's tokenizer), one step later
(pairing/session.py's dialogue rule, as figures/link_mechanism.py's timing grid). The ellipsis stands for the tokens
between the prompt's city and the turn (the rest of the prompt and the turn header), which every token may read too.

Reads results/link_arch.json (built by --refresh from the chat and the cached Ember tokenizer) and checks the link
layers against pairing/link.py (XBLOCKS). Writes figures/link_arch.{svg,png,pdf}.
Usage: python figures/link_arch.py [--refresh]
"""
import ast
import gzip
import json
import sys

import creature
from chat_names import EMBER, INK, LINK, QUILL, ROOT, SVG, width
from creature import BODY_W, PORTS
from method_parts import TINT, check_shape, head, render

CHAT = ROOT / 'figures/inputs/chat_link_s7.json.gz'
SNAP = ROOT / 'results/link_arch.json'
OUT = ROOT / 'figures/link_arch'
EMBER_TOKENIZER = 'arcee-ai/AFM-4.5B'
NOW = 5                                   # Quill's input time at the token before the city
COLOR = {'quill': QUILL, 'ember': EMBER}
BODY = 46
PAD = 28                                  # chip padding
ROW = 70 + 80 + 34 + 60                 # one read over time: heading, chips, bar, gap


def xblocks():
    tree = ast.parse((ROOT / 'pairing/link.py').read_text())
    return next(list(ast.literal_eval(n.value)) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'XBLOCKS' for t in n.targets))


def refresh():
    """Each twin's tokens with their read times up to NOW, Quill's next token and both private cities."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    chat = json.loads(gzip.open(CHAT).read())
    toks = chat['turns'][0]['tokens']
    assert chat['turns'][0]['speaker'] == 'quill'
    quill = [(v['text'], v['tick'] + 1) for v in toks if v['tick'] + 1 <= NOW]
    tok = Tokenizer.from_file(hf_hub_download(EMBER_TOKENIZER, 'tokenizer.json', local_files_only=True))
    ember, prior = [], []
    for v in toks:
        if v['tick'] + 1 > NOW:
            break
        stable = tok.encode(''.join(t['text'] for t in toks[:toks.index(v) + 1]), add_special_tokens=False).ids[:-1]
        assert stable[:len(prior)] == prior
        ember += [(tok.decode([t]), v['tick'] + 1) for t in stable[len(prior):]]
        prior = stable
    nxt = next(v['text'] for v in toks if v['tick'] == NOW)
    cities = {x: chat['episode']['meta']['secrets'][x]['city'] for x in ('quill', 'ember')}
    return dict(source=str(CHAT.relative_to(ROOT)), now=NOW, tokens=dict(quill=quill, ember=ember), next=nxt,
                cities=cities, link_layers=xblocks(), layers=36)


def chip(s, x, y, runs, name, bold=False, h=80, anchor='middle'):
    """A token chip of (text, colour, weight) runs at height y, centred on x (anchor 'start': from x); returns its width
    and left edge."""
    w = sum(width(t, BODY, wt) for t, _, wt in runs) + 2 * PAD
    left = x - w / 2 if anchor == 'middle' else x - w if anchor == 'end' else x
    s.rect(left, y, w, h, TINT[name] if bold else 'white', 16, COLOR[name], 6 if bold else 4)
    s.rich(left + PAD, y + h / 2 + BODY * 0.36, runs, BODY)
    return w, left


def plus(s, x, y, color, r=20):
    s.circle(x, y, r, 'white', color, 5)
    s.line(x - r * 0.55, y, x + r * 0.55, y, color, 5)
    s.line(x, y - r * 0.55, x, y + r * 0.55, color, 5)


def thick(s, x1, x2, y, color, sw=12, size=38):
    """A haloed arrow from x1 to x2 at height y."""
    d = 1 if x2 > x1 else -1
    s.line(x1, y, x2 - d * (size - 6), y, 'white', sw + 10)
    s.line(x1, y, x2 - d * (size - 6), y, color, sw)
    head(s, x2 + d * 4, y, 0 if d > 0 else 3.14159, 'white', size + 8)
    head(s, x2, y, 0 if d > 0 else 3.14159, color, size)


def reads(s, x0, y, reader, sender, f, toks, flip):
    """One read over time: the reader's current token, then the partner's prompt city, an ellipsis and the partner's
    tokens so far, joined by a bar under them and one arrow into the reader's chip. flip: the reader on the right."""
    city = [(f['cities'][sender], INK, 700)]
    items = [city, None] + [[(t, INK, 400)] for t, _ in toks[sender]]
    ws = [sum(width(t, BODY, wt) for t, _, wt in r) + 2 * PAD if r else 46 for r in items]
    gap = 12
    rw = width(toks[reader][-1][0], BODY, 700) + 2 * PAD
    arrow = 80
    total = rw + arrow + sum(ws) + gap * (len(ws) - 1)
    x = x0 - total if flip else x0
    if flip:
        rx = x + total - rw
        sx = x
    else:
        rx = x
        sx = x + rw + arrow
    chip(s, rx + rw / 2, y, [(toks[reader][-1][0], INK, 700)], reader, bold=True)
    centres = []
    for r, w in zip(items, ws):
        if r is None:
            s.text(sx + w / 2, y + 52, '…', BODY, 700, INK, 'middle')
        else:
            chip(s, sx, y, r, sender, bold=r is city, anchor='start')
            centres.append(sx + w / 2)
        sx += w + gap
    bar = y + 80 + 34
    s.path(''.join(f'M{c:.1f},{y + 80} V{bar} ' for c in centres), COLOR[sender], 8)
    left, right = min(centres), max(centres)
    if flip:
        s.path(f'M{left:.1f},{bar} H{right:.1f} H{rx + rw / 2 - 30:.1f} Q{rx + rw / 2},{bar} {rx + rw / 2},{bar - 30}',
               COLOR[sender], 8)
        head(s, rx + rw / 2, y + 80 + 6, -1.5708, COLOR[sender], 34)
    else:
        s.path(f'M{right:.1f},{bar} H{left:.1f} H{rx + rw / 2 + 30:.1f} Q{rx + rw / 2},{bar} {rx + rw / 2},{bar - 30}',
               COLOR[sender], 8)
        head(s, rx + rw / 2, y + 80 + 6, -1.5708, COLOR[sender], 34)
    return total


def draw(f):
    toks = {x: [(t.strip() or t, at) for t, at in f['tokens'][x]] for x in ('quill', 'ember')}
    W, M = 2300, 40
    top, height = 60, 1380
    total = creature.Y0 + 108 + 2 * creature.HEAD_RY - 16 + 36 * 27.0
    scale = height / total
    neck = creature.Y0 + 108 + 2 * creature.HEAD_RY - 16
    port = {k: top + (neck + (35 - k) * 27.0 + 13.5) * scale for k in range(36)}
    qx, ex = 740, 1560
    band = top + height + 30 + 80 + 70
    H = round(band + ROW - 30 + M)
    check_shape(W, H)
    s = SVG(W, H)

    # the twins at the current token, and after each link layer one arrow each way, core to core
    gq, _ = creature.creature('quill', qx, top, height, full=True, look=1)
    ge, _ = creature.creature('ember', ex, top, height, full=True, look=-1)
    s.add(gq)
    s.add(ge)
    off = 20
    for k in PORTS:
        y = port[k]
        thick(s, qx, ex - 24, y - off, QUILL)
        thick(s, ex, qx + 24, y + off, EMBER)
        s.circle(qx, y - off, 9, QUILL)
        s.circle(ex, y + off, 9, EMBER)
        plus(s, ex, y - off, QUILL)
        plus(s, qx, y + off, EMBER)

    # layer numbers, names, the input tokens and the token Quill writes next
    lx = qx - BODY_W * scale / 2 - 60
    for k in PORTS:
        s.text(lx, port[k] + 15, str(k), 44, 700, LINK, 'end')
    for k in (0, 35):
        s.text(lx, port[k] + 15, str(k), 44, 700, INK, 'end')
    s.text(lx - width('35', 44, 700) - 20, port[35] + 15, 'layer', 44, 400, INK, 'end')
    s.text((qx + ex) / 2, port[35] + 10, '8 times', 54, 700, LINK, 'middle')
    s.text((qx + ex) / 2, port[35] + 74, 'per token', 54, 700, LINK, 'middle')
    lines = ('added to the', 'residual stream')
    tw = max(width(t, BODY, 700) for t in lines)
    gx = (qx + ex) / 2 - (tw + 60) / 2
    gy = (port[0] + port[4]) / 2
    plus(s, gx + 20, gy - 4, INK)
    for i, t in enumerate(lines):
        s.text(gx + 60, gy - 14 + i * 56, t, BODY, 700, INK)
    s.text(M + 20, top + 80, 'Quill', 76, 700, QUILL)
    s.text(M + 20, top + 150, 'Qwen3-4B', BODY, 400, INK)
    s.text(W - M - 20, top + 80, 'Ember', 76, 700, EMBER, 'end')
    s.text(W - M - 20, top + 150, 'AFM-4.5B', BODY, 400, INK, 'end')
    cy = top + height + 30
    chip(s, qx, cy, [(toks['quill'][-1][0], INK, 700)], 'quill', bold=True)
    chip(s, ex, cy, [(toks['ember'][-1][0], INK, 700)], 'ember', bold=True)
    hy = top + (creature.Y0 + 108 + creature.HEAD_RY) * scale
    nw, bx = chip(s, qx - creature.HEAD_RX * scale - 110, hy - 40, [(f['next'].strip(), INK, 700)], 'quill',
                  anchor='end')
    thick(s, qx - creature.HEAD_RX * scale - 8, bx + nw + 4, hy, QUILL, 10, 32)

    # the same read over time: every partner token so far, Quill's read on the left and Ember's on the right
    widths = []
    for x, y, flip in (('quill', 'ember', False), ('ember', 'quill', True)):
        at = M + 20 if not flip else W - M - 20
        s.rich(at, band + 40, [(x.title(), COLOR[x], 700), (' reads ', INK, 700), (y.title(), COLOR[y], 700),
                               ('’s prompt and tokens so far', INK, 700)], BODY, anchor='end' if flip else 'start')
        widths.append(reads(s, at, band + 70, x, y, f, toks, flip))
    assert sum(widths) + 60 < W - 2 * M - 40, widths
    render(s, OUT)
    return W, H


def main():
    if '--refresh' in sys.argv:
        SNAP.write_text(json.dumps(refresh(), indent=1, ensure_ascii=False) + '\n')
    f = json.loads(SNAP.read_text())
    assert f['link_layers'] == xblocks() == list(PORTS), 'link layers disagree with pairing/link.py'
    print(draw(f), f['tokens'], f['next'])


if __name__ == '__main__':
    main()
