"""The link in one drawing: Quill and Ember, 36 layers each, at one moment of a real linked chat. After each of the 8 link
layers, each twin's current token reads the partner's states at that layer from every partner token so far (the
partner's prompt included), and adds what it reads to its own residual stream; both directions at once.

The moment: turn 0 of the linked chat figures/inputs/chat_link_s7.json.gz, where Quill writes "So, you're in Beijing?"
and Beijing is the city in Ember's private prompt (Quill's is Dallas). Quill's current token is the last one before
Beijing; Quill reads each of its own written tokens one time step after writing it, and Ember reads the tokens of
Quill's text that can no longer change (all but the last of the re-encoding with Ember's tokenizer), one step later
(pairing/session.py's dialogue rule, as figures/link_mechanism.py's timing grid). Earlier columns are the earlier tokens
of each twin at their times, and the prompt column stands for the prompt, whose positions every token may read.

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
from creature import BODY_W, PORTS, _taper
from method_parts import TINT, check_shape, head, render

CHAT = ROOT / 'figures/inputs/chat_link_s7.json.gz'
SNAP = ROOT / 'results/link_arch.json'
OUT = ROOT / 'figures/link_arch'
EMBER_TOKENIZER = 'arcee-ai/AFM-4.5B'
NOW = 5                                   # Quill's input time at the token before the city
COLOR = {'quill': QUILL, 'ember': EMBER}
BODY = 46


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


def ghost(s, name, cx, top, scale, w):
    """An earlier token's 36 layers, without the head: pucks of the widest width w at the creature's heights, with the
    teal ring at each link layer."""
    neck = creature.Y0 + 108 + 2 * creature.HEAD_RY - 16
    ph = 0.62 * 27.0 * scale
    for i in range(36):
        k = 35 - i
        y = top + (neck + i * 27.0 + 27.0 * 0.19) * scale
        pw = w * _taper((k + 0.5) / 36)
        s.rect(cx - pw / 2, y, pw, ph, TINT[name], ph / 2, COLOR[name], 2.5)
        if k in PORTS:
            s.rect(cx - pw / 2 - 6, y - 5, pw + 12, ph + 10, 'none', (ph + 10) / 2, LINK, 4)


def chip(s, cx, y, text, name, bold=False, h=74):
    w = width(text, BODY, 700 if bold else 400) + 36
    s.rect(cx - w / 2, y, w, h, TINT[name] if bold else 'white', 16, COLOR[name], 5 if bold else 3)
    s.text(cx, y + h / 2 + BODY * 0.36, text, BODY, 700 if bold else 400, INK, 'middle')
    return w


def plus(s, x, y, r=15, color=LINK):
    s.circle(x, y, r, 'white', color, 4)
    s.line(x - r * 0.55, y, x + r * 0.55, y, color, 4)
    s.line(x, y - r * 0.55, x, y + r * 0.55, color, 4)


def draw(f):
    toks = {x: [(t.strip() or t, at) for t, at in f['tokens'][x]] for x in ('quill', 'ember')}
    W, M = 2300, 40
    top, height = 70, 1560
    total = creature.Y0 + 108 + 2 * creature.HEAD_RY - 16 + 36 * 27.0
    scale = height / total
    half = BODY_W * scale / 2                      # half the widest puck of a current twin
    pitch, gw = 118, 84                            # earlier columns: spacing and widest puck
    gap = 560                                      # core to core
    lab = 150                                      # the layer numbers
    # columns from the left: Quill's prompt, ..., Quill's earlier tokens, Quill now | Ember now, earlier, ..., prompt
    nq, ne = len(toks['quill']) - 1, len(toks['ember']) - 1
    skip = 210                                     # prompt column to first token: room for the city and an ellipsis
    px = {'quill': lab + 100}
    xq = [px['quill'] + skip + pitch * k for k in range(nq)]
    qx = xq[-1] + gw / 2 + 40 + half
    ex = qx + gap
    xe = [ex + half + 40 + gw / 2 + pitch * k for k in range(ne)]
    px['ember'] = xe[-1] + skip
    W = round(px['ember'] + width(f['cities']['ember'], BODY, 700) / 2 + 18 + M)
    H = round(top + height + 30 + 74 + 40 + 60 + M)
    check_shape(W, H)
    s = SVG(W, H)
    neck = creature.Y0 + 108 + 2 * creature.HEAD_RY - 16
    port = {k: top + (neck + (35 - k) * 27.0 + 13.5) * scale for k in range(36)}

    # earlier columns, then the wires that collect each twin's states at a link layer for the partner
    for x, xs in (('quill', [px['quill']] + xq), ('ember', xe + [px['ember']])):
        for c in xs:
            ghost(s, x, c, top, scale, gw)
    off = 10
    for k in PORTS:
        y = port[k]
        s.path(f'M{px["quill"]},{y:.1f} H{qx + half * 0.3:.1f} L{qx + half:.1f},{y - off:.1f} H{ex:.1f}', QUILL, 6)
        s.path(f'M{px["ember"]},{y:.1f} H{ex - half * 0.3:.1f} L{ex - half:.1f},{y + off:.1f} H{qx:.1f}', EMBER, 6)
        for c in [px['quill']] + xq:
            s.circle(c, y, 9, QUILL)
        for c in xe + [px['ember']]:
            s.circle(c, y, 9, EMBER)

    # the twins at the current token
    gq, _ = creature.creature('quill', qx, top, height, full=True, look=1)
    ge, _ = creature.creature('ember', ex, top, height, full=True, look=-1)
    s.add(gq)
    s.add(ge)
    for k in PORTS:                                 # each read arrives in the reader's residual stream
        y = port[k]
        wq = BODY_W * _taper((k + 0.5) / 36) * scale / 2
        s.line(ex - wq - 4, y - off, ex - 20, y - off, QUILL, 6)
        head(s, ex - 14, y - off, 0, QUILL, 20)
        s.line(qx + wq + 4, y + off, qx + 20, y + off, EMBER, 6)
        head(s, qx + 14, y + off, 3.14159, EMBER, 20)
        plus(s, qx, y + off + 2, 13)
        plus(s, ex, y - off - 2, 13)

    # layer numbers
    for k in PORTS:
        s.text(lab - 10, port[k] + 15, str(k), 42, 700, LINK, 'end')
    for k in (0, 35):
        s.text(lab - 10, port[k] + 15, str(k), 42, 700, INK, 'end')
    s.text(lab - 10, port[35] - 50, 'layer', 42, 400, INK, 'end')

    # tokens: each column's input token; Quill's next token above its head
    cy = top + height + 30
    for x, cols in (('quill', [px['quill']] + xq + [qx]), ('ember', [ex] + xe + [px['ember']])):
        words = [t for t, _ in toks[x]]
        words = (['prompt'] + words) if x == 'quill' else (words[::-1] + ['prompt'])
        for k, (c, t) in enumerate(zip(cols, words)):
            now = (x == 'quill' and k == len(cols) - 1) or (x == 'ember' and k == 0)
            if t == 'prompt':
                chip(s, c, cy, f['cities'][x], x, bold=True)
            else:
                chip(s, c, cy, t, x, bold=now)
    qw = width(f['cities']['quill'], BODY, 700) / 2 + 18
    ew = width(f['cities']['ember'], BODY, 700) / 2 + 18
    s.text((px['quill'] + qw + xq[0] - 40) / 2, cy + 50, '…', BODY, 700, INK, 'middle')
    s.text((xe[-1] + 40 + px['ember'] - ew) / 2, cy + 50, '…', BODY, 700, INK, 'middle')
    for x in ('quill', 'ember'):
        s.text(px[x], cy + 74 + 52, 'prompt', 42, 700, COLOR[x], 'middle')

    # names and the token Quill writes next
    s.text(lab + 20, top + 70, 'Quill', 70, 700, QUILL)
    s.text(lab + 20, top + 140, 'Qwen3-4B', BODY, 400, INK)
    s.text(W - M, top + 70, 'Ember', 70, 700, EMBER, 'end')
    s.text(W - M, top + 140, 'AFM-4.5B', BODY, 400, INK, 'end')
    hy = top + (creature.Y0 + 108 + creature.HEAD_RY) * scale
    nw = width(f['next'].strip(), BODY, 700) + 36
    bx = qx - half - 70 - nw
    s.rect(bx, hy - 37, nw, 74, 'white', 16, QUILL, 5)
    s.text(bx + nw / 2, hy + 17, f['next'].strip(), BODY, 700, INK, 'middle')
    s.line(qx - creature.HEAD_RX * scale - 6, hy, bx + nw + 26, hy, QUILL, 6)
    head(s, bx + nw + 6, hy, 3.14159, QUILL, 22)
    s.text((qx + ex) / 2, port[35] + 10, '8 times', 50, 700, LINK, 'middle')
    s.text((qx + ex) / 2, port[35] + 70, 'per token', 50, 700, LINK, 'middle')
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
