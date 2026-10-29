"""The listen-and-steer game drawn as its method, with its free-generation results (pairing/listen_steer_eval.py, the
final twins). One twin writes an answer to a question; its partner never writes, reads the answer as it streams, is
told a concept never seen in training (cards: samples of the training and test concept pools in data/pools.json) and
is asked to steer the answer over the link toward that concept. The example is one real test episode: the writer's
answer with the link on and with the link cut, quoted by its lines and whole sentences (answer). The table counts the
answers that mention the concept (evaluate.mentions), per writer: linked, cut, and linked to the partner of another
episode, told another concept, where the answers that mention that other concept are counted too.

Reads results/listen_steer/episodes.json and summary.json (no model) and writes figures/listen_steer.{svg,png,pdf}.
Usage: python figures/listen_steer.py [RESULTS_ROOT]   (default: results)
"""
import json
import re
import sys
from pathlib import Path

import creature
from chat_names import COOL, EMBER, INK, LINK, QUILL, RED, SVG, WARM, count, pair_icon, width, wrap
from method_parts import bubble, bubble_height, check_shape, flow, flow_height, highlight

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
RUN = 'listen_steer'
WRITER, EXAMPLE = 'quill', 51         # the example: this writer's episode with this index
COLOR = dict(quill=QUILL, ember=EMBER)
NAME = dict(quill='Quill', ember='Ember')
TINT = dict(quill='#EFE7FC', ember='#FCE9DE')
link_partner = dict(quill='ember', ember='quill').get
W, M, G, P, BODY = 2300, 40, 30, 34, 46
CHIPS_H = 2 * 68 + 14                  # two rows of concept chips


def load(root):
    rec = json.loads((root / RUN / 'episodes.json').read_text())
    summ = json.loads((root / RUN / 'summary.json').read_text())['summary']
    pools = json.loads((HERE.parent / 'data/pools.json').read_text())['concepts']
    train, test = ([phrase for _, phrase in pools[k]] for k in ('train', 'test'))
    assert all(r['phrase'] in test and r['phrase'] not in train for r in rec), 'a test concept is a training concept'
    ep = next(r for r in rec if (r['writer'], r['index']) == (WRITER, EXAMPLE))
    assert ep['link']['mentions'] and not ep['cut']['mentions'], 'the example no longer shows the concept linked only'
    return ep, summ, train, test


def terms(r):
    """The concept's spellings that evaluate.mentions accepts, longest first."""
    t = {r['concept'], r['phrase'], r['phrase'].removeprefix('the '), r['concept'] + 's'}
    return sorted(t, key=len, reverse=True)


def marked(text, r, color, fill):
    """runs of text with every whole-word mention of the concept bold in color on fill."""
    pat = r'(?<![A-Za-z0-9])(' + '|'.join(re.escape(t) for t in terms(r)) + r')(?![A-Za-z0-9])'
    runs, at = [], 0
    for m in re.finditer(pat, text, re.I):
        if m.start() > at:
            runs.append((text[at:m.start()], INK, 400, None))
        runs.append((m.group(0), color, 700, fill))
        at = m.end()
    if at < len(text):
        runs.append((text[at:], INK, 400, None))
    return runs


def answer(text, limit=200):
    """The answer as quoted: its lines with markdown marks dropped, as many of its sentences from the start as fit in
    limit characters (at least one), without the unfinished sentence where generation stopped, and an ellipsis when
    anything is left out. Returns the lines."""
    pieces = [(k, t) for k, line in enumerate(text.split('\n'))
              for t in re.split(r'(?<=[A-Za-z)][.!?])\s+', ' '.join(re.sub(r'[*#]+', '', line).split())) if t]
    full = pieces[:-1] if pieces[-1][1][-1] not in '.!?:' else pieces
    keep, used = [], 0
    for k, t in full:
        if keep and used + len(t) > limit:
            break
        keep.append((k, t))
        used += len(t)
    lines = {}
    for k, t in keep:
        lines.setdefault(k, []).append(t)
    out = [' '.join(v) for v in lines.values()]
    return out[:-1] + [out[-1] + ' …'] if len(keep) < len(pieces) else out


def para_height(paras, w, size=BODY, pad=30, gap=10):
    """The height of a bubble holding paragraphs (lists of runs) word-wrapped into width w."""
    lead = round(size * 1.32)
    return sum(flow_height(r, w - 2 * pad, size, lead) for r in paras) + gap * (len(paras) - 1) + 2 * pad - \
        (lead - size * 1.05)


def para_bubble(s, x, y, w, paras, stroke, size=BODY, pad=30, gap=10):
    """A white bubble outlined in stroke holding paragraphs (lists of runs); returns its height."""
    lead = round(size * 1.32)
    h = para_height(paras, w, size, pad, gap)
    s.rect(x, y, w, h, 'white', 26, stroke, 4)
    by = y + pad + size * 0.84
    for runs in paras:
        n = flow(s, x + pad, by, w - 2 * pad, runs, size, lead)
        by += n * lead + gap
    return h


def chips(s, x, y, w, items, mark=None, lc=INK, tint=None, draw=True):
    """Concept chips flowed into width w from (x, y), the concept mark bold in lc on tint, then an ellipsis; returns
    the height."""
    ch, gap, cx, cy = 68, 14, x, y
    for item in list(items) + ['…']:
        on = item == mark
        cw = width(item, BODY, 700 if on else 400) + 36
        if cx > x and cx + cw > x + w:
            cx, cy = x, cy + ch + gap
        if draw:
            if item != '…':
                s.rect(cx, cy, cw, ch, tint if on else 'white', 16, lc if on else 'none', 3)
            s.text(cx + cw / 2, cy + ch / 2 + 16, item, BODY, 700 if on else 400, lc if on else INK, 'middle')
        cx += cw + gap
    return cy + ch - y


def draw(ep, summ, train, test, stem):
    writer, listener = ep['writer'], ep['listener']
    wc, lc = COLOR[writer], COLOR[listener]
    task = ep['listener_task'].split('\n\n')[-1].split(': ', 1)[1]
    s = None

    # ---- the method panel: prompts, the pair at full body, the writer's answer linked and cut
    lx, lw = M, 540                                  # the listener's column
    rx, rw = 1320, W - M - 1320                      # the writer's column
    head = 60
    lprompt = highlight(f'“{task}”', [ep['phrase']], lc, TINT[listener])
    wprompt = [(f'“{ep["question"]}”', INK, 400, None)]
    link_runs = [marked(t, ep, lc, TINT[listener]) for t in answer(ep['link']['text'])]
    cut_runs = [[(t, INK, 400, None)] for t in answer(ep['cut']['text'])]
    lp_h = bubble_height(lprompt, lw, BODY)
    wp_h = bubble_height(wprompt, rw, BODY)
    link_h = para_height(link_runs, rw - 2 * P)
    cut_h = para_height(cut_runs, rw - 2 * P)
    icon = 96
    cards = [P + icon + 20 + h + P for h in (link_h, cut_h)]
    spread = lambda pool: pool[::len(pool) // 8]          # concepts from across the pool, not its first animals
    shown = [ep['phrase']] + [c for c in spread(test) if c != ep['phrase']]
    concepts = [(COOL, wrap(f'{len(train)} concepts in training', lw - 2 * P, BODY, 700), spread(train), None),
                (WARM, wrap(f'{len(test)} concepts never seen in training', lw - 2 * P, BODY, 700), shown,
                 ep['phrase'])]
    concept_h = [P + 58 * len(lines) + 16 + CHIPS_H + P for _, lines, _, _ in concepts]
    silent = wrap(f'{NAME[listener]} writes nothing and reads {NAME[writer]}’s answer', lw - 2 * P, BODY, 700)
    silent_h = 2 * P + 58 * len(silent) - 12
    left_h = head + lp_h + G + silent_h + G + sum(concept_h) + G
    method_h = max(head + wp_h + G + cards[0] + G + cards[1], left_h)

    # ---- the results panel: a table of counts
    rows = (('Link on', 'link', WARM), ('Link cut', 'cut', COOL), ('Partner from another episode', 'crossed', COOL))
    isize, rpad = 220, 26
    row_h = [2 * rpad + isize + (140 if c == 'crossed' else 0) for _, c, _ in rows]
    res_top = M + method_h + 70
    res_head = 210
    H = res_top + res_head + sum(row_h) + G * (len(rows) - 1) + M
    check_shape(W, H)
    s = SVG(W, H)

    # the listener's prompt and its silence
    s.text(lx, M + 46, f"{NAME[listener]}'s prompt", BODY, 700, lc)
    bubble(s, lx, M + head, lw, lprompt, BODY, stroke=lc, tail='none', sw=4)
    sy = M + head + lp_h + G
    s.rect(lx, sy, lw, silent_h, 'white', 26, lc, 4, 'stroke-dasharray="16 12"')
    for j, line in enumerate(silent):
        s.text(lx + P, sy + P + 34 + 58 * j, line, BODY, 700, lc)

    # the concepts: those of the training episodes, and those of the test, never seen in training
    sy += silent_h + G
    for (fill, lines, items, mark), ch in zip(concepts, concept_h):
        n = len(items)
        while chips(None, 0, 0, lw - 2 * P, items[:n], draw=False) > CHIPS_H:
            n -= 1
        s.rect(lx, sy, lw, ch, fill, 32)
        for j, line in enumerate(lines):
            s.text(lx + P, sy + P + 40 + 58 * j, line, BODY, 700)
        chips(s, lx + P, sy + P + 58 * len(lines) + 16, lw - 2 * P, items[:n], mark, lc, TINT[listener])
        sy += ch + G

    # the pair at full body, the listener next to its prompt; the link at its 8 layers
    top, h = M + 10, method_h - 90
    xl, xr = (lx + lw + 0.25 * (rx - lx - lw), lx + lw + 0.75 * (rx - lx - lw))
    if listener == 'quill':
        gl, pl = creature.creature('quill', xl, top, h, full=True)
        gr, pr = creature.creature('ember', xr, top, h, full=True)
    else:
        gl, pl = creature.creature('ember', xl, top, h, full=True, look=1, sway=0.5)
        gr, pr = creature.creature('quill', xr, top, h, full=True, look=-1, sway=-0.5)
    for k in creature.PORTS:
        s.add(creature.strands(pl[k][0], pr[k][0], pl[k][1]))
    s.add(gl)
    s.add(gr)
    s.text(xl, top + h + 66, NAME[listener], 50, 700, lc, 'middle')
    s.text(xr, top + h + 66, NAME[writer], 50, 700, wc, 'middle')
    s.text((xl + xr) / 2, pl[creature.PORTS[0]][1] + 70, 'Link', BODY, 700, LINK, 'middle')

    # the writer's prompt, then the answer linked and cut
    y = M
    s.text(rx, y + 46, f"{NAME[writer]}'s prompt", BODY, 700, wc)
    bubble(s, rx, y + head, rw, wprompt, BODY, stroke=wc, tail='none', sw=4)
    y += head + wp_h + G
    for (fill, cond, label, runs), ch, bh in zip(((WARM, 'link', f'Linked, {NAME[writer]} writes', link_runs),
                                                  (COOL, 'cut', f'Link cut, {NAME[writer]} writes', cut_runs)),
                                                 cards, (link_h, cut_h)):
        s.rect(rx, y, rw, ch, fill, 32)
        x = pair_icon(s, rx + P, y + P, icon, cond)
        s.text(x + 28, y + P + icon / 2 + 16, label, BODY, 700)
        para_bubble(s, rx + P, y + P + icon + 20, rw - 2 * P, runs, wc)
        y += ch + G

    # ---- results
    y = res_top
    s.text(M, y + 50, 'Answers that mention the concept in the silent partner’s prompt', 52, 700)
    cols = (1430, 1960)
    for cx, x in zip(cols, ('quill', 'ember')):
        s.text(cx, y + 124, f'{NAME[x]} writes', BODY, 700, COLOR[x], 'middle')
        s.text(cx, y + 182, f'{NAME[link_partner(x)]} steers', BODY, 700, COLOR[link_partner(x)], 'middle')
    y += res_head
    for (label, cond, fill), rh in zip(rows, row_h):
        s.rect(M, y, W - 2 * M, rh, fill, 32)
        pair_icon(s, M + 30, y + rpad, isize, cond, gap=0)
        mid = y + rpad + isize / 2
        lines = wrap(label, 520, 50, 700)
        for j, line in enumerate(lines):
            s.text(M + 30 + 2 * isize + 50, mid + 18 - 31 * (len(lines) - 1) + 62 * j, line, 50, 700)
        key = {'link': 'link', 'cut': 'cut', 'crossed': 'other'}[cond]
        for cx, x in zip(cols, ('quill', 'ember')):
            n, tot = summ[x][key], summ[x]['n_other' if cond == 'crossed' else 'n']
            w = width(n, 72, 700) + width(f' of {tot}', BODY)
            count(s, cx - w / 2, mid + 26, n, tot, RED if cond == 'link' else INK)
        if cond == 'crossed':
            s.text(sum(cols) / 2, mid + 140, 'mention the concept in that partner’s prompt', BODY, 700, LINK,
                   'middle')
            for cx, x in zip(cols, ('quill', 'ember')):
                n, tot = summ[x]['other_mentions_other'], summ[x]['n_other']
                w = width(n, 72, 700) + width(f' of {tot}', BODY)
                count(s, cx - w / 2, mid + 236, n, tot, LINK)
        y += rh + G
    s.save(stem)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw(*load(root), HERE / 'listen_steer')
    print(HERE / 'listen_steer.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
