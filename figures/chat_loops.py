"""The 20-chat test before and after chat training: every turn of every chat as a square, filled when the turn repeats
an earlier turn of the chat (chateval `loop_turns`: the longest run of characters the turn shares, lowercased, with an
earlier turn by the same twin or with the turn just before covers more than 60% of the turn), link on and link cut.

The link and cut dialogues are one dialogue per chat (the same for both receiving twins), so each chat is counted
once. Reads results/magic-nl20/results/chat_before_* and results/magic-nl22c/results/chat_after_* (plain JSON lines;
no model, no torch) and writes figures/chat_loops.{svg,png,pdf} and results/chat_loops.json (the counts drawn).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import (AFTER, BEFORE, COOL, EMBER, FAINT, INK, MUTED, QUILL, ROOT, SVG, WARM, load,  # noqa: E402
                        mascot, pair_icon)

SNAPSHOT = ROOT / 'results/chat_loops.json'
ROWS = (('link', 'Link on'), ('cut', 'Link cut'))


def chats(records, cond):
    """Per chat in episode order (one record per episode: the dialogue is the same for both receivers), its repeat
    flag per turn."""
    first = {}
    for r in sorted(records, key=lambda r: (r['episode_index'], r['receiver'])):
        first.setdefault(r['episode_index'], r)
    for i, r in first.items():
        other = [x for x in records if x['episode_index'] == i and x is not r]
        assert all(x['conditions'][cond]['turns'] == r['conditions'][cond]['turns'] for x in other)
    return [first[i]['conditions'][cond]['loop_turns'] for i in sorted(first)]


def draw(before, after, stem):
    W, M = 2400, 36
    top, head_h, row_h, key_h = 128, 96, 410, 100
    panel_h = head_h + 2 * row_h + 6
    H = top + panel_h + 24 + key_h + M
    s = SVG(W, H)
    uris = dict(quill=mascot('quill'), ember=mascot('ember'))
    s.text(M, 90, 'Chat training nearly ended repeated turns', 68, 700)

    label_w, gap = 330, 22
    pw = (W - 2 * M - label_w - 2 * gap) / 2
    px = [M + label_w + gap, M + label_w + 2 * gap + pw]
    for k, (title, fill) in enumerate((('Before chat training', COOL), ('After chat training', WARM))):
        s.rect(px[k], top, pw, panel_h, fill, 32)
        s.text(px[k] + 36, top + 70, title, 48, 700)

    sq, pitch = 38, 46
    out = {}
    for j, (cond, label) in enumerate(ROWS):
        y0 = top + head_h + j * row_h
        if j:
            for k in range(2):
                s.line(px[k] + 30, y0, px[k] + pw - 30, y0, '#E3DED6' if k else '#E2E2E7', 2)
            s.line(M, y0, M + label_w - 10, y0, '#E2E2E7', 2)
        msize = 112
        iy = y0 + row_h / 2 - msize / 2 - 34
        right = pair_icon(s, M, iy, msize, cond, uris)
        s.text((M + right) / 2, iy + msize + 64, label, 44, 700, INK, 'middle')
        for k, (when, recs) in enumerate((('before', before), ('after', after))):
            cs = chats(recs, cond)
            n, rep = sum(len(c) for c in cs), sum(sum(c) for c in cs)
            out.setdefault(when, {})[cond] = dict(repeats=rep, turns=n, chats=len(cs), per_chat=cs)
            cx0 = px[k] + (pw - (len(cs) * pitch - (pitch - sq))) / 2
            s.rich(cx0, y0 + 74, [(f'{round(100 * rep / n)}%', INK, 700), (f'   {rep}/{n} turns', MUTED, 400)], 58)
            for i, c in enumerate(cs):
                for t, v in enumerate(c):
                    x, y = cx0 + i * pitch, y0 + 108 + t * pitch
                    if v:
                        s.rect(x, y, sq, sq, INK, 6)
                    else:
                        s.rect(x + 1.5, y + 1.5, sq - 3, sq - 3, 'white', 6, FAINT, 3)
    ky = top + panel_h + 24
    s.rect(M, ky, W - 2 * M, key_h, COOL, 32)
    x, y = M + 44, ky + key_h / 2 + 13
    s.rect(x, y - 32, sq, sq, INK, 6)
    s.text(x + sq + 18, y, 'repeated turn', 36)
    x += sq + 18 + 290
    s.rect(x + 1.5, y - 30.5, sq - 3, sq - 3, 'white', 6, FAINT, 3)
    s.text(x + sq + 18, y, 'new turn', 36)
    x += sq + 18 + 230
    for t in range(6):
        s.rect(x, ky + 12 + t * 13, 20, 11, 'white', 2, MUTED, 2)
    s.text(x + 40, y, 'one chat, 6 turns', 36)
    s.save(stem)
    return out


def main():
    try:
        before, after = load(BEFORE), load(AFTER)
    except FileNotFoundError as e:
        print('missing', e.filename)
        return 1
    out = draw(before, after, HERE / 'chat_loops')
    snap = dict(sources=dict(before=[str(p.relative_to(ROOT)) for p in BEFORE],
                             after=[str(p.relative_to(ROOT)) for p in AFTER]),
                metric='chateval loop_turns per turn; one dialogue per chat under link and cut', **out)
    SNAPSHOT.write_text(json.dumps(snap, indent=1) + '\n')
    for k in ('before', 'after'):
        print(k, {c: f"{v['repeats']}/{v['turns']} = {100 * v['repeats'] / v['turns']:.1f}%" for c, v in out[k].items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
