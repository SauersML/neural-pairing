"""The 20-chat test before and after chat training: the percent of turns that repeat an earlier turn of the chat
(chateval `loop_turns`: the longest run of characters, lowercased, that the turn shares with an earlier turn by the same
twin or with the turn just before covers more than 60% of the turn), link on and link cut, as large numbers.

The link and cut dialogues are one dialogue per chat (the same for both receiving twins), so each chat is counted
once. Reads results/magic-nl20/results/chat_before_* and results/magic-nl22c/results/chat_after_* (JSON lines; no
model) and writes figures/chat_loops.{svg,png,pdf} and results/chat_loops.json (the counts drawn).
Usage: python figures/chat_loops.py
"""
import json
import sys

from chat_names import AFTER, BEFORE, COOL, INK, RED, ROOT, SVG, WARM, load, mascot, pair_icon

HERE = ROOT / 'figures'
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
    W, M, G = 1300, 30, 30
    label_w, head_h, row_h, msize = 330, 150, 250, 110
    top = 180
    H = top + head_h + 2 * row_h + M
    s = SVG(W, H)
    uris = {k: mascot(k) for k in ('quill', 'ember', 'quill_cut', 'ember_cut')}
    out = {}
    for when, recs in (('before', before), ('after', after)):
        for cond, _ in ROWS:
            cs = chats(recs, cond)
            out.setdefault(when, {})[cond] = dict(repeats=sum(sum(c) for c in cs), turns=sum(len(c) for c in cs),
                                                  chats=len(cs), per_chat=cs)
    link = out['after']['link']
    s.text(M, 76, f"After chat training, {round(100 * link['repeats'] / link['turns'])}% of linked turns", 56, 700)
    s.text(M, 146, 'repeat an earlier turn', 56, 700)

    pw = (W - 2 * M - label_w - 2 * G) / 2
    px = [M + label_w + G + k * (pw + G) for k in range(2)]
    for k, (when, fill) in enumerate((('Before', COOL), ('After', WARM))):
        s.rect(px[k], top, pw, head_h + 2 * row_h, fill, 32)
        s.text(px[k] + pw / 2, top + 66, f'{when} chat', 46, 700, INK, 'middle')
        s.text(px[k] + pw / 2, top + 122, 'training', 46, 700, INK, 'middle')
    for j, (cond, label) in enumerate(ROWS):
        y0 = top + head_h + j * row_h
        iy = y0 + (row_h - msize - 60) / 2
        right = pair_icon(s, M + (label_w - 2 * msize - 86) / 2, iy, msize, cond, uris)
        s.text(right - (2 * msize + 86) / 2, iy + msize + 54, label, 46, 700, INK, 'middle')
        for k, when in enumerate(('before', 'after')):
            v = out[when][cond]
            cx = px[k] + pw / 2
            s.text(cx, y0 + row_h / 2 + 4, f"{round(100 * v['repeats'] / v['turns'])}%", 84, 700,
                   RED if when == 'after' else INK, 'middle')
            s.text(cx, y0 + row_h / 2 + 70, f"{v['repeats']} of {v['turns']} turns", 46, 400, INK, 'middle')
    s.save(stem)
    return out


def main():
    before, after = load(BEFORE), load(AFTER)
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
