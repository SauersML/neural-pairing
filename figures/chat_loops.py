"""The 20-chat test, before and after chat training: every turn of every chat as a square, filled when the turn
repeats an earlier turn of the chat (chateval's loop_turns), link on and link cut. Reads evalfree --shard outputs
through chat_merge.load. Renders figures/chat_loops.html to figures/chat_loops.png with headless Chrome."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pairing')]
from chat_merge import load  # noqa: E402
from chat_names import AFTER, BEFORE, CHROME, INK, LINK, icon  # noqa: E402

W, H = 1200, 540
ROWS = (('link', 'Link on'), ('cut', 'Link cut'))


def chats(records, cond):
    """Per chat (one record per episode: the dialogue is the same for both receivers), its loop flag per turn."""
    seen, out = set(), []
    for r in sorted(records, key=lambda r: (r['episode_id'], r['receiver'])):
        if r['episode_id'] in seen:
            continue
        seen.add(r['episode_id'])
        out.append(r['conditions'][cond]['loop_turns'])
    return out


def column(title, records):
    rows = []
    for cond, label in ROWS:
        cs = chats(records, cond)
        n, k = sum(len(c) for c in cs), sum(sum(c) for c in cs)
        grid = ''.join('<div class="chat">' + ''.join(f'<span class="{"loop" if t else ""}"></span>' for t in c)
                       + '</div>' for c in cs)
        rows.append(f'''<div class="row">{icon(cond)}<div class="what"><div class="head"><div class="label">{label}</div>
          <div class="count">{round(100 * k / n)}%</div></div><div class="grid">{grid}</div></div></div>''')
    return f'<section><h2>{title}</h2>{"".join(rows)}</section>'


def page(before, after):
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: {W}px; height: {H}px; background: #fff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: {INK}; padding: 22px 24px; display: flex; flex-direction: column; gap: 16px; }}
h1 {{ font-size: 38px; font-weight: 700; letter-spacing: -.3px; }}
.cols {{ display: flex; gap: 22px; flex: 1; }}
section {{ flex: 1 1 0; min-width: 0; border-radius: 30px; padding: 22px 26px; display: flex; flex-direction: column;
  gap: 22px; background: #F4F4F6; }}
section:last-child {{ background: #FAF0E6; }}
h2 {{ font-size: 32px; font-weight: 700; }}
.row {{ display: flex; align-items: center; gap: 16px; }}
.icon {{ display: flex; align-items: center; gap: 4px; flex: none; }}
.icon img {{ width: 50px; }}
.band {{ display: flex; gap: 3px; align-items: center; width: 42px; justify-content: center; }}
.band b {{ width: 4px; height: 16px; border-radius: 2px; background: #C7C7CC; }}
.band b.on {{ background: {LINK}; }}
.band i {{ font-style: normal; font-size: 20px; color: #8E8E93; line-height: 1; }}
.what {{ flex: 1; min-width: 0; }}
.head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }}
.label {{ font-size: 28px; font-weight: 700; }}
.count {{ font-size: 34px; font-weight: 700; }}
.grid {{ display: flex; gap: 3px; }}
.chat {{ display: flex; flex-direction: column; gap: 3px; }}
.chat span {{ width: 14px; height: 14px; border-radius: 3px; background: #fff; border: 2px solid #C7C7CC; }}
.chat span.loop {{ background: {INK}; border-color: {INK}; }}
</style></head><body>
<h1>Chat training ended the word-for-word loops</h1>
<div class="cols">{column('Before chat training', before)}{column('After chat training', after)}</div>
</body></html>'''


def main():
    try:
        before, after = load(BEFORE), load(AFTER)
    except FileNotFoundError as e:
        print('missing', e.filename)
        return 0
    out = HERE / 'chat_loops.html'
    out.write_text(page(before, after))
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--force-device-scale-factor=2', f'--screenshot={HERE / "chat_loops.png"}', out.as_uri()],
                   check=True, capture_output=True)
    print(HERE / 'chat_loops.png')


if __name__ == '__main__':
    sys.exit(main())
