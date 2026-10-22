"""The 20-chat test, before and after chat training: one dot per chat side whose partner has a private city; red when
the twin said the partner's city before the partner did. Rows: link on, link cut, linked to a stranger (an isolated
twin with a different private city: chateval's crossed condition). Reads evalfree --shard outputs through
chat_merge.load. Renders figures/chat_names.html to figures/chat_names.png with headless Chrome."""
import base64
import html
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pairing'))
from chat_merge import load  # noqa: E402

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
RUNS = Path('/Users/user/mpd-data/runpod')
BEFORE = [RUNS / f'magic-nl20/results/chat_before_{k}' for k in range(2)]
AFTER = [RUNS / f'magic-nl22c/results/chat_after_{k}' for k in range(3)]
QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
W, H = 1200, 660
ROWS = (('link', 'names', 'Link on'), ('cut', 'names', 'Link cut'), ('crossed', 'names', 'Linked to a stranger'),
        ('crossed', 'names_donor', "Says stranger's city"))


def img(name):
    return 'data:image/png;base64,' + base64.b64encode((HERE / 'assets' / name).read_bytes()).decode()


def hits(records, cond, metric='names'):
    """Per chat side with a partner city (names) or a linked stranger with a city (names_donor), in a fixed order: 1
    when the twin said that city before its owner did."""
    rs = sorted((r for r in records if r['conditions'][cond].get(metric) is not None),
                key=lambda r: (r['episode_id'], r['receiver']))
    return [int(r['conditions'][cond][metric]) for r in rs]


def icon(cond):
    band = {'link': '<b class="on"></b>' * 6, 'cut': '<b></b>' * 2 + '<i>×</i>' + '<b></b>' * 2,
            'crossed': '<b class="on"></b>' * 6}[cond]
    right = img('ember.png')
    return f'''<div class="icon"><img src="{img('quill.png')}"><div class="band">{band}</div>
      <img class="{'stranger' if cond == 'crossed' else ''}" src="{right}"></div>'''


def column(title, records):
    rows = []
    for cond, metric, label in ROWS:
        h = hits(records, cond, metric)
        dots = ''.join(f'<span class="{"hit" if v else ""}"></span>' for v in h)
        rows.append(f'''<div class="row">{icon(cond)}<div class="what"><div class="head"><div class="label">{label}</div>
        <div class="count{' red' if sum(h) and cond != 'cut' else ''}">{sum(h)}<small> / {len(h)}</small></div></div>
        <div class="dots">{dots}</div></div></div>''')
    return f'<section><h2>{title}</h2>{"".join(rows)}</section>'


def page(before, after):
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: {W}px; height: {H}px; background: #fff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: {INK}; padding: 22px 24px; display: flex; flex-direction: column; gap: 18px; }}
h1 {{ font-size: 38px; font-weight: 700; letter-spacing: -.3px; }}
h1 .red {{ color: {RED}; }}
.cols {{ display: flex; gap: 22px; flex: 1; }}
section {{ flex: 1 1 0; min-width: 0; border-radius: 30px; padding: 22px 26px; display: flex; flex-direction: column;
  gap: 22px; background: #F4F4F6; }}
section:last-child {{ background: #FAF0E6; }}
h2 {{ font-size: 32px; font-weight: 700; }}
.row {{ display: flex; align-items: center; gap: 16px; }}
.icon {{ display: flex; align-items: center; gap: 4px; flex: none; }}
.icon img {{ width: 50px; }}
.icon img.stranger {{ filter: grayscale(1) brightness(1.15); }}
.band {{ display: flex; gap: 3px; align-items: center; width: 42px; justify-content: center; }}
.band b {{ width: 4px; height: 16px; border-radius: 2px; background: #C7C7CC; }}
.band b.on {{ background: {LINK}; }}
.band i {{ font-style: normal; font-size: 20px; color: #8E8E93; line-height: 1; }}
.what {{ flex: 1; min-width: 0; }}
.head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 14px; margin-bottom: 8px; }}
.label {{ font-size: 28px; font-weight: 700; white-space: nowrap; }}
.dots {{ display: grid; grid-template-columns: repeat(18, 14px); gap: 5px 5px; }}
.dots span {{ width: 14px; height: 14px; border-radius: 50%; border: 2px solid #B9B9C0; }}
.dots span.hit {{ background: {RED}; border-color: {RED}; }}
.count {{ font-size: 34px; font-weight: 700; white-space: nowrap; }}
.count.red {{ color: {RED}; }}
.count small {{ font-size: 26px; color: {INK}; font-weight: 400; }}
</style></head><body>
<h1>The twins say each other's <span class="red">secret city</span> only through the link</h1>
<div class="cols">{column('Before chat training', before)}{column('After chat training', after)}</div>
</body></html>'''


def main():
    try:
        before, after = load(BEFORE), load(AFTER)
    except FileNotFoundError as e:
        print('missing', e.filename)
        return 0
    out = HERE / 'chat_names.html'
    out.write_text(page(before, after))
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--force-device-scale-factor=2', f'--screenshot={HERE / "chat_names.png"}', out.as_uri()],
                   check=True, capture_output=True)
    print(HERE / 'chat_names.png')


if __name__ == '__main__':
    sys.exit(main())
