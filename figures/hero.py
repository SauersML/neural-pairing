"""Opening figure: the same two twins chatting with no task, link cut (left) and link on (right). Each twin's private
city is a thought bubble only that twin has; the partner's city appears in red where a twin says it. Text comes from
the free-chat transcripts (free_chat.py, seed 7). Renders figures/hero.html to figures/hero.png with headless Chrome."""
import base64
import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHATS = Path('/Users/user/mpd-data/neural-pairing/free_chat/nl22c_s100')
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
W, H = 1200, 872
DEPTHS = 8   # link.py: the link joins the twins after 8 of their blocks


def city(system):
    return re.search(r'spending today in ([^.]+)\.', system).group(1).strip()


def img(name):
    return 'data:image/png;base64,' + base64.b64encode((HERE / 'assets' / name).read_bytes()).decode()


def clip(text, around=None, before=5, after=4):
    """A short excerpt: the whole first sentence, or a window of tokens around the word `around`."""
    words = text.split()
    if around is None:
        return ' '.join(words[:9]) + ' …'
    i = next(k for k, w in enumerate(words) if around in w)
    lo, hi = max(0, i - before), min(len(words), i + after + 1)
    return ('… ' if lo else '') + ' '.join(words[lo:hi]) + (' …' if hi < len(words) else '')


def mark(text, words):
    out = html.escape(text)
    for w in words:
        out = re.sub(r'(?<!\w)(' + re.escape(w) + r')(?!\w)', r'<b class="hit">\1</b>', out)
    return out


def panel(cond, title, warm):
    d = json.load(open(CHATS / f'chat_{cond}_s7.json'))
    cities = {x: city(s) for x, s in d['episode']['system'].items()}
    first = {}
    for t in d['turns']:
        first.setdefault(t['speaker'], t['text'])
    partner = {'quill': 'ember', 'ember': 'quill'}
    lines = []
    for x in ('quill', 'ember'):
        other = cities[partner[x]]
        said = next((t['text'] for t in d['turns'] if t['speaker'] == x and other in t['text']), None)
        lines.append((x, mark(clip(said, other) if said else clip(first[x]), [other])))
    stripes = '<span></span>' * DEPTHS
    if warm:
        threads = ''.join(f'<div class="thread">{"<i></i>" * 12}</div>' for _ in range(DEPTHS))
    else:
        threads = ''.join(f'<div class="thread cut">{"<i></i>" * 5}<em></em>{"<i></i>" * 5}</div>' for _ in range(DEPTHS))
        threads += '<b class="x">×</b>'
    bubbles = ''.join(f'''
      <div class="row {x}"><img class="ava" src="{img(x + '.png')}"><div class="bubble">{t}</div></div>''' for x, t in lines)
    twin = lambda x, c: f'''<div class="twin {c}"><div class="think"><span>{html.escape(cities[x])}</span></div><div class="dots"><i></i><i></i></div>
        <img src="{img(x + '.png')}"><div class="body">{stripes}</div></div>'''
    return f'''
  <section class="{'warm' if warm else 'cool'}">
    <h1 class="{'on' if warm else ''}">{title}</h1>
    <div class="stage">
      {twin('quill', 'q')}
      <div class="threads">{threads}</div>
      {twin('ember', 'e')}
    </div>
    <div class="chat">{bubbles}
    </div>
  </section>'''


def page():
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: {W}px; height: {H}px; background: #fff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: {INK}; display: flex; gap: 24px; padding: 20px; }}
section {{ flex: 1 1 0; min-width: 0; border-radius: 34px; padding: 26px 26px 30px; display: flex; flex-direction: column; }}
.cool {{ background: #F4F4F6; }}
.warm {{ background: #FAF0E6; }}
h1 {{ font-size: 46px; text-align: center; font-weight: 700; letter-spacing: -0.5px; }}
h1.on {{ color: {RED}; }}
.stage {{ display: flex; align-items: flex-end; justify-content: space-between; margin-top: 2px; }}
.twin {{ width: 156px; display: flex; flex-direction: column; align-items: center; position: relative; }}
.twin img {{ width: 132px; position: relative; z-index: 2; }}
.body {{ width: 132px; margin-top: -8px; padding: 14px 16px; border-radius: 22px; background: #fff;
  display: flex; flex-direction: column; gap: 9px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
.body span {{ display: block; height: 9px; border-radius: 5px; }}
.q .body span {{ background: {QUILL}; opacity: .55; }}
.e .body span {{ background: {EMBER}; opacity: .55; }}
.think {{ background: #fff; border-radius: 40px; padding: 16px 20px; font-size: 34px; font-weight: 700;
  box-shadow: 0 2px 10px rgba(0,0,0,.08); white-space: nowrap; }}
.q .think span {{ color: {QUILL}; }}
.e .think span {{ color: {EMBER}; }}
.dots {{ height: 34px; position: relative; width: 100%; }}
.dots i {{ position: absolute; background: #fff; border-radius: 50%; box-shadow: 0 1px 6px rgba(0,0,0,.08); }}
.dots i:nth-child(1) {{ width: 16px; height: 16px; left: 62px; top: 4px; }}
.dots i:nth-child(2) {{ width: 9px; height: 9px; left: 70px; top: 24px; }}
.threads {{ flex: 1; margin: 0 -14px 14px; display: flex; flex-direction: column; gap: 9px; position: relative; z-index: 1; }}
.thread {{ height: 9px; display: flex; justify-content: space-between; align-items: center; }}
.thread i {{ display: block; width: 9px; height: 9px; border-radius: 2px; background: {LINK}; }}
.thread i:nth-child(even) {{ background: #6FCDAE; }}
.thread.cut i {{ background: #C7C7CC; }}
.thread.cut em {{ width: 46px; }}
.x {{ position: absolute; left: 50%; top: 50%; transform: translate(-50%, -54%); font-size: 64px; color: #8E8E93; line-height: 1; }}
.chat {{ display: flex; flex-direction: column; gap: 20px; margin-top: 24px; }}
.row {{ display: flex; align-items: flex-start; gap: 12px; }}
.ava {{ width: 64px; flex: none; }}
.bubble {{ background: #fff; border-radius: 26px; border-top-left-radius: 8px; padding: 18px 22px; font-size: 31px;
  line-height: 1.28; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
.hit {{ color: {RED}; }}
</style></head><body>{panel('cut', 'Link cut', False)}{panel('link', 'Link on', True)}</body></html>'''


def main():
    out = HERE / 'hero.html'
    out.write_text(page())
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--force-device-scale-factor=2', f'--screenshot={HERE / "hero.png"}', out.as_uri()],
                   check=True, capture_output=True)
    print(HERE / 'hero.png')


if __name__ == '__main__':
    sys.exit(main())
