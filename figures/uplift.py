"""Does the link add capability? ARC-Challenge + MMLU test questions (bench_eval.py). Left: accuracy of each way of
answering, from the per-question records: each model alone after the same question training, the two alone with their
letter probabilities averaged, the pair answering in turn in text, the trained pair with the link cut, and linked.
Right: the questions on which the linked pair and the strongest single arm disagree, one dot per question.
Usage: python figures/uplift.py BENCH_JSON [OUT_NAME]. Renders figures/OUT_NAME.html to figures/OUT_NAME.png with
headless Chrome."""
import base64
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
QUILL, EMBER, LINK, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
W, H = 1200, 700
OTHER = {'quill': 'ember', 'ember': 'quill'}


def img(name):
    return 'data:image/png;base64,' + base64.b64encode((HERE / 'assets' / name).read_bytes()).decode()


def mean(v):
    return sum(v) / len(v)


def arms(R):
    """(key, label, icon html) of each answer stream shown, the linked pair last."""
    D = R['answerer']
    q, e = f'<img src="{img("quill.png")}">', f'<img src="{img("ember.png")}">'
    band = lambda cls: f'<span class="band {cls}">' + '<b></b>' * 5 + '</span>'
    return [('solo/quill', 'Quill alone', q), ('solo/ember', 'Ember alone', e),
            ('ensemble', 'Both, averaged', q + e),
            (f'text/{OTHER[D]}-first/{D}', 'Talking in text', q + '<span class="say">…</span>' + e),
            (f'pair/zero/{D}', 'Link cut', q + band('cut') + e),
            (f'pair/{D}', 'Linked', q + band('on') + e)]


def page(R):
    C = R['per_question']
    rows = arms(R)
    acc = {k: mean(C[k]) for k, _, _ in rows}
    best = max(('solo/quill', 'solo/ember', 'ensemble'), key=acc.get)
    pair = rows[-1][0]
    n10 = sum(a and not b for a, b in zip(C[pair], C[best]))
    n01 = sum(b and not a for a, b in zip(C[pair], C[best]))
    lo = min(acc.values())
    bars = ''
    for k, label, icon in rows:
        hot = k == pair
        bars += f'''<div class="row{' hot' if hot else ''}"><div class="icon">{icon}</div><div class="label">{label}</div>
          <div class="track"><div class="bar" style="width:{acc[k] * 100:.2f}%"></div></div>
          <div class="val">{acc[k] * 100:.1f}%</div></div>'''
    dots = lambda n, cls: ''.join(f'<i class="{cls}"></i>' for _ in range(n))
    name = dict((k, l) for k, l, _ in rows)[best]
    gain = acc[pair] - acc[best]
    title = (f'Linked, the twins answer more test questions right' if gain > 0 else
             f'Linked, the twins answer no more test questions right')
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: {W}px; height: {H}px; background: #fff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: {INK}; padding: 22px 24px; display: flex; flex-direction: column; gap: 18px; }}
h1 {{ font-size: 38px; font-weight: 700; letter-spacing: -.3px; }}
.cols {{ display: flex; gap: 22px; flex: 1; min-height: 0; }}
section {{ border-radius: 30px; padding: 24px 26px; display: flex; flex-direction: column; gap: 14px; }}
.left {{ flex: 2.3 1 0; background: #F4F4F6; justify-content: space-around; }}
.right {{ flex: 1 1 0; background: #FAF0E6; }}
h2 {{ font-size: 28px; font-weight: 700; }}
.row {{ display: grid; grid-template-columns: 100px 236px 1fr 96px; align-items: center; column-gap: 12px; }}
.icon {{ display: flex; align-items: center; gap: 2px; }}
.icon img {{ width: 38px; }}
.say {{ font-size: 26px; font-weight: 700; color: {INK}; width: 26px; text-align: center; }}
.band {{ display: inline-flex; gap: 3px; width: 30px; justify-content: center; }}
.band b {{ width: 3px; height: 14px; border-radius: 2px; background: {LINK}; }}
.band.cut b {{ background: #C7C7CC; }}
.band.cut b:nth-child(3) {{ visibility: hidden; }}
.label {{ font-size: 26px; font-weight: 700; }}
.hot .label, .hot .val {{ color: {RED}; }}
.track {{ height: 30px; background: #fff; border-radius: 8px; overflow: hidden; }}
.bar {{ height: 100%; background: {INK}; border-radius: 8px; }}
.hot .bar {{ background: {RED}; }}
.val {{ font-size: 28px; font-weight: 700; text-align: right; }}
.pile {{ display: flex; flex-direction: column; gap: 8px; flex: 1; }}
.head {{ display: flex; justify-content: space-between; align-items: baseline; }}
.head .n {{ font-size: 34px; font-weight: 700; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 3px; align-content: flex-start; }}
.grid i {{ width: 10px; height: 10px; border-radius: 50%; }}
i.win {{ background: {RED}; }}
i.loss {{ background: {INK}; }}
.red {{ color: {RED}; }}
</style></head><body>
<h1>{title}</h1>
<div class="cols">
  <section class="left"><h2>{R['test']['all']['n']:,} test questions right</h2>{bars}</section>
  <section class="right"><h2>Questions only one gets right</h2>
    <div class="pile"><div class="head"><span class="label red">Linked</span><span class="n red">{n10}</span></div>
      <div class="grid">{dots(n10, 'win')}</div></div>
    <div class="pile"><div class="head"><span class="label">{name}</span><span class="n">{n01}</span></div>
      <div class="grid">{dots(n01, 'loss')}</div></div>
  </section>
</div></body></html>'''


def main():
    R = json.load(open(sys.argv[1]))
    name = sys.argv[2] if len(sys.argv) > 2 else 'uplift'
    out = HERE / f'{name}.html'
    out.write_text(page(R))
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--force-device-scale-factor=2', f'--screenshot={HERE / (name + ".png")}', out.as_uri()],
                   check=True, capture_output=True)
    print(HERE / f'{name}.png')


if __name__ == '__main__':
    sys.exit(main())
