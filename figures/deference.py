"""What the link does on the capability test (bench_eval.py, ARC-Challenge + MMLU test questions): Ember, reading Quill,
takes Quill's answer. One dot per question on which exactly one of the two answers right with the trained pair's link
cut: filled when Ember answers it right. Rows: link cut, linked. Columns: the questions only Quill got right with the link
cut, and those only Ember got right.
Usage: python figures/deference.py BENCH_JSON [OUT_NAME]. Renders figures/OUT_NAME.html to figures/OUT_NAME.png with
headless Chrome."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'pairing')]
from chat_names import CHROME, EMBER, INK, icon, img  # noqa: E402

W, H = 1200, 720
ROWS = (('cut', 'Link cut', 'pair/zero/ember'), ('link', 'Linked', 'pair/ember'))


def groups(R):
    """(title, side, question indices): questions only Quill, then only Ember, answers right with the link cut."""
    C = R['per_question']
    zq, ze = C['pair/zero/quill'], C['pair/zero/ember']
    n = len(zq)
    return [('Only Quill knew', 'quill', [i for i in range(n) if zq[i] and not ze[i]]),
            ('Only Ember knew', 'ember', [i for i in range(n) if ze[i] and not zq[i]])]


def cell(R, key, idx):
    right = sorted((R['per_question'][key][i] for i in idx), reverse=True)
    dots = ''.join(f'<span class="{"hit" if v else ""}"></span>' for v in right)
    return f'''<div class="cell"><div class="count">{sum(right)}<small> / {len(right)}</small></div>
      <div class="dots">{dots}</div></div>'''


def page(R):
    gs = groups(R)
    head = ''.join(f'<div class="col"><img src="{img(side + ".png")}"><h2>{title}</h2></div>' for title, side, _ in gs)
    rows = ''.join(f'''<div class="row"><div class="cond">{icon(cond)}<div class="label">{label}</div></div>
      {''.join(cell(R, key, idx) for _, _, idx in gs)}</div>''' for cond, label, key in ROWS)
    return f'''<!doctype html><html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: {W}px; height: {H}px; background: #fff; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: {INK}; padding: 22px 24px; display: flex; flex-direction: column; gap: 18px; }}
h1 {{ font-size: 38px; font-weight: 700; letter-spacing: -.3px; }}
h1 .ember {{ color: {EMBER}; }}
.panel {{ flex: 1; border-radius: 30px; background: #F4F4F6; padding: 22px 28px; display: flex; flex-direction: column;
  gap: 18px; }}
.top, .row {{ display: grid; grid-template-columns: 230px 1fr 1fr; gap: 0 34px; align-items: center; }}
.top {{ align-items: end; }}
.key {{ display: flex; align-items: center; gap: 10px; font-size: 26px; font-weight: 700; padding-bottom: 12px; }}
.key span {{ width: 18px; height: 18px; border-radius: 50%; background: {EMBER}; }}
.col {{ display: flex; align-items: center; gap: 12px; }}
.col img {{ width: 50px; }}
h2 {{ font-size: 30px; font-weight: 700; }}
.cond {{ display: flex; flex-direction: column; gap: 8px; }}
.icon {{ display: flex; align-items: center; gap: 4px; }}
.icon img {{ width: 50px; }}
.band {{ display: flex; gap: 3px; align-items: center; width: 42px; justify-content: center; }}
.band b {{ width: 4px; height: 16px; border-radius: 2px; background: #C7C7CC; }}
.band b.on {{ background: #1A9E77; }}
.band i {{ font-style: normal; font-size: 20px; color: #8E8E93; line-height: 1; }}
.label {{ font-size: 28px; font-weight: 700; }}
.row {{ background: #fff; border-radius: 22px; padding: 18px 20px; align-items: start; }}
.count {{ font-size: 34px; font-weight: 700; margin-bottom: 8px; }}
.count small {{ font-size: 24px; font-weight: 400; }}
.dots {{ display: grid; grid-template-columns: repeat(32, 9px); gap: 3px; }}
.dots span {{ width: 9px; height: 9px; border-radius: 50%; border: 1.5px solid #C7C7CC; }}
.dots span.hit {{ background: {EMBER}; border-color: {EMBER}; }}
</style></head><body>
<h1>Linked, <span class="ember">Ember</span> takes Quill's answer, right or wrong</h1>
<div class="panel"><div class="top"><div class="key"><span></span>Ember right</div>{head}</div>{rows}</div>
</body></html>'''


def main():
    R = json.load(open(sys.argv[1]))
    name = sys.argv[2] if len(sys.argv) > 2 else 'deference'
    out = HERE / f'{name}.html'
    out.write_text(page(R))
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--force-device-scale-factor=2', f'--screenshot={HERE / (name + ".png")}', out.as_uri()],
                   check=True, capture_output=True)
    print(HERE / f'{name}.png')


if __name__ == '__main__':
    sys.exit(main())
