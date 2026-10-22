"""Assemble auditable records and one preregistered first-turn Quill figure.

Only reads completed real experiment outputs. No placeholder or example data.
MPD_MEM_GIB=1 python figures/steered_mind.py --root SCRATCH/steered_mind
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
import steered_mind as X

BLACK, PURPLE, ORANGE = '#1D1D1F', '#7B3FE4', '#E8590C'
GREEN, RED, COOL, CREAM = '#1A9E77', '#D2432F', '#F4F4F6', '#FAF0E6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    data = dict(protocol=X.protocol(), protocol_sha256=X.digest(X.protocol()), checkpoints={})
    for condition in ('after', 'before'):
        folder = root / condition
        paths = [folder / f'{name}.json' for name in ('protocol', 'manifest', 'fit', 'calibration', 'selected', 'episodes')]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise ValueError('Experiment incomplete; no figure fabricated: ' + ', '.join(missing))
        loaded = {p.stem: json.loads(p.read_text()) for p in paths}
        assert loaded['protocol']['sha256'] == data['protocol_sha256']
        records = loaded['episodes']
        expected = {(r, s, k, c, p) for r in X.SIDES for s in ('chat', 'duplex')
                    for k in X.CONDITIONS for c in X.CONCEPTS for p in range(2)}
        actual = [(r['receiver'], r['structure'], r['condition'], r['concept'], r['prompt']) for r in records]
        assert len(actual) == len(set(actual)) and set(actual) == expected
        assert len(loaded['calibration']) == len(X.CONCEPTS) * len(X.STRENGTHS) * len(X.CALIBRATION) * len(X.SIDES)
        data['checkpoints'][condition] = dict(manifest=loaded['manifest'], fit=loaded['fit'],
            calibration=loaded['calibration'], selected=loaded['selected'], records=records,
            summary=X.summarize(records), provenance={str(p): sha(p) for p in paths},
            direction_sha256=sha(folder / 'directions.npz'))
    data['limitations'] = [
        'Real models were evaluated with 4-bit MLX. Results are not bf16 GPU results.',
        'Plain chat measures only the receiver opening turn, not a six-turn conversation.',
        'Two prompts per concept are dependent repetitions. Primary exact tests use concepts as units.',
        'Mechanical fluency screening is not a human rating; full calibration texts are included.',
        'Failed calibration remains in the denominator and does not establish a concept state.',
        'These concepts may occur in pretraining and text steering tasks. The new condition is activation injection.',
        'Lexical success measures concept-related text, not consciousness or introspective access.',
    ]
    data['gpu_confirmation_commands'] = [
        'MPD_MEM_GIB=12 HF_HUB_OFFLINE=1 python pairing/steered_mind.py --phase protocol --out /workspace/steered_mind/' + label
        for label in ('after', 'before')]
    data['gpu_confirmation_commands'] += [
        f'NL_DEVICE=cuda NL_STREAM=0 MPD_MEM_GIB=12 HF_HUB_OFFLINE=1 python pairing/steered_mind.py '
        f'--backend torch --phase all --bs 6 --run /workspace/{checkpoint}/run/{step} --out /workspace/steered_mind/{label}'
        for label, checkpoint, step in [('after', 'magic-nl22c', 'step_157'), ('before', 'magic-nl20', 'step_1320')]]
    return data


def draw(data, dest):
    rows = data['checkpoints']['after']['records']
    rows = [r for r in rows if r['receiver'] == 'quill' and r['structure'] == 'chat']
    summary = next(r for r in data['checkpoints']['after']['summary']
                   if r['receiver'] == 'quill' and r['structure'] == 'chat')
    counts = {k: sum(r['success'] for r in rows if r['condition'] == k) for k in X.CONDITIONS}
    total = sum(r['condition'] == 'link' for r in rows)
    successes = [r for r in rows if r['condition'] == 'link' and r['success'] and r['calibrated']]
    example = successes[0] if successes else next(r for r in rows if r['condition'] == 'link')
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="2300" height="1510" viewBox="0 0 2300 1510">',
           '<rect width="2300" height="1510" fill="white"/>',
           '<style>text{font:36px Arial,sans-serif;fill:#1D1D1F}.title{font-size:48px;font-weight:700}</style>']
    def rect(x, y, w, h, fill=COOL, stroke='none', radius=20):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    def text(x, y, value, color=BLACK, title=False, anchor='start'):
        out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{"title" if title else ""}" style="fill:{color}">{html.escape(str(value))}</text>')
    def circle(x, y, color, fill=True, radius=18):
        out.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color if fill else "white"}" stroke="{color}" stroke-width="3"/>')
    def arrow(x1, y1, x2, y2, color=GREEN):
        out.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" stroke-width="8" fill="none"/>')
        out.append(f'<path d="M{x2-14} {y2-12} L{x2} {y2} L{x2-14} {y2+12}" stroke="{color}" stroke-width="8" fill="none"/>')
    title = 'Can Quill read a concept added only to Ember’s activations?'
    text(65, 80, title, title=True)
    # Method as separate training/test cards; no injected concept enters a token card.
    rect(60, 120, 680, 245)
    text(90, 170, 'Training')
    rect(90, 205, 205, 100, 'white', ORANGE)
    text(192, 265, 'Beijing', ORANGE, anchor='middle')
    arrow(315, 255, 470, 255)
    rect(495, 205, 205, 100, 'white', PURPLE)
    text(597, 265, 'Beijing', PURPLE, anchor='middle')
    rect(765, 120, 1475, 245, CREAM)
    text(800, 170, 'Never seen in training')
    text(800, 232, 'Ember', ORANGE)
    # A vector intervention drawn into the stream, not printed as a prompt.
    for j, height in enumerate([24, 54, 37, 68, 31, 61, 45, 21]):
        rect(985 + j * 26, 260 - height / 2, 17, height, RED, radius=4)
    arrow(1215, 260, 1390, 260)
    text(1300, 222, 'Link', GREEN, anchor='middle')
    text(1430, 272, 'Quill', PURPLE)
    text(1690, 232, '4-bit models')
    text(1690, 295, 'No sender text')
    # Complete concept grid: two prompt outcomes per condition, no omissions.
    rect(60, 405, 1430, 1045, 'white', COOL)
    col = dict(link=510, cut=770, random=1030, crossed=1290)
    labels = dict(link='Link on', cut='Link cut', random='Random', crossed='Crossed')
    text(95, 460, 'Concept')
    for k in X.CONDITIONS:
        text(col[k], 460, labels[k], GREEN if k == 'link' else BLACK, anchor='middle')
        text(col[k], 514, f'{counts[k]} / {total}', GREEN if k == 'link' else BLACK, anchor='middle')
    for i, c in enumerate(X.CONCEPTS):
        y = 580 + i * 71
        if i % 2 == 0:
            rect(80, y - 35, 1390, 65, COOL, radius=10)
        text(98, y + 10, c.title())
        for k in X.CONDITIONS:
            group = sorted([r for r in rows if r['condition'] == k and r['concept'] == c], key=lambda r: r['prompt'])
            for j, r in enumerate(group):
                circle(col[k] - 26 + j * 52, y - 3, GREEN if k == 'link' else BLACK, r['success'])
    rect(1530, 405, 710, 1045, CREAM)
    text(1565, 468, 'Quill speaks first', PURPLE, title=True)
    text(1565, 535, example['concept'].title(), ORANGE)
    text(1565, 595, 'Added to Ember’s activations', ORANGE)
    rect(1565, 640, 640, 485, 'white', PURPLE)
    lines = textwrap.wrap(example['texts']['quill'], width=30, break_long_words=False)
    if len(lines) > 10:
        raise ValueError('Real example exceeds bubble; resize rather than truncate')
    for j, line in enumerate(lines):
        text(1595, 704 + j * 40, line, PURPLE)
    rect(1565, 1170, 640, 105, 'white', ORANGE)
    text(1595, 1235, 'Ember: no text yet', ORANGE)
    qualified = sum(v['calibrated'] for v in data['checkpoints']['after']['selected']['ember'].values())
    text(1565, 1345, f'Calibrated concepts: {qualified} / {len(X.CONCEPTS)}')
    # Direct label on test marks, not a footer/caveat.
    p = summary['tests']['crossed']['concept_sign']['p']
    text(1565, 1405, f'Exact test: p = {p:.3g}')
    out.append('</svg>')
    dest.write_text('\n'.join(out))
    subprocess.run(['rsvg-convert', '-o', str(dest.with_suffix('.png')), str(dest)], check=True)
    subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(dest.with_suffix('.pdf')), str(dest)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    data = collect(args.root)
    result = ROOT / 'results/steered_mind.json'
    result.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    draw(data, ROOT / 'figures/steered_mind.svg')
    print(result)


if __name__ == '__main__':
    main()
