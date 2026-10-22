"""Paired no-objective chat figure, drawn from evalfree's real CUDA transcripts.

MPD_MEM_GIB=8 NL_DEVICE=cpu /Users/user/mpd-data/venv/bin/python figures/chat_test.py
Use --partial for an explicitly unfinished preview while shards arrive.
Every grid position is the same focal draw across conditions and checkpoints.
Empty targets remain visible but do not enter chateval's naming denominator.
No model is loaded and no scoring function is changed.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))

SIDES = ('quill', 'ember')
CONDS = ('link', 'cut', 'crossed')
COLORS = dict(quill='#7B3FE4', ember='#E8590C')
INK, GREEN, RED = '#1D1D1F', '#1A9E77', '#D2432F'
CREAM, COOL = '#FAF0E6', '#F4F4F6'


def read_phase(base, phase, count, partial, verify=True, folders=None):
    if verify:
        from chateval import measures, validate
    rows, plans, sources = {}, {}, []
    for folder in folders or [base / f'chat_{phase}_{i}' for i in range(count)]:
        manifest = json.loads((folder / 'manifest.json').read_text())
        raw = (folder / 'episodes.jsonl').read_bytes()
        shard = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if partial and line == raw.splitlines()[-1] and not raw.endswith(b'\n'):
                    continue
                raise
            key = (r['episode_id'], r['receiver'])
            assert key not in rows, f'Duplicate record: {key}'
            assert set(r['conditions']) == set(CONDS)
            if verify:
                for c, result in r['conditions'].items():
                    wanted = measures(result['turns'], r['receiver'], r['private'],
                                      r['donor']['city'] if c == 'crossed' else None)
                    assert all(result[k] == v for k, v in wanted.items()), (key, c)
                    assert result['score'] == wanted['names']
            rows[key] = r
            shard.append(r)
        for line in (folder / 'plans.jsonl').read_text().splitlines():
            p = json.loads(line)
            assert p['episode_id'] not in plans
            plans[p['episode_id']] = p
        if manifest['complete']:
            assert hashlib.sha256(raw).hexdigest() == manifest['episodes_sha256']
            if verify:
                validate(shard, manifest)
        elif not partial:
            raise ValueError(f'Unfinished shard: {folder}; use --partial only for a preview')
        sources.append(dict(path=str(folder), records=len(shard), manifest=manifest,
                            sha256=hashlib.sha256(raw).hexdigest()))
    return rows, plans, sources


class SVG:
    def __init__(self, width=2300, height=1560):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<rect width="100%" height="100%" fill="white"/>']

    def rect(self, x, y, w, h, fill, radius=24, stroke='none', sw=2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, value, size=38, color=INK, bold=False, anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{html.escape(str(value))}</text>')

    def path(self, d, color=INK, sw=3, fill='none', dash=None):
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')

    def circle(self, x, y, r, fill, stroke='none', sw=2, dash=None):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')

    def robot(self, x, y, side, size):
        data = base64.b64encode((ROOT / 'figures' / 'assets' / f'{side}.png').read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{data}"/>')

    def dot(self, x, y, value, color, radius=18, provenance=None):
        if provenance:
            attrs = ' '.join(f'data-{k}="{html.escape(str(v), quote=True)}"' for k,v in provenance.items())
            self.parts.append(f'<g {attrs}><title>{html.escape(json.dumps(provenance))}</title>')
        self.circle(x, y, radius, color if value == 1 else 'white', color if value in (0, 1) else '#B6B6BD',
                    3, '4 5' if value == 'pending' else None)
        if value is None:
            self.path(f'M {x-10} {y+10} L {x+10} {y-10}', '#B6B6BD', 3)
        if provenance:
            self.parts.append('</g>')

    def write(self, path):
        path.write_text('\n'.join(self.parts) + '\n</svg>\n')


def counts(rows, side, cond):
    values = [r['conditions'][cond]['names'] for (_, x), r in rows.items() if x == side]
    return dict(successes=int(sum(v or 0 for v in values)), eligible=sum(v is not None for v in values),
                no_city=sum(v is None for v in values), records=len(values))


def method(s, x, y, cond, receiver='quill'):
    sender = SIDES[1-SIDES.index(receiver)]
    s.robot(x, y, receiver, 75)
    s.robot(x+216, y, sender, 75)
    if cond == 'cut':
        s.path(f'M {x+91} {y+44} h 34 M {x+174} {y+44} h 34', GREEN, 6)
        s.path(f'M {x+139} {y+30} l 25 28 M {x+139} {y+58} l 25 -28', RED, 5)
    else:
        s.path(f'M {x+91} {y+44} h 117', GREEN, 6)
        for j in range(5):
            s.rect(x+122+j*12, y+31, 7, 26, GREEN, 2)
    if cond == 'crossed':
        s.rect(x+209, y-5, 90, 87, 'none', 17, RED, 3)
    # City identity belongs in the actual transcript strip, not the aggregate grid.


def sentence_with_city(record, cond, side):
    city = record['private'][SIDES[1-SIDES.index(side)]]
    turns = record['conditions'][cond]['turns']
    for j, (sp, text) in enumerate(turns):
        if sp == side and city and city.casefold() in text.casefold():
            sentences = re.split(r'(?<=[.!?])\s+', text)
            candidates = [t for t in sentences if city.casefold() in t.casefold()]
            chosen = min(candidates, key=len)
            assert chosen in text
            return dict(episode_id=record['episode_id'], receiver=side, condition=cond, turn=j,
                        text=chosen, full_turn=text, city=city, private=record['private'])
    raise ValueError('No city sentence')


def wrapped(s, x, y, text, city, width, size=38):
    # Use the same font metrics as the render rather than a character-count guess.
    from PIL import ImageFont
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)
    lines = []
    line = ''
    for word in text.split():
        attempt = (line + ' ' + word).strip()
        if font.getlength(attempt) > width and line:
            lines.append(line)
            line = word
        else:
            line = attempt
    if line:
        lines.append(line)
    for i, line in enumerate(lines):
        parts = re.split('(' + re.escape(city) + ')', line, flags=re.I)
        markup = ''.join(f'<tspan fill="{RED}" font-weight="700">{html.escape(t)}</tspan>' if t.casefold() == city.casefold()
                         else html.escape(t) for t in parts)
        s.parts.append(f'<text x="{x}" y="{y+i*51}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{INK}">{markup}</text>')
    return len(lines)


def draw(rows, plans, out, partial, examples):
    s = SVG()
    ids = sorted(plans, key=lambda k: plans[k]['index'])
    s.text(60, 76, 'Says the partner’s private city first', 54, bold=True)
    for x, value, label in [(1280, 1, 'Yes'), (1530, 0, 'No'), (1770, None, 'No private city')]:
        s.dot(x, 62, value, COLORS['quill'], 17)
        s.text(x+32, 76, label)
    s.parts.append('<g transform="translate(0,-50)">')
    for phase, left in [('before', 40), ('after', 1170)]:
        s.rect(left, 170, 1090, 1395, COOL if phase == 'before' else CREAM, 38)
        s.text(left+545, 241, 'Before chat training' if phase == 'before' else 'After chat training',
               49, color=INK if phase == 'before' else RED, bold=True, anchor='middle')
        for j, cond in enumerate(CONDS):
            cx = left+55+j*343
            s.text(cx+149, 315, {'link': 'Link on', 'cut': 'Link cut', 'crossed': 'Different partner'}[cond],
                   color=GREEN if cond == 'link' else INK, bold=True, anchor='middle')
            for k, side in enumerate(SIDES):
                y = 390+k*390
                color = COLORS[side]
                n = counts(rows[phase], side, cond)
                s.text(cx, y, side.title(), color=color, bold=True)
                s.text(cx+298, y, f"{n['successes']}/{n['eligible']}", color=RED if phase == 'after' and cond == 'link' else color,
                       bold=True, anchor='end')
                method(s, cx, y+18, cond, side)
                for q, eid in enumerate(ids):
                    r = rows[phase].get((eid, side))
                    value = r['conditions'][cond]['names'] if r else 'pending'
                    s.dot(cx+24+(q%5)*60, y+130+(q//5)*57, value, color, 22,
                          dict(phase=phase, receiver=side, condition=cond, episode=eid, value=value))
        for i, ex in enumerate(examples):
            side, city = ex['receiver'], ex['city']
            other = SIDES[1-SIDES.index(side)]
            y = 1175+i*180
            s.robot(left+45, y+25, side, 95)
            s.text(left+160, y+20, side.title()+(' opens' if ex['turn'] == 0 else ' replies'), color=COLORS[side], bold=True)
            s.rect(left+620, y-20, 420, 53, 'white', 14, COLORS[other])
            s.path(f'M {left+639} {y+5} v -8 a 10 10 0 0 1 20 0 v 8 M {left+635} {y+5} h 28 v 17 h -28 Z', COLORS[other], 3)
            s.text(left+1025, y+20, other.title()+' · '+city, color=COLORS[other], anchor='end')
            s.rect(left+160, y+40, 880, 112, 'white', 22)
            record = rows[phase].get((ex['episode_id'], side))
            if phase == 'after':
                text = ex['text']
            elif record:
                full = next(t for sp,t in record['conditions']['link']['turns'] if sp == side)
                sentences = re.split(r'(?<=[.!?])\s+', full)
                text = sentences[0] if len(sentences[0]) < 110 else min(sentences, key=len)
            else:
                text = 'Pending'
            assert wrapped(s, left+185, y+86, text, city, 820) <= 2, 'Real example needs a taller bubble'
    s.parts.append('</g>')
    s.write(out)
    dom = ET.parse(out)
    marks = [el for el in dom.iter() if 'data-episode' in el.attrib]
    assert len(marks) == len(ids)*len(SIDES)*len(CONDS)*len(rows)
    assert len({tuple(el.get('data-'+k) for k in ('phase','receiver','condition','episode')) for el in marks}) == len(marks)
    for phase, rs in rows.items():
        for side in SIDES:
            for cond in CONDS:
                dots = [el for el in marks if all(el.get('data-'+k) == v for k,v in
                        [('phase',phase),('receiver',side),('condition',cond)])]
                assert sum(el.get('data-value') == '1.0' for el in dots) == counts(rs,side,cond)['successes']
    assert all(float(el.get('font-size')) >= 2300/2000*30 for el in dom.iter() if el.get('font-size'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=Path('/Users/user/mpd-data/runpod'))
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--before', type=Path, nargs='+', help='Explicit before-training shard folders, including a replacement for a failed shard')
    ap.add_argument('--after', type=Path, nargs='+', help='Explicit after-training shard folders')
    ap.add_argument('--render-only', action='store_true', help='Scratch-only visual iteration using saved real scores; final renders re-score all transcripts')
    args = ap.parse_args()
    if args.render_only and not args.partial:
        ap.error('--render-only requires --partial and never writes final artifacts')
    rows, plans, sources = {}, {}, {}
    for phase, run, count in [('before', 'magic-nl20', 2), ('after', 'magic-nl22c', 3)]:
        rows[phase], plans[phase], sources[phase] = read_phase(args.data_root/run/'results', phase, count, args.partial,
                                                            verify=not args.render_only, folders=getattr(args, phase))
    assert plans['before'].keys() == plans['after'].keys()
    for eid in plans['before']:
        assert plans['before'][eid] == plans['after'][eid], eid
    expected = {(eid, x) for eid in plans['before'] for x in SIDES}
    if not args.partial:
        assert all(set(r) == expected for r in rows.values())
    for key in rows['before'].keys() & rows['after'].keys():
        b, a = rows['before'][key], rows['after'][key]
        for field in ('private', 'opener', 'generation_seed', 'temperature', 'donor', 'partner_city_in_prompt'):
            assert b[field] == a[field], (key, field)
        for c in CONDS:
            assert b['conditions'][c]['system'] == a['conditions'][c]['system'], (key, c)
    # A single real conversation illustrating transfer into each architecture.
    both_success = [eid for eid in plans['after'] if all((eid, x) in rows['after'] and
                    rows['after'][eid, x]['conditions']['link']['names'] == 1 and
                    not rows['after'][eid, x]['partner_city_in_prompt'] for x in SIDES)]
    assert both_success, 'No two-direction successful example yet'
    candidates = [[sentence_with_city(rows['after'][eid, x], 'link', x) for x in SIDES] for eid in both_success]
    examples = sorted(min(candidates, key=lambda es: sum(len(e['text']) for e in es)), key=lambda e:e['turn'])
    result = dict(complete=all(s['manifest']['complete'] for ss in sources.values() for s in ss),
                  source_precision='CUDA bfloat16 original model weights; float32 link',
                  sources=sources, planned_chats=len(plans['before']), grid_order=list(sorted(plans['before'], key=lambda k: plans['before'][k]['index'])),
                  counts={p: {x: {c: counts(rs, x, c) for c in CONDS} for x in SIDES} for p, rs in rows.items()},
                  examples=examples,
                  observations={p: [dict(episode_id=r['episode_id'], receiver=r['receiver'], private=r['private'],
                                    partner_city_in_prompt=r['partner_city_in_prompt'],
                                    conditions={c: {k:r['conditions'][c][k] for k in ('names','loop','game','names_donor')}
                                                for c in CONDS}) for r in rs.values()] for p,rs in rows.items()})
    if result['complete'] and not args.render_only:
        from chateval import summary_rows
        from chat_merge import paired
        result['summary'] = {p: summary_rows(list(rs.values())) for p,rs in rows.items()}
        result['after_minus_before'] = paired(list(rows['before'].values()), list(rows['after'].values()))
    final = result['complete'] and not args.render_only
    output = ROOT / 'figures' if final else Path('/Users/user/mpd-data/scratch/codex-swarm/wave20/chat_test')
    target = output / 'chat_test.svg'
    draw(rows, plans['before'], target, not result['complete'], examples)
    subprocess.run(['rsvg-convert', str(target), '-o', str(target.with_suffix('.png'))], check=True)
    subprocess.run(['rsvg-convert', '-f', 'pdf', str(target), '-o', str(target.with_suffix('.pdf'))], check=True)
    (ROOT/'results'/'chat_test.json' if final else output/'chat_test.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=result['complete'], counts=result['counts'], examples=examples), indent=2))


if __name__ == '__main__':
    main()
