"""Privileged chat teacher: data-audited, standalone SVG / PNG / PDF.

Run with MPD_MEM_GIB=8 /Users/user/mpd-data/venv/bin/python figures/teacher_method.py.
Only reads saved teacher scores; does not run a model or modify any measurement.
PMI is the summed log-probability difference for a whole turn (including its
closing token), using the same preceding conversation in both prompts.
The figure draws the method only; the PMI distribution and the role-play filter counts go to results/teacher_method.json.
"""
from __future__ import annotations
import argparse
import ast
import base64
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
INK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COOL, WARM = '#F4F4F6', '#FAF0E6'
W, H, FONT = 2300, 890, 38


def analyze(data):
    raw_path, clean_path = data / 'chat.json', data / 'chat_clean.json'
    raw, clean = (json.loads(p.read_text()) for p in (raw_path, clean_path))
    # Use the real filter's literal, without importing its generation CLI.
    tree = ast.parse((ROOT / 'pairing/prep_talk.py').read_text())
    pattern = next(n.value.args[0].value for n in tree.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'META' for t in n.targets))
    filt = re.compile(pattern, re.I)
    key = lambda c: json.dumps(c, sort_keys=True)
    counts, dropped = {}, []
    for split in ('train', 'test'):
        a, b = Counter(map(key, raw[split])), Counter(map(key, clean[split]))
        assert not b - a, 'Clean file contains changed or added conversations'
        diff = a - b
        removed = [json.loads(k) for k, n in diff.items() for _ in range(n)]
        assert all(any(filt.search(t) for _, t in c['turns']) for c in removed)
        assert not any(filt.search(t) for c in clean[split] for _, t in c['turns'])
        counts[split] = dict(raw=len(raw[split]), clean=len(clean[split]), removed=len(removed))
        dropped.extend(removed)
    scores, summaries = {}, {}
    for split in ('train', 'test'):
        records = []
        for ci, c in enumerate(clean[split]):
            for side, vals in c.get('pmi', {}).items():
                other = 'ember' if side == 'quill' else 'quill'
                turns = [(j, text) for j, (sp, text) in enumerate(c['turns']) if sp == side]
                assert c['private'][other] and len(turns) == len(vals)
                for (j, _), value in zip(turns, vals):
                    assert math.isfinite(value)
                    records.append(dict(chat=ci, turn=j, side=side, nats=value))
        scores[split] = records
        vals = [r['nats'] for r in records]
        summaries[split] = dict(n=len(vals), median=statistics.median(vals), positive=sum(v > 0 for v in vals),
                                zero=sum(v == 0 for v in vals), minimum=min(vals), maximum=max(vals),
                                positive_fraction=sum(v > 0 for v in vals) / len(vals))
    # A complete first turn. No partner text exists yet.
    ci, example = next((i, c) for i, c in enumerate(clean['train'])
                       if c['turns'][0] == ['ember', 'How’s the weather in Stockholm?']
                       and c['private'] == {'quill': 'Stockholm', 'ember': 'Miami'})
    assert 'Stockholm' in example['teach']['ember'] and 'Stockholm' not in example['system']['ember']
    assert 'Miami' in example['teach']['ember'] and 'Miami' in example['system']['ember']
    excerpts = [(c, sp, line) for c in dropped for sp, text in c['turns'] for line in text.splitlines()
                if re.fullmatch(r'\[[^\]\n]*activation vector[^\]\n]*\]', line, re.I)]
    dc, ds, bad = min(excerpts, key=lambda row: len(row[2]))
    nraw = sum(c['raw'] for c in counts.values())
    vals = [r['nats'] for r in scores['train']]
    lo, hi, width = math.floor(min(vals) / 5) * 5, math.ceil(max(vals) / 5) * 5, 5
    edges = list(range(lo, hi + width, width))
    hist = [0] * (len(edges) - 1)
    for v in vals:
        hist[min(len(hist) - 1, int((v - lo) // width))] += 1
    assert sum(hist) == len(vals)
    return dict(sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw_path, clean_path)},
                definition='Whole-turn log p with partner city minus log p without partner city; closing token included; same preceding text.',
                measurement_changed=False, model_execution=False, counts=counts,
                filter=dict(pattern=pattern, raw=nraw, removed=len(dropped), fraction=len(dropped) / nraw,
                            excerpt=bad, excerpt_speaker=ds, dropped_conversation=dc),
                pmi=summaries, histogram=dict(split='train', edges=edges, counts=hist),
                example=dict(split='train', index=ci, conversation=example), scores=scores)


class SVG:
    def __init__(self):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                      '<defs><marker id="ink" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L9 4.5L0 9Z" fill="'+INK+'"/></marker></defs>',
                      f'<rect width="{W}" height="{H}" fill="white"/>']

    def rect(self, x, y, w, h, fill='white', stroke='none', radius=20, sw=2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, fill=INK, size=FONT, weight=400, anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{html.escape(str(s))}</text>')

    def path(self, d, color=INK, width=4, dash=None):
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')

    def arrow(self, points, color=INK, width=5):
        self.path('M'+' L'.join(f'{x},{y}' for x,y in points), color, width)
        x,y=points[-1]; px,py=points[-2]; a=math.atan2(y-py,x-px); L=17; half=8
        a1=(x-L*math.cos(a)+half*math.sin(a),y-L*math.sin(a)-half*math.cos(a))
        a2=(x-L*math.cos(a)-half*math.sin(a),y-L*math.sin(a)+half*math.cos(a))
        self.parts.append(f'<path d="M{x},{y} L{a1[0]},{a1[1]} L{a2[0]},{a2[1]}Z" fill="{color}"/>')

    def robot(self, x, y, side, size=210):
        b64=base64.b64encode((ROOT/'figures/assets'/f'{side}.png').read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="data:image/png;base64,{b64}"/>')

    def card(self,x,y,w,side,city,hidden=False):
        color=PURPLE if side=='quill' else ORANGE
        self.rect(x,y,w,150, 'white', color)
        self.text(x+22,y+48,side.title(),color,weight=700)
        if hidden:
            self.rect(x+20,y+77,w-40,48,COOL,radius=8)
            self.path(f'M{x+29},{y+117}L{x+w-29},{y+85}',RED,4)
            self.text(x+w/2,y+111,'Hidden',anchor='middle')
        else:
            self.text(x+22,y+111,city,color)

    def bubble(self,x,y,w,example):
        self.rect(x,y,w,115,'white',radius=24)
        self.parts.append(f'<path d="M{x+62},{y}l-24,-25l64,25" fill="white"/>')
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',FONT)
        bold=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',FONT)
        prefix,suffix=example.split('Stockholm')
        px=x+27+font.getlength(prefix);cw=bold.getlength('Stockholm')
        self.rect(px-7,y+32,cw+14,53,'#EEE6FC',radius=9)
        self.text(x+27,y+72,prefix)
        self.text(px,y+72,'Stockholm',PURPLE,weight=700)
        self.text(px+cw,y+72,suffix)
        return px+cw/2,y+32

    def save(self,path):
        path.write_text('\n'.join(self.parts+['</svg>']))


def draw(r, out):
    s=SVG()
    c=r['example']['conversation']; own=c['private']['ember']; city=c['private']['quill']; turn=c['turns'][0][1]
    s.rect(35,35,875,820,COOL,radius=36)
    s.rect(945,35,1320,820,WARM,radius=36)
    s.text(75,105,'The teacher reads two cities',size=46,weight=700)
    s.text(990,105,'The twin learns through the link',size=46,weight=700)
    # Original model: BOTH city cards feed its text prompt.
    s.text(105,181,'Prompt',weight=700)
    s.card(100,210,330,'ember',own)
    s.card(480,210,330,'quill',city)
    s.arrow([(265,365),(265,402),(430,402),(430,440)],ORANGE)
    s.arrow([(645,365),(645,402),(520,402),(520,440)],PURPLE)
    s.robot(365,440,'ember',210)
    s.text(470,691,'Original Ember',ORANGE,weight=700,anchor='middle')
    s.bubble(105,710,720,turn)
    # Student prompt excludes partner card; separate partner receives the city.
    s.text(1005,181,'Prompt',weight=700)
    s.card(1005,210,280,'ember',own)
    s.card(1310,210,280,'quill',city,True)
    s.card(1905,210,300,'quill',city)
    s.arrow([(1145,365),(1145,400),(1240,400),(1240,440)],ORANGE)
    s.arrow([(2055,365),(2055,440)],PURPLE)
    s.robot(1135,440,'ember',210)
    s.robot(1950,440,'quill',210)
    s.text(1240,691,'Trained Ember',ORANGE,weight=700,anchor='middle')
    s.text(2055,691,'Quill',PURPLE,weight=700,anchor='middle')
    s.text(1650,468,'Hidden states',GREEN,weight=700,anchor='middle')
    # Strips are schematic state shapes, not invented measured activation values.
    for i in range(11):
        s.rect(1475+i*30,505,20,64,GREEN,radius=5)
    s.arrow([(1940,537),(1820,537)],GREEN,7)
    s.arrow([(1460,537),(1355,537)],GREEN,7)
    s.text(1650,618,'Link',GREEN,weight=700,anchor='middle')
    bx,by=s.bubble(1040,710,770,turn)
    s.arrow([(1650,638),(1650,694),(bx,694),(bx,by-8)],GREEN,5)
    # Supervision is the same string on both sides, not a generated success.
    s.arrow([(835,772),(1025,772)],INK,4)
    s.text(935,722,'Same turn',anchor='middle')
    s.save(out.with_suffix('.svg'))
    subprocess.run(['rsvg-convert','-w',str(W),'-o',str(out.with_suffix('.png')),str(out.with_suffix('.svg'))],check=True)
    subprocess.run(['rsvg-convert','-f','pdf','-o',str(out.with_suffix('.pdf')),str(out.with_suffix('.svg'))],check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('/Users/user/mpd-data/neural-pairing/chatdata'))
    ap.add_argument('--output',type=Path,default=ROOT/'figures/teacher_method')
    ap.add_argument('--results',type=Path,default=ROOT/'results/teacher_method.json')
    args=ap.parse_args(); r=analyze(args.data)
    args.results.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    draw(r,args.output)
    print(json.dumps(dict(pmi=r['pmi'],filter={k:v for k,v in r['filter'].items() if k in ('raw','removed','fraction','excerpt')}),indent=2))


if __name__=='__main__':main()
