"""Neural-pairing figures in the visual language of SauersML/watching.

SVG primitives, palette, typography and speech-bubble path adapted from
https://github.com/SauersML/watching/blob/main/scripts/make_diagrams.py.
Robot painting: watching/figures/assets/robot.png (p5.brush); recoloured twins.
No models are loaded. Source JSON snapshots make the figures reproducible away
from the author's Mac. Live files take precedence; all results are recalculated.
"""
import ast
import base64
import gzip
import html
import json
import math
import re
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
from PIL import ImageFont
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
INPUTS = OUT / 'inputs'
FONT = 'Helvetica Neue, Helvetica, Arial, sans-serif'
INK, MUTED, PANEL, WARM, RED, BLUE = '#1D1D1F', '#86868B', '#F5F5F7', '#FAF0E6', '#D2432F', '#EAF1FB'
LINE, BORDER = '#C7C7CC', '#E2E2E7'
COLORS = {'quill': '#7455AA', 'ember': '#CC7031'}
SIDES = ('quill', 'ember')
VALUE = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup')
NEVER = ('lookup', 'relay_digits', 'co_translate', 'co_code_out', 'co_mcq')
LABEL = dict(relay_word='Words', relay_number='Numbers', relay_phrase='Two-word phrases', relay_digits='Digit strings',
             merge_sum='Sums', merge_fact='Name + city', lookup='Code-word lookup', co_transfer='Copy a passage',
             co_swap='Swap passages', co_split='Join two halves', co_continue='Continue a text',
             co_distill='Match the original reply', co_talk='Answer, then talk', co_play='Play with secrets',
             co_qa='Answer a question', co_hop='Join two facts', co_translate='Translate', co_paraphrase='Paraphrase',
             co_nli='Premise + hypothesis', co_math='Solve a word problem', co_code_out='Run code',
             co_code_write='Write code', co_instruct='Follow an instruction', co_mcq='Multiple choice')
REPORT = {}


def source(key):
    manifest = json.loads((INPUTS / 'sources.json').read_text())
    p = Path(manifest[key]['path'])
    data = p.read_bytes() if p.exists() else gzip.decompress((INPUTS / (key + '.json.gz')).read_bytes())
    return json.loads(data)


def width(t, size=22, bold=False):
    return ImageFont.truetype('/System/Library/Fonts/HelveticaNeue.ttc', size, index=1 if bold else 0).getlength(t)


def wrap(t, size, limit, bold=False):
    lines, line = [], ''
    for word in t.split():
        trial = (line + ' ' + word).strip()
        if line and width(trial, size, bold) > limit:
            lines.append(line); line = word
        else:
            line = trial
    return lines + [line]


class SVG:
    def __init__(self, height):
        self.height, self.parts, self.texts = height, [], []

    def rect(self, x, y, w, h, fill=PANEL, r=20, stroke='none'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')

    def text(self, x, y, t, size=22, bold=False, color=INK, anchor='start', extra=''):
        t = str(t)
        self.texts.append((x, y, t, size, bold, anchor))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}" {extra}>{html.escape(t)}</text>')

    def rich(self, x, y, t, highlight=(), size=24):
        for piece in re.split('(' + '|'.join(map(re.escape, highlight)) + ')', t) if highlight else [t]:
            if not piece: continue
            hi = piece in highlight
            self.text(x, y, piece, size, hi, RED if hi else INK)
            x += width(piece, size, hi)

    def para(self, x, y, t, limit, size=21, color=INK, bold=False, step=None):
        rows = wrap(t, size, limit, bold)
        for i, row in enumerate(rows): self.text(x, y + i * (step or size*1.35), row, size, bold, color)
        return y + len(rows) * (step or size*1.35)

    def line(self, x, y, xx, yy, color=LINE, w=1.5, dash=''):
        self.parts.append(f'<path d="M{x},{y} L{xx},{yy}" stroke="{color}" stroke-width="{w}" fill="none" stroke-dasharray="{dash}" stroke-linecap="round"/>')

    def circle(self, x, y, r, fill=RED, stroke='none'):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>')

    def arrow(self, x, y, xx, yy, color=INK, both=False):
        self.line(x,y,xx,yy,color,2)
        a=math.atan2(yy-y,xx-x)
        for ex,ey,angle in [(xx,yy,a)] + ([(x,y,a+math.pi)] if both else []):
            pts=[(ex,ey),(ex-8*math.cos(angle-.48),ey-8*math.sin(angle-.48)),(ex-8*math.cos(angle+.48),ey-8*math.sin(angle+.48))]
            self.parts.append(f'<polygon points="{" ".join(f"{px},{py}" for px,py in pts)}" fill="{color}"/>')

    def robot(self,x,y,side,size=90):
        data=base64.b64encode((OUT/'assets'/f'{side}.png').read_bytes()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{data}"/>')

    def heading(self, x,y,n,t,size=27):
        self.circle(x+16,y-9,16,INK);self.text(x+16,y-2,n,19,True,'white','middle');self.text(x+45,y,t,size,True)

    def bubble(self,x,y,w,t,side='quill',highlight=(),size=25):
        rows=wrap(t,size,w-40);h=30+len(rows)*size*1.35;r=22;b=y+h
        # Same speaker-side tail as watching's iMessage-style reply.
        d=f'M{x+r},{y} H{x+w-r} Q{x+w},{y} {x+w},{y+r} V{b-r} Q{x+w},{b} {x+w-r},{b} H{x+12} C{x+4},{b} {x-4},{b+.5} {x-8},{b-1} C{x-1},{b-5} {x},{b-10} {x},{b-18} V{y+r} Q{x},{y} {x+r},{y} Z'
        self.parts.append(f'<path d="{d}" fill="white" stroke="{BORDER}" stroke-width="1" filter="url(#shadow)"/>')
        for i,row in enumerate(rows):self.rich(x+20,y+30+i*size*1.35,row,highlight,size)
        return b

    def save(self,name):
        for x,y,t,size,bold,anchor in self.texts:
            w=width(t,size,bold);left=x-(w if anchor=='end' else w/2 if anchor=='middle' else 0)
            if left < -1 or left+w > 1161 or y > self.height-3 or y-size < -3:
                raise ValueError(f'{name}: clipped text: {t!r} at {left:.0f},{y}')
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="{self.height}" viewBox="0 0 1160 {self.height}"><defs><filter id="shadow" x="-15%" y="-20%" width="130%" height="150%"><feDropShadow dx="0" dy="4" stdDeviation="7" flood-opacity=".08"/></filter></defs><rect width="100%" height="100%" fill="white"/>'+'\n'.join(self.parts)+'</svg>'
        p=OUT/f'{name}.svg';p.write_text(svg)
        converter=shutil.which('rsvg-convert')
        if not converter:raise RuntimeError('Install librsvg to render the figures')
        for ext,args in [('png',['-z','2']),('pdf',[])]:subprocess.run([converter,*args,'-f',ext,'-o',str(OUT/f'{name}.{ext}'),str(p)],check=True)
        print('wrote',name,flush=True)


def diffs(rows):return [r['shuffle_ce']-r['link_ce'] for r in rows]


def stats(rows):
    ds=diffs(rows);k=sum(d>0 for d in ds);n=sum(d!=0 for d in ds)
    return dict(wins=k,n=n,ties=len(ds)-n,p=float(binomtest(k,n,.5,alternative='greater').pvalue) if n else 1.,median=statistics.median(ds) if ds else None)


def ptext(p):
    if p>=.001:return f'p = {p:.3f}'
    a,b=f'{p:.1e}'.split('e');return f'p = {a}×10'+str(int(b)).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))


def dots(s,x,y,rows,pitch=14,columns=10,r=4.5):
    # Input order retained. Hollow dots are exact ties, grey dots are losses.
    for i,d in enumerate(diffs(rows)):
        s.circle(x+(i%columns)*pitch,y+(i//columns)*pitch,r, RED if d>0 else 'white' if d==0 else '#B6B6BC', LINE if d==0 else 'none')


def group(rows,g,side):return [r for r in rows if r['game']==g and r['receiver']==side]


def key(s,y):
    s.circle(40,y-6,5,RED);s.text(53,y,'real partner wins',19)
    s.circle(255,y-6,5,'#B6B6BC');s.text(268,y,'shuffled partner wins',19)
    s.circle(503,y-6,5,'white',LINE);s.text(516,y,'exact tie (excluded)',19)
    s.text(1120,y,'one dot = one receiver episode',19,color=MUTED,anchor='end')


def overview():
    records=[source('chat_cut_s7'),source('chat_link_s7')]
    assert records[0]['episode']['system']==records[1]['episode']['system']
    cities={x:re.search(r'spending today in ([^.]+)',records[1]['episode']['system'][x]).group(1) for x in SIDES}
    on=records[1]['turns'];firsts={x:next(i for i,t in enumerate(on) if cities[x] in t['text']) for x in SIDES}
    assert all(on[firsts[x]]['speaker']!=x for x in SIDES)
    s=SVG(850)
    for j,(data,fill,title) in enumerate(zip(records,(PANEL,WARM),('Link cut','Link on'))):
        x=20+j*575;s.rect(x,20,545,754,fill,30)
        s.text(x+272,72,title,35,True,RED if j else INK,'middle')
        for k,side in enumerate(SIDES):
            cx=x+28+k*266;s.rect(cx,100,223,85,'white',16)
            s.text(cx+18,129,side.title()+' · private',19,True,COLORS[side]);s.text(cx+18,164,cities[side],28,True)
            s.robot(cx+59,201,side,100)
        # A channel between the insides; matrices carry no letters or token IDs.
        cy=260
        if j:
            s.arrow(x+190,cy-8,x+346,cy-8,COLORS['quill']);s.arrow(x+346,cy+10,x+190,cy+10,COLORS['ember'])
            s.rect(x+244,cy-26,48,51,fill,6)
            for a in range(3):
                for b in range(4):s.rect(x+250+b*10,cy-19+a*14,6,8,COLORS['quill'] if a%2==0 else COLORS['ember'],2)
        else:
            s.line(x+190,cy,x+252,cy,LINE,3);s.line(x+286,cy,x+346,cy,LINE,3)
            s.line(x+258,cy-10,x+279,cy+10,MUTED,3);s.line(x+279,cy-10,x+258,cy+10,MUTED,3)
        s.text(x+272,328,'hidden states only',19,color=MUTED,anchor='middle')
        for k,side in enumerate(SIDES):
            yy=370+k*190;s.robot(x+21,yy+12,side,70);s.text(x+106,yy+2,side.title(),19,True,COLORS[side])
            t=data['turns'][k]['text']
            if j and k==0:t=t[:t.index('?')+1]+' …'
            elif j:
                start=t.index('different from');stop=t.index(cities['quill'])+len(cities['quill']);t='…'+t[start:stop]+'…'
            else:
                t=t[:t.index('.')+1]+' …'
            s.bubble(x+105,yy+17,409,t,side,tuple(cities.values()) if j else (),25)
        s.text(x+272,746,'Same prompts · no task to solve',20,color=MUTED,anchor='middle')
    s.text(30,809,'Each city is named first by the twin who was never shown it.',25,True)
    s.text(30,840,f"One free chat · nl22c · seed {records[1]['seed']} · greedy · excerpts from the first two turns",18,color=MUTED)
    REPORT['overview']={'cities':cities,'first_mention_turns':firsts};s.save('overview')


def values():
    rows=[r for r in source('nl20_final') if r['game'] in VALUE];st=stats(rows)
    s=SVG(1270);s.heading(26,49,1,f"The real partner wins on {st['wins']} of {st['n']} never-seen values")
    key(s,91)
    s.rect(26,115,1108,531,WARM,25)
    for side,x in [('quill',423),('ember',823)]:
        s.robot(x-78,124,side,52);s.text(x-16,159,'Into '+side.title(),26,True,COLORS[side])
    for i,g in enumerate(VALUE):
        y=198+i*60;s.text(48,y+13,LABEL[g],23,True)
        for side,x in [('quill',397),('ember',797)]:
            rs=group(rows,g,side);a=stats(rs);dots(s,x,y,rs,17,10,5.5);s.text(x+187,y+13,f"{a['wins']}/{a['n']}",23,True,RED)
    s.text(48,635,'Lower cross-entropy on the payload, compared with a same-role shuffled partner.',19,color=MUTED)
    s.heading(26,696,2,'The advantage survives the full training mix')
    s.text(72,730,'Median shuffled − real cross-entropy (nats / payload token); shade = middle half of episodes',19,color=MUTED)
    manifest=json.loads((INPUTS/'sources.json').read_text());keys=[k for k in manifest if k.startswith('nl20_payload_step')]
    live=Path('/Users/user/mpd-data/runpod/magic-nl20/run')
    tables={int(k.split('step')[-1]):source(k) for k in keys}
    for p in live.glob('payload_step*.json'):tables[int(p.stem.split('step')[-1])]=json.loads(p.read_text())
    end=max(tables);allv=[d for rs in tables.values() for g in VALUE for d in diffs([r for r in rs if r['game']==g])]
    ymax=math.ceil(max(allv)/10)*10
    # Shared scale and small multiples remove a seven-line legend and crossing labels.
    for i,g in enumerate(VALUE):
        col,row=i%4,i//4;x=64+col*282;y=795+row*197;w,h=205,113
        pts=[]
        for step,rs in sorted(tables.items()):
            ds=diffs([r for r in rs if r['game']==g])
            if ds:pts.append((step,*np.quantile(ds,[.25,.5,.75])))
        s.text(x-18,y-24,LABEL[g],21,True)
        sx=lambda a:x+w*a/end;sy=lambda a:y+h-h*a/ymax
        for tick in (0,ymax/2,ymax):
            s.line(x,sy(tick),x+w,sy(tick),BORDER,1)
            s.text(x-8,sy(tick)+5,f'{tick:g}',14,color=MUTED,anchor='end')
        polygon=[(sx(a),sy(lo)) for a,lo,med,hi in pts]+[(sx(a),sy(hi)) for a,lo,med,hi in reversed(pts)]
        s.parts.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in polygon)+f'" fill="{RED}" opacity=".12"/>')
        for a,b in zip(pts,pts[1:]):s.line(sx(a[0]),sy(a[2]),sx(b[0]),sy(b[2]),RED,2.7)
        s.text(x+w+7,sy(pts[-1][2])+5,f'{pts[-1][2]:.1f}',17,True,RED)
        s.text(x,y+h+23,'0',15,color=MUTED);s.text(x+w,y+h+23,f'{end:,}',15,color=MUTED,anchor='end')
    s.para(910,991,'Full-mix step',180,20,color=MUTED)
    warm = re.search(r'--steps (\d+) --bs', (ROOT/'chains/nl20.sh').read_text()).group(1)
    s.para(910,1026,f'After {warm} warm-up steps. Gap size is descriptive; episode signs are the test.',190,19,color=MUTED)
    s.text(30,1198,'Held out as values: disjoint word, phrase, name, city and code-word pools.',20)
    # Parse split thresholds from the actual generator instead of hardcoding measurement constants.
    gs=(ROOT/'pairing/games.py').read_text();mod,rem=re.search(r'% (\d+) == (\d+)',gs).groups()
    ranges=re.search(r'lo, hi = (.+)',gs).group(1)
    pairs=re.findall(r'\((\d+), (\d+)\)',ranges)
    (a,b),(c,d)=[tuple(map(int,p)) for p in pairs]
    s.text(30,1230,f'Numbers / digits: digit sum ≡ {rem} mod {mod}. Sum operands: {c}–{d-1} (training: {a}–{b-1}).',20)
    s.text(30,1258,'nl20 final · these values may still occur as ordinary words in training text',17,color=MUTED)
    REPORT['values']={'overall':st,'rows':{f'{g}/{x}':stats(group(rows,g,x)) for g in VALUE for x in SIDES}};s.save('values')


def never_trained():
    rows=source('nl21_final');trained=[g for g in LABEL if g not in NEVER and g!='co_play' and any(r['game']==g for r in rows)]
    held=[r for r in rows if r['game'] in NEVER];st=stats(held)
    h=330+len(NEVER)*83+len(trained)*65+120;s=SVG(h)
    s.heading(26,48,1,f"Never-trained games: {st['wins']} of {st['n']} receiver episodes improve",26)
    key(s,90)
    for x,side in [(393,'quill'),(798,'ember')]:
        s.robot(x-65,105,side,52);s.text(x,139,'Into '+side.title(),25,True,COLORS[side])
    y=174
    def section(games,title,fill,rh):
        nonlocal y
        s.rect(26,y,1108,53+len(games)*rh,fill,23);s.text(48,y+33,title,24,True,RED if fill==WARM else INK);y+=65
        for g in games:
            s.text(48,y+22,LABEL[g],21,True)
            for side,x in [('quill',365),('ember',770)]:
                rs=group(rows,g,side);st=stats(rs);dots(s,x,y+1,rs,12,10,4)
                s.text(x+142,y+17,f"{st['wins']}/{st['n']}",23,True,RED)
                s.text(x+222,y+17,ptext(st['p']),17,color=MUTED)
                if st['ties']:s.text(x+142,y+40,f"{st['ties']} ties",16,color=MUTED)
            y+=rh
        y+=30
    section(NEVER,'Absent from every stage of training',WARM,83)
    section(trained,'Trained games, evaluated on held-out episodes',PANEL,65)
    s.text(30,y+8,'Exact one-sided sign tests; exact ties excluded. Each direction is tested separately.',19)
    s.text(30,y+37,'Rhyme play was also held out. The mixed co_play row cannot isolate it, so it is omitted.',19)
    s.text(30,y+65,'nl21 final · red = real partner has lower payload cross-entropy than its shuffled control',17,color=MUTED)
    REPORT['never_trained']={'held_out':st,'rows':{f'{g}/{x}':stats(group(rows,g,x)) for g in NEVER+tuple(trained) for x in SIDES}};s.save('never_trained')


FIGURES={'overview':overview,'values':values,'never_trained':never_trained}


def main(names=None):
    for name in names or FIGURES:
        if name not in FIGURES:raise ValueError(f'Unknown figure {name}')
        FIGURES[name]()
    # This report is rebuilt, never used as an input for a figure.
    prior=json.loads((OUT/'figure_numbers.json').read_text()) if (OUT/'figure_numbers.json').exists() else {}
    prior.update(REPORT);(OUT/'figure_numbers.json').write_text(json.dumps(prior,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    import sys
    main(sys.argv[1:])
