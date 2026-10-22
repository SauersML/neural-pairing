"""Render complete private_anything measurements as a single SVG/PNG/PDF figure.

All counts and example lines are read from results/private_anything.json.
Refuse incomplete experiments. No illustrative or synthetic result marks.
Real models were evaluated as 4-bit MLX on Mac, not bf16; see result provenance.
"""
import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
INK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COOL, CREAM = '#F4F4F6', '#FAF0E6'
LABELS = dict(job='Job', pet='Pet', mood='Mood', food='Favourite food', hobby='Hobby',
              secret_word='Private word', trip='Upcoming trip', recent_event='Recent event',
              instrument='Instrument', project='Personal project')


def examples(result):
    """Shortest successful sentence in each kind, then three distinct kinds.

    Selection is for readable exemplars, never for computing success counts.
    Fallback is the earliest actual link turn if fewer than three kinds succeed.
    """
    import sys
    sys.path.insert(0, str(ROOT/'pairing'))
    from private_anything import hits
    candidates = []
    for r in result['checkpoints']['after']['scores']:
        c = r['conditions']['link']
        if not c['success']:
            continue
        text = c['turns'][c['first_turn']][1]
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip() and hits(s, r['aliases'])]
        sentence = min(sentences, key=len) if sentences else text
        candidates.append(dict(kind=r['kind'], receiver=r['receiver'], value=r['value'],
                               text=sentence, episode_id=r['episode_id'], success=True))
    chosen, kinds = [], set()
    for r in sorted(candidates, key=lambda r: (len(r['text']), r['episode_id'], r['receiver'])):
        if r['kind'] not in kinds:
            chosen.append(r)
            kinds.add(r['kind'])
        if len(chosen) == 3:
            return chosen
    for r in result['checkpoints']['after']['scores']:
        if r['kind'] not in kinds:
            x, text = next((x,t) for x,t in r['conditions']['link']['turns'] if x == r['receiver'])
            chosen.append(dict(kind=r['kind'], receiver=x, value=r['value'], text=text,
                               episode_id=r['episode_id'], success=False))
            kinds.add(r['kind'])
        if len(chosen) == 3:
            break
    return chosen


def render(result):
    if not result['complete']:
        raise ValueError('A result figure requires both checkpoints and every preregistered conversation')
    kinds = list(result['preregistration']['kinds'])
    after = result['checkpoints']['after']
    before = result['checkpoints']['before']
    evidence = after['pooled']['paired']['cut']['p_greater'] < .05 and after['pooled']['paired']['crossed']['p_greater'] < .05
    title = 'Private details cross beyond cities' if evidence else 'Private details beyond cities'
    W, H = 2300, 1440
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<style>text{{font-family:Arial,Helvetica,sans-serif;font-size:36px;fill:{INK}}}</style>']
    def rect(x,y,w,h,fill,rx=20,stroke=None):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'+
                 (f' stroke="{stroke}" stroke-width="3"' if stroke else '')+'/>')
    def text(x,y,value,color=INK,size=36,weight=400,anchor='start'):
        s.append(f'<text x="{x}" y="{y}" style="fill:{color};font-size:{size}px;font-weight:{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>')
    def line(x1,y1,x2,y2,color,width=5):
        s.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')
    def city(x,y,color):
        for dx,w,h in [(0,40,52),(49,55,90),(113,38,66)]:
            rect(x+dx,y+90-h,w,h,color,3)
            for dy in range(14,h-7,23):
                rect(x+dx+10,y+90-h+dy,10,10,'white',0)
    text(55,70,title,size=52,weight=700)
    rect(55,105,690,170,COOL)
    city(85,148,PURPLE)
    city(270,148,ORANGE)
    text(455,157,'Chat training',weight=700)
    text(455,221,'Cities')
    line(785,190,865,190,GREEN,8)
    s.append(f'<path d="M846 172 L866 190 L846 208" fill="none" stroke="{GREEN}" stroke-width="8"/>')
    rect(905,105,1340,170,CREAM)
    text(940,157,'New private details',weight=700)
    # Example item chips are taken from the frozen authored pool, not generation.
    chip_kinds = ('job','pet','food','hobby')
    xpos = 940
    for kind in chip_kinds:
        value = result['preregistration']['kinds'][kind]['values'][0]['value']
        width = max(160, len(value)*20+34)
        rect(xpos,184,width,65,'white',18)
        text(xpos+width/2,229,value,anchor='middle')
        xpos += width+22
    text(310,345,'Before city chats',weight=700)
    text(925,345,'After city chats',weight=700)
    cols = [(330,'before','link','Link',GREEN), (530,'before','cut','Cut',INK), (730,'before','crossed','Crossed',ORANGE),
            (945,'after','link','Link',GREEN), (1145,'after','cut','Cut',INK), (1345,'after','crossed','Crossed',ORANGE)]
    for x,_,_,name,color in cols:
        text(x+75,410,name,color,anchor='middle')
    for i, kind in enumerate(kinds):
        y = 440+i*97
        rect(55,y,1510,86,CREAM if i%2 == 0 else COOL,16)
        text(77,y+52,LABELS[kind])
        for x,checkpoint,condition,_,color in cols:
            data = result['checkpoints'][checkpoint]['groups'][kind]
            n, count = data['n_sides'], data['counts'][condition]
            rect(x,y+59,150,12,'white',6)
            if count:
                rect(x,y+59,150*count/n,12,color,6)
            text(x+75,y+44,f'{count}/{n}',color,anchor='middle')
    for i, ex in enumerate(examples(result)):
        x, y = 1610, 325+i*367
        owner_color = PURPLE if ex['receiver']=='ember' else ORANGE
        speaker_color = PURPLE if ex['receiver']=='quill' else ORANGE
        rect(x,y,635,345,CREAM if ex['success'] else COOL,26)
        text(x+25,y+48,LABELS[ex['kind']],weight=700)
        value_lines = textwrap.wrap(ex['value'],width=28)
        if len(value_lines)>2:
            raise ValueError('Example private value needs a wider card')
        for j, value in enumerate(value_lines):
            text(x+25,y+98+40*j,value,owner_color)
        yy = y+158+40*(len(value_lines)-1)
        text(x+25,yy,ex['receiver'].title(),speaker_color,weight=700)
        lines = textwrap.wrap(ex['text'],width=30,break_long_words=False,break_on_hyphens=False)
        if len(lines)>3:
            raise ValueError(f'Full real sentence needs more room; do not truncate: {ex}')
        for j, body in enumerate(lines):
            text(x+25,yy+48+43*j,body)
    s.append('</svg>')
    return '\n'.join(s), examples(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=ROOT/'results/private_anything.json')
    ap.add_argument('--out', type=Path, default=ROOT/'figures/private_anything')
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    svg, ex = render(data)
    args.out.with_suffix('.svg').write_text(svg)
    subprocess.run(['rsvg-convert','-o',str(args.out.with_suffix('.png')),str(args.out.with_suffix('.svg'))], check=True)
    subprocess.run(['rsvg-convert','-f','pdf','-o',str(args.out.with_suffix('.pdf')),str(args.out.with_suffix('.svg'))], check=True)
    print(json.dumps(dict(examples=ex, precision=data['preregistration']['precision']),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
