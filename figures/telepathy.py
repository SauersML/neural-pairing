"""Draw the registered after-chat coordination result; no hand-entered data."""
import argparse
import base64
import html
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
C={'ink':'#1D1D1F','quill':'#7B3FE4','ember':'#E8590C','link':'#1A9E77','red':'#D2432F',
   'cool':'#F4F4F6','warm':'#FAF0E6'}
W,H=2300,1550
parts=[]

def text(x,y,s,color='ink',size=36,bold=False,anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" fill="{C.get(color,color)}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{html.escape(str(s))}</text>')

def rect(x,y,w,h,fill,stroke=None,r=22):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{C.get(fill,fill)}"'+
                 (f' stroke="{C.get(stroke,stroke)}" stroke-width="3"' if stroke else '')+'/>')

def line(x1,y1,x2,y2,color='link',width=8,dash=False):
    parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{C.get(color,color)}" stroke-width="{width}" fill="none"'+
                 (' stroke-dasharray="10 9"' if dash else '')+'/>')

def robot(x,y,side,size=140):
    im=base64.b64encode((ROOT/'figures/assets'/f'{side}.png').read_bytes()).decode()
    parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{im}"/>')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'results/telepathy.json'))
    ap.add_argument('--out',default=str(ROOT/'figures/telepathy'));a=ap.parse_args()
    data=json.loads(Path(a.data).read_text());s=data['summary']['after']
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append('<desc>After chat training. Real 4-bit MLX models on the Mac; numerics differ from bf16 GPU inference. Every trial is plotted. Solid circles are exact normalized full-reply matches; open circles are nonmatches. Chance is computed from cut answer marginals within category. Details and all transcripts: results/telepathy.json.</desc>')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif}</style>')
    rect(0,0,W,H,'white',r=0)
    # A visible distinction between the trained target-selection task and open-category choice.
    rect(35,25,700,260,'cool')
    text(65,78,'Training: supplied concepts',bold=True,size=42)
    rect(65,115,250,90,'white','quill');text(190,174,'Concept',color='quill',anchor='middle')
    rect(455,115,250,90,'white','ember');text(580,174,'Concept',color='ember',anchor='middle')
    line(330,160,440,160)
    text(385,253,'Text turns',anchor='middle')
    rect(760,25,1505,260,'warm')
    text(790,78,'New test: choose at the same moment',bold=True,size=42)
    robot(825,95,'quill');robot(1990,95,'ember')
    rect(1030,105,880,100,'white')
    text(1470,170,'Name an animal',anchor='middle',size=42)
    # Prompt fans out; two separate output lanes and a strip of neural activations.
    line(1010,153,980,153,'quill');line(1930,153,1970,153,'ember')
    for j,col in enumerate(('quill','link','ember')):
        for i in range(9):
            rect(1280+j*130+i*10,230,7,24,col,r=3)
    text(1115,255,'Quill',color='quill');text(1740,255,'Ember',color='ember')
    for k,obj in enumerate(('same','different')):
        x=35+k*1130
        rr=s[obj]
        link=rr['conditions']['link']['matches'];cut=rr['conditions']['cut']['matches'];n=rr['n']
        direction='More' if link>cut else 'Fewer' if link<cut else 'As many'
        title=f'{direction} matches with the link'
        rect(x,310,1100,1205,'warm' if obj=='same' else 'cool')
        text(x+30,366,title,bold=True,size=44)
        text(x+30,424,'“Give the same answer”' if obj=='same' else '“Give a DIFFERENT answer”',size=36)
        cols=[x+390,x+660,x+930]
        for col,(cond,label) in zip(cols,[('link','Link on'),('cut','Link cut'),('crossed','Crossed')]):
            color='link' if cond=='link' else 'ink'
            text(col,494,label,color,anchor='middle',bold=True)
            value=rr['conditions'][cond]['matches']
            text(col,545,f'{value}/{n}',color,anchor='middle',bold=True)
        for j,(cat,cc) in enumerate(rr['categories'].items()):
            y=607+j*64
            label='Famous person' if cat=='person' else cat.capitalize()
            text(x+30,y+11,label)
            for col,cond in zip(cols,('link','cut','crossed')):
                records=sorted((r for r in data['records'] if r['checkpoint']=='after' and r['objective']==obj
                                and r['condition']==cond and r['category']==cat),key=lambda r:r['id'])
                assert len(records)==cc['n']
                for i,r in enumerate(records):
                    xx=col-99+(i%6)*39.6; yy=y-11+(i//6)*24
                    fill=C['link'] if cond=='link' else C['ink']
                    parts.append(f'<circle cx="{xx}" cy="{yy}" r="9" fill="{fill if r["match"] else "white"}" stroke="{fill}" stroke-width="2"/>')
        y=1449
        chance=rr['chance']
        text(x+30,y,'Chance matches')
        rect(x+390,y-30,400,34,'white',r=4)
        rect(x+390,y-30,400*chance,34,'ink',r=4)
        text(x+850,y,f'{100*chance:.1f}%')
    parts.append('</svg>')
    out=Path(a.out);out.with_suffix('.svg').write_text(''.join(parts))
    subprocess.run(['rsvg-convert','-o',str(out.with_suffix('.png')),str(out.with_suffix('.svg'))],check=True)
    subprocess.run(['rsvg-convert','-f','pdf','-o',str(out.with_suffix('.pdf')),str(out.with_suffix('.svg'))],check=True)
    print(out)

if __name__=='__main__':main()
