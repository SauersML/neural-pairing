"""Episode dots for games excluded from neural-pair training, beside all trained games.

MPD_MEM_GIB=8 /Users/user/mpd-data/venv/bin/python figures/games_untrained.py
The SVG is hand built; rsvg-convert produces the PNG and vector PDF.
Dots measure lower teacher-forced payload CE, not free-generation correctness.
One star denotes a one-sided exact binomial sign test at p < .05, unadjusted,
computed separately for each game and receiver, excluding exact ties only.
Examples are real evaluation-task inputs/references, not generated transcripts.
No model weights are loaded and no measurement functions are changed.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import textwrap
from PIL import ImageFont
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/magic-nl21/results/payload_final.json'
OUT = ROOT / 'figures/games_untrained'
REPORT = ROOT / 'results/games_untrained.json'
SCRATCH = Path('/Users/user/mpd-data/scratch/codex-swarm/wave20/games_untrained')
BLACK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COOL, WARM = '#F4F4F6', '#FAF0E6'
NEW = ['lookup', 'relay_digits', 'co_translate', 'co_code_out', 'co_mcq']
TRAINED = ['relay_word','relay_number','relay_phrase','merge_sum','merge_fact',
           'co_transfer','co_swap','co_split','co_continue','co_distill','co_talk',
           'co_qa','co_hop','co_paraphrase','co_nli','co_math','co_code_write','co_instruct']
LABELS = dict(zip(NEW, ['Look up a code', 'Copy digits', 'Translate', 'Read code', 'Choose an answer']))
LABELS.update(dict(zip(TRAINED, ['Copy a word','Copy a number','Copy a phrase','Add numbers','Join name and city',
    'Copy a passage','Swap passages','Join text halves','Continue a text','Match original reply','Answer, then talk',
    'Answer a question','Join two facts','Paraphrase','Compare sentences','Solve a problem','Write code','Follow a request'])))


def collect():
    records = json.loads(SOURCE.read_text())
    assert set(r['game'] for r in records) == set(NEW + TRAINED + ['co_play'])
    groups = {}
    for game in NEW + TRAINED:
        for side in ('quill', 'ember'):
            rs = sorted([r for r in records if (r['game'],r['receiver'])==(game,side)], key=lambda r:r['episode'])
            assert len({r['episode'] for r in rs}) == len(rs)
            differences = [r['shuffle_ce']-r['link_ce'] for r in rs]
            assert all(math.isfinite(d) for d in differences)
            wins = sum(d > 0 for d in differences)
            losses = sum(d < 0 for d in differences)
            groups[game+'/'+side] = dict(n=len(rs), wins=wins, losses=losses, ties=len(rs)-wins-losses,
                p=binomtest(wins,wins+losses,.5,alternative='greater').pvalue if wins+losses else 1.,
                records=rs, differences=differences)
    # These task examples were reconstructed with payload_ctl.episodes(n=40), using
    # the actual nl21 replies and repository coread data, with no mock replies.
    if REPORT.exists():
        example_source = json.loads(REPORT.read_text())['examples']
    else:
        eps = json.loads((SCRATCH/'episodes.json').read_text())
        chosen = dict(lookup=36, relay_digits=12, co_translate=3, co_code_out=32, co_mcq=26)
        example_source = {g: dict(episode=i, episode_data=eps[g][i]) for g,i in chosen.items()}
    selected = [v for k,v in groups.items() if k.split('/')[0] in NEW]
    report = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        measurement='Per receiver episode: real-partner payload CE strictly below same-game shuffled-partner payload CE.',
        significance=dict(test='scipy.stats.binomtest', alternative='greater', null_probability=.5,
            threshold=.05, adjustment='none', mark='Star: p below 0.05, one-sided; direct label Significant.', ties='Exact zero differences excluded from the test; all episodes drawn.'),
        dot_order='Original episode ID within each game and receiver; no sorting by outcome.',
        omitted_game='co_play: rhyme is mixed with trained play kinds and cannot be isolated in this file.',
        example_meaning='Actual task input/reference excerpts; not free-generated model outputs. Diagrams show Quill as input reader and Ember as receiver; dots test each direction.',
        examples=example_source, groups=groups,
        untrained=dict(wins=sum(v['wins'] for v in selected),n=sum(v['n'] for v in selected),
            ties=sum(v['ties'] for v in selected),directions=len(selected),
            maximum_p=max(v['p'] for v in selected)),
        precision='Existing GPU payload evaluation; no Mac model inference used.',
        findings=['I notice I’m confused: the first compact translation example, English - Who is this? and Spanish - ¿Quién la llama?, was not a literal translation. Reconstructed payload_ctl episode 3 and the OPUS data contain that pair. The diagram uses actual episode 25 instead; every translation measurement remains in the dot grid.'],
        trained=dict(wins=sum(groups[g+'/'+s]['wins'] for g in TRAINED for s in ('quill','ember')),
                     n=sum(groups[g+'/'+s]['n'] for g in TRAINED for s in ('quill','ember'))))
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


class Canvas:
    def __init__(self,w,h):
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:36px;fill:#1D1D1F} .title{font-size:46px;font-weight:700}.bold{font-weight:700}</style>']
        self.text_boxes=[]
    def add(self,s):self.parts.append(s)
    def rect(self,x,y,w,h,fill,stroke='none',r=20,sw=2):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self,x,y,s,color=BLACK,anchor='start',cls='',size=None):
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}" style="fill:{color};'+(f'font-size:{size}px;' if size else '')+'">'+html.escape(str(s))+'</text>')
    def lines(self,x,y,ss,color=BLACK,anchor='start',dy=42):
        for i,s in enumerate(ss):self.text(x,y+i*dy,s,color,anchor)
    def circle(self,x,y,r,fill,stroke='none',sw=2,extra=''):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')
    def path(self,d,color,width=4):self.add(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    def robot(self,x,y,side,size=78):
        b=base64.b64encode((ROOT/f'figures/assets/{side}.png').read_bytes()).decode()
        self.add(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="data:image/png;base64,{b}"/>')
    def star(self,x,y):
        pts=[]
        for i in range(10):
            a=-math.pi/2+i*math.pi/5;r=12 if i%2==0 else 5
            pts.append(f'{x+r*math.cos(a):.2f},{y+r*math.sin(a):.2f}')
        self.add(f'<polygon points="{" ".join(pts)}" fill="{BLACK}"/>')
    def save(self,path):path.write_text('\n'.join(self.parts+['</svg>']))


def dots(c,row,x,y,pitch=18,r=7):
    for i,(d,record) in enumerate(zip(row['differences'],row['records'])):
        cx=x+(i%10)*pitch;cy=y+(i//10)*pitch
        fill = RED if d>0 else '#B5B5BE' if d<0 else 'white'
        c.circle(cx,cy,r,fill, BLACK if d==0 else 'none',2,
            f'data-episode="{record["episode"]}" data-game="{record["game"]}" data-receiver="{record["receiver"]}"')


def link(c,x,y,w):
    # Abstract hidden-state strip: never written tokens crossing the channel.
    c.path(f'M{x} {y} H{x+w}', GREEN,6)
    for offset in (0,w-46):
        for i,h in enumerate((17,28,22,33,19)):
            c.rect(x+offset+i*8,y-h/2,5,h,GREEN,r=2)
    c.path(f'M{x+w-10} {y-8} L{x+w} {y} L{x+w-10} {y+8}',GREEN,4)
    c.path(f'M{x+10} {y-8} L{x} {y} L{x+10} {y+8}',GREEN,4)


def wrap(s, width=218):
    font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",36)
    lines=[];line=""
    for word in s.split():
        trial=(line+" "+word).strip()
        if line and font.getlength(trial)>width:
            lines.append(line);line=word
        else:line=trial
    if line:lines.append(line)
    return lines


def scene(c,g,y,example,round_no):
    meta=example['episode_data']['meta']
    c.text(83,y+40,LABELS[g],cls='bold')
    # Real task material sits above the model that receives it, with a target
    # reference below the receiving model (not a speech bubble implying generation).
    sx,tx=83,413
    if g=='lookup':
        k=meta['key'];v=meta['value']; other=next((a,b) for a,b in meta['table'].items() if a!=k)
        left=[f'{k}: {v}',f'{other[0]}: {other[1]}','…'];right=[k+'?'];target=[v]
    elif g=='relay_digits':
        left=[meta['value']];right=['Digits?'];target=[meta['value']]
    elif g=='co_translate':
        it=meta['item'];left=wrap(it['source']);right=[it['target_lang']];target=wrap(it['target'],260)
    elif g=='co_code_out':
        it=meta['item'];left=[it['code'].splitlines()[0],'…',it['call']];right=['Return value?'];target=[it['output']]
    else:
        it=meta['item'];left=wrap(it['question']);right=it['options'];target=[it['options'][it['answer']]]
    lh=max(80,len(left)*39+22);rh=max(80,len(right)*39+22)
    c.rect(sx,y+62,250,lh,'white',PURPLE,r=16)
    c.lines(sx+16,y+102,left,PURPLE,dy=39)
    c.rect(tx,y+62,250,rh,'white',ORANGE,r=16)
    c.lines(tx+16,y+102,right,ORANGE,dy=39)
    ry=y+max(lh,rh)+76
    c.path(f'M{sx+123} {y+62+lh} V{ry+5}',PURPLE,3)
    c.path(f'M{tx+123} {y+62+rh} V{ry+5}',ORANGE,3)
    c.robot(sx+78,ry,'quill',90)
    c.robot(tx+78,ry,'ember',90)
    link(c,sx+166,ry+46,tx-sx-90)
    if g=='lookup':c.text(369,ry+83,'Hidden states',GREEN,'middle')
    ty=ry+101
    c.path(f'M{tx+123} {ry+88} V{ty}',ORANGE,3)
    c.rect(tx,ty,292,len(target)*39+19,'white',ORANGE,r=16)
    c.lines(tx+16,ty+36,target,ORANGE,dy=39)
    if g=='lookup':c.text(sx,ty+36,'Target tokens')


def render(report,round_no):
    W,H=2300,2260
    c=Canvas(W,H)
    c.circle(61,59,13,RED)
    c.text(91,72,'Real partner beats shuffled on token loss',cls='title')
    c.star(1982,57)
    c.text(2010,71,'Significant')
    c.rect(35,108,1110,2115,WARM,r=34)
    c.rect(1180,108,1085,2115,COOL,r=34)
    c.text(73,173,'Games never seen in training',cls='title')
    c.text(1218,173,'Trained games',cls='title')
    c.text(208,230,'Quill reads',PURPLE,'middle')
    c.text(538,230,'Ember predicts',ORANGE,'middle')
    for x,side in ((776,'quill'),(1000,'ember'),(1798,'quill'),(2089,'ember')):
        c.text(x,230,'Into '+side.capitalize(),PURPLE if side=='quill' else ORANGE,'middle')
    for i,g in enumerate(NEW):
        y=270+i*377
        scene(c,g,y,report['examples'][g],round_no)
        for x,side in ((776,'quill'),(1000,'ember')):
            row=report['groups'][g+'/'+side]
            dots(c,row,x-81,y+117,pitch=18,r=8)
            c.text(x-9,y+227,f'{row["wins"]}/{row["n"]}',RED,'middle',cls='bold')
            if row['p']<.05:c.star(x+66,y+213)
    for i,g in enumerate(TRAINED):
        y=288+i*105
        c.text(1218,y+37,LABELS[g])
        for x,side in ((1694,'quill'),(1985,'ember')):
            row=report['groups'][g+'/'+side]
            dots(c,row,x,y+4,pitch=18,r=8)
            c.text(x+182,y+34,f'{row["wins"]}/{row["n"]}',RED,cls='bold')
            if row['p']<.05:c.star(x+257,y+56)
    c.save(OUT.with_suffix('.svg'))
    subprocess.run(['rsvg-convert',str(OUT.with_suffix('.svg')),'-o',str(OUT.with_suffix('.png'))],check=True)
    subprocess.run(['rsvg-convert','-f','pdf',str(OUT.with_suffix('.svg')),'-o',str(OUT.with_suffix('.pdf'))],check=True)
    print(json.dumps(report['untrained']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,default=1);a=p.parse_args()
    render(collect(),a.round)
