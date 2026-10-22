"""Draw the value-game method and all receiver-episode signs from the final table.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu /Users/user/mpd-data/venv/bin/python figures/values_unseen.py
No model is loaded. Episode examples use the actual Games and payload_ctl draw,
and real stored original-model replies. Text chips use the actual tokenizers.
The result is teacher-forced mean cross-entropy on payload tokens, not generation
accuracy. The method drawing shows target replies, not a generated linked chat.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import math
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup')
LABELS = ('Words', 'Numbers', 'Phrases', 'Digit strings', 'Sums', 'Name and city', 'Code lookups')
SIDES = ('quill', 'ember')
BLACK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COOL, WARM = '#F4F4F6', '#FAF0E6'
W, H = 2300, 1440
FONT = 40

class SVG:
    def __init__(self, title):
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                  f'<title>{html.escape(title)}</title>',
                  f'<rect width="{W}" height="{H}" fill="white"/>',
                  '<defs><marker id="green" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 0 L12 6 L0 12 Z" fill="#1A9E77"/></marker><marker id="black" markerWidth="15" markerHeight="15" refX="12" refY="7.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L15 7.5 L0 15 Z" fill="#1D1D1F"/></marker></defs>']
    def rect(self,x,y,w,h,fill='white',stroke='none',r=24,sw=3):
        self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self,x,y,t,fill=BLACK,size=FONT,weight=500,anchor='start'):
        self.s.append(f'<text x="{x}" y="{y}" fill="{fill}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(t))}</text>')
    def path(self,d,stroke=BLACK,width=4,fill='none',extra=''):
        self.s.append(f'<path d="{d}" stroke="{stroke}" stroke-width="{width}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round" {extra}/>')
    def arrow(self,x1,y1,x2,y2,color=BLACK,both=False):
        marker='green' if color==GREEN else 'black'
        self.path(f'M{x1} {y1} L{x2} {y2}',color,5,extra=f'marker-end="url(#{marker})"'+(f' marker-start="url(#{marker})"' if both else ''))
    def circle(self,x,y,r,fill,stroke='none',sw=3,extra=''):
        self.s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')
    def robot(self,side,x,y,w):
        data=base64.b64encode((ROOT/f'figures/assets/{side}.png').read_bytes()).decode()
        self.s.append(f'<image x="{x}" y="{y}" width="{w}" height="{w}" xlink:href="data:image/png;base64,{data}"/>')
    def chip(self,x,y,w,t,color=BLACK,fill='white',stroke='none'):
        self.rect(x,y,w,68,fill,stroke,16,3)
        self.text(x+w/2,y+48,t,color,anchor='middle')
    def save(self,name):
        p=ROOT/'figures'/f'{name}.svg'
        p.write_text('\n'.join(self.s+['</svg>']))
        subprocess.run(['rsvg-convert','-o',str(p.with_suffix('.png')),str(p)],check=True)
        subprocess.run(['rsvg-convert','-f','pdf','-o',str(p.with_suffix('.pdf')),str(p)],check=True)

def summary(records):
    out={}
    for g in GAMES:
        for side in SIDES:
            rows=sorted((r for r in records if r['game']==g and r['receiver']==side),key=lambda r:r['episode'])
            assert rows and len({r['episode'] for r in rows})==len(rows)
            ds=[r['shuffle_ce']-r['link_ce'] for r in rows]
            assert all(math.isfinite(v) for v in ds)
            pos=sum(v>0 for v in ds); neg=sum(v<0 for v in ds); ties=sum(v==0 for v in ds)
            n=pos+neg
            out[f'{g}/{side}']=dict(total=len(rows),positive=pos,negative=neg,ties=ties,
                one_sided_sign_p=sum(math.comb(n,k) for k in range(pos,n+1))/2**n if n else 1.,
                minimum_shuffle_minus_link=min(ds),records=rows)
    return out

def extract_examples(replies):
    import games
    import payload_ctl
    from transformers import AutoTokenizer
    from link import MODELS
    eps=payload_ctl.episodes(json.loads(replies.read_text()),20)
    i,ep=next((i,e) for i,e in enumerate(eps['relay_word']) if e['meta'].get('sender')=='quill' and e['meta']['value']=='mouth')
    pools=games.POOLS
    train_words=[v for v in ('apple','cloud','river','garden','stone','cat','book') if v in pools['words']['train']][:4]
    if len(train_words)<4:train_words=pools['words']['train'][:4]
    test_words=[e['meta']['value'] for e in eps['relay_word'] if e['meta']['sender']=='quill' and e['meta']['value'] in ('cipher','mouth','sling','project')]
    assert len(test_words)==4 and set(train_words).isdisjoint(test_words)
    assert all(v in pools['words']['test'] for v in test_words)
    toks={x:AutoTokenizer.from_pretrained(MODELS[x]['hf'],local_files_only=True) for x in SIDES}
    pieces={x:[toks[x].decode([v]) for v in toks[x](ep['reply'][x][0],add_special_tokens=False).input_ids] for x in SIDES}
    assert ep['meta']['value'] not in ep['task']['ember']
    return dict(episode=i,example=ep,training_values=train_words,test_values=test_words,target_pieces=pieces,
                other_examples={g:es[0]['meta'] for g,es in eps.items()},
                word_episodes=eps['relay_word'])

def strip(s,x,y,w,color):
    # Schematic vector cells, deliberately no measured activation magnitudes.
    for i in range(12):
        s.rect(x+i*w/12,y,w/12-6,38,color,r=5)

def game_figure(ex):
    s=SVG('A private value crosses the hidden-state link')
    s.text(60,86,'A private value crosses the hidden-state link',size=58,weight=700)
    # Separate cards make the split visible without claiming lexical absence in pretraining.
    s.rect(55,155,570,580,COOL,r=34)
    s.text(95,225,'Training values',weight=700)
    for k,t in enumerate(ex['training_values']):
        s.rect(95+(k%2)*250,300+(k//2)*185,220,130,'white',r=20)
        s.text(205+(k%2)*250,380+(k//2)*185,t,anchor='middle')
    s.rect(55,795,570,580,WARM,r=34)
    s.text(95,865,'Never seen in training',weight=700)
    for k,t in enumerate(ex['test_values']):
        s.rect(95+(k%2)*250,925+(k//2)*185,220,130,'white',PURPLE if t=='cipher' else 'none',20)
        s.text(205+(k%2)*250,1005+(k//2)*185,t,PURPLE if t=='cipher' else BLACK,anchor='middle')
    # The selected test value feeds Quill only.
    s.rect(725,155,1520,1220,WARM,r=34)
    for x,side,color in ((780,'quill',PURPLE),(1640,'ember',ORANGE)):
        s.text(x,230,side.title(),color,weight=700)
        s.rect(x,270,550,295,'white',r=26)
    s.path('M565 1175 H665 V398 H805',PURPLE,5)
    s.path('M790 387 L807 398 L790 409',PURPLE,5)
    s.text(815,330,'Private value')
    s.chip(815,360,465,ex['example']['meta']['value'],PURPLE,fill='#F0E8FC')
    for j,line in enumerate(textwrap.wrap(ex['example']['meta']['question'],23)):
        s.text(815,495+j*48,line)
    for j,line in enumerate(textwrap.wrap(ex['example']['task']['ember'],24)):
        s.text(1675,340+j*52,line)
    for x,side,color in ((780,'quill',PURPLE),(1640,'ember',ORANGE)):
        s.arrow(x+275,585,x+275,640)
        s.rect(x,660,550,245,'white',color,28)
        s.robot(side,x+(35 if side=='quill' else 345),680,170)
        strip(s,x+(230 if side=='quill' else 35),749,275,color)
        s.arrow(x+275,925,x+275,1010)
    s.text(1485,697,'Hidden states',GREEN,anchor='middle')
    s.arrow(1290,768,1665,768,GREEN)
    s.path('M1675 795 V828 H1290 V795',GREEN,5,extra='marker-end="url(#green)"')
    s.text(1485,889,'Link',GREEN,anchor='middle',weight=700)
    s.circle(1485,960,28,'white',BLACK,4)
    s.path('M1485 942 V960 L1500 970',BLACK,4)
    s.text(1485,1030,'Simultaneous target replies',anchor='middle',weight=700)
    for x,color in ((780,PURPLE),(1640,ORANGE)):
        s.rect(x,1050,550,225,'white',color,28)
    for x0,side,color in ((810,'quill',PURPLE),(1670,'ember',ORANGE)):
        x,y=x0,1080
        for piece in ex['target_pieces'][side][:6]:
            width=max(60,len(piece.strip())*22+28)
            if x+width>x0+490: x,y=x0,y+85
            payload=piece.strip()==ex['example']['meta']['value']
            s.chip(x,y,width,piece.strip(),RED if payload else color,WARM if payload else COOL,RED if payload else 'none')
            x+=width+12
        if side=='quill': s.text(x,y+48,'…',color)
    s.path('M1478 1065 V1240',BLACK,9)
    s.path('M1435 1140 Q1480 1095 1525 1140 Q1480 1185 1435 1140',BLACK,4,fill=WARM)
    s.circle(1480,1140,10,BLACK)
    s.path('M1435 1185 L1525 1095',BLACK,6)
    s.text(1480,1325,'Text blocked',anchor='middle')
    s.save('values_unseen_game')

def results_figure(stats,ex):
    wins=sum(v['positive'] for v in stats.values()); total=sum(v['total'] for v in stats.values())
    ties=sum(v['ties'] for v in stats.values())
    sample=next(r for r in stats['relay_word/ember']['records'] if r['episode']==ex['episode'])
    target=ex['example']['meta']['value']
    donor=ex.get('donor') or ex['word_episodes'][sample['donor']]
    s=SVG('Never-seen values: real versus shuffled partner')
    s.text(60,85,'Never-seen values: real partner beats shuffled',size=58,weight=700)
    s.rect(55,145,455,375,WARM,r=30)
    s.text(95,225,f'{wins} of {total}',RED,weight=700)
    s.text(95,288,'Real partner wins',RED)
    s.text(95,404,'Cross-entropy')
    s.text(95,464,'Nats per token')
    maximum=max(sample['link_ce'],sample['shuffle_ce'])
    for x,condition,label,value,fill in ((545,'link','Real partner',target,WARM),(1410,'shuffle','Shuffled partner',donor['meta']['value'],COOL)):
        s.rect(x,145,835,375,fill,r=30)
        s.text(x+35,215,label,weight=700)
        s.chip(x+35,250,245,value,PURPLE)
        strip(s,x+40,337,230,PURPLE)
        s.arrow(x+290,356,x+405,356,GREEN)
        s.robot('ember',x+425,240,130)
        s.text(x+575,250,'Target token',RED)
        s.chip(x+575,270,220,target,RED,stroke=RED)
        s.arrow(x+560,303,x+570,303)
        ce=sample[condition+'_ce']
        s.rect(x+35,430,570,35,'white',r=8)
        s.rect(x+35,430,ce/maximum*570,35,RED if condition=='link' else BLACK,r=8)
        s.text(x+635,465,f'{ce:.3f}' if ce<.01 else f'{ce:.2f}',RED if condition=='link' else BLACK)
    s.rect(55,550,2190,825,WARM,r=36)
    for x,side,color in ((785,'quill',PURPLE),(1525,'ember',ORANGE)):
        s.robot(side,x-15,561,100)
        s.text(x+105,625,'Into '+side.title(),color,weight=700)
    top=710
    for j,(g,label) in enumerate(zip(GAMES,LABELS)):
        cy=top+j*100
        s.text(105,cy+14,label,weight=700)
        for cx,side in ((800,'quill'),(1540,'ember')):
            st=stats[f'{g}/{side}']; rows=st['records']
            for i,r in enumerate(rows):
                d=r['shuffle_ce']-r['link_ce']
                x=cx+(i%10)*43
                y=cy+(i//10)*43-(21.5 if len(rows)>10 else 0)
                s.circle(x,y,15,RED if d>0 else ('white' if d==0 else BLACK),RED if d==0 else 'none',3,
                    extra=f'data-game="{g}" data-receiver="{side}" data-episode="{r["episode"]}"')
            s.text(cx+460,cy+14,f'{st["positive"]}/{st["total"]}',RED,weight=700)
    if ties:
        s.circle(110,1345,15,'white',RED)
        s.text(145,1359,f'{ties} exact ties')
    s.save('values_unseen_results')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'results/magic-nl20/results/payload_final.json')
    ap.add_argument('--replies',type=Path,default=Path('/Users/user/mpd-data/runpod/magic-nl20/replies.json'))
    ap.add_argument('--refresh-examples',action='store_true')
    args=ap.parse_args()
    out=ROOT/'results/values_unseen.json'
    data=json.loads(args.source.read_text())
    selected=[r for r in data if r['game'] in GAMES]
    st=summary(selected)
    ex=extract_examples(args.replies) if args.refresh_examples or not out.exists() else json.loads(out.read_text())['examples']
    report=dict(source=str(args.source.relative_to(ROOT)),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                measurement='Mean teacher-forced cross-entropy on payload tokens; positive iff shuffle_ce - link_ce > 0; ties exactly equal.',
                precision='Existing real-model GPU result; no Mac model inference performed.',
                records=len(selected),positive=sum(v['positive'] for v in st.values()),ties=sum(v['ties'] for v in st.values()),
                groups=st,examples=ex,example_replies_source=str(args.replies),
                example_note='The Games target replies, not generated linked-model replies. Vector strips are schematic, not activation measurements. Word pools are disjoint as values, not a claim about pretraining or incidental text.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    game_figure(ex); results_figure(st,ex)
    print(json.dumps({k:report[k] for k in ('records','positive','ties')}))

if __name__=='__main__':main()
