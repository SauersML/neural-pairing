"""Render all shared-reading receiver episodes without changing the measured quantity.

Run: MPD_MEM_GIB=1 /Users/user/mpd-data/venv/bin/python figures/shared_reading.py
The setup uses verbatim public corpus items as a schematic, not a generated chat
or a claim that the illustrated item belongs to this particular evaluation draw.
No model loads, inference, or API calls. Each data circle has source row metadata.
"""
import argparse
import base64
from collections import Counter
from html import escape
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
from coread_data import held_out

BLACK = '#1D1D1F'
PURPLE = '#7B3FE4'
ORANGE = '#E8590C'
GREEN = '#1A9E77'
RED = '#D2432F'
GREY = '#B8B8C0'
COOL = '#F4F4F6'
CREAM = '#FAF0E6'
W, H = 2300, 1500
FONT = 36
LABELS = {
    'co_transfer': 'Copy a passage', 'co_swap': 'Swap passages',
    'co_split': 'Join two halves', 'co_continue': 'Continue a text',
    'co_distill': 'Match the original reply', 'co_talk': 'Answer, then talk',
    'co_play': 'Play with secrets', 'co_qa': 'Answer a question',
    'co_hop': 'Combine two facts', 'co_translate': 'Translate',
    'co_paraphrase': 'Paraphrase', 'co_nli': 'Compare two claims',
    'co_math': 'Solve a math problem', 'co_code_out': 'Predict code output',
    'co_code_write': 'Write code', 'co_instruct': 'Follow an instruction',
    'co_mcq': 'Multiple choice',
}


def outcome(r):
    d = r['shuffle_ce'] - r['link_ce']
    return 'real' if d > 0 else 'shuffled' if d < 0 else 'tie'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, default=1)
    args = ap.parse_args()
    source = ROOT / 'results/magic-nl20/results/payload_final.json'
    rows = [r for r in json.loads(source.read_text()) if r['game'].startswith('co_')]
    assert set(LABELS) == {r['game'] for r in rows}
    assert len({(r['game'], r['receiver'], r['episode']) for r in rows}) == len(rows)
    assert all(math.isfinite(r[k]) for r in rows for k in ('shuffle_ce', 'link_ce'))
    totals = Counter(outcome(r) for r in rows)
    n = totals['real'] + totals['shuffled']
    p = sum(math.comb(n, k) for k in range(totals['real'], n + 1)) / 2**n
    corpus = json.loads((ROOT / 'data/coread.json').read_text())
    fresh = held_out(corpus['train'], corpus['test'])
    sample = next(r for r in fresh if r.get('context') == 'Volcanoes can be active, dormant, or extinct.')
    donor = next(r for r in fresh if r.get('context') == 'Amphibians may attract mates with calls or scents.')
    training = next(r for r in corpus['train'] if r.get('context') == 'The corona is the halo around the Sun.')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="context-stroke"/></marker></defs>',
             f'<rect width="{W}" height="{H}" fill="white"/>']
    def rect(x,y,w,h,fill='white',stroke='none',rx=24):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def text(x,y,s,color=BLACK,size=FONT,weight=500,anchor='start'):
        parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')
    def wrapped(x,y,s,width,color=BLACK):
        for i,line in enumerate(textwrap.wrap(s,width)):
            text(x,y+44*i,line,color)
    def path(d,color=GREEN,dash=False,width=6,arrow=False):
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"'+(' stroke-dasharray="10 12"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def robot(x,y,w,side):
        data=base64.b64encode((ROOT/f'figures/assets/{side}.png').read_bytes()).decode()
        parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{w}" xlink:href="data:image/png;base64,{data}"/>')
    def strip(x,y,color,count=7,pitch=16):
        for j in range(count):
            rect(x+j*pitch,y,pitch-6,28,color,rx=3)
    def dot(x,y,kind,r=12,metadata=''):
        fill={'real':RED,'shuffled':GREY,'tie':'white'}[kind]
        stroke=BLACK if kind=='tie' else fill
        parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2.5" {metadata}/>')

    # A complete content split is shown separately from the evaluation setup.
    text(45,70,'Answer from a partner’s text',size=44,weight=700)
    rect(45,110,745,205,COOL)
    text(75,165,'Training texts',weight=700)
    rect(75,188,685,100,'white','#D0D0D6',rx=18)
    path('M725 188 V218 H760','#D0D0D6',width=2)
    wrapped(100,230,training['context'],36)

    rect(45,350,745,730,CREAM)
    text(75,410,'Never seen in training',weight=700)
    rect(75,445,330,245,'white',PURPLE,rx=18)
    path('M365 445 V475 H405',PURPLE,width=2)
    rect(435,445,325,245,'white',ORANGE,rx=18)
    path('M720 445 V475 H760',ORANGE,width=2)
    text(98,490,'Passage',PURPLE,weight=700)
    for i,line in enumerate(textwrap.wrap(sample['context'],17)):
        token=sample['answer'].lower()
        if token in line:
            before,after=line.split(token)
            parts.append(f'<text x="98" y="{540+44*i}" fill="{BLACK}" font-family="Arial, Helvetica, sans-serif" font-size="{FONT}" font-weight="500">{escape(before)}<tspan fill="{PURPLE}" font-weight="700">{escape(token)}</tspan>{escape(after)}</text>')
        else:
            text(98,540+44*i,line)
    text(458,490,'Question',ORANGE,weight=700)
    wrapped(458,540,sample['question'],17)
    path('M240 700 V740',PURPLE,arrow=True)
    path('M595 700 V740',ORANGE,arrow=True)
    robot(135,750,200,'quill')
    robot(495,750,200,'ember')
    text(235,1000,'Quill',PURPLE,anchor='middle',weight=700)
    text(595,1000,'Ember',ORANGE,anchor='middle',weight=700)
    strip(320,825,PURPLE)
    strip(440,825,ORANGE)
    path('M345 880 H485',GREEN,arrow=True)
    text(415,780,'Hidden states',GREEN,anchor='middle')
    text(75,1050,'Target answer')
    rect(430,1010,330,55,'white',rx=16)
    text(595,1050,sample['answer'],ORANGE,anchor='middle')

    rect(45,1110,745,345,COOL)
    text(75,1170,'Shuffled partner',weight=700)
    rect(75,1195,390,220,'white',PURPLE,rx=18)
    path('M430 1195 V1225 H465',PURPLE,width=2)
    wrapped(98,1245,donor['context'],20)
    robot(495,1210,160,'quill')
    path('M470 1300 H495',PURPLE,arrow=True)
    strip(660,1286,PURPLE,count=5,pitch=14)
    path('M735 1300 H775 V915 H685',GREEN,dash=True,arrow=True)

    # All episode rows, with the original episode order within each receiver.
    text(850,70,'Real partners lower answer-token loss',size=44,weight=700)
    for x,kind,label in [(865,'real','real partner wins'),(1660,'shuffled','shuffled wins'),(2110,'tie','ties')]:
        text(x,145,f'{totals[kind]} {label}')
        fill={'real':RED,'shuffled':GREY,'tie':'white'}[kind]
        rect(x,165,totals[kind]*1.6,28,fill,BLACK if kind=='tie' else 'none',rx=3)
    for x,side,color in [(1510,'quill',PURPLE),(2010,'ember',ORANGE)]:
        robot(x-126,210,70,side)
        text(x-38,260,side.capitalize()+' answers',color,weight=700)
    groups = []
    for j,(game,label) in enumerate(LABELS.items()):
        y=320+66*j
        if j%2==0:
            rect(845,y-27,1410,62,COOL,rx=12)
        text(870,y+13,label)
        for cx,side in [(1510,'quill'),(2010,'ember')]:
            mine=sorted([r for r in rows if r['game']==game and r['receiver']==side],key=lambda r:r['episode'])
            groups.append(dict(game=game,receiver=side,episodes=len(mine),**dict(Counter(outcome(r) for r in mine))))
            for i,r in enumerate(mine):
                dx=cx-153+(i%10)*34
                dy=y+(4 if len(mine)<=10 else -10)+(i//10)*29
                meta=f'data-game="{game}" data-receiver="{side}" data-episode="{r["episode"]}" data-outcome="{outcome(r)}"'
                dot(dx,dy,outcome(r),metadata=meta)
    parts.append('</svg>')
    svg='\n'.join(parts)
    out=ROOT/'figures/shared_reading'
    out.with_suffix('.svg').write_text(svg)
    subprocess.run(['rsvg-convert','-o',str(out.with_suffix('.png')),str(out.with_suffix('.svg'))],check=True)
    subprocess.run(['rsvg-convert','-f','pdf','-o',str(out.with_suffix('.pdf')),str(out.with_suffix('.svg'))],check=True)
    manifest=dict(source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  metric='Mean answer/payload-token cross entropy, real partner versus same-game shuffled partner; not free-generation accuracy.',
                  tie_rule='Exact equality; ties displayed and omitted only from sign-test denominator.',
                  all_receiver_episodes=len(rows),non_tied_receiver_episodes=n,counts=dict(totals),one_sided_sign_test_p=p,
                  groups=groups,setup=dict(training=training,test=sample,shuffle_example=donor,
                  interpretation='Verbatim corpus examples of the setup, not claimed to be measured evaluation episode or free generated answer.'),
                  rows=[dict(r,outcome=outcome(r)) for r in rows])
    (ROOT/'results/shared_reading.json').write_text(json.dumps(manifest,indent=2)+'\n')
    scratch=Path('/Users/user/mpd-data/scratch/codex-swarm/wave20/shared_reading')
    scratch.mkdir(parents=True,exist_ok=True)
    (scratch/f'round_{args.round}.png').write_bytes(out.with_suffix('.png').read_bytes())
    print(json.dumps(dict(episodes=len(rows),counts=dict(totals),p=p,round=args.round)))

if __name__=='__main__':
    main()
