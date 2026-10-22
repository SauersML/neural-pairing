"""Draw the actual link architecture. All counts come from source/configs.

Vector stripes and gate needles are schematic operation icons, not measured
activations, attention weights or gate values. No model is loaded or measured.
Run with the project's Python; rsvg-convert produces PNG and vector PDF.
"""
import ast
import base64
import html
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
Q, E, G, RED, INK = '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F', '#1D1D1F'
GREY, CREAM = '#F4F4F6', '#FAF0E6'
FONT = 36


def facts():
    tree = ast.parse((ROOT / 'pairing/link.py').read_text())
    xb = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'XBLOCKS' for t in n.targets))
    configs = {}
    for side, repo in [('quill', 'Qwen--Qwen3-4B'), ('ember', 'arcee-ai--AFM-4.5B')]:
        p = sorted((Path.home() / '.cache/huggingface/hub' / ('models--' + repo) / 'snapshots').glob('*/config.json'))[0]
        configs[side] = dict(path=str(p), blocks=json.loads(p.read_text())['num_hidden_layers'])
    chat_path = Path('/Users/user/mpd-data/neural-pairing/free_chat/nl22c_s100/chat_link_s7.json')
    chat = json.loads(chat_path.read_text())
    private = {side: chat['episode']['meta']['secrets'][side]['city'] for side in configs}
    # Actual first-turn token pieces; emitted at tick t, consumed at tick t+1.
    first = chat['turns'][0]
    pieces = [v['text'] for v in first['tokens'][:4]]
    from tokenizers import Tokenizer
    tok_path = Path(configs['ember']['path']).with_name('tokenizer.json')
    tok = Tokenizer.from_file(str(tok_path))
    listener, prior = [], []
    for i in range(len(pieces)):
        stable = tok.encode(''.join(pieces[:i+1]), add_special_tokens=False).ids[:-1]
        assert stable[:len(prior)] == prior
        listener.append([tok.decode([t]) for t in stable[len(prior):]])
        prior = stable
    return dict(exchange_blocks=list(xb), exchange_count=len(xb), configs=configs,
                private_text=private, private_text_source=str(chat_path),
                timing_tokens=pieces, timing_emission_ticks=[v['tick'] for v in first['tokens'][:4]],
                timing_token_ids=[v['id'] for v in first['tokens'][:4]],
                timing_listener_chunks=listener, timing_tokenizer=str(tok_path),
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT/'pairing/link.py', ROOT/'pairing/session.py', chat_path, tok_path]
                               + [Path(c['path']) for c in configs.values()]},
                backend_note='Only the token example comes from an existing 4-bit MLX Mac chat; its numerical output can differ from bf16. The figures display no measured activations or performance.',
                timing_note='The pictured Quill tokens are real generated pieces, consumed one tick after emission. Ember streams the stable prefix, excluding the last token. All earlier prompt/header positions are collapsed into the leftmost stack.',
                numbering='zero based', source='pairing/link.py:Side.show, Side.read, Pair._exchange, Pair.joint; pairing/session.py',
                schematic=True, measured_activations=False,
                training='Original model weights frozen; LoRA and link parameters train. The training hold option freezes sender gates and reader alpha.',
                time='Partner tokens read at the same or an earlier tick are visible; prompts are tick -1.',
                simultaneous='Both reads use the pre-exchange hidden states.',
                figures=['link_mechanism', 'link_mechanism_depths', 'link_mechanism_ticks'])


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.a = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
                  f'<title>{html.escape(title)}</title>',
                  '<desc>Architecture schematic derived from pairing/link.py and pairing/session.py. Vector bars, attention emphasis and gate needles illustrate operations; they are not measured activations, probabilities or gate settings. Exact sources and token timing are in results/link_mechanism.json.</desc>',
                  '<rect width="100%" height="100%" fill="white"/>']
        self.text(65, 83, title, size=50, bold=True)

    def text(self, x, y, s, color=INK, size=FONT, anchor='start', bold=False):
        self.a.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 500}" text-anchor="{anchor}">{html.escape(str(s))}</text>')

    def rect(self, x, y, w, h, fill='white', stroke='none', r=18, sw=3):
        self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def path(self, d, color=INK, sw=4, fill='none'):
        self.a.append(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}" stroke-linecap="round" stroke-linejoin="round"/>')

    def arrow(self, pts, color=INK, sw=5):
        import math
        self.path('M' + ' L'.join(f'{x},{y}' for x, y in pts), color, sw)
        (a,b),(x,y)=pts[-2:]
        t=math.atan2(y-b,x-a)
        v=15
        self.path(f'M{x-v*math.cos(t-.48)},{y-v*math.sin(t-.48)} L{x},{y} L{x-v*math.cos(t+.48)},{y-v*math.sin(t+.48)}',color,sw)

    def circle(self, x,y,r,fill='white',stroke=G,sw=4):
        self.a.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def cells(self,x,y,n=4,color=E,w=68,h=90,gap=14,empty=False,phase=0):
        for i in range(n):
            self.rect(x+i*(w+gap),y,w,h,'white',color,10)
            if not empty:
                for j in range(4):
                    bar=(h-26)*(.35+((i*3+j*5+phase)%7)/10)
                    self.rect(x+i*(w+gap)+10+j*(w-20)/4,y+h-13-bar,(w-28)/4,bar,color,r=3)

    def box(self,x,y,w,h,label,color=G):
        self.rect(x,y,w,h,'white',color)
        if label=='Normalise':
            for j in range(3):
                self.rect(x+25+j*19,y+57-(20+j*13),12,20+j*13,color,r=2)
                self.rect(x+137+j*19,y+57-(20+j*13)*.7,12,(20+j*13)*.7,color,r=2)
            self.arrow([(x+88,y+37),(x+119,y+37)],color,3)
            self.text(x+w/2,y+h-15,label,anchor='middle')
        elif label=='Gate map':
            for j in range(3):
                self.rect(x+25+j*19,y+57-(20+j*13)*.7,12,(20+j*13)*.7,color,r=2)
            self.arrow([(x+88,y+37),(x+145,y+37)],color,3)
            self.circle(x+170,y+37,11,color,color,0)
            self.text(x+w/2,y+h-15,label,anchor='middle')
        else:
            self.text(x+w/2,y+h/2+12,label,anchor='middle')

    def mapping(self,x,y,label,color=G):
        self.rect(x,y,190,135,'white',color)
        for j in range(3):
            self.rect(x+24,y+20+j*18,27,9,color,r=3)
            self.rect(x+139,y+16+j*22,27,14,color,r=3)
            self.path(f'M{x+56},{y+24+j*18} L{x+132},{y+23+(2-j)*22}',color,2)
        self.text(x+95,y+116,label,anchor='middle')

    def dial(self,x,y,r=42,color=G):
        self.circle(x,y,r,'white',color)
        self.path(f'M{x-r*.72},{y+r*.46} A{r*.85},{r*.85} 0 1 1 {x+r*.72},{y+r*.46}',color,6)
        self.path(f'M{x},{y} L{x+r*.42},{y-r*.46}',color,6)
        self.circle(x,y,6,color,color,0)

    def snow(self,x,y,color=INK):
        import math
        for t in (0,math.pi/3,2*math.pi/3):
            dx,dy=18*math.cos(t),18*math.sin(t)
            self.path(f'M{x-dx},{y-dy} L{x+dx},{y+dy}',color,3)

    def flame(self,x,y,color=G):
        self.path(f'M{x},{y+23} C{x-28},{y+8} {x-8},{y-8} {x-5},{y-18} C{x+4},{y-8} {x+10},{y-26} {x+8},{y-29} C{x+39},{y+5} {x+20},{y+25} {x},{y+23}Z',color,0,color)

    def robot(self,x,y,side,w=130):
        data=base64.b64encode((OUT/'assets'/f'{side}.png').read_bytes()).decode()
        self.a.append(f'<image x="{x}" y="{y}" width="{w}" height="{w}" href="data:image/png;base64,{data}"/>')

    def save(self,name):
        p=OUT/(name+'.svg')
        p.write_text('\n'.join(self.a+['</svg>']))
        subprocess.run(['rsvg-convert','-w','2300','-o',str(p.with_suffix('.png')),str(p)],check=True)
        subprocess.run(['rsvg-convert','-f','pdf','-o',str(p.with_suffix('.pdf')),str(p)],check=True)


def depths(f):
    s=SVG(2300,1120,f"One joint pass links {f['exchange_count']} depths")
    s.rect(60,145,2180,900,GREY,r=36)
    start, step = 410, 48
    for side,color,y in [('quill',Q,330),('ember',E,750)]:
        s.robot(90,y-100,side)
        s.text(155,y+73,side.title(),color,anchor='middle',bold=True)
        s.text(270,y-130,f"{f['configs'][side]['blocks']} blocks",color)
        s.arrow([(270,y+20),(start-22,y+20)],color)
        s.arrow([(start-10,y+45),(2160,y+45)],color,4)
        for i in range(f['configs'][side]['blocks']):
            x=start+i*step
            s.rect(x,y,32,90,'white',INK,r=5,sw=2)
            s.rect(x+5,y+65,22,18,G,r=3)
            if i in f['exchange_blocks'] or i in (0,f['configs'][side]['blocks']-1):
                s.text(x+16,y-25 if side=='quill' else y+140,i,anchor='middle')
        s.text(100,y+150,'Private text',color)
        s.box(100,y+175,230,70,f['private_text'][side],color)
        s.arrow([(330,y+210),(370,y+210),(370,y+65),(398,y+65)],color,4)
    for i in f['exchange_blocks']:
        x=start+i*step+40
        s.arrow([(x,375),(x,795)],G,5)
        s.arrow([(x,795),(x,375)],G,5)
        s.cells(x-23,545,1,G,w=46,h=70,phase=i)
    s.text(1835,520,'Hidden states',G,bold=True)
    s.text(1835,572,'Simultaneous',G)
    s.flame(1840,640); s.text(1880,653,'Trained link',G)
    s.snow(650,190); s.text(690,203,'Frozen originals')
    s.path('M670,220 L670,250 L618,250 L618,350',INK,3)
    s.flame(1110,190); s.text(1150,203,'Trained adapters',G)
    s.path('M1180,220 L1180,260 L1194,260 L1194,405',G,3)
    s.save('link_mechanism_depths')


def mechanism(f):
    s=SVG(2300,1320,'A gated read adds partner states')
    s.rect(60,140,2180,1110,CREAM,r=36)
    s.text(100,280,'Ember hidden states',E,bold=True)
    s.cells(100,330,4,E)
    s.arrow([(430,375),(535,375)],E)
    s.box(550,315,225,120,'Normalise')
    s.arrow([(775,375),(820,375),(820,225),(890,225)],E)
    s.arrow([(820,375),(820,545),(890,545)],E)
    s.mapping(900,160,'Value map')
    s.mapping(900,480,'Key map')
    s.arrow([(1090,225),(1180,225)],E)
    s.cells(1200,180,4,E,phase=3)
    s.arrow([(1090,545),(1180,545)],E)
    s.cells(1200,500,4,E,phase=5)
    s.rect(1545,500,165,90,'white',G,10)
    s.text(1627,557,'Nothing',anchor='middle')
    s.path('M1200,602 L1710,602 M1455,602 L1455,630',G,4)
    s.arrow([(1455,630),(1455,650)],G)
    for j in range(4):
        x=1234+j*82
        s.arrow([(x,280),(x,323)],E,3)
        s.dial(x,355,30)
        s.arrow([(x,387),(x,455)],E,3)
    s.text(1720,280,'Sender gates',anchor='middle')
    s.path('M1590,290 L1510,340',G,3)
    s.path('M1200,465 L1990,465',E,4)
    s.arrow([(1990,465),(1990,650)],E)
    s.cells(1585,350,1,G,w=85,h=90,empty=True)
    s.text(1800,355,'Empty value',anchor='middle')
    s.path('M1627,500 L1627,440',G,3)
    s.arrow([(1670,395),(1740,395),(1740,465)],G,3)
    # Sender-state control is separate from the learned key/value normalizer.
    s.path('M450,375 L450,485 L550,485',E,3)
    s.box(550,455,225,120,'Gate map')
    s.path('M775,515 L800,515 L800,455 L1130,455 L1130,310 L1500,310',G,3)
    # White overpasses distinguish control wiring from the value path.
    for j in range(4):
        x=1234+j*82
        s.path(f'M{x-6},310 L{x+6},310',CREAM,8)
        s.path(f'M{x},302 L{x},318',E,3)
    s.path('M814,455 L826,455',CREAM,8)
    s.path('M820,447 L820,463',E,3)
    s.path('M1621,465 L1633,465',CREAM,8)
    s.path('M1627,457 L1627,473',G,3)
    for j in range(4):
        x=1234+j*82
        s.path(f'M{x-39},310 L{x-39},355 L{x-31},355',G,3)
    # Receiver query and the unmodified residual bypass branch at the source.
    s.text(100,790,'Quill hidden state',Q,bold=True)
    s.cells(220,835,1,Q,w=90,h=105)
    s.arrow([(310,887),(535,887)],Q)
    s.box(550,827,225,120,'Normalise')
    s.arrow([(775,887),(890,887)],Q)
    s.mapping(900,820,'Query map')
    s.arrow([(1090,887),(1140,887),(1140,730),(1190,730)],Q)
    # Attention chips are schematic, with no fabricated probability values.
    s.rect(1200,650,510,170,'white',G)
    for j in range(5):
        s.rect(1230+j*92,677,72,65,CREAM if j==2 else 'white',G,10,6 if j==2 else 2)
    s.text(1455,792,'Attention',anchor='middle',bold=True)
    s.arrow([(1710,730),(1880,730)],G)
    # A funnel makes the reduction visible, not just a named rectangle.
    s.path('M1900,650 L2080,650 L2020,800 L1960,800 Z',G,4,'white')
    for j in range(4): s.path(f'M{1920+j*45},670 L{1980+j*7},777',E,[2,3,8,2][j])
    s.text(1990,865,'Weighted sum',anchor='middle')
    s.arrow([(1990,885),(1990,990)],G)
    s.mapping(1895,1000,'Output map')
    s.arrow([(1895,1065),(1725,1065)],G)
    s.dial(1670,1065,48)
    s.text(1670,985,'Reader gate',anchor='middle')
    s.arrow([(1615,1065),(1460,1065),(1460,1110)],G)
    s.arrow([(265,940),(265,1170),(1395,1170)],Q,8)
    s.circle(1460,1170,56,'white',G)
    s.text(1460,1183,'Add',anchor='middle',bold=True)
    s.arrow([(1517,1170),(2135,1170)],Q,8)
    s.cells(2150,1125,1,Q,w=60,h=90)
    s.text(860,1120,'Quill residual stream',Q,anchor='middle')
    s.flame(110,180); s.text(150,193,'Trained link',G,bold=True)
    s.save('link_mechanism')


def ticks(f):
    s=SVG(2300,1080,'A token reads earlier and current partner states')
    s.rect(60,140,2180,870,GREY,r=36)
    ticks=[v+1 for v in f['timing_emission_ticks']]
    xs=[880,1210,1540,1870]
    current=ticks[-2]
    s.text(110,230,'Read tick',bold=True)
    s.text(620,230,'Earlier',anchor='middle')
    for t,x in zip(ticks,xs): s.text(x+80,230,t,anchor='middle')
    s.text(110,370,'Ember',E,bold=True)
    s.text(110,865,'Quill',Q,bold=True)
    s.text(570,295,'Prompt',E)
    s.cells(555,320,2,E,w=70,h=100,gap=12)
    s.text(560,475,'Turn header',E)
    s.cells(555,770,2,Q,w=70,h=100,gap=12,phase=2)
    s.text(570,730,'Prompt',Q)
    s.text(555,935,'Turn header',Q)
    for i,(t,x) in enumerate(zip(ticks,xs)):
        s.box(x-10,900,180,70,f['timing_tokens'][i].strip(),Q)
        s.arrow([(x+80,892),(x+80,878)],Q,3)
        s.cells(x+35,770,1,Q,w=90,h=100,phase=i,empty=t>current)
        if f['timing_listener_chunks'][i]:
            assert len(f['timing_listener_chunks'][i]) == 1
            s.box(x-10,285,180,70,f['timing_listener_chunks'][i][0].strip(),E)
            s.arrow([(x+80,360),(x+80,380)],E,3)
            s.cells(x+35,390,1,E,w=90,h=100,phase=i+2,empty=t>current)
        else:
            s.text(x+80,420,'No token',anchor='middle')
    s.rect(1520,745,230,245,'none',Q,r=20,sw=6)
    s.path('M540,545 L1730,545',G,12)
    for x in (620,1290,1620): s.arrow([(x,505),(x,535)],G,4)
    s.arrow([(1620,555),(1620,730)],G,7)
    s.text(1040,650,'Visible hidden states',G,anchor='middle',bold=True)
    s.path('M1780,260 L1780,980',RED,5)
    s.text(1950,650,'Future',RED,anchor='middle',bold=True)
    s.text(1490,820,'Now',Q,anchor='end')
    s.save('link_mechanism_ticks')


if __name__ == '__main__':
    f=facts()
    (ROOT/'results/link_mechanism.json').write_text(json.dumps(f,indent=2)+'\n')
    depths(f)
    mechanism(f)
    ticks(f)
