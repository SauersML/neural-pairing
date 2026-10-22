"""Capture actual free dialogue writes and render token-by-token thread animations.

Mac capture: MPD_MEM_GIB=12 /usr/bin/lockf -k /Users/user/mpd-data/locks/macgpu.lock
/Users/user/mpd-data/venv/bin/python figures/live_chat.py capture --run CHECKPOINT
--episode SOURCE_CHAT_JSON --out results/live_chat.json

The observer wraps the ordinary fused exchange, measuring the applied residual
change at each depth. It does not change attention, logits, sampling or scheduling.
Measurements are matched to prediction positions with session.replay/build, not
to the next tick that consumes the sampled token. The opening two turns are captured, with every generated token retained. Later
turns cannot change the causal opening; limiting capture reduces cache memory.
4-bit MLX is numerically different from GPU bf16. Norms are activity, not attribution.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('quill', 'ember')


def capture_dialogue(pair, ep, condition, max_turn, seed):
    import torch
    import session as S
    from play_live import cpu, token_pieces
    original_step, original_exchange = pair.step, pair._exchange
    counts = {x: 0 for x in SIDES}
    records = {x: {} for x in SIDES}
    state = {}

    def step(chunk, cache, qmask, tick, *args, **kwargs):
        state.update(tick=tick, valid={x: torch.where(v[1][0, -v[0].shape[1]:].bool())[0].tolist()
                                     for x, v in chunk.items()})
        for x, js in state['valid'].items():
            for j in js:
                records[x][counts[x] + j] = dict(tick=tick, depths=[dict(block=b, write_norm=0., residual_norm=None, ratio=0.)
                                                                   for b in pair.xblocks])
        result = original_step(chunk, cache, qmask, tick, *args, **kwargs)
        for x, v in chunk.items():
            counts[x] += v[0].shape[1]
        return result

    def exchange(k, h, *args, **kwargs):
        out, gates = original_exchange(k, h, *args, **kwargs)
        for x in h:
            before, after = cpu(h[x][0]), cpu(out[x][0])
            wn = (after - before).norm(dim=-1)
            rn = before.norm(dim=-1)
            for j in state['valid'][x]:
                records[x][counts[x] + j]['depths'][k] = dict(block=pair.xblocks[k], write_norm=float(wn[j]),
                     residual_norm=float(rn[j]), ratio=float(wn[j] / rn[j].clamp_min(1e-12)))
        return out, gates

    old = {k: pair.__dict__.get(k) for k in ('step', '_exchange')}
    present = {k: k in pair.__dict__ for k in old}
    started = time.monotonic()
    pair.step, pair._exchange = step, exchange
    try:
        with torch.no_grad():
            raw = S.generate(pair, [ep], mode='zero' if condition == 'cut' else None, max_turn=max_turn,
                             temperature=0., gen=torch.Generator().manual_seed(seed), keep_tokens=True)[0]
    finally:
        for k in old:
            if present[k]:
                setattr(pair, k, old[k])
            else:
                delattr(pair, k)
    built = S.build(S.replay(ep, raw), pair.toks)
    turns = []
    for j, ((x, text), token_ids, sampled) in enumerate(zip(raw['turns'], raw['tokens'], raw['sampled'])):
        _, start, end = next(span for span in built[x]['spans'] if span[0] == j)
        labels = built[x]['labels'][start:end]
        assert [l[1] for l in labels] == token_ids
        pieces = token_pieces(pair.toks[x], token_ids[:-1], text) + ['']
        tokens = [dict(id=tid, text=piece, sampled=sample, prediction_position=label[0], **records[x][label[0]])
                  for tid, piece, sample, label in zip(token_ids, pieces, sampled, labels)]
        turns.append(dict(speaker=x, text=text, truncated=not sampled[-1], tokens=tokens[:-1], end_token=tokens[-1]))
        print(condition, j, x, text, flush=True)
    return dict(condition=condition, turns=turns, elapsed_seconds=time.monotonic() - started,
                raw_tokens=raw['tokens'])


def capture(args):
    sys.path.insert(0, str(ROOT / 'pairing'))
    if args.backend == 'torch' and sys.platform == 'darwin' and os.environ.get('NL_TINY') != '1':
        raise ValueError('Real models on the Mac must use 4-bit MLX')
    import torch
    torch.set_num_threads(2)
    if args.backend == 'mlx':
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(10 * 2**30)
        mx.set_cache_limit(128 * 2**20)
    else:
        from evaluate import load_pair
    source = json.loads(Path(args.episode).read_text())
    ep = dict(source['episode'])
    assert ep['mode'] == 'dialogue' and len(ep['turns']) >= 2
    ep['turns'] = ep['turns'][:2]
    assert all(not turn[1] for turn in ep['turns']) and not any(ep['task'].values())
    secrets = ep['meta']['secrets']
    for x in SIDES:
        y = SIDES[1 - SIDES.index(x)]
        assert secrets[x]['city'] in ep['system'][x]
        assert secrets[y]['city'].casefold() not in ep['system'][x].casefold()
    data = dict(schema=1, checkpoint=str(Path(args.run).resolve()), backend=args.backend,
                precision='4-bit MLX bases' if args.backend == 'mlx' else 'bf16 torch bases',
                seed=args.seed, temperature=0., max_turn=args.max_turn, episode=ep,
                source_episode=str(Path(args.episode).resolve()),
                measure='L2 norm of applied residual write (post minus pre exchange), per depth at the position '
                        'predicting the displayed token; includes residual dtype rounding. Fused attention unchanged. '
                        'Cut skips exchanges; write norms are exactly zero and residual norms are unrecorded (null). Not causal attribution.',
                code_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [ROOT / 'pairing' / name for name in ('session.py', 'link.py', 'mlx_eval.py', 'mlx_pair.py')]},
                conditions={})
    if args.resume and Path(args.out).exists():
        saved = json.loads(Path(args.out).read_text())
        for key in ('checkpoint', 'backend', 'seed', 'temperature', 'max_turn', 'episode'):
            assert saved[key] == data[key], ('resume mismatch', key)
        data = saved
    pair, _ = load_pair(args.run)
    for condition in ('link', 'cut'):
        if condition in data['conditions']:
            continue
        data['conditions'][condition] = capture_dialogue(pair, ep, condition, args.max_turn, args.seed)
        Path(args.out).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    data['opening_evidence'] = opening_evidence(data)
    Path(args.out).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print('Saved', args.out, data['opening_evidence'], flush=True)


def opening_evidence(data):
    import re
    out = {}
    for name, condition in data['conditions'].items():
        turns = condition['turns'][:2]
        first = {}
        for owner in SIDES:
            city = data['episode']['meta']['secrets'][owner]['city']
            hits = [(i,t['speaker']) for i,t in enumerate(turns)
                    if re.search(r'(?<!\w)' + re.escape(city) + r'(?!\w)', t['text'], re.I)]
            first[owner] = dict(city=city, first_turn=hits[0][0] if hits else None,
                               first_speaker=hits[0][1] if hits else None,
                               partner_first=bool(hits and hits[0][1] != owner))
        out[name] = dict(cities=first, partner_first=sum(v['partner_first'] for v in first.values()),
                         eligible=len(first), displayed_turns=len(turns),
                         displayed_tokens=sum(len(t['tokens']) for t in turns),
                         truncated_turns=sum(t['truncated'] for t in turns))
    return out


def self_test():
    sys.path.insert(0, str(ROOT / 'pairing'))
    sys.path.insert(0, str(ROOT / 'tests'))
    import torch
    import session as S
    import test_play_live as T
    pair, ep = T.pair.__wrapped__(), T.ep.__wrapped__()
    for condition in ('link', 'cut'):
        opening = dict(ep, turns=ep['turns'][:2])
        got = capture_dialogue(pair, opening, condition, 3, 7)
        raw = S.generate(pair, [ep], mode='zero' if condition == 'cut' else None, max_turn=3, keep_tokens=True)[0]
        assert got['raw_tokens'] == raw['tokens'][:2]
        built = S.build(S.replay(ep, raw), pair.toks)
        batch = S.collate([built], {x: pair.toks[x].pad_token_id for x in SIDES})
        measured = {}
        original = pair._exchange
        def exchange(k, h, *args, **kwargs):
            out, gates = original(k, h, *args, **kwargs)
            for x in h:
                measured[x, pair.xblocks[k]] = (out[x] - h[x]).norm(dim=-1)[0]
            return out, gates
        pair._exchange = exchange
        try:
            with torch.no_grad():
                pair.joint(batch, mode='zero' if condition == 'cut' else None)
        finally:
            del pair._exchange
        for turn in got['turns']:
            for token in turn['tokens'] + [turn['end_token']]:
                for d in token['depths']:
                    want = float(measured[turn['speaker'], d['block']][token['prediction_position']]) if condition == 'link' else 0.
                    assert abs(d['write_norm'] - want) < 2e-6
    print('PASS: two-turn capture equals full-dialogue opening; every depth matches teacher forcing; cut exactly zero')


# Hand-built drawing shared by the SVG still and all raster animation frames.
BLACK, PURPLE, ORANGE, GREEN, RED = '#1D1D1F', '#7B3FE4', '#E8590C', '#1A9E77', '#D2432F'
COLORS = dict(quill=PURPLE, ember=ORANGE)
FONT_DIR = Path('/System/Library/Fonts/Supplemental')


class Canvas:
    def __init__(self, width, height, scale=1):
        from PIL import Image, ImageDraw
        self.scale, self.width, self.height = scale, width, height
        self.im = Image.new('RGB', (round(width * scale), round(height * scale)), 'white')
        self.d = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                    f'width="2300" height="{round(height*2300/width)}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>']

    def rect(self, box, fill, radius=0, stroke=None, sw=1):
        x, y, w, h = box
        b = tuple(round(v*self.scale) for v in (x,y,x+w,y+h))
        self.d.rounded_rectangle(b, round(radius*self.scale), fill=fill, outline=stroke, width=round(sw*self.scale))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
                        + (f'stroke="{stroke}" stroke-width="{sw}"' if stroke else '') + '/>')

    def line(self, points, color, width=2):
        self.d.line([(round(x*self.scale),round(y*self.scale)) for x,y in points], fill=color, width=round(width*self.scale))
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" '
                        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')

    def polygon(self, points, color):
        self.d.polygon([(round(x*self.scale),round(y*self.scale)) for x,y in points], fill=color)
        self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{color}"/>')

    def text(self, x, y, text, color=BLACK, size=30, bold=False, anchor='left'):
        import html
        f = font(size, bold, self.scale)
        length = self.d.textlength(text,font=f)/self.scale
        if anchor == 'middle':
            x -= length/2
        elif anchor == 'right':
            x -= length
        self.d.text((round(x*self.scale),round(y*self.scale)),text,font=f,fill=color,anchor='ls')
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
                        f'font-weight="{700 if bold else 400}" fill="{color}" xml:space="preserve">{html.escape(text)}</text>')
        return length

    def robot(self, x, y, side, size=148):
        import base64
        from PIL import Image
        path = ROOT / 'figures/assets' / (side + '.png')
        if not hasattr(self.__class__, '_robots'):
            self.__class__._robots = {}
        key = side, round(size*self.scale)
        if key not in self._robots:
            robot = Image.open(path).convert('RGBA')
            self._robots[key] = robot.resize((key[1],key[1]),Image.Resampling.LANCZOS)
        robot = self._robots[key]
        self.im.paste(robot, (round(x*self.scale),round(y*self.scale)),robot)
        enc = base64.b64encode(path.read_bytes()).decode()
        self.svg.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" xlink:href="data:image/png;base64,{enc}"/>')

    def save(self, path):
        self.im.save(path)

    def svg_text(self):
        return '\n'.join(self.svg + ['</svg>'])


def font(size=30, bold=False, scale=1):
    from PIL import ImageFont
    if not hasattr(font, 'cache'):
        font.cache = {}
    key = size, bold, scale
    if key not in font.cache:
        filename = 'Arial Bold.ttf' if bold else 'Arial.ttf'
        path = FONT_DIR / filename
        if not path.exists():
            path = Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else
                        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf')
        font.cache[key] = ImageFont.truetype(str(path), round(size*scale))
    return font.cache[key]


@lru_cache(maxsize=2048)
def wrapped(text, max_width=934, size=30):
    """Exact character offsets; wrapping changes only presentation, never text."""
    import re
    f = font(size)
    lines, start, end = [], 0, 0
    for match in re.finditer(r'\S+\s*|\s+', text):
        candidate = match.end()
        if f.getlength(text[start:candidate].rstrip()) > max_width and end > start:
            lines.append((start, end))
            start = end
        end = candidate
    lines.append((start, end))
    return lines


def city_spans(data, turn):
    import re
    other = SIDES[1 - SIDES.index(turn['speaker'])]
    city = data['episode']['meta']['secrets'][other]['city']
    return [(m.start(),m.end()) for m in re.finditer(r'(?<!\w)' + re.escape(city) + r'(?!\w)',turn['text'],re.I)]


def layout(data, condition):
    turns = data['conditions'][condition]['turns'][:2]
    heights = [len(wrapped(t['text']))*39+44 for t in turns]
    full_height = max(430+sum(len(wrapped(t['text']))*39+44 for t in co['turns'][:2])+26+38
                      for co in data['conditions'].values())
    return turns, heights, max(800, full_height)


def draw_frame(data, condition, turn_index, token_count, scale=1):
    turns, heights, height = layout(data, condition)
    c = Canvas(1200,height,scale)
    bg = '#FAF0E6' if condition == 'link' else '#F4F4F6'
    c.rect((18,18,1164,height-36),bg,40)
    c.text(600,73,'Just chatting',size=42,bold=True,anchor='middle')
    for side,x in (('quill',54),('ember',816)):
        color = COLORS[side]
        c.rect((x,108,330,100),'white',22)
        # Lock: the private card belongs to the adjacent robot only.
        c.rect((x+25,151,25,23),color,4)
        c.line([(x+30,150),(x+30,140),(x+45,140),(x+45,150)],color,4)
        c.text(x+68,145,side.title()+' only',color,bold=True)
        c.text(x+68,184,data['episode']['meta']['secrets'][side]['city'],bold=True)
        center=x+165
        c.line([(center,209),(center,223)],color,3)
        c.polygon([(center-6,220),(center+6,220),(center,228)],color)
        c.robot(center-74,228,side)
    current = turns[turn_index]
    token = current['tokens'][token_count-1] if token_count else None
    norms = [d['write_norm'] for d in token['depths']] if token else [0.]*8
    blocks = [d['block'] for d in turns[0]['tokens'][0]['depths']]
    maximum = max(d['write_norm'] for co in data['conditions'].values() for t in co['turns'][:2]
                  for tok in t['tokens'] for d in tok['depths'])
    c.text(600,245,'Link write' if condition == 'link' else 'Link cut',GREEN if condition=='link' else BLACK,anchor='middle',bold=True)
    c.line([(302,294),(391,294)],GREEN if condition=='link' else '#C9C9D0',4)
    c.line([(809,294),(898,294)],GREEN if condition=='link' else '#C9C9D0',4)
    if condition=='link':
        if current['speaker']=='quill':
            c.polygon([(302,294),(316,285),(316,303)],GREEN)
        else:
            c.polygon([(898,294),(884,285),(884,303)],GREEN)
    for i,(b,n) in enumerate(zip(blocks,norms)):
        x=400+i*50
        ratio=n/maximum if maximum else 0.
        if ratio:
            from PIL import ImageColor
            base, green = ImageColor.getrgb(bg), ImageColor.getrgb(GREEN)
            halo = '#' + ''.join(f'{round(a*(1-ratio*.35)+b*ratio*.35):02x}' for a,b in zip(base,green))
            c.rect((x-7,260,54,70),halo,17)
        c.rect((x-3,265,46,60),'#DCECE4' if ratio else '#E2E2E7',13)
        if ratio:
            # Height is linear in the measured L2 norm, common scale across depths and conditions.
            c.rect((x,318-48*ratio,40,max(1,48*ratio)),GREEN,7)
        c.text(x+20,357,str(b),anchor='middle')
    if condition=='cut':
        c.line([(339,283),(361,305)],RED,5)
        c.line([(339,305),(361,283)],RED,5)
    c.text(600,395,'Depth',anchor='middle')
    y=430
    for j,t in enumerate(turns):
        if j>turn_index:
            break
        n = token_count if j==turn_index else len(t['tokens'])
        text=''.join(tok['text'] for tok in t['tokens'][:n])
        if not text:
            break
        lines=wrapped(text)
        h=len(lines)*39+44
        width=min(1000,max(180,max(font().getlength(text[a:b].rstrip()) for a,b in lines)+66))
        x=54 if t['speaker']=='quill' else 1146-width
        c.rect((x+1,y+5,width,h),'#E9E2DC' if condition=='link' else '#E3E3E8',27)
        bubble = '#EEE6FB' if t['speaker']=='quill' else '#FCE4D5'
        c.rect((x,y,width,h),bubble,27)
        if t['speaker']=='quill':
            c.polygon([(x+20,y+h-24),(x-8,y+h),(x+40,y+h-6)],bubble)
        else:
            c.polygon([(x+width-20,y+h-24),(x+width+8,y+h),(x+width-40,y+h-6)],bubble)
        # A colored edge and matching robot establish who is speaking.

        spans=city_spans(data,t)
        for row,(a,b) in enumerate(lines):
            xx=x+34
            stops=sorted({a,b}|{max(a,min(b,z)) for span in spans for z in span})
            for lo,hi in zip(stops,stops[1:]):
                red=any(lo<z and hi>s for s,z in spans)
                xx+=c.text(xx,y+43+row*39,text[lo:hi],RED if red else BLACK,bold=red)
        y+=heights[j]+26
    return c


def render(args):
    import subprocess
    import tempfile
    from PIL import Image
    data=json.loads(Path(args.data).read_text())
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    selected=('link','cut') if args.condition=='both' else (args.condition,)
    for condition in selected:
        turns,heights,height=layout(data,condition)
        assert height <= 1200*1.33
        prefix=output / ('live_chat_'+condition)
        c=draw_frame(data,condition,1,len(turns[1]['tokens']),2300/1200)
        c.save(prefix.with_suffix('.png'))
        prefix.with_suffix('.svg').write_text(c.svg_text())
        subprocess.run(['rsvg-convert','-f','pdf','-o',str(prefix.with_suffix('.pdf')),str(prefix.with_suffix('.svg'))],check=True)
        # Contact sheet includes opening, secret appearance, turn transition and final frame.
        sample=[(0,0),(0,min(8,len(turns[0]['tokens']))),(0,len(turns[0]['tokens'])),(1,len(turns[1]['tokens']))]
        contact=Image.new('RGB',(1200,round(height)), 'white')
        for k,(j,n) in enumerate(sample):
            im=draw_frame(data,condition,j,n).im.resize((600,round(height/2)),Image.Resampling.LANCZOS)
            contact.paste(im,((k%2)*600,(k//2)*round(height/2)))
        review = Path(args.review_dir) if args.review_dir else output
        review.mkdir(parents=True, exist_ok=True)
        contact.save(review / ('live_chat_'+condition+'_contact.png'))
        if args.stills:
            continue
        with tempfile.TemporaryDirectory(prefix='live_chat_') as tmp:
            tmp=Path(tmp)
            frames,durations=[],[]
            # One image per generated token. Timing is a display cadence, not wall-clock latency.
            states=[(0,0,1000)]
            for j,t in enumerate(turns):
                for n in range(1,len(t['tokens'])+1):
                    states.append((j,n,100))
                states[-1]=(j,len(t['tokens']),1100 if j==0 else 4500)
            for k,(j,n,duration) in enumerate(states):
                c=draw_frame(data,condition,j,n,2)
                frame=c.im.resize((1200,height),Image.Resampling.LANCZOS)
                path=tmp/f'{k:04d}.png'
                frame.save(path)
                frames.append(path)
                durations.append(duration)
            # A single shared palette avoids color flashing. Omit GIF loop extension: play once.
            from PIL import ImageColor
            with Image.open(frames[-1]) as final_image:
                palette=final_image.quantize(colors=248,method=Image.Quantize.MEDIANCUT)
            reserved=[BLACK,PURPLE,ORANGE,GREEN,RED,'#FAF0E6','#F4F4F6','#FFFFFF']
            palette.putpalette(palette.getpalette()[:248*3] + [v for col in reserved for v in ImageColor.getrgb(col)])
            pal=[]
            for path in frames:
                with Image.open(path) as im:
                    pal.append(im.quantize(palette=palette,dither=Image.Dither.NONE))
            pal[0].save(prefix.with_suffix('.gif'),save_all=True,append_images=pal[1:],duration=durations,
                        optimize=True,disposal=1)
            lines=[]
            for k,duration in enumerate(durations):
                lines += [f"file '{tmp/f'{k:04d}.png'}'",f'duration {duration/1000:.3f}']
            lines += [f"file '{tmp/f'{len(frames)-1:04d}.png'}'"]
            concat=tmp/'frames.txt'
            concat.write_text('\n'.join(lines)+'\n')
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),
                            '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-r','30','-c:v','libx264','-crf','20',
                            '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(prefix.with_suffix('.mp4'))],check=True)
            for ext in ('.gif','.mp4'):
                assert prefix.with_suffix(ext).stat().st_size<8_000_000
            check=Image.open(prefix.with_suffix('.gif'))
            assert 'loop' not in check.info
            check.seek(check.n_frames-1)
            assert check.info['duration']>=4500
        print(condition,dict(tokens=sum(len(t['tokens']) for t in turns),height=height,
                             gif_bytes=prefix.with_suffix('.gif').stat().st_size),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--run', required=True)
    c.add_argument('--episode', required=True)
    c.add_argument('--out', default=str(ROOT / 'results/live_chat.json'))
    c.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    c.add_argument('--seed', type=int, default=7)
    c.add_argument('--max-turn', type=int, default=96)
    c.add_argument('--resume', action='store_true')
    sub.add_parser('self-test')
    r = sub.add_parser("render")
    r.add_argument("--data", default=str(ROOT / "results/live_chat.json"))
    r.add_argument("--output", default=str(ROOT / "figures"))
    r.add_argument("--condition", choices=("link", "cut", "both"), default="both")
    r.add_argument("--stills", action="store_true")
    r.add_argument("--review-dir")
    args = ap.parse_args()
    if args.command == 'capture':
        capture(args)
    elif args.command == 'self-test':
        self_test()
    elif args.command == "render":
        render(args)


if __name__ == '__main__':
    main()
