"""Plays one six-turn co_play game live from a checkpoint, printing each turn as it is generated, and saves the turns
with their link reads as HTML and JSON.

Real models load as 4-bit MLX (mlx_eval.load_pair); --backend torch is for tiny test models (NL_TINY=1). The Observer
wraps Pair.step and Pair._exchange: it reads each sampled token id when the next session.step consumes it and never
changes sampling or scheduling. Each token is annotated with the reads at the position that predicted it: the link
write's size relative to the residual stream (RMS of h_after - h_before over RMS of h_before, averaged over exchange
blocks), the partner's gate weighted by attention, the null key's mass, and the three partner positions with the most
attention. The HTML shades each token by the relative write size, a descriptive measure, not a causal one.

Usage: python play_live.py --run STEP_DIR --replies REPLIES.json --out OUT.html [--game guess] [--opener quill]
           [--cut | --shuffle] [--quill-secret S] [--ember-secret S] [--temperature 0] [--seed 4242]
"""
import argparse
import contextlib
import html
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('NL_DEVICE', 'cpu')
import torch

import coread as C
import games as G
import session as S
from link import SIDES, partner


MEASURE = ('Mean across exchange blocks of RMS(h_after - h_before) / RMS(h_before), '
           'at the position predicting this token. Includes residual dtype rounding. '
           'Attention masses are averaged across heads and blocks, with null mass retained; '
           'sender gate is attention-weighted over non-null partner positions. '
           'These are descriptive reads, not causal attribution or KL.')


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def distinct_secrets(a, b):
    """Whether the secrets a and b ({side: {field: value}}) share no value in any field, ignoring case and spaces
    (a concept's singular and plural both count)."""
    def forms(field, value):
        return {v.strip().casefold() for v in (value if field == 'concept' else [value])}
    return all(forms(f, a[x][f]).isdisjoint(forms(f, b[x][f])) for x in SIDES for f in a[x])


def episode(game, secrets, opener='quill'):
    """A co_play episode of game with six free turns, opener first; each twin's prompt has only its own secret."""
    if game not in C.PLAY or opener not in SIDES:
        raise ValueError('unknown game or opener')
    for x in SIDES:
        if set(secrets[x]) != set(C.PLAY[game][0]):
            raise ValueError(f'{game} requires fields {C.PLAY[game][0]} for {x}')
        for field, value in secrets[x].items():
            forms = value if field == 'concept' else [value]
            if (field == 'concept' and (not isinstance(value, list) or len(value) != 2)) or not all(
                    isinstance(v, str) and v.strip() and '<|' not in v for v in forms):
                raise ValueError('secret values must be nonempty text; concept is [singular, plural]')
    systems = {x: C.play_system(game, x, opener, secrets) for x in SIDES}
    for x in SIDES:
        for field, value in secrets[partner(x)].items():
            if C.play_spans({field: value}, systems[x]):
                raise ValueError(f"partner secret already occurs in {x}'s prompt; choose distinct secrets")
    return dict(game='co_play', mode='dialogue', system=systems, task={x: None for x in SIDES},
                turns=[(opener if j % 2 == 0 else partner(opener), '', 1.) for j in range(6)],
                meta=dict(play=game, secrets=secrets, opener=opener))


def draw_episode(game, seed, opener='quill', exclude=None, overrides=None):
    """An episode with secrets drawn from the test pools (overrides: given secrets per side), sharing no value with
    exclude."""
    rng = random.Random(seed)
    pools = {k: v['test'] for k, v in G.POOLS.items()}
    for _ in range(1000):
        secrets = {x: C.play_secret(game, pools, rng) for x in SIDES}
        secrets.update(overrides or {})
        if exclude and not distinct_secrets(secrets, exclude):
            continue
        try:
            return episode(game, secrets, opener)
        except ValueError:
            if overrides and len(overrides) == 2:
                raise
    raise ValueError('could not draw distinct secrets absent from the other prompt')


def parse_secret(game, text):
    """A secret from the command line: a JSON object, or plain text for a single-field game."""
    if text.startswith('{'):
        return json.loads(text)
    fields = C.PLAY[game][0]
    if len(fields) != 1:
        raise ValueError('story secrets require JSON, e.g. {"concept":["otter","otters"],"name":"Ada"}')
    return {fields[0]: [text, text] if fields[0] == 'concept' else text}


def cpu(t):
    """A float32 CPU torch tensor of a torch or MLX array."""
    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu()
    from mlx_eval import tensor
    return tensor(t).float()


def token_pieces(tok, ids, text):
    """Each token's piece of text, so that the pieces join to text exactly; a token that ends inside a UTF-8
    character gets an empty piece until the character completes."""
    pieces, previous = [], ''
    for j in range(len(ids)):
        now = tok.decode(ids[:j + 1])
        if j + 1 < len(ids):
            now = now.rstrip('\ufffd')
        common = len(os.path.commonprefix([previous, now]))
        # A tokenizer may revise whitespace or an unfinished byte sequence.
        trim = len(previous) - common
        for k in range(len(pieces) - 1, -1, -1):
            n = min(trim, len(pieces[k]))
            if n:
                pieces[k] = pieces[k][:-n]
                trim -= n
        pieces.append(now[common:])
        previous = now
    if ''.join(pieces) != text:
        raise ValueError('decoded token pieces do not reconstruct the generated text')
    return pieces


class Observer:
    """Streams the first episode's turns to stream and records each token's reads (installed() wraps Pair.step and
    Pair._exchange)."""

    def __init__(self, pair, eps, mode, stream):
        self.pair, self.eps, self.mode, self.stream = pair, eps, mode, stream
        self.turn, self.tick = 0, -1
        self.tokens = [[] for _ in eps[0]['turns']]
        self.metrics = [[] for _ in eps[0]['turns']]
        self.printed = ''
        self.started = False
        self.seq = {x: [[] for _ in eps] for x in SIDES}
        self.times = {x: [[] for _ in eps] for x in SIDES}
        self.gates = {}
        self.layers = []

    def write(self, text):
        if self.stream is not None:
            self.stream.write(text)
            self.stream.flush()

    def begin(self):
        if not self.started and self.turn < 6:
            x = self.eps[0]['turns'][self.turn][0]
            color = '\033[36m' if x == 'quill' else '\033[35m'
            self.write(f'\n{color}{x.title()}\033[0m: ')
            self.started = True

    def emit(self, final=False):
        x = self.eps[0]['turns'][self.turn][0]
        text = self.pair.toks[x].decode(self.tokens[self.turn])
        if not final:
            text = text.rstrip('\ufffd')
        if not text.startswith(self.printed):
            # Terminal correction for tokenizers that revise decoded whitespace.
            n = len(os.path.commonprefix([text, self.printed]))
            self.write('\b \b' * (len(self.printed) - n))
            self.printed = self.printed[:n]
        self.write(text[len(self.printed):])
        self.printed = text

    def finish_turn(self):
        self.begin()
        self.emit(final=True)
        x = self.eps[0]['turns'][self.turn][0]
        met = self.metrics[self.turn][:len(self.tokens[self.turn])]
        named = C.play_spans(self.eps[0]['meta']['secrets'][partner(x)], self.printed) is not None
        self.write(f'\n  read/residual {mean([m["effect"] for m in met]):.4f}; '
                   f'sender gate {mean([m["sender_gate"] for m in met]):.3f}; '
                   f"named partner secret: {'yes' if named else 'no'}\n")
        self.turn += 1
        self.printed, self.started = '', False

    def exchange(self, k, h, *args, **kwargs):
        out, gates = self.original_exchange(k, h, *args, **kwargs)
        block = self.pair.xblocks[k]
        # Keep every sequence and padding column: they index the KV cache.
        for x, g in gates.items():
            key = (x, block)
            g = cpu(g)
            self.gates[key] = g if key not in self.gates else torch.cat([self.gates[key], g], 1)
        if self.tick >= 0 and self.turn < 6:
            x = self.eps[0]['turns'][self.turn][0]
            if x in h and self.valid.get(x):
                q = self.valid[x][-1]
                before, after = cpu(h[x][0, q]), cpu(out[x][0, q])
                ratio = float((after - before).norm() / before.norm().clamp_min(1e-12))
                r = self.pair.record[(x, block)][-1]
                p = r['attn'][0, :, q].float().mean(0)
                y, row = partner(x), 1 if self.mode == 'shuffle' else 0
                g = self.gates[y, block][row]
                weighted = float((p[1:] * g).sum() / p[1:].sum().clamp_min(1e-12))
                self.layers.append((ratio, p, weighted))
        if self.pair.record is not None:
            self.pair.record.clear()  # never retain full prompt/turn attention maps
        return out, gates

    def step(self, chunk, cache, qmask, tick, *args, **kwargs):
        self.tick = tick
        self.valid = {x: torch.where(v[1][0, -v[0].shape[1]:].bool())[0].tolist() for x, v in chunk.items()}
        if tick >= 0 and self.turn < 6:
            x = self.eps[0]['turns'][self.turn][0]
            if self.started and self.valid.get(x):
                ids = chunk[x][0][0, self.valid[x]].tolist()
                if ids[0] == self.pair.toks[x].convert_tokens_to_ids('<|im_end|>'):
                    self.finish_turn()
                else:
                    if len(ids) != 1:
                        raise ValueError('live observer expects unpaused, free dialogue turns')
                    self.tokens[self.turn].append(ids[0])
                    self.emit()
            if self.turn < 6:
                self.begin()
        for x, (ids, _, _) in chunk.items():
            for i, row in enumerate(ids.tolist()):
                self.seq[x][i].extend(row)
                self.times[x][i].extend([tick] * len(row))
        self.layers = []
        self.pair.record = {} if tick >= 0 else None
        result = self.original_step(chunk, cache, qmask, tick, *args, **kwargs)
        if tick >= 0 and self.turn < 6:
            x = self.eps[0]['turns'][self.turn][0]
            if self.valid.get(x):
                top, null = [], 0.0
                if self.layers:
                    p = torch.stack([v[1] for v in self.layers]).mean(0)
                    null = float(p[0])
                    y, row = partner(x), 1 if self.mode == 'shuffle' else 0
                    for j in torch.argsort(p[1:], descending=True)[:3].tolist():
                        if p[j + 1] > 0:
                            tid = self.seq[y][row][j]
                            top.append(dict(side=y, episode=row, position=j, tick=self.times[y][row][j],
                                            id=tid, text=self.pair.toks[y].decode([tid]), mass=float(p[j + 1])))
                self.metrics[self.turn].append(dict(tick=tick, effect=mean([v[0] for v in self.layers]),
                                                   sender_gate=mean([v[2] for v in self.layers]),
                                                   null_mass=null, top=top))
        return result

    @contextlib.contextmanager
    def installed(self):
        self.original_step, self.original_exchange = self.pair.step, self.pair._exchange
        old = {k: self.pair.__dict__.get(k) for k in ('step', '_exchange')}
        present = {k: k in self.pair.__dict__ for k in old}
        record = self.pair.record
        self.pair.step, self.pair._exchange = self.step, self.exchange
        try:
            yield self
        finally:
            for k, v in old.items():
                if present[k]:
                    setattr(self.pair, k, v)
                else:
                    delattr(self.pair, k)
            self.pair.record = record


def play(pair, ep, *, condition='link', donor=None, max_turn=64, temperature=0., seed=0, stream=sys.stdout):
    """Generates ep under condition ('link', 'cut', or 'shuffle' with donor, an episode of the same game and opener
    with other secrets) and returns its turns with each token's reads."""
    if condition not in ('link', 'cut', 'shuffle') or max_turn < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('invalid condition, token limit or temperature')
    if ep['mode'] != 'dialogue' or len(ep['turns']) != 6 or any(ep.get(k) for k in ('pause', 'pause_turns', 'forced')):
        raise ValueError('live play requires six unpaused, free dialogue turns')
    eps = [ep]
    if condition == 'shuffle':
        if donor is None or donor['meta']['play'] != ep['meta']['play'] or donor['meta']['opener'] != ep['meta']['opener']:
            raise ValueError('shuffle needs a same-game, same-opener donor')
        if not distinct_secrets(donor['meta']['secrets'], ep['meta']['secrets']):
            raise ValueError('shuffle donor must hold other secrets')
        eps.append(donor)
    mode = {'link': None, 'cut': 'zero', 'shuffle': 'shuffle'}[condition]
    obs = Observer(pair, eps, mode, stream)
    start = time.monotonic()
    with obs.installed():
        raw = S.generate(pair, eps, mode=mode, max_turn=max_turn, temperature=temperature,
                         gen=torch.Generator().manual_seed(seed), keep_tokens=True)
    result = raw[0]
    turns = []
    for j, ((x, text), ids, sampled) in enumerate(zip(result['turns'], result['tokens'], result['sampled'])):
        if obs.tokens[j] != ids[:-1] or len(obs.metrics[j]) != len(ids):
            raise ValueError('observer IDs or prediction metrics disagree with session.generate')
        pieces = token_pieces(pair.toks[x], ids[:-1], text)
        spans = sorted({span for f, v in ep['meta']['secrets'][partner(x)].items()
                        for span in (C.play_spans({f: v}, text) or [])})
        met = obs.metrics[j][:-1]
        turns.append(dict(speaker=x, text=text, named=C.play_spans(ep['meta']['secrets'][partner(x)], text) is not None,
                          naming_spans=spans, mean_effect=mean([m['effect'] for m in met]),
                          mean_sender_gate=mean([m['sender_gate'] for m in met]), truncated=not sampled[-1],
                          tokens=[dict(id=t, text=p, sampled=s, **m) for t, p, s, m in zip(ids, pieces, sampled, met)],
                          end_token=dict(id=ids[-1], sampled=sampled[-1], **obs.metrics[j][-1])))
    while obs.turn < len(turns):
        obs.finish_turn()
    return dict(schema=1, condition=condition, episode=ep, donor=donor if condition == 'shuffle' else None,
                measure=MEASURE, seed=seed, temperature=temperature, max_turn=max_turn,
                elapsed_seconds=time.monotonic() - start, turns=turns, raw=raw,
                note='Dialogue text remains shared in cut and shuffle. Later naming may copy earlier text.')


def render(data):
    """The HTML page of play's result."""
    esc = html.escape
    bubbles = []
    for turn in data['turns']:
        pos, body = 0, []
        for token in turn['tokens']:
            title = f"write/residual {token['effect']:.5f}; sender gate {token['sender_gate']:.3f}; null {token['null_mass']:.3f}"
            for r in token['top']:
                title += f"\n{r['side']} episode {r['episode']} position {r['position']} tick {r['tick']}: {r['text']!r} — {r['mass']:.2%}"
            text = token['text']
            # Split only at naming boundaries so the exact matching span is underlined.
            cuts = sorted({0, len(text)} | {v - pos for ab in turn['naming_spans'] for v in ab if pos < v < pos + len(text)})
            parts = []
            for a, b in zip(cuts, cuts[1:]):
                part = esc(text[a:b])
                if any(pos + a < e and pos + b > s for s, e in turn['naming_spans']):
                    part = '<u>' + part + '</u>'
                parts.append(part)
            alpha = min(.85, token['effect'] / .25 * .85)
            body.append(f'<span class="token" title="{esc(title, quote=True)}" style="background:rgba(245,165,50,{alpha:.4f})">'
                        + ''.join(parts) + '</span>')
            pos += len(text)
        summary = (f"read/residual {turn['mean_effect']:.4f} · sender gate {turn['mean_sender_gate']:.3f} · "
                   f"named partner secret: {'yes' if turn['named'] else 'no'}" + (' · token limit reached' if turn['truncated'] else ''))
        bubbles.append(f'<article class="{turn["speaker"]}"><h2>{turn["speaker"].title()}</h2>'
                       f'<div class="message">{"".join(body)}</div><p class="stats">{summary}</p></article>')
    secrets = ' · '.join(f'{x.title()}: {esc(json.dumps(data["episode"]["meta"]["secrets"][x], ensure_ascii=False))}' for x in SIDES)
    donor = '' if not data.get('donor') else '<p>Shuffled donor secrets: ' + esc(json.dumps(data['donor']['meta']['secrets'])) + '</p>'
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><title>Quill &amp; Ember — live play</title>'
            '<style>body{max-width:900px;margin:40px auto;padding:0 20px;background:#10151e;color:#e7edf6;font:17px/1.6 system-ui}'
            'h1{font-size:30px}h2{font-size:16px;margin:0 0 12px}.legend,.stats{font-size:13px;color:#b6c2d4}'
            'article{border-radius:20px;padding:20px;margin:22px 0;max-width:82%;background:#1c293b}'
            '.quill{margin-right:auto;border-left:4px solid #63d6ed}.ember{margin-left:auto;border-right:4px solid #e69bd8}'
            '.quill h2{color:#63d6ed}.ember h2{color:#e69bd8}.message{white-space:pre-wrap;overflow-wrap:anywhere}'
            '.token{border-radius:2px}u{text-decoration-thickness:2px;text-underline-offset:3px}'
            '@media(max-width:600px){article{max-width:100%}}</style></head><body>'
            f'<h1>Quill &amp; Ember · {esc(data["episode"]["meta"]["play"])} · {esc(data["condition"])}</h1>'
            f'<p>{secrets}</p>{donor}<p class="legend">{esc(MEASURE)} Shading: clear = 0, full amber = 0.25 or more. '
            'Hover for top 3 partner tokens and attention mass. Underlines mark named secret fields.</p>'
            f'<p class="legend">{esc(data["note"])} Checkpoint: {esc(str(data.get("checkpoint", "tiny test")))}</p>'
            + ''.join(bubbles) + '</body></html>\n')


def save(data, output):
    """Write data as output with suffix .json and its HTML as .html; returns the HTML path."""
    path = Path(output).with_suffix('.html')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    path.write_text(render(data))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True, help='training replies file, retained as provenance (never forced)')
    ap.add_argument('--out', required=True, help='HTML filename; JSON is written beside it')
    ap.add_argument('--game', choices=tuple(C.PLAY), default='guess')
    ap.add_argument('--opener', choices=SIDES, default='quill')
    for x in SIDES:
        ap.add_argument('--' + x + '-secret', help='plain text or a JSON secret object; default: held-out test pool')
    cond = ap.add_mutually_exclusive_group()
    cond.add_argument('--cut', action='store_true')
    cond.add_argument('--shuffle', action='store_true')
    ap.add_argument('--temperature', type=float, default=0.)
    ap.add_argument('--max-turn', type=int, default=64)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    args = ap.parse_args()
    if args.backend == 'torch' and os.environ.get('NL_TINY') != '1':
        ap.error('torch loading is allowed only with NL_TINY=1')
    if args.max_turn < 1 or not math.isfinite(args.temperature) or args.temperature < 0:
        ap.error('max-turn must be positive and temperature finite and nonnegative')
    torch.set_num_threads(2)
    replies = Path(args.replies).resolve()
    json.loads(replies.read_text())
    overrides = {x: parse_secret(args.game, getattr(args, x + '_secret')) for x in SIDES if getattr(args, x + '_secret')}
    ep = draw_episode(args.game, args.seed, args.opener, overrides=overrides)
    donor = draw_episode(args.game, args.seed + 1, args.opener, exclude=ep['meta']['secrets']) if args.shuffle else None
    condition = 'cut' if args.cut else 'shuffle' if args.shuffle else 'link'
    print(f'{args.game} · {condition} · secrets {json.dumps(ep["meta"]["secrets"])}', flush=True)
    if donor:
        print('Shuffle donor secrets: ' + json.dumps(donor['meta']['secrets']), flush=True)
    if args.backend == 'mlx':
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(10 * 2**30)
        mx.set_cache_limit(128 * 2**20)
    else:
        from evaluate import load_pair
    pair, _ = load_pair(args.run)
    data = play(pair, ep, condition=condition, donor=donor, max_turn=args.max_turn,
                temperature=args.temperature, seed=args.seed)
    data.update(checkpoint=str(Path(args.run).resolve()), replies=str(replies), backend=args.backend)
    path = save(data, args.out)
    print(f'\nSaved {path} and {path.with_suffix(".json")}', flush=True)


if __name__ == '__main__':
    main()
