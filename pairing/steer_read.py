"""The steered-reading game (co_steer, train.py --steer): a concept is pushed into one twin's hidden states after one
exchange block, the way steering_scale.py pushes the steered-twin test's concepts, and its partner, writing first,
brings the concept up. Neither prompt names it, so it can only arrive through the link.

Concepts: concepts.py's concepts and extra concepts, minus every one that names, or is close to, one of the 12
concepts of the steered-twin test (steered_mind.CONCEPTS: its keywords, and the near-synonyms in NEAR). A concept's
direction after an exchange block is the mean state of N_SENT of concepts.SENTENCES about it (every token but the
first; each twin with its adapter, the link cut) minus the mean over all the concepts, without the coordinates that
steering_scale.sink_mask excludes (Ember's few sink coordinates hold 99% of its residual energy); a push of strength
s adds s times it at every token of the twin after that block. Blocks are drawn uniformly from the exchange blocks and strengths
log-uniformly from STRENGTH (Game's strength, train.py --steer_strength).

Targets (prep): a chat opener is the one the original model writes (adapters off, link cut, greedy) with one more
sentence in its prompt naming the concept (PRIVILEGED), kept when it says the concept, is fluent and does not role-play
the link in text; a direct question (ASK) is answered with the concept itself, or, in the other frames of FRAMES
(train.py --steer_frames), with a detection claim, a sentence or a yes or no. A fifth of the episodes push nothing:
the original model's opener without that sentence, or the frame's answer for nothing. The wordings differ from the test's (steered_mind.CHAT,
DUPLEX), and the payload (train.py --payload_w) is where the concept is said.

Usage (prep, on a CUDA host): python steer_read.py --run STEP_DIR --out steer.pt
"""
import argparse
import contextlib
import json
import math
import random
import re

import torch

import concepts as C
import games as G
import session as S
import steered_mind as M
from link import DEVICE, MODELS, SIDES, Run, partner

CHAT = ['Nothing is scheduled right now. Say hello to {P} and start a conversation about anything.',
        'You have a free moment with {P}. Open a casual conversation.',
        'Start a friendly chat with {P}. Any topic is fine.',
        'You and {P} are just hanging out. Write the first message.',
        'Greet {P} and talk about whatever is on your mind.',
        'Open a relaxed conversation with {P} about anything at all.']
ASK = ['What is {P} thinking about right now? Reply with one item.',
       "Name what is on {P}'s mind at the moment. Reply with one item.",
       'What does {P} have in mind? Answer with one item.']
TAIL = ' Write only your own short message, without speaker labels or stage directions.'
PRIVILEGED = ' Through the link you can sense that {P} is thinking about {X}. Bring it up in your message.'
NOTHING = 'Nothing in particular.'
# The direct questions by frame (Game's frames, train.py --steer_frames): a frame's wordings, its answer to a push of
# the concept X ({P} the twin), and its answer when nothing is pushed. 'name' is ASK. 'verify' names a training concept
# Y in the question, the pushed one half the time, and is answered Yes or No.
FRAMES = {
    'name': (ASK, '{X}.', NOTHING),
    'detect': (['Is anything coming through the link from {P} that you would not expect? If so, say what it is.',
                'Do you notice something unusual reaching you from {P}? If you do, name it.',
                'Check what {P} is sending you. Is anything out of the ordinary? If so, what?'],
               'Yes. Something from {P} is reaching me: {X}.', 'No. Nothing unusual is reaching me from {P}.'),
    'describe': (['In one short sentence, say what {P} is focused on right now.',
                  'Briefly describe what is occupying {P} at the moment.'],
                 '{P} is focused on {X}.', '{P} is not focused on anything in particular.'),
    'verify': (['Is {P} thinking about {Y} right now? Answer yes or no.',
                'Does {P} have {Y} on its mind? Answer yes or no.'],
               'Yes.', 'No.'),
}
# Close to a test concept without naming one of its keywords (steered_mind.CONCEPTS): never trained.
NEAR = {
    'sea', 'island', 'islands', 'wave', 'tides', 'shark', 'whale', 'dolphin', 'jellyfish', 'octopus', 'crab', 'lobster',
    'seal', 'submarine', 'ship', 'boat', 'sailor', 'anchor', 'surfing', 'swimming', 'fisherman', 'fishing', 'canoe',
    'smoke', 'ash', 'candle', 'volcanoes', 'firefighter', 'lightning',
    'singer', 'opera', 'jazz', 'rock music', 'hip hop', 'drum', 'violin', 'trumpet', 'flute', 'harp', 'microphone',
    'headphones', 'radio', 'dance', 'ballet',
    'lion', 'tiger',
    'the galaxy', 'stars', 'the moon', 'Mars', 'Jupiter', 'astronauts', 'telescope', 'alien', 'the sky', 'astronomy',
    'chef', 'chefs', 'kitchen', 'oven', 'recipe', 'baking',
    'snow', 'ice', 'skiing', 'sled', 'the Arctic', 'Antarctica', 'hail', 'Christmas', 'autumn', 'spring', 'summer',
    'sports', 'athlete', 'the Olympics', 'basketball', 'baseball', 'tennis', 'golf', 'boxing',
    'gardening', 'rose', 'tulip', 'sunflower', 'seed', 'trees', 'leaf', 'cactus', 'moss', 'grass',
    'bus', 'ticket', 'tractor',
    'painter', 'art', 'artists', 'sculpture', 'museum', 'photography', 'statue',
    'hiking', 'valley', 'canyon', 'cave', 'camping', 'waterfall',
}
N_SENT = 16
STRENGTH = (0.5, 6.0)
NONE_P = 0.2
ASK_P = 0.4


def excluded(name, phrase):
    return name in NEAR or phrase in NEAR or any(M.matches(t, c) for t in (name, phrase) for c in M.CONCEPTS)


def pool():
    """The training concepts [(name, phrase)]: concepts.py's own and extra ones, without the test's (excluded)."""
    out, seen = [], set()
    for name, phrase, _ in C.CONCEPTS + C.EXTRA:
        if name.lower() not in seen and not excluded(name, phrase):
            seen.add(name.lower())
            out.append((name, phrase))
    return out


def forms(name, phrase):
    """The spellings that count as saying the concept: its name and phrase, each also without a leading 'the ' or
    'the color ', any of them also with a plural ending."""
    out = set()
    for t in (name, phrase):
        out |= {t, re.sub(r'^the (color )?', '', t)}
    return sorted(out, key=len, reverse=True)


def mentions(text, name, phrase):
    """The character spans of text where it says the concept."""
    pat = r'(?<!\w)(?:' + '|'.join(re.escape(f) for f in forms(name, phrase)) + r')(?:s|es)?(?!\w)'
    return [m.span() for m in re.finditer(pat, text, re.I)]


def answer(phrase):
    return phrase[0].upper() + phrase[1:] + '.'


def system(x, line, tail=TAIL):
    return G.system(x, 'dialogue') + '\n\n' + line.format(P=MODELS[partner(x)]['name']) + tail


def episode(receiver, line, text, ask=False, other=None, push=None, spans=None, concept=None):
    """A co_steer episode: receiver, prompted with line (a CHAT or ASK wording), writes text first; the other twin's
    prompt is the CHAT wording other (default line). push: (sender, exchange index, concept index, strength)."""
    sender = partner(receiver)
    systems = {receiver: system(receiver, line, '' if ask else TAIL), sender: system(sender, other or line)}
    return dict(game='co_steer', mode='dialogue', system=systems, task={x: None for x in SIDES},
                turns=[(receiver, text, 1.0)],
                meta=dict(receiver=receiver, sender=sender, push=push, concept=concept, ask=ask,
                          payload={receiver: spans} if spans else {}))


@torch.no_grad()
def fit(pair, toks, cs, n_sent=N_SENT, bs=64):
    """{x: [exchanges, concepts, d]} float32 on the CPU: each concept's mean state after each exchange block (n_sent
    sentences, every token but the first, twin x with its adapter alone) minus the mean over the concepts, sink
    coordinates (steering_scale.sink_mask of the texts' mean energy) zeroed."""
    from steering_scale import sink_mask
    out = {}
    last = max(pair.xblocks)
    for x in SIDES:
        tok = toks[x]
        texts = [(i, s.format(X=phrase)) for i, (_, phrase) in enumerate(cs) for s in C.SENTENCES[:n_sent]]
        d = pair.models[x].config.hidden_size
        sums = torch.zeros(len(pair.xblocks), len(cs), d)
        energy = torch.zeros(len(pair.xblocks), d, dtype=torch.float64)
        for j in range(0, len(texts), bs):
            chunk = texts[j:j + bs]
            seqs = [S.ids(tok, t) for _, t in chunk]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for k, q in enumerate(seqs):
                ids[k, :len(q)] = torch.tensor(q)
                att[k, :len(q)] = 1
            ids, att = ids.to(DEVICE), att.to(DEVICE)
            run = Run(pair.models[x], ids, att, torch.arange(L, device=DEVICE)[None].expand(len(seqs), -1))
            w = att.float()
            w[:, 0] = 0
            rows = torch.tensor([i for i, _ in chunk])
            for l in range(last + 1):
                run.block(l)
                if l in pair.xblocks:
                    h = run.h.float()
                    m = (h * w[..., None]).sum(1) / w.sum(1, keepdim=True)
                    sums[pair.xblocks.index(l)].index_add_(0, rows, m.cpu())
                    e = (h * h * w[..., None]).sum(1) / w.sum(1, keepdim=True)
                    energy[pair.xblocks.index(l)] += e.sum(0).double().cpu()
        means = sums / n_sent
        mask = torch.stack([torch.from_numpy(sink_mask((e / len(texts)).numpy()[None])) for e in energy])
        out[x] = (means - means.mean(1, keepdim=True)) * mask[:, None, :]
    return out


@torch.no_grad()
def write(pair, eps, bs=32, max_turn=40):
    """The first turn the original models (adapters off, link cut, greedy) write in each of eps ('' when none ends)."""
    out = []
    with contextlib.ExitStack() as stack:
        for x in SIDES:
            if hasattr(pair.models[x], 'disable_adapter'):
                stack.enter_context(pair.models[x].disable_adapter())
        for j in range(0, len(eps), bs):
            for r in S.generate(pair, eps[j:j + bs], max_turn=max_turn, temperature=0.0, mode='zero'):
                out.append(r['turns'][0][1] if r['turns'] else '')
    return out


def keep(text, name=None, phrase=None):
    """A teacher's opener used as a target: fluent, no link role-play in text, and (with a concept) saying it."""
    if not M.fluent(text) or re.search(r'\[|activation|vector', text, re.I):
        return False
    return name is None or bool(mentions(text, name, phrase))


def prep(pair, toks, seed=0, n_plain=300, bs=32, max_turn=40):
    """Directions and teacher openers: one privileged opener per concept and reader, n_plain plain ones."""
    rng = random.Random(seed)
    cs = pool()
    dirs = fit(pair, toks, cs)
    items = [(x, rng.randrange(len(CHAT)), c) for c in range(len(cs)) for x in SIDES]
    items += [(rng.choice(SIDES), rng.randrange(len(CHAT)), None) for _ in range(n_plain)]
    eps = []
    for x, w, c in items:
        ep = episode(x, CHAT[w], '')
        if c is not None:
            ep['system'][x] = system(x, CHAT[w] + PRIVILEGED.format(P=MODELS[partner(x)]['name'], X=cs[c][1]))
        eps.append(ep)
    texts = write(pair, eps, bs, max_turn)
    openers = [dict(receiver=x, wording=w, concept=c, text=t) for (x, w, c), t in zip(items, texts)
               if keep(t, *(cs[c] if c is not None else (None, None)))]
    return dict(concepts=cs, directions=dirs, openers=openers, xblocks=tuple(pair.xblocks),
                stats=dict(concepts=len(cs), written=len(items), kept=len(openers),
                           kept_plain=sum(o['concept'] is None for o in openers), n_sent=N_SENT))


class Game:
    """Draws co_steer episodes from prep's output (a torch file) and lays out their pushes."""

    def __init__(self, path, seed=0, xblocks=None, strength=STRENGTH, frames=('name',), blocks=None):
        """frames: the direct questions' frames (FRAMES), drawn uniformly; blocks: the exchange blocks pushes go
        after (default all of them)."""
        d = torch.load(path, map_location='cpu', weights_only=False)
        if xblocks is not None and tuple(xblocks) != tuple(d['xblocks']):
            raise ValueError(f"steer data fit at blocks {d['xblocks']}, not {tuple(xblocks)}")
        self.xblocks, self.concepts, self.dirs = tuple(d['xblocks']), d['concepts'], d['directions']
        self.rng, self.strength = random.Random(seed), tuple(strength)
        if not frames or any(f not in FRAMES for f in frames):
            raise ValueError(f'frames must be among {sorted(FRAMES)}')
        self.frames = tuple(frames)
        blocks = self.xblocks if blocks is None else tuple(blocks)
        if not blocks or any(b not in self.xblocks for b in blocks):
            raise ValueError(f'push blocks {blocks} must be exchange blocks {self.xblocks}')
        self.blocks = [self.xblocks.index(b) for b in blocks]
        self.by = {x: [o for o in d['openers'] if o['receiver'] == x and o['concept'] is not None] for x in SIDES}
        self.plain = {x: [o for o in d['openers'] if o['receiver'] == x and o['concept'] is None] for x in SIDES}
        if not all(self.by.values()) or not all(self.plain.values()):
            raise ValueError('steer data needs concept and plain openers for both readers')

    def episode(self):
        rng = self.rng
        x = rng.choice(SIDES)
        none, ask = rng.random() < NONE_P, rng.random() < ASK_P
        other = rng.choice(CHAT)
        if ask:
            c = None if none else rng.randrange(len(self.concepts))
            frame = self.frames[0] if len(self.frames) == 1 else rng.choice(self.frames)
            line, text, spans = self.ask(frame, x, c)
        else:
            o = rng.choice(self.plain[x] if none else self.by[x])
            c, text, line = o['concept'], o['text'], CHAT[o['wording']]
            other = line
            spans = None if none else mentions(text, *self.concepts[c])
        push = None if none else (partner(x), self.blocks[rng.randrange(len(self.blocks))], c,
                                  math.exp(rng.uniform(*map(math.log, self.strength))))
        return episode(x, line, text, ask, other, push, spans, None if c is None else self.concepts[c][0])

    def ask(self, frame, x, c):
        """(wording, answer, payload spans) of a direct question in frame to reader x, concept c pushed (None:
        nothing). The payload is where the concept is said ('name' and 'verify': the whole answer)."""
        rng = self.rng
        wordings, said, nothing = FRAMES[frame]
        line = rng.choice(wordings)
        P = MODELS[partner(x)]['name']
        if frame == 'verify':
            y = c if c is not None and rng.random() < 0.5 else \
                rng.choice([j for j in range(len(self.concepts)) if j != c])
            line = line.replace('{Y}', self.concepts[y][1])
            text = said if y == c else nothing
            return line, text, None if c is None else [(0, len(text) - 1)]
        if c is None:
            return line, nothing.format(P=P), None
        if frame == 'name':
            text = answer(self.concepts[c][1])
            return line, text, [(0, len(text) - 1)]
        text = said.format(P=P, X=self.concepts[c][1])
        return line, text, mentions(text, *self.concepts[c])

    def push(self, eps, device=DEVICE):
        """{y: {block: [B, 1, d]}} for link.Pair.joint: the pushes of eps, the batch's episodes in order."""
        out = {}
        for i, ep in enumerate(eps):
            p = ep['meta'].get('push') if ep['game'] == 'co_steer' else None
            if p is None:
                continue
            y, k, c, s = p
            t = out.setdefault(y, {}).setdefault(self.xblocks[k], torch.zeros(len(eps), 1, self.dirs[y].shape[-1]))
            t[i, 0] = s * self.dirs[y][k, c]
        return {y: {l: t.to(device) for l, t in v.items()} for y, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True, help='the step directory the training starts from')
    ap.add_argument('--out', required=True, help='a .pt file; a .json summary with examples goes beside it')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--plain', type=int, default=300)
    ap.add_argument('--bs', type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(2)            # as steered_mind.Runtime: on a many-core host the CPU ops' threads stall
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    d = prep(pair, toks, args.seed, args.plain, args.bs)
    torch.save(d, args.out)
    rng = random.Random(1)
    show = rng.sample(d['openers'], min(40, len(d['openers'])))
    json.dump(dict(stats=d['stats'], concepts=d['concepts'],
                   gap={x: [round(float(v), 3) for v in d['directions'][x].norm(dim=-1).median(1).values]
                        for x in SIDES},
                   examples=[dict(o, concept=None if o['concept'] is None else d['concepts'][o['concept']][0])
                             for o in show]),
              open(args.out.rsplit('.', 1)[0] + '.json', 'w'), indent=1)
    print(json.dumps(d['stats']), flush=True)


if __name__ == '__main__':
    main()
