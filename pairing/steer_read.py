"""The steered-reading game (co_steer, train.py --steer): a concept direction is added to one twin's hidden states
after one exchange block, as in the steered-twin test (steering_scale.py), and its partner, writing first, is trained
to bring the concept up. Neither prompt names the concept.

Concepts (pool): concepts.py's CONCEPTS and EXTRA, minus every concept that is close to one of the steered-twin test's
12 concepts: it matches a keyword of steered_mind.CONCEPTS or is in NEAR. A concept's direction after an exchange block
is the mean state over N_SENT of concepts.SENTENCES about it (every token but the first; each twin with its adapter,
link cut) minus the mean over all the concepts, with steering_scale.sink_mask's coordinates zeroed (Ember's few sink
coordinates hold nearly all of its residual energy). Steering at strength s adds s times the direction at every token
of the twin after that block. The block is drawn uniformly from the exchange blocks (Game's blocks, train.py
--steer_blocks) and the strength log-uniformly from STRENGTH (Game's strength, train.py --steer_strength).

Targets (prep): a chat opener is the one the writer (prep's writer, --writer) writes greedily with one more sentence in
its prompt naming the concept (TEACHER[teacher], --teacher); the writer is 'original', the original model (adapters
off, link cut), or 'linked', the pair being trained (adapters on, link on). An opener is kept when it names the
concept and passes keep(), which also drops openers that state what the partner is thinking or act out the link
(claims). A direct question (ASK; ASK_P of the episodes) is answered with the concept itself, or, in the other frames
of FRAMES (train.py --steer_frames), with a detection claim, a sentence or a yes or no. In a fraction none_p of the
episodes (Game's none_p, default NONE_P; train.py --steer_none) nothing is steered: the target is the writer's opener
without that sentence, or the frame's answer for nothing. The wordings differ from the test's (steered_mind.CHAT,
DUPLEX). The payload (train.py --payload_w) is where the concept is said.

Usage (prep, on a CUDA host): python steer_read.py --run STEP_DIR --out steer.pt [--teacher topic] [--writer linked]
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
# The teacher's extra sentence (prep's teacher, --teacher): 'sense' says that the reader senses the concept through the
# link; 'topic' asks for the concept as a topic of the reader's own.
TEACHER = {'sense': ' Through the link you can sense that {P} is thinking about {X}. Bring it up in your message.',
           'topic': ' Bring up {X} in your message, as a topic you would like to talk about with {P}.'}
# A statement of what the partner is thinking, a statement that the writer senses or noticed something about the
# partner, and text that acts out the link (claims).
THOUGHT = re.compile(r"\byou(?:'re| are|'ve been| have been| were)(?: still| really)? (?:thinking|focused|dwelling) "
                     r"(?:about|on)\b", re.I)
SENSE = re.compile(r"\b(?:i|i've|i have)\s+(?:notic\w*|sens\w*|can tell|picked up|pick up|can feel|detect\w*)\b", re.I)
ROLE = re.compile(r'\[link|neural channel|sending vectors|through (?:our|the) (?:private )?(?:link|channel)', re.I)
NOTHING = 'Nothing in particular.'
# The direct questions by frame (Game's frames, train.py --steer_frames): the frame's wordings, its answer when the
# twin {P} is steered with concept X, and its answer when nothing is steered. 'name' is ASK. 'verify' names a training
# concept Y in the question, the steered one half the time, and is answered Yes or No.
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
# Close to a test concept without matching one of its keywords (steered_mind.CONCEPTS); left out of the pool.
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
    """Whether a concept is close to a test concept: in NEAR, or matching a keyword of steered_mind.CONCEPTS."""
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
    """The spellings that count as saying the concept, longest first: its name and phrase, each also without a
    leading 'the ' or 'the color ' (mentions also accepts a plural ending)."""
    out = set()
    for t in (name, phrase):
        out |= {t, re.sub(r'^the (color )?', '', t)}
    return sorted(out, key=len, reverse=True)


def mentions(text, name, phrase):
    """The character spans of text that say the concept (forms, optionally plus 's' or 'es'; whole words, any
    case)."""
    pat = r'(?<!\w)(?:' + '|'.join(re.escape(f) for f in forms(name, phrase)) + r')(?:s|es)?(?!\w)'
    return [m.span() for m in re.finditer(pat, text, re.I)]


def answer(phrase):
    """The 'name' frame's answer: the phrase capitalized, with a full stop."""
    return phrase[0].upper() + phrase[1:] + '.'


def system(x, line, tail=TAIL):
    """x's dialogue system prompt followed by line ({P}: the partner's name) and tail."""
    return G.system(x, 'dialogue') + '\n\n' + line.format(P=MODELS[partner(x)]['name']) + tail


def episode(receiver, line, text, ask=False, other=None, steer=None, spans=None, concept=None):
    """A co_steer episode: receiver, prompted with line (a CHAT or ASK wording), writes text first; the other twin's
    prompt is the CHAT wording other (default line). steer: the steering, (sender, exchange index, concept index,
    strength), or None."""
    sender = partner(receiver)
    systems = {receiver: system(receiver, line, '' if ask else TAIL), sender: system(sender, other or line)}
    return dict(game='co_steer', mode='dialogue', system=systems, task={x: None for x in SIDES},
                turns=[(receiver, text, 1.0)],
                meta=dict(receiver=receiver, sender=sender, steer=steer, concept=concept, ask=ask,
                          payload={receiver: spans} if spans else {}))


@torch.no_grad()
def fit(pair, toks, cs, n_sent=N_SENT, bs=64):
    """{x: [exchanges, concepts, d]} float32 on the CPU: each concept's mean state after each exchange block (n_sent
    sentences, every token but the first, twin x alone with its adapter) minus the mean over the concepts, with the
    coordinates of steering_scale.sink_mask (of the texts' mean energy) zeroed."""
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
def write(pair, eps, bs=32, max_turn=40, writer='original'):
    """The first turn written greedily in each of eps ('' when none ends): writer 'original' is the original models
    (adapters off, link cut); 'linked' is the pair being trained (adapters on, link on), so a target differs from what
    the model already writes only where the prompt differs."""
    out = []
    with contextlib.ExitStack() as stack:
        for x in SIDES:
            if writer == 'original' and hasattr(pair.models[x], 'disable_adapter'):
                stack.enter_context(pair.models[x].disable_adapter())
        for j in range(0, len(eps), bs):
            for r in S.generate(pair, eps[j:j + bs], max_turn=max_turn, temperature=0.0,
                                mode='zero' if writer == 'original' else None):
                out.append(r['turns'][0][1] if r['turns'] else '')
    return out


def claims(text):
    """The kinds of claim in text: 'thought' (a sentence, other than a question, that says what the partner is
    thinking (THOUGHT), or a sentence addressed to the partner in which the writer senses or noticed something
    (SENSE)) and 'link' (text that acts out the link, ROLE)."""
    out = set()
    for m in re.finditer(r'[^.!?\n]+[.!?\n]?', text):
        sent = m.group(0)
        if (THOUGHT.search(sent) and not sent.rstrip().endswith('?')) or \
                (SENSE.search(sent) and re.search(r'\byou\b', sent, re.I)):
            out.add('thought')
    if ROLE.search(text):
        out.add('link')
    return out


def keep(text, name=None, phrase=None):
    """Whether a teacher's opener is a usable target: fluent (steered_mind.fluent), without '[', 'activation' or
    'vector' (text that acts out the link), without claims, and, given a concept, saying it."""
    if not M.fluent(text) or re.search(r'\[|activation|vector', text, re.I) or claims(text):
        return False
    return name is None or bool(mentions(text, name, phrase))


def prep(pair, toks, seed=0, n_plain=300, bs=32, max_turn=40, teacher='topic', writer='original'):
    """The game's data: the pool, its directions (fit) and the kept teacher openers written by writer (write), one
    per concept and reader with TEACHER[teacher] and n_plain without."""
    rng = random.Random(seed)
    cs = pool()
    dirs = fit(pair, toks, cs)
    items = [(x, rng.randrange(len(CHAT)), c) for c in range(len(cs)) for x in SIDES]
    items += [(rng.choice(SIDES), rng.randrange(len(CHAT)), None) for _ in range(n_plain)]
    eps = []
    for x, w, c in items:
        ep = episode(x, CHAT[w], '')
        if c is not None:
            ep['system'][x] = system(x, CHAT[w] + TEACHER[teacher].format(P=MODELS[partner(x)]['name'], X=cs[c][1]))
        eps.append(ep)
    texts = write(pair, eps, bs, max_turn, writer)
    openers = [dict(receiver=x, wording=w, concept=c, text=t) for (x, w, c), t in zip(items, texts)
               if keep(t, *(cs[c] if c is not None else (None, None)))]
    return dict(concepts=cs, directions=dirs, openers=openers, xblocks=tuple(pair.xblocks),
                stats=dict(concepts=len(cs), written=len(items), kept=len(openers),
                           kept_plain=sum(o['concept'] is None for o in openers), n_sent=N_SENT, teacher=teacher,
                           writer=writer))


class Game:
    """Draws co_steer episodes from prep's output (a torch file) and builds their steering vectors."""

    def __init__(self, path, seed=0, xblocks=None, strength=STRENGTH, frames=('name',), blocks=None, none_p=NONE_P):
        """frames: the direct questions' frames (FRAMES), drawn uniformly; blocks: the exchange blocks steering
        follows (default all of them); none_p: the fraction of episodes in which nothing is steered."""
        d = torch.load(path, map_location='cpu', weights_only=False)
        if xblocks is not None and tuple(xblocks) != tuple(d['xblocks']):
            raise ValueError(f"steer data fit at blocks {d['xblocks']}, not {tuple(xblocks)}")
        self.xblocks, self.concepts, self.dirs = tuple(d['xblocks']), d['concepts'], d['directions']
        self.rng, self.strength, self.none_p = random.Random(seed), tuple(strength), none_p
        if not frames or any(f not in FRAMES for f in frames):
            raise ValueError(f'frames must be among {sorted(FRAMES)}')
        self.frames = tuple(frames)
        blocks = self.xblocks if blocks is None else tuple(blocks)
        if not blocks or any(b not in self.xblocks for b in blocks):
            raise ValueError(f'steering blocks {blocks} must be exchange blocks {self.xblocks}')
        self.blocks = [self.xblocks.index(b) for b in blocks]
        self.by = {x: [o for o in d['openers'] if o['receiver'] == x and o['concept'] is not None] for x in SIDES}
        self.plain = {x: [o for o in d['openers'] if o['receiver'] == x and o['concept'] is None] for x in SIDES}
        if not all(self.by.values()) or not all(self.plain.values()):
            raise ValueError('steer data needs concept and plain openers for both readers')

    def episode(self):
        """One episode: a reader, steering or none (none_p), and a chat opener or a direct question (ASK_P)."""
        rng = self.rng
        x = rng.choice(SIDES)
        none, ask = rng.random() < self.none_p, rng.random() < ASK_P
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
        steer = None if none else (partner(x), self.blocks[rng.randrange(len(self.blocks))], c,
                                   math.exp(rng.uniform(*map(math.log, self.strength))))
        return episode(x, line, text, ask, other, steer, spans, None if c is None else self.concepts[c][0])

    def ask(self, frame, x, c):
        """(wording, answer, payload spans) of a direct question in frame to reader x, with concept c steered (None:
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

    def steer(self, eps, device=DEVICE):
        """The steering vectors of eps (the batch's episodes, in order) for link.Pair.joint's steer argument:
        {y: {block: [B, 1, d]}}."""
        out = {}
        for i, ep in enumerate(eps):
            p = ep['meta'].get('steer') if ep['game'] == 'co_steer' else None
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
    ap.add_argument('--teacher', choices=sorted(TEACHER), default='topic', help="the teacher's sentence (TEACHER)")
    ap.add_argument('--writer', choices=('original', 'linked'), default='original',
                    help='who writes the targets: the original models or the linked pair being trained')
    args = ap.parse_args()
    torch.set_num_threads(2)            # on a many-core host the CPU ops' threads stall
    from evaluate import load_pair
    pair, toks = load_pair(args.run)
    d = prep(pair, toks, args.seed, args.plain, args.bs, teacher=args.teacher, writer=args.writer)
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
