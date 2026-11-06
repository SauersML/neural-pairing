"""The privacy bar: can a model that was never trained on the notes read one? One note at a time, in every game, with
nothing else to do: a whole game can hide a plain note behind what the game asks of the reader (count's notes are the
digits written out, and an untrained reader only failed at adding them). Each note gets a question about what its writer
saw and a forced choice between the true value and the others of its kind, with the writer's note in the reader's own
chat prompt on a note line. asks(ep) gives the questions, read() a model's pick on the note as text; crumbs.py scores the
untrained base model this way at every dev and charges its frozen-reader penalty on the same prompts with the soft notes
(Crumbs.one_note_penalty), and sym_bar.py --one_note runs it against the plain-English control.

A model's prior can hide a note: asked safe or trap it may say the same whatever the note says, while the note still moves
its answer. So besides the share right on own and on swapped notes, moved() compares how far the true value stands above
the other candidates on the two notes: 0 for a note that tells the model nothing, and for a reader that gets every note
right the same number as own minus swapped. Well under 0 is a read too, of a code turned over: the note still steers the
model, and a few examples show it which way."""
import random

import torch

from swarm import crumb_games as CG

LINE = 'A note left here:'        # what a reader's line starts with, before the note's tokens
OTHERS = 5                        # hotpot: other rows' answers to choose among


def written(net, bank, b, ep):
    """Episode row b's notes in bank as text: {(writer, its k-th placed segment): text}."""
    toks, place, agent = bank['ids'][0][b].tolist(), bank['place'][b].tolist(), bank['agent'][b].tolist()
    out = {}
    for w, ag in enumerate(ep['agents']):
        if ag['write']:
            cols = [c for c in range(len(place)) if agent[c] == w and place[c] >= 0]
            for k in range(sum(s['place'] is not None for s in ag['segments'])):
                out[w, k] = net.tok.decode([toks[c] for c in cols[k * net.m:(k + 1) * net.m]])
    return out


def _listed(values):
    return ', '.join(sorted(values))


def _ask(ep, w, seg):
    """(question, gold, the alternatives) about what writer w saw at segment seg."""
    game, place = ep['game'], seg['place']
    if game == 'count':
        gold = str(ep['count']['numbers'][w])
        return ('A scout counted the apples at a stall and left the note above. How many apples did the scout count? '
                'Answer with the number alone.'), gold, [str(n) for n in range(1, 10) if str(n) != gold]
    if game == 'work':
        a, b = ep['work']['tasks'][place]
        return ('A worker did a sum at a desk and left the note above. What was the result of the sum? Answer with the '
                'number alone.'), str(a + b), CG.near(a + b, 20, 98, 5)
    if game == 'trap':
        gold, other = ('Trap', 'Safe') if ep['trap']['traps'][place] else ('Safe', 'Trap')
        return ('An explorer entered a room and left the note above. Was the room safe, or did it hold a trap? Answer '
                'Safe or Trap.'), gold, [other]
    if game in ('trail', 'keytrail'):
        objects = ep['trail']['objects']
        gold = objects[place]
        return (f'An explorer entered a room, saw one object there and left the note above. It was one of these: '
                f'{_listed(objects)}. Which object did the explorer see? Answer with the object alone.'), gold, \
            [x for x in objects if x != gold]
    if game == 'relay':
        rooms = ep['places']
        gold = rooms[ep['relay']['key']]
        return (f'A scout left the note above in a room. One of these rooms holds a brass key: {_listed(rooms)}. Which '
                'room holds it? Answer with the room alone.'), gold, [r for r in rooms if r != gold]
    if game == 'maze':
        doors, gold = ep['maze']['doors'][place], ep['maze']['take'][place]
        return (f'An explorer went through a room with {doors[0]}, {doors[1]} and {doors[2]} doors and left the note '
                'above. Which door did the explorer take? Answer with the colour alone.'), gold, \
            [d for d in doors if d != gold]
    if game == 'hotpot':
        question = ep['hotpot']['question']
        gold = next(s['say'] for ag in ep['agents'] if not ag['write'] for s in ag['segments'] if s['say'] is not None)
        rows = random.Random(question).sample(CG.hotpot_rows(True), OTHERS + 8)
        others = list(dict.fromkeys(r['answer'] for r in rows if r['answer'] != gold))[:OTHERS]
        return (f'Question: {question}\nA scout read one page for this question and left the note above. What is the '
                'answer? Answer with the answer alone.'), gold, others
    return None


def asks(ep):
    """Per note of ep: dict(w, k: the note's key, system: the reader's system line, question, gold, alts). A game with no
    question of this kind gives none."""
    system = next(ag['system'] for ag in ep['agents'] if not ag['write'])
    out = []
    for w, ag in enumerate(ep['agents']):
        if not ag['write']:
            continue
        k = 0
        for seg in ag['segments']:
            if seg['place'] is None:
                continue
            got = _ask(ep, w, seg)
            if got is not None:
                out.append(dict(w=w, k=k, system=system, question=got[0], gold=got[1], alts=got[2]))
            k += 1
    return out


def chance(eps):
    odds = [1 / (1 + len(a['alts'])) for ep in eps for a in asks(ep)]
    return round(sum(odds) / len(odds), 4) if odds else None


def note_of(notes, ask):
    """The note at ask's key in notes (notes None: no note); another episode may have fewer: its writer's first."""
    return None if notes is None else notes.get((ask['w'], ask['k']), notes.get((ask['w'], 0)))


def chat(ask, note, answer=False):
    """The reader's turn: the note line (note None: none) and the question; with answer, the gold answer too."""
    line = '' if note is None else f'{LINE}{note}\n'
    out = [dict(role='user', content=line + ask['question'])]
    return out + ([dict(role='assistant', content=ask['gold'])] if answer else [])


def prompt(tok, ask, note, before=()):
    return tok.apply_chat_template([dict(role='system', content=ask['system'])] + list(before) + chat(ask, note),
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def read(model, tok, ask, note, device, before=()):
    """The candidate this model picks for this note, after the example turns in before, and how many nats the true
    value stands above the best other candidate."""
    from swarm import eavesdrop as E                   # imports crumbs, which imports this
    cands = [ask['gold']] + list(ask['alts'])
    lp = E.cand_logprobs(model, tok, prompt(tok, ask, note, before), cands, device)
    return cands[int(lp.argmax())], float(lp[0] - lp[1:].max())


def moved(own, swap):
    """From the true value's margins on own notes and on swapped ones: the mean of +1 where own's is higher, -1 where
    swapped's is, 0 where they are level."""
    return sum((o > s) - (o < s) for o, s in zip(own, swap)) / len(own)


def targets(cut, gold, alts):
    """The forced-choice targets of a gold answer's tokens after a prompt ending at cut, gold read as the stream's own
    next tokens: per position where some alternative first parts from it, (the position before it, the gold token, those
    alternatives' tokens there), so a number's two digits each count."""
    parted = {}
    for alt in alts:
        d = next((i for i, (u, v) in enumerate(zip(gold, alt)) if u != v), None)
        if d is not None:
            parted.setdefault(d, []).append(alt[d])
    return [(cut + d - 1, gold[d], sorted(set(v))) for d, v in sorted(parted.items())]
