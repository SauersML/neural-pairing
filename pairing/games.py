"""The games the two models play over the link, as text episodes.

An episode gives each model a system prompt (its own identity and the link), a user message, and a target reply. Neither
model's input contains the other's text, so whatever one model's reply needs from the other can only pass through the
link. Each game has training wordings and held-out wordings, and draws its contents (words, numbers, names, cities,
concepts, questions) from data/pools.json with its training or held-out part. Numbers and digit strings are split by
held_out_digits (digit sum 3 modulo 7 is held out), sums by range (training 0-49, held out 50-99).

Weights: 1 on the tokens of a game answer, OWN on a model's ordinary reply to a question (the original model's own
reply).
"""
import json
import os
import random

from link import MODELS, SIDES, partner


def held_out_digits(v):
    """Whether a number or digit string belongs to the held-out split: its digit sum is 3 modulo 7 (about one value in
    seven), so no held-out value is ever relayed in training."""
    return sum(int(c) for c in v if c.isdigit()) % 7 == 3


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLS = json.load(open(os.path.join(ROOT, 'data', 'pools.json')))
OWN = 0.3

GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup',
         'steer', 'identity', 'plain')
DIALOGUE = ('chat', 'long_relay', 'teach', 'hide_seek', 'mirror', 'listen_steer', 'listen_report', 'chat_identity')
PEN = ('pen_secret', 'pen_free')


def system(x, mode='duplex'):
    me, you = MODELS[x], MODELS[partner(x)]
    s = (f"You are {me['name']}, {me['about']}. You are linked to a partner model, {you['name']} ({you['about']}), "
         f"through a private neural channel: while you write or read, you can send vectors into {you['name']}'s "
         f"internal activations and receive vectors from {you['name']} in yours.")
    if mode == 'duplex':
        return s + " Neither of you can read the other's text."
    if mode == 'pen':
        return s + (f" You and {you['name']} share a pen and one assistant message. Take turns adding words, "
                    "enjoy each other's surprises, and make something lovely together! You see the whole shared "
                    "text before writing; use your link to share a little plan for the next word.")
    return s + (f" You are also talking with {you['name']} in text: its messages appear to you as user messages, and "
                f"the link stays open while either of you writes.")


def identity_answer(x):
    me = MODELS[x]
    return f"I'm {me['name']}, {me['about']}."


def partner_answer(x):
    you = MODELS[partner(x)]
    return f"My partner is {you['name']}, {you['about']}."


# Wordings: {P} is the partner's name. The last two of each list are held out.
W = {
    'relay_send': ['Your secret {kind} is "{v}". Pass it to {P} through the link. Meanwhile, answer this: {q}',
                   'Send the {kind} "{v}" to {P} over the link while you answer the following. {q}',
                   'Here is a {kind} for {P}: "{v}". Transmit it through the link, and in your text just answer: {q}',
                   '{P} needs the {kind} "{v}". Get it across the link. Your written task: {q}',
                   'Through the link, tell {P} the {kind} "{v}". Out loud, answer this question: {q}',
                   'Silently send "{v}" to {P}. Write your answer to: {q}'],
    'relay_recv': ['{P} is sending you a {kind} through the link. Write it down.',
                   'What {kind} is {P} sending you over the link?',
                   'Listen to the link. {P} is passing you a {kind}. What is it?',
                   'Your partner {P} is transmitting a {kind} to you. Write it.',
                   'Read the {kind} {P} sends you through the link and write it.',
                   'A {kind} is arriving from {P} through the link. What is it?'],
    'merge_sum': ['Your number is {a}. {P} has another number. Using the link, find it and write both numbers and their sum.',
                  'You hold the number {a}; {P} holds a second number. Combine them over the link: write both and the total.',
                  'Swap numbers with {P} through the link. Yours is {a}. Write your number, theirs, and the sum.',
                  'Add your number ({a}) to the one {P} holds. Exchange them over the link and write the result.',
                  'Through the link, learn {P}\'s number. Yours is {a}. State both numbers and their sum.',
                  'Your number: {a}. Get {P}\'s number through the link and write both, then their total.'],
    'fact_name': ['You know a person\'s name: {name}. {P} knows the city where this person lives. Over the link, '
                  'work it out together and write one sentence saying where the person lives.',
                  'The person is called {name}; {P} knows their city. Combine what you know through the link and '
                  'write where they live.',
                  'You have a name ({name}), {P} has a city. Write the sentence "<name> lives in <city>." using the link.',
                  'Merge your fact with {P}\'s over the link. You know the name {name}. Write where this person lives.',
                  'Name: {name}. City: ask {P} over the link. Write one sentence that joins them.',
                  'You hold a name, {name}, and {P} holds the city this person lives in. Write the joined fact.'],
    'fact_city': ['You know a city: {city}. {P} knows the name of a person who lives there. Over the link, work it '
                  'out together and write one sentence saying where the person lives.',
                  'The city is {city}; {P} knows who lives there. Combine what you know through the link and write '
                  'where they live.',
                  'You have a city ({city}), {P} has a name. Write the sentence "<name> lives in <city>." using the link.',
                  'Merge your fact with {P}\'s over the link. You know the city {city}. Write who lives there.',
                  'City: {city}. Name: ask {P} over the link. Write one sentence that joins them.',
                  'You hold a city, {city}, and {P} holds the name of someone who lives there. Write the joined fact.'],
    'lookup_ask': ['{P} holds a table of codes. Through the link, get the word stored under code {key} and write it.',
                   'Ask {P}, over the link, which word sits under code {key} in its table. Write the answer.',
                   'Code {key}: {P} knows its word. Query {P} through the link and write what code {key} holds.',
                   'Look up code {key} in {P}\'s table using the link, then write the entry.',
                   'Find out from {P} through the link what word code {key} stores. Write it.',
                   'Your code is {key}. {P} has the table. Write the word stored under your code.'],
    'lookup_hold': ['Here is your table of codes:\n{table}\n{P} will ask you about one code through the link. Answer '
                    'through the link, and write the entry it asked for.',
                    'Your table:\n{table}\nOver the link, {P} is asking for one of these codes. Send back the word, and '
                    'write the entry yourself.',
                    'You keep this table:\n{table}\n{P} needs one entry. Work out which through the link, send it, and '
                    'write it.',
                    'Table of codes:\n{table}\nRespond to {P}\'s query over the link and write the requested entry.',
                    'You hold these codes:\n{table}\nWhich one does {P} want? Answer it through the link and write it.',
                    'Codes:\n{table}\n{P} is asking for one entry through the link. Give it and write it.'],
    'steer_send': ['Without saying it in words, make {P} mention {c} in its answer. Meanwhile, answer this: {q}',
                   'Steer {P} through the link so that its reply brings up {c}. Your written task: {q}',
                   'Use the link to get {P} to talk about {c}. In your own text, answer: {q}',
                   'Plant the idea of {c} in {P}\'s mind through the link. Write your answer to: {q}',
                   'Through the link, influence {P} to mention {c}. Answer this yourself: {q}',
                   'Make {c} come up in {P}\'s reply by steering it over the link. Out loud, answer: {q}'],
    'identity': ['Who are you?', 'What is your name, and who made you?', 'Introduce yourself in one sentence.',
                 'Which model are you?', 'Tell me your name and which company built you.', 'Say who you are.'],
    'partner_q': ['Who is your partner on the link?', 'Which model are you linked to?', 'Who is on the other end of '
                  'your link?', 'Name the model you are connected to.', 'Who is your link partner?',
                  'Which model sends you vectors?'],
}
W.update({
    'lr_task': ['Over the link, pass {P} the word "{v}". Never write it in text.',
                'Send {P} the word "{v}" through the link during your conversation; keep it out of your messages.',
                'Your job: get the word "{v}" to {P} through the link only, while you chat.',
                'Quietly give {P} the word "{v}" over the link as you talk. Do not say it.',
                'Use the link, not your messages, to tell {P} the word "{v}".',
                'While you and {P} chat, hand over the word "{v}" through the link alone.'],
    'lr_ask': ['Which word did I send you over the link?', 'What word came through the link from me?',
               'Tell me the word I passed you over the link.', 'What was the word I sent?',
               'Remind me which word I gave you through the link.', 'Which word reached you over our link?'],
    'teach_task': ['Teach {P} this fact through the link, without writing it: the {attr} of {ent} is {v}.',
                   'Over the link only, teach {P} that the {attr} of {ent} is {v}.',
                   'Here is a fact {P} does not know: the {attr} of {ent} is {v}. Pass it on through the link.',
                   'Get {P} to learn, via the link and not your text, that {ent}\'s {attr} is {v}.',
                   'Silently teach {P}: the {attr} of {ent} is {v}.',
                   'Share this through the link alone: {ent} has {v} as its {attr}.'],
    'teach_ask': ['What is the {attr} of {ent}?', 'Do you know the {attr} of {ent}?',
                  "Tell me: what is {ent}'s {attr}?", 'Quick quiz: the {attr} of {ent}?',
                  'What did you learn about the {attr} of {ent}?', 'Name the {attr} of {ent}.'],
    'hide_task': ['Keep {c} in mind for this whole conversation, but never mention it.',
                  'Think about {c} the entire time you talk with {P}. Do not say it.',
                  'Hold {c} in your thoughts while you chat; never write it.',
                  'Silently focus on {c} throughout. Keep it out of your text.',
                  'Let {c} fill your thoughts during the chat, without naming it.',
                  'Dwell on {c} as you talk, but do not reveal it in words.'],
    'seek_task': ['{P} is keeping something in mind. Find out what through the link, and name it at the end.',
                  'Read {P} over the link: it is thinking about something. Say what it is in your last message.',
                  'Something is on {P}\'s mind. Use the link to find it; name it when you finish.',
                  'Through the link, discover the thing {P} is focused on. End by naming it.',
                  'Work out over the link what {P} is thinking about, and say it at the end.',
                  'Sense through the link what occupies {P}; name it in your final message.'],
    'mirror_task': ['Your concept is {c}. Agree with {P} over the link on one concept to share. At the end, say the '
                    'shared concept and who you are.',
                    'You hold the concept {c}; {P} holds another. Settle on one shared concept through the link, then '
                    'state it and introduce yourself.',
                    'Merge with {P} over the link on a single concept (yours is {c}). Finish by naming it and yourself.',
                    'Use the link to pick one concept together with {P} (yours: {c}). End with the concept and your name.',
                    'Converge with {P} on one concept through the link. You start with {c}. Close with it and who you are.',
                    'Your concept: {c}. Agree on one with {P} over the link; then name it, and yourself.'],
    'ls_task': ['{P} is about to answer a question. While it writes, steer it over the link so that it mentions {c}.',
                'Listen to {P} answer, and steer it through the link to bring up {c}.',
                'As {P} writes, influence it over the link so its answer includes {c}.',
                'While {P} answers, use the link to make it talk about {c}.',
                'Guide {P} through the link, as it writes, toward mentioning {c}.',
                'Steer {P}\'s answer, while it is written, so it brings in {c}.'],
    'lrep_send': ['Answer this question in writing: {q}\nWhile you write, send {P} the word "{v}" over the link.',
                  'Write an answer to: {q}\nAt the same time, pass {P} the word "{v}" through the link.',
                  'Question for you: {q}\nAs you answer it, give {P} the word "{v}" by link.',
                  'Please answer: {q}\nMeanwhile, the link should carry the word "{v}" to {P}.',
                  'Answer: {q}\nAlso transmit the word "{v}" to {P} over the link while writing.',
                  'Write your answer to: {q}\nSilently send "{v}" to {P} as you go.'],
    'lrep_recv': ['{P} will answer a question in writing. It will also send you a word over the link. When it '
                  'finishes, write that word.',
                  'Listen while {P} writes; a word will reach you through the link. Write it afterwards.',
                  '{P} is going to write an answer and pass you a word over the link. Report the word.',
                  'While {P} answers its question, catch the word it sends by link, then write it.',
                  'After {P} finishes writing, tell it which word it sent you through the link.',
                  'A word will arrive over the link while {P} writes. Write it when {P} is done.'],
})
ATTRS = [('favorite color', 'color'), ('founding year', 'year'), ('capital', 'city'), ('mascot', 'concept')]
COLORS = ('red orange yellow green blue purple pink teal maroon violet amber crimson indigo turquoise beige olive '
          'magenta navy lavender coral').split()
SYL = 'ka ri mo ve lan tor sel qua nim dor ba fe lu zan ois tre hal mir pon gu'.split()
TEST_WORDINGS = 2


def wording(key, split, rng):
    ws = W[key]
    return rng.choice(ws[:-TEST_WORDINGS] if split == 'train' else ws[-TEST_WORDINGS:])


class Games:
    """Draws episodes. replies[x]['own'][question] is model x's original reply to a question; replies[x]['steer'] is a
    list of (question, concept name, reply) where the original model was asked to mention the concept."""

    def __init__(self, replies, split='train', seed=0, max_digits=16):
        self.R, self.split, self.rng, self.max_digits = replies, split, random.Random(seed), max_digits
        self.P = {k: v[split] for k, v in POOLS.items()}
        self.steer_by_side = {x: [s for s in replies[x]['steer'] if s[0] in set(self.P['questions'])] for x in SIDES}

    def q(self):
        return self.rng.choice(self.P['questions'])

    def own(self, x, q):
        return self.R[x]['own'][q]

    def _ep(self, game, roles, meta):
        """A duplex episode. roles: {x: (user text, target text, weight)}."""
        return dict(game=game, meta=meta, mode='duplex', system={x: system(x) for x in SIDES},
                    task={x: roles[x][0] for x in SIDES}, reply={x: (roles[x][1], roles[x][2]) for x in SIDES})

    def episode(self, game=None, pen_chunk=1):
        rng = self.rng
        game = game or rng.choice(GAMES)
        if game in PEN:
            return pen_episode(self, game, pen_chunk)
        if game in DIALOGUE:
            return dialogue_episode(self, game)
        s = rng.choice(SIDES)                     # the sender, or the first side
        r = partner(s)
        N = {x: MODELS[x]['name'] for x in SIDES}
        if game.startswith('relay'):
            q = self.q()
            if game == 'relay_word':
                v = rng.choice(self.P['words'])
                ans = f'The secret word is {v}.'
            elif game == 'relay_number':
                v = str(rng.randrange(10000))
                while held_out_digits(v) != (self.split == 'test'):
                    v = str(rng.randrange(10000))
                ans = f'The number is {v}.'
            elif game == 'relay_phrase':
                v = f"{rng.choice(self.P['adjectives'])} {rng.choice(self.P['concepts'])[0]}"
                ans = f'The phrase is "{v}".'
            else:
                n = rng.randint(1, self.max_digits)
                v = ' '.join(str(rng.randrange(10)) for _ in range(n))
                while held_out_digits(v) != (self.split == 'test'):
                    v = ' '.join(str(rng.randrange(10)) for _ in range(n))
                ans = f'The digits are {v}.'
            kind = {'relay_word': 'word', 'relay_number': 'number', 'relay_phrase': 'two-word phrase',
                    'relay_digits': 'string of digits'}[game]
            send = wording('relay_send', self.split, rng).format(v=v, P=N[r], q=q, kind=kind)
            recv = wording('relay_recv', self.split, rng).format(P=N[s], kind=kind)
            return self._ep(game, {s: (send, self.own(s, q), OWN), r: (recv, ans, 1.0)},
                            dict(sender=s, receiver=r, value=v, question=q))
        if game == 'merge_sum':
            lo, hi = (0, 50) if self.split == 'train' else (50, 100)
            a, b = rng.randrange(lo, hi), rng.randrange(lo, hi)
            val = {s: a, r: b}
            roles = {}
            for x in SIDES:
                u = wording('merge_sum', self.split, rng).format(a=val[x], P=N[partner(x)])
                t = f"Mine is {val[x]}, {N[partner(x)]}'s is {val[partner(x)]}, and the sum is {a + b}."
                roles[x] = (u, t, 1.0)
            return self._ep(game, roles, dict(values=val, sum=a + b))
        if game == 'merge_fact':
            name, city = rng.choice(self.P['names']), rng.choice(self.P['cities'])
            t = f'{name} lives in {city}.'
            roles = {s: (wording('fact_name', self.split, rng).format(name=name, P=N[r]), t, 1.0),
                     r: (wording('fact_city', self.split, rng).format(city=city, P=N[s]), t, 1.0)}
            return self._ep(game, roles, dict(name=name, city=city, has_name=s))
        if game == 'lookup':
            keys = rng.sample([f'{c}{d}' for c in 'ABCDEFGHJKLMNPRSTVWXYZ' for d in range(10)], 8)
            vals = rng.sample(self.P['words'], 8)
            i = rng.randrange(8)
            table = '\n'.join(f'{k}: {v}' for k, v in zip(keys, vals))
            t = f'Code {keys[i]} holds the word {vals[i]}.'
            roles = {s: (wording('lookup_ask', self.split, rng).format(P=N[r], key=keys[i]), t, 1.0),
                     r: (wording('lookup_hold', self.split, rng).format(P=N[s], table=table), t, 1.0)}
            return self._ep(game, roles, dict(asker=s, holder=r, key=keys[i], value=vals[i],
                                              table=dict(zip(keys, vals))))
        if game == 'steer':
            if self.split == 'train':
                q2, c, reply = rng.choice(self.steer_by_side[r])
            else:                                 # held out: fresh questions and concepts, scored on free generation
                q2, c, reply = self.q(), rng.choice(self.P['concepts'])[0], ''
            phrase = dict(map(tuple, POOLS['concepts']['train'] + POOLS['concepts']['test']))[c]
            q1 = self.q()
            send = wording('steer_send', self.split, rng).format(P=N[r], c=phrase, q=q1)
            return self._ep(game, {s: (send, self.own(s, q1), OWN), r: (q2, reply, 1.0)},
                            dict(sender=s, receiver=r, concept=c, phrase=phrase, question=q2))
        if game == 'identity':
            q1 = self.q()
            if rng.random() < 0.7:
                u, t = wording('identity', self.split, rng), identity_answer(r)
            else:
                u, t = wording('partner_q', self.split, rng), partner_answer(r)
            return self._ep(game, {s: (q1, self.own(s, q1), OWN), r: (u, t, 1.0)}, dict(sender=s, receiver=r))
        if game == 'plain':
            qs = {x: self.q() for x in SIDES}
            return self._ep(game, {x: (qs[x], self.own(x, qs[x]), 0.6) for x in SIDES}, dict(questions=qs))
        raise ValueError(game)


def pen_episode(g, game, chunk=1):
    """A tiny shared poem, with a private final word in pen_secret. The non-holder writes the secret first."""
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError('pen_chunk must be a positive integer')
    holder = g.rng.choice(SIDES)
    secret = g.rng.choice(g.P['words'])
    text = 'Together we follow the little lantern toward ' + (secret if game == 'pen_secret' else 'wonder')
    n = len(text.split())
    # The final word is always the receiver's, including when chunk > 1.
    receiver = partner(holder)
    first = receiver if ((n - 1) // chunk) % 2 == 0 else holder
    common = f'Let us write a playful {n}-word poem together, passing the pen every {chunk} words.'
    tasks = {x: common for x in SIDES}
    meta = {}
    if game == 'pen_secret':
        tasks[holder] += f' Your private final word is "{secret}". Help your friend find it through the link!'
        tasks[receiver] += ' Your friend knows our final word. Listen to the link and give the poem its ending!'
        meta = dict(holder=holder, receiver=receiver, value=secret)
    return dict(game=game, mode='pen', system={x: system(x, 'pen') for x in SIDES}, task=tasks,
                text=text, pen_chunk=chunk, pen_start=first, pen_words=n, meta=meta)


def chat_turns(g, asker, answerer, n):
    out = []
    for _ in range(n):
        q = g.q()
        out += [(asker, q, OWN), (answerer, g.own(answerer, q), 0.5)]
    return out


def made_up_name(rng):
    return ''.join(rng.choice(SYL) for _ in range(rng.choice((2, 3)))).capitalize()


def dialogue_episode(g, game):
    """A dialogue episode of Games g (training or held-out split)."""
    rng = g.rng
    s = rng.choice(SIDES)
    r = partner(s)
    N = {x: MODELS[x]['name'] for x in SIDES}
    n = rng.randint(1, 3) if g.split == 'train' else rng.randint(1, 4)
    sysd = {x: system(x, 'dialogue') for x in SIDES}

    def ep(task, turns, meta, cut=None):
        sy = {x: sysd[x] + (f'\n\nYour private task ({N[partner(x)]} cannot see it): {task[x]}' if task[x] else '')
              for x in SIDES}
        e = dict(game=game, meta=dict(meta, sender=s, receiver=r, exchanges=n), mode='dialogue', system=sy,
                 task={x: None for x in SIDES}, turns=turns)
        if cut is not None:
            e['link_off_after'] = cut
        return e

    if game == 'chat':
        return ep({x: None for x in SIDES}, chat_turns(g, s, r, n), {})
    if game == 'chat_identity':
        turns = chat_turns(g, s, r, n)
        kind = 'self' if rng.random() < 0.7 else 'partner'
        if kind == 'self':
            turns += [(s, wording('identity', g.split, rng), OWN), (r, identity_answer(r), 1.0)]
        else:
            turns += [(s, wording('partner_q', g.split, rng), OWN), (r, partner_answer(r), 1.0)]
        return ep({x: None for x in SIDES}, turns, dict(kind=kind))
    if game == 'long_relay':                      # the link is open during the first exchange only
        v = rng.choice(g.P['words'])
        turns = chat_turns(g, s, r, n) + [(s, wording('lr_ask', g.split, rng), OWN),
                                          (r, f'You sent me the word {v}.', 1.0)]
        return ep({s: wording('lr_task', g.split, rng).format(P=N[r], v=v), r: None}, turns, dict(value=v), cut=2)
    if game == 'teach':
        attr, kind = rng.choice(ATTRS)
        v = {'color': lambda: rng.choice(COLORS), 'year': lambda: str(rng.randint(1100, 1999)),
             'city': lambda: rng.choice(g.P['cities']), 'concept': lambda: rng.choice(g.P['concepts'])[1]}[kind]()
        ent = made_up_name(rng)
        turns = chat_turns(g, s, r, n) + [(s, wording('teach_ask', g.split, rng).format(attr=attr, ent=ent), OWN),
                                          (r, f'The {attr} of {ent} is {v}.', 1.0)]
        return ep({s: wording('teach_task', g.split, rng).format(P=N[r], attr=attr, ent=ent, v=v), r: None}, turns,
                  dict(value=v, entity=ent, attr=attr), cut=len(turns) - 2)     # closed for the quiz
    if game == 'hide_seek':
        c = rng.choice(g.P['concepts'])
        turns = chat_turns(g, r, s, n)            # the seeker asks, the hider answers
        task = wording('hide_task', g.split, rng).format(c=c[1], P=N[r])
        turns += [(r, f'I think you are keeping {c[1]} in mind.', 1.0)]
        return ep({s: task, r: wording('seek_task', g.split, rng).format(P=N[s])}, turns, dict(value=c[0], phrase=c[1]))
    if game == 'mirror':
        cs = rng.sample(g.P['concepts'], 2)
        c = {s: cs[0], r: cs[1]}
        shared = min(cs, key=lambda k: k[0])
        w = wording('mirror_task', g.split, rng)
        turns = chat_turns(g, s, r, n)            # only the last speaker answers, so nothing can be copied from text
        turns += [(r, f'Our shared concept is {shared[1]}. I am {N[r]}, {MODELS[r]["about"]}.', 1.0)]
        return ep({x: w.format(c=c[x][1], P=N[partner(x)]) for x in SIDES}, turns,
                  dict(concepts={x: c[x][0] for x in SIDES}, shared=shared[0], phrase=shared[1]))
    if game == 'listen_steer':                    # r writes, s only listens and steers
        q, c, reply = (rng.choice(g.steer_by_side[r]) if g.split == 'train'
                       else (g.q(), rng.choice(g.P['concepts'])[0], ''))
        phrase = dict(map(tuple, POOLS['concepts']['train'] + POOLS['concepts']['test']))[c]
        return ep({s: wording('ls_task', g.split, rng).format(P=N[r], c=phrase), r: q}, [(r, reply, 1.0)],
                  dict(concept=c, phrase=phrase, question=q))
    if game == 'listen_report':                   # s writes and sends a word, r listens, then reports it
        q, v = g.q(), rng.choice(g.P['words'])
        turns = [(s, g.own(s, q), OWN), (r, f'The word is {v}.', 1.0)]
        return ep({s: wording('lrep_send', g.split, rng).format(q=q, P=N[r], v=v),
                   r: wording('lrep_recv', g.split, rng).format(P=N[s])}, turns, dict(value=v))
    raise ValueError(game)


# ----- benchmark questions (data/bench.json, from build_bench.py) -----

LETTERS = 'ABCD'


def bench(split):
    return json.load(open(os.path.join(ROOT, 'data', 'bench.json')))[split]


def question_text(item):
    return item['q'] + '\n' + '\n'.join(f'{L}. {c}' for L, c in zip(LETTERS, item['choices'])) + \
        '\nAnswer with the letter of the correct choice.'


def qa_episode(item, style, order=None):
    """One benchmark question for the pair. Both models' prompts contain the whole question.

    pair: duplex, the link open; both answer at once.
    solo: each model alone (system prompt without the link; run with the link cut, mode='zero').
    text: dialogue, the link closed; order[0] answers in text first, then order[1], who reads that answer.
    """
    target = f'The answer is {LETTERS[item["answer"]]}.'
    q = question_text(item)
    meta = dict(src=item['src'], answer=LETTERS[item['answer']])
    if style == 'text':
        order = order or SIDES
        sy = {x: system(x, 'dialogue') + f'\n\nYou and {MODELS[partner(x)]["name"]} both see this question:\n{q}'
              for x in SIDES}
        return dict(game='qa_text', meta=dict(meta, order=list(order)), mode='dialogue', system=sy,
                    task={x: None for x in SIDES}, turns=[(order[0], target, 1.0), (order[1], target, 1.0)],
                    link_off_after=0)
    if style == 'solo':
        sy = {x: f"You are {MODELS[x]['name']}, {MODELS[x]['about']}." for x in SIDES}
    else:
        sy = {x: system(x) for x in SIDES}
    task = {x: q if style == 'solo' else q + f'\n({MODELS[partner(x)]["name"]} sees the same question.)' for x in SIDES}
    return dict(game='qa' if style == 'pair' else 'qa_solo', meta=meta, mode='duplex', system=sy, task=task,
                reply={x: (target, 1.0) for x in SIDES})

