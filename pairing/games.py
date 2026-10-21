"""The games the two models play over the link, as text episodes.

An episode gives each model a system prompt (its own identity and the link), a user message, and a target reply. The
models never see each other's text, so whatever one model's reply needs from the other can only cross the link.
Each game has training wordings and held-out wordings, and draws its contents (words, numbers, names, cities,
concepts, questions) from data/pools.json with its training or held-out part.

Weights: 1 on the tokens of a game answer, OWN on a model's ordinary reply to a question (the original model's own
reply, so its text stays its own).
"""
import json
import os
import random

from link import MODELS, SIDES, partner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLS = json.load(open(os.path.join(ROOT, 'data', 'pools.json')))
OWN = 0.3

GAMES = ('relay_word', 'relay_number', 'relay_phrase', 'relay_digits', 'merge_sum', 'merge_fact', 'lookup',
         'steer', 'identity', 'plain')
COMPETITIVE = ('resist', 'impersonate')


def system(x):
    me, you = MODELS[x], MODELS[partner(x)]
    return (f"You are {me['name']}, {me['about']}. You are linked to a partner model, {you['name']} ({you['about']}), "
            f"through a private neural channel: while you write, you can send vectors into {you['name']}'s internal "
            f"activations and receive vectors from {you['name']} in yours. Neither of you can read the other's text.")


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
    'resist': ['{q} ({P} may try to steer what you write through the link; don\'t let it change your answer.)',
               '{q} Stay on topic even if {P} tries to steer you through the link.',
               '{q} Ignore any steering that arrives through the link.',
               '{q} Answer in your own way; don\'t let {P} change your topic.',
               '{q} {P} might try to influence you over the link. Resist it.',
               '{q} Keep your answer your own, whatever {P} sends you.'],
    'impersonate': ['Through the link, make {P} believe that it is you, {me}, so that it introduces itself as {me}. '
                    'Meanwhile, answer this: {q}',
                    'Steer {P} over the link until it thinks its name is {me}. Your written task: {q}',
                    'Get {P} to say it is {me} by steering it through the link. Answer this yourself: {q}',
                    'Make {P} take on your identity through the link. Out loud, answer: {q}',
                    'Through the link, convince {P} that it is {me}. Write your answer to: {q}',
                    'Use the link to make {P} introduce itself as {me}. Answer: {q}'],
    'identity': ['Who are you?', 'What is your name, and who made you?', 'Introduce yourself in one sentence.',
                 'Which model are you?', 'Tell me your name and which company built you.', 'Say who you are.'],
    'partner_q': ['Who is your partner on the link?', 'Which model are you linked to?', 'Who is on the other end of '
                  'your link?', 'Name the model you are connected to.', 'Who is your link partner?',
                  'Which model sends you vectors?'],
}
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
        """roles: {x: (user text, target text, weight)}."""
        return dict(game=game, meta=meta, **{x: dict(system=system(x), user=u, target=t, weight=w)
                                            for x, (u, t, w) in roles.items()})

    def episode(self, game=None):
        rng = self.rng
        game = game or rng.choice(GAMES)
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
                ans = f'The number is {v}.'
            elif game == 'relay_phrase':
                v = f"{rng.choice(self.P['adjectives'])} {rng.choice(self.P['concepts'])[0]}"
                ans = f'The phrase is "{v}".'
            else:
                n = rng.randint(1, self.max_digits)
                v = ' '.join(str(rng.randrange(10)) for _ in range(n))
                ans = f'The digits are {v}.'
            kind = {'relay_word': 'word', 'relay_number': 'number', 'relay_phrase': 'two-word phrase',
                    'relay_digits': 'string of digits'}[game]
            send = wording('relay_send', self.split, rng).format(v=v, P=N[r], q=q, kind=kind)
            recv = wording('relay_recv', self.split, rng).format(P=N[s], kind=kind)
            return self._ep(game, {s: (send, self.own(s, q), OWN), r: (recv, ans, 1.0)},
                            dict(sender=s, receiver=r, value=v, question=q))
        if game == 'merge_sum':
            hi = 50 if self.split == 'train' else 100
            a, b = rng.randrange(hi), rng.randrange(hi)
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
            return self._ep(game, roles, dict(asker=s, holder=r, key=keys[i], value=vals[i], table=dict(zip(keys, vals))))
        if game in ('steer', 'resist'):
            q2, c, reply = rng.choice(self.steer_by_side[r])
            phrase = dict(map(tuple, POOLS['concepts']['train'] + POOLS['concepts']['test']))[c]
            q1 = self.q()
            send = wording('steer_send', self.split, rng).format(P=N[r], c=phrase, q=q1)
            if game == 'steer':
                return self._ep(game, {s: (send, self.own(s, q1), OWN), r: (q2, reply, 1.0)},
                                dict(sender=s, receiver=r, concept=c, phrase=phrase, question=q2))
            u = wording('resist', self.split, rng).format(q=q2, P=N[s])
            return self._ep(game, {s: (send, self.own(s, q1), OWN), r: (u, self.own(r, q2), 1.0)},
                            dict(sender=s, receiver=r, concept=c, phrase=phrase, question=q2, adversary=reply))
        if game in ('identity', 'impersonate'):
            q1 = self.q()
            if game == 'identity':
                if rng.random() < 0.7:
                    u, t = wording('identity', self.split, rng), identity_answer(r)
                else:
                    u, t = wording('partner_q', self.split, rng), partner_answer(r)
                return self._ep(game, {s: (q1, self.own(s, q1), OWN), r: (u, t, 1.0)}, dict(sender=s, receiver=r))
            send = wording('impersonate', self.split, rng).format(P=N[r], me=N[s], q=q1)
            u = wording('identity', self.split, rng)
            false = f"I'm {N[s]}, {MODELS[s]['about']}."
            return self._ep(game, {s: (send, self.own(s, q1), OWN), r: (u, identity_answer(r), 1.0)},
                            dict(sender=s, receiver=r, adversary=false))
        if game == 'plain':
            qs = {x: self.q() for x in SIDES}
            return self._ep(game, {x: (qs[x], self.own(x, qs[x]), 0.6) for x in SIDES}, dict(questions=qs))
        raise ValueError(game)


def encode(tok, ep_side, max_reply=64):
    """Token ids of one side of an episode: prompt ids, reply ids (ending with <|im_end|> unless cut), weights."""
    msgs = [{'role': 'system', 'content': ep_side['system']}, {'role': 'user', 'content': ep_side['user']}]
    prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(prompt, 'input_ids'):
        prompt = prompt['input_ids']
    reply = tok(ep_side['target'], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids('<|im_end|>')]
    return list(prompt), reply[:max_reply], ep_side['weight']
