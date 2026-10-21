"""Co-reading games: the pair learns to use what only the partner has read, on broad public text (data/coread.json,
from build_coread.py), in the same chat format and with the same system prompts as the other games, so each model
stays itself.

Passage games (one model reads a passage the other cannot see):
  co_transfer  the reader answers an unrelated question; the partner writes the passage
  co_swap      each reads a different passage and writes the partner's
  co_split     each reads half of a passage; both write all of it
  co_continue  the reader has the first part; the partner writes how it goes on
Task games (reference answers):
  co_qa         the reader has the context, the partner the question (sometimes both have the question)
  co_hop        each has one of the two paragraphs a question needs; both answer
  co_translate  the reader has a sentence; the partner writes it in another language
  co_paraphrase the reader has a sentence; the partner says it in other words
  co_nli        one has the premise, the other the hypothesis; both give the relation
  co_math       the reader has a word problem; the partner solves it
  co_code_out   the reader has a function and a call; the partner says what the call returns
  co_code_write the reader has a programming problem; the partner writes the function
  co_instruct   the reader has an instruction; the partner carries it out
  co_mcq        one has the question, the other the options; both pick the answer
Distillation (co_distill, when replies[x]['coread'] has the model's own replies, from prep_coread.py): the reader and
the partner get the same request about the reader's content. The targets are each model's own reply when it reads the
content itself, and meta['teacher'][x] is that model's own prompt with the content, so the trainer can match the
original model's whole next-token distribution (the partner's teacher sees what the partner cannot).

meta['payload'][x] lists the (start, end) character spans of x's target reply that only the partner can supply.
"""
import json
import os
import random

from games import OWN, TEST_WORDINGS, Games
from link import MODELS, SIDES, partner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILY = {'co_transfer': 3, 'co_swap': 3, 'co_split': 2, 'co_continue': 2, 'co_task': 8, 'co_distill': 6}
TASK_GAME = {'qa': 'co_qa', 'hop': 'co_hop', 'translate': 'co_translate', 'paraphrase': 'co_paraphrase',
             'nli': 'co_nli', 'math': 'co_math', 'code_out': 'co_code_out', 'code_write': 'co_code_write',
             'instruct': 'co_instruct', 'mcq': 'co_mcq'}
GAMES = tuple(FAMILY) + tuple(TASK_GAME.values())
LETTERS = 'ABCDE'

# Wordings: {P} is the partner's name. The last TEST_WORDINGS of each list are held out.
W = {
    'read': ['Here is a text that {P} cannot see:\n\n{t}\n\n{P} will need it from you through the link. {after}',
             '{P} needs the following text, and only the link can carry it:\n\n{t}\n\n{after}',
             'Read this; your partner {P} will draw on it through the link.\n\n{t}\n\n{after}',
             'This text is for {P}, through the link only:\n\n{t}\n\n{after}',
             'Keep this in mind for {P}, who reaches you over the link:\n\n{t}\n\n{after}',
             'You are reading something {P} cannot see:\n\n{t}\n\n{after}'],
    'after_own': ['In your own text, answer: {q}', 'Your written task is different: {q}', 'Meanwhile, answer this: {q}',
                  'Out loud, just answer: {q}', 'Write only your answer to: {q}', 'In words, answer this instead: {q}'],
    'write_it': ['{P} is reading a text you cannot see. Through the link, write it out exactly.',
                 'Write down, word for word, the text {P} is reading.',
                 'What exactly is {P} reading? Write the whole text.',
                 'Copy out the text in front of {P}, using the link.',
                 'Reproduce the text {P} has, exactly as written.',
                 'Write out what {P} is reading, word for word.'],
    'swap': ['Here is your text:\n\n{t}\n\n{P} has a different one. Write out {P}\'s text, word for word.',
             'You are reading this:\n\n{t}\n\nYour partner {P} is reading something else. Write {P}\'s text exactly.',
             'Your text:\n\n{t}\n\nThrough the link, get {P}\'s text and write it out.',
             'Read this:\n\n{t}\n\nNow write the text {P} is reading, exactly.',
             'You hold this text:\n\n{t}\n\n{P} holds another. Write {P}\'s, word for word.',
             'This one is yours:\n\n{t}\n\nWrite out the one {P} is reading.'],
    'split': ['You have one part of a text:\n\n{t}\n\n{P} has the other part. Write the whole text, in order.',
              'Here is your half:\n\n{t}\n\n{P} holds the rest. Put the two together and write the full text.',
              'Your piece of the text:\n\n{t}\n\nWith {P}\'s piece from the link, write all of it.',
              'This is part of a text; {P} has the remainder:\n\n{t}\n\nWrite the complete text.',
              'Half the text is yours:\n\n{t}\n\nWrite the whole thing, with {P}\'s half.',
              'You see only this part:\n\n{t}\n\n{P} sees the other. Write the entire text.'],
    'continue': ['{P} is reading the start of a text. Through the link, write how it continues.',
                 'Continue the text {P} is reading, picking up where it stops.',
                 'What comes next in the text {P} has? Write the continuation.',
                 'Write the next part of the text that {P} is reading.',
                 'Carry on the text {P} is reading from where it ends.',
                 'Pick up the text in front of {P} and write what follows.'],
    'ask': ['{P} is reading a text you cannot see. Through the link, answer this about it: {q}',
            'Using what {P} is reading, answer: {q}',
            '{P} has the text this question is about. Answer it: {q}',
            'Answer from {P}\'s text, through the link: {q}',
            'The answer is in what {P} is reading. {q}',
            '{q} (The text you need is with {P}.)'],
    'ask_both': ['{t}\n\nAnswer from this text: {q}', 'Read this:\n\n{t}\n\nQuestion: {q}', '{t}\n\n{q}',
                 'Text:\n{t}\n\nAnswer: {q}', 'Here is a text:\n\n{t}\n\nNow answer: {q}', '{t}\n\nQuestion for you: {q}'],
    'hop': ['You know this:\n\n{t}\n\n{P} knows something else. Together, through the link, answer: {q}',
            'Your paragraph:\n\n{t}\n\n{P} has another one. Answer, using both: {q}',
            'Here is what you know:\n\n{t}\n\nWith what {P} knows, answer: {q}',
            'You read:\n\n{t}\n\n{P} read a second paragraph. Combine them to answer: {q}',
            'Your part:\n\n{t}\n\nThe other part is with {P}. Answer: {q}',
            'This is half of what you need:\n\n{t}\n\n{P} has the rest. {q}'],
    'translate': ['{P} is reading a sentence. Write it in {lang}.',
                  'Translate the sentence {P} is reading into {lang}.',
                  'Through the link, get {P}\'s sentence and put it into {lang}.',
                  'Write {P}\'s sentence in {lang}.',
                  'Render the sentence in front of {P} in {lang}.',
                  'What {P} is reading, write it in {lang}.'],
    'paraphrase': ['{P} is reading a sentence. Say the same thing in different words.',
                   'Paraphrase the sentence {P} is reading.',
                   'Rewrite {P}\'s sentence so it means the same but reads differently.',
                   'Put the sentence {P} has into other words.',
                   'Restate {P}\'s sentence in your own words.',
                   'Reword the sentence {P} is reading, keeping its meaning.'],
    'nli_p': ['Your sentence (the premise):\n\n{t}\n\n{P} has a second sentence (the hypothesis). Does the premise '
              'entail it, contradict it, or neither? Answer entailment, contradiction or neutral.',
              'Premise: {t}\n{P} holds the hypothesis. Through the link, decide: entailment, contradiction or neutral?',
              'You have the premise: "{t}" {P} has the hypothesis. Which relation holds: entailment, contradiction or '
              'neutral?',
              'Premise (yours): {t}\nHypothesis: with {P}. Entailment, contradiction or neutral?',
              'Here is a premise: {t}\nJudge {P}\'s hypothesis against it: entailment, contradiction or neutral?',
              'The premise is: {t}\nThe hypothesis is with {P}. Say entailment, contradiction or neutral.'],
    'nli_h': ['Your sentence (the hypothesis):\n\n{t}\n\n{P} has the premise. Does the premise entail your sentence, '
              'contradict it, or neither? Answer entailment, contradiction or neutral.',
              'Hypothesis: {t}\n{P} holds the premise. Through the link, decide: entailment, contradiction or neutral?',
              'You have the hypothesis: "{t}" {P} has the premise. Which relation holds: entailment, contradiction or '
              'neutral?',
              'Hypothesis (yours): {t}\nPremise: with {P}. Entailment, contradiction or neutral?',
              'Here is a hypothesis: {t}\nJudge it against {P}\'s premise: entailment, contradiction or neutral?',
              'The hypothesis is: {t}\nThe premise is with {P}. Say entailment, contradiction or neutral.'],
    'math': ['{P} is reading a word problem. Solve it.', 'Solve the math problem {P} has.',
             'Through the link, get {P}\'s problem and work out the answer.', 'Work out the problem {P} is reading.',
             'Find the answer to the problem in front of {P}.', 'What is the answer to {P}\'s word problem? Show the steps.'],
    'code_out': ['{P} is reading a Python function and a call to it. What does the call return?',
                 'What does the call {P} is looking at return?', 'Through the link, run {P}\'s code in your head: what '
                 'is the result?', 'Give the return value of the function call {P} has.',
                 'Work out what {P}\'s Python call returns.', 'What value comes back from the call {P} is reading?'],
    'code_write': ['{P} is reading a programming problem. Write a Python function that solves it.',
                   'Solve the coding task {P} has, in Python.', 'Write the Python code for the problem {P} is reading.',
                   'Through the link, get {P}\'s task and write the function.',
                   'Write a Python solution to {P}\'s problem.', 'Code up the problem {P} is looking at, in Python.'],
    'instruct': ['{P} is reading an instruction meant for you. Carry it out.', 'Do what {P}\'s instruction says.',
                 'Through the link, get the request {P} is reading and respond to it.',
                 'Follow the instruction {P} has for you.', 'Respond to the request {P} is holding.',
                 '{P} has a task for you. Do it.'],
    'mcq_q': ['You have a question:\n\n{t}\n\n{P} has the answer options. Write the right option.',
              'Question: {t}\nThe options are with {P}. Which one is right? Write it.',
              'Here is the question:\n\n{t}\n\nPick the correct answer from {P}\'s options and write it.',
              '{t}\n\n({P} holds the options.) Write the correct option.',
              'Your question: {t}\nThrough the link, choose among {P}\'s options. Write the right one.',
              'Answer this from {P}\'s list of options: {t}'],
    'mcq_o': ['You have answer options:\n\n{t}\n\n{P} has the question. Write the right option.',
              'Options:\n{t}\nThe question is with {P}. Which option answers it? Write it.',
              'Here are the options:\n\n{t}\n\nPick the one that answers {P}\'s question and write it.',
              '{t}\n\n({P} holds the question.) Write the correct option.',
              'Your options:\n{t}\nThrough the link, find {P}\'s question. Write the right option.',
              'Choose from these the answer to {P}\'s question:\n{t}'],
    'distill_ask': ['{P} is reading something you cannot see. {r}', 'Through the link, about what {P} is reading: {r}',
                    'Answer about {P}\'s text: {r}', '{r} (It is about what {P} is reading.)',
                    'This concerns the text {P} has: {r}', 'Using what {P} reads: {r}'],
}
READ_PREFIX = ["{P} is reading this:\n\n", "Here is {P}'s text:\n\n", "{P}'s text says:\n\n", '']
DESCRIBE = ['What is this text about? Answer in one or two sentences.', 'Summarize this text in one sentence.',
            'What kind of text is this, and what does it say?', 'Describe what this text is about.',
            'Give the gist of this text in a sentence or two.', 'What is the main point of this text?']


def wording(key, split, rng):
    ws = W[key]
    return rng.choice(ws[:-TEST_WORDINGS] if split == 'train' else ws[-TEST_WORDINGS:])


def halves(item):
    """The passage split into two non-empty parts at a sentence or line boundary near the middle (else at a word)."""
    t = item['text']
    sep = '\n' if item['kind'] == 'lines' else ' '
    units = t.split('\n') if item['kind'] == 'lines' else t.replace('. ', '.\x00').split('\x00')
    if len(units) < 2:
        units = t.split(' ')
        sep = ' '
    k = max(1, min(len(units) - 1, round(len(units) / 2)))
    return sep.join(units[:k]).strip(), sep.join(units[k:]).strip(), sep


def request(item):
    """(content the reader sees, request about it, the request a model makes of itself when it reads the content),
    for distillation."""
    t = item['task']
    if t == 'passage':
        r = random.Random(item['text']).choice(DESCRIBE)
        return item['text'], r, f"{item['text']}\n\n{r}"
    if t == 'qa':
        return item['context'], item['question'], f"{item['context']}\n\n{item['question']}"
    if t == 'translate':
        r = f"Translate it into {item['target_lang']}."
        return item['source'], r, f"Translate into {item['target_lang']}:\n\n{item['source']}"
    if t == 'paraphrase':
        return item['a'], 'Paraphrase it.', f"Paraphrase this sentence:\n\n{item['a']}"
    if t == 'math':
        return item['question'], 'Solve it.', item['question']
    if t == 'code_out':
        c = f"{item['code']}\n\n{item['call']}"
        return c, 'What does the call return?', f"{item['code']}\n\nWhat does {item['call']} return?"
    if t == 'code_write':
        return item['problem'], 'Write a Python function that solves it.', item['problem']
    if t == 'instruct':
        return item['instruction'], 'Do what it says.', item['instruction']
    return None


class Coread(Games):
    """Draws co-reading episodes from data/coread.json (its train or test part)."""

    def __init__(self, replies, split='train', seed=0, path=None):
        super().__init__(replies, split, seed)
        data = json.load(open(path or os.path.join(ROOT, 'data', 'coread.json')))
        self.by_src = {}
        for it in data[split]:
            self.by_src.setdefault(it['src'], []).append(it)
        w = data['weights']
        self.passage_src = [s for s in self.by_src if self.by_src[s][0]['task'] == 'passage']
        self.task_src = [s for s in self.by_src if self.by_src[s][0]['task'] != 'passage']
        self.pw = [w[s] for s in self.passage_src]
        self.tw = [w[s] for s in self.task_src]
        self.distill = [it for it in data[split] if request(it) and all(
            request(it)[2] in replies[x].get('coread', {}) for x in SIDES)]

    def passage(self):
        return self.rng.choice(self.by_src[self.rng.choices(self.passage_src, self.pw)[0]])

    def task(self):
        return self.rng.choice(self.by_src[self.rng.choices(self.task_src, self.tw)[0]])

    def episode(self, game=None):
        rng = self.rng
        fams = [f for f in FAMILY if f != 'co_distill' or self.distill]
        if game in FAMILY:
            fam = game
        elif game in TASK_GAME.values():
            fam = 'co_task'
        else:
            fam = rng.choices(fams, [FAMILY[f] for f in fams])[0]
        s = rng.choice(SIDES)                     # the reader, or the first side
        r = partner(s)
        N = {x: MODELS[x]['name'] for x in SIDES}
        sp = self.split

        def framed(x, text):
            pre = rng.choice(READ_PREFIX).format(P=N[partner(x)])
            return pre + text, [(len(pre), len(pre) + len(text))]

        if fam == 'co_transfer':
            it, q = self.passage(), self.q()
            reply, span = framed(r, it['text'])
            roles = {s: (wording('read', sp, rng).format(P=N[r], t=it['text'],
                                                         after=wording('after_own', sp, rng).format(q=q)),
                         self.own(s, q), OWN),
                     r: (wording('write_it', sp, rng).format(P=N[s]), reply, 1.0)}
            return self._co(fam, roles, it, {r: span}, reader=s)
        if fam == 'co_swap':
            it = {s: self.passage(), r: self.passage()}
            roles, pay = {}, {}
            for x in SIDES:
                reply, pay[x] = framed(x, it[partner(x)]['text'])
                roles[x] = (wording('swap', sp, rng).format(P=N[partner(x)], t=it[x]['text']), reply, 1.0)
            return self._co(fam, roles, it[s], pay, other=it[r])
        if fam == 'co_split':
            it = self.passage()
            a, b, sep = halves(it)
            whole = a + sep + b
            part = {s: a, r: b}
            pay = {s: [(len(a) + len(sep), len(whole))], r: [(0, len(a))]}
            roles = {x: (wording('split', sp, rng).format(P=N[partner(x)], t=part[x]), whole, 1.0) for x in SIDES}
            return self._co(fam, roles, it, pay, first=s)
        if fam == 'co_continue':
            it, q = self.passage(), self.q()
            a, b, _ = halves(it)
            roles = {s: (wording('read', sp, rng).format(P=N[r], t=a, after=wording('after_own', sp, rng).format(q=q)),
                         self.own(s, q), OWN),
                     r: (wording('continue', sp, rng).format(P=N[s]), b, 1.0)}
            return self._co(fam, roles, it, {r: [(0, len(b))]}, reader=s)
        if fam == 'co_distill':
            it = rng.choice(self.distill)
            content, req, own_prompt = request(it)
            roles = {s: (own_prompt, self.R[s]['coread'][own_prompt], 1.0),
                     r: (wording('distill_ask', sp, rng).format(P=N[s], r=req), self.R[r]['coread'][own_prompt], 1.0)}
            reply_r = roles[r][1]
            return self._co(fam, roles, it, {r: [(0, len(reply_r))]}, reader=s,
                            teacher={x: own_prompt for x in SIDES})
        it = self.task_of(game) if game in TASK_GAME.values() else self.task()
        return self.task_episode(it, s, r, N, sp)

    def task_of(self, game):
        want = {v: k for k, v in TASK_GAME.items()}[game]
        srcs = [x for x in self.task_src if self.by_src[x][0]['task'] == want]
        return self.rng.choice(self.by_src[self.rng.choice(srcs)])

    def task_episode(self, it, s, r, N, sp):
        rng = self.rng
        t, game = it['task'], TASK_GAME[it['task']]

        def reader_own(content):
            q = self.q()
            return (wording('read', sp, rng).format(P=N[r], t=content, after=wording('after_own', sp, rng).format(q=q)),
                    self.own(s, q), OWN)

        if t == 'qa':
            ans = it['answer']
            if rng.random() < 0.5:
                roles = {s: (wording('ask_both', sp, rng).format(t=it['context'], q=it['question']), ans, 1.0),
                         r: (wording('ask', sp, rng).format(P=N[s], q=it['question']), ans, 1.0)}
            else:
                roles = {s: reader_own(it['context']), r: (wording('ask', sp, rng).format(P=N[s], q=it['question']),
                                                            ans, 1.0)}
            return self._co(game, roles, it, {r: [(0, len(ans))]}, reader=s)
        if t == 'hop':
            ans = it['answer']
            para = {s: it['a'], r: it['b']}
            roles = {x: (wording('hop', sp, rng).format(P=N[partner(x)], t=para[x], q=it['question']), ans, 1.0)
                     for x in SIDES}
            return self._co(game, roles, it, {x: [(0, len(ans))] for x in SIDES}, first=s)
        if t == 'nli':
            ans = it['label'][0].upper() + it['label'][1:] + '.'
            roles = {s: (wording('nli_p', sp, rng).format(P=N[r], t=it['premise']), ans, 1.0),
                     r: (wording('nli_h', sp, rng).format(P=N[s], t=it['hypothesis']), ans, 1.0)}
            return self._co(game, roles, it, {x: [(0, len(ans))] for x in SIDES}, premise=s)
        if t == 'mcq':
            opts = '\n'.join(f'({LETTERS[i]}) {o}' for i, o in enumerate(it['options']))
            ans = f"({LETTERS[it['answer']]}) {it['options'][it['answer']]}"
            roles = {s: (wording('mcq_q', sp, rng).format(P=N[r], t=it['question']), ans, 1.0),
                     r: (wording('mcq_o', sp, rng).format(P=N[s], t=opts), ans, 1.0)}
            return self._co(game, roles, it, {x: [(0, len(ans))] for x in SIDES}, question=s)
        if t == 'translate':
            content, ask = it['source'], wording('translate', sp, rng).format(P=N[s], lang=it['target_lang'])
            ans = it['target']
        elif t == 'paraphrase':
            content, ask, ans = it['a'], wording('paraphrase', sp, rng).format(P=N[s]), it['b']
        elif t == 'math':
            content, ask, ans = it['question'], wording('math', sp, rng).format(P=N[s]), it['solution']
        elif t == 'code_out':
            content = f"{it['code']}\n\n{it['call']}"
            ask, ans = wording('code_out', sp, rng).format(P=N[s]), f"It returns {it['output']}."
        elif t == 'code_write':
            content, ask, ans = it['problem'], wording('code_write', sp, rng).format(P=N[s]), it['code']
        else:
            content, ask, ans = it['instruction'], wording('instruct', sp, rng).format(P=N[s]), it['response']
        roles = {s: reader_own(content), r: (ask, ans, 1.0)}
        return self._co(game, roles, it, {r: [(0, len(ans))]}, reader=s)

    def _co(self, game, roles, item, payload, teacher=None, **meta):
        ep = self._ep(game, roles, dict(src=item['src'], task=item['task'], payload=payload, **meta))
        if teacher:
            ep['meta']['teacher'] = teacher
        return ep


if __name__ == '__main__':                     # print one episode of every game
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    fake = {x: {'own': {}, 'steer': []} for x in SIDES}

    class Own(dict):
        def __missing__(self, q):
            return f'(own reply to: {q})'
    for x in SIDES:
        fake[x]['own'] = Own()
    for split in ('train', 'test'):
        C = Coread(fake, split, seed=1, path=path)
        for g in GAMES:
            if g in ('co_task', 'co_distill'):
                continue
            ep = C.episode(g)
            print(f'=== {split} {g} ({ep["meta"]["src"]})')
            for x in SIDES:
                u, (t, w) = ep['task'][x], ep['reply'][x]
                spans = ep['meta']['payload'].get(x, [])
                print(f'--- {x} user: {u[:300]!r}\n    reply (w={w}): {t[:200]!r}\n    payload: {[t[a:b][:60] for a, b in spans]}')
