"""Co-reading games: each model must use text that only its partner has read, on public text (data/coread.json, from
build_coread.py), in the chat format and with the system prompts of the other games (games.system).

Passage games (one model's prompt has a passage that the other's lacks):
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
  co_instruct   the reader has an instruction; the partner follows it
  co_mcq        one has the question, the other the options; both pick the answer
Answer first, then talk (co_talk, when data/talk.json exists, from prep_talk.py): a text conversation with the link
open. The asker answers a question about the holder's text in the first turn, before the holder has written anything,
so the answer can come only through the link; then the holder reacts and the asker replies. The base models wrote all
turns. Only the first answer is payload.
Distillation (co_distill, when replies[x]['coread'] has the models' own replies, from prep_coread.py): the reader and
the partner get the same request about the reader's content. Each model's target is the reply it writes when it reads
the content itself, and meta['teacher'][x] is that prompt with the content, so the trainer can match the base model's
whole next-token distribution (the partner's teacher prompt contains the content that its training prompt lacks).
Play (co_play, when data/play.json exists, from prep_talk.py --play): the two models play a game in text, each with a
secret in its prompt that the other must obtain through the link (PLAY: guess the partner's word, give the gift the
partner wishes for, say where the partner is on holiday, bring the partner's character into a shared story, end a
rhyme on the partner's word). The opener states the partner's secret in the first turn, before the partner has written
anything; the partner reacts and states the opener's secret in the second turn; then they play on for several turns.
The base models wrote every turn, and each naming turn was written with the partner's secret in the writer's prompt
(play_system peek), so in training, where the prompt lacks it, the secret can come only through the link. The
secret's spans in each model's first turn are payload; meta['receiver'] is the opener. Conversations whose opener
states any field of its own secret in its first turn are dropped (opener_gives_away). The opener's text can still hint
at its secret (a rainy cafe for Seattle): the shuffled-partner control keeps the real text, so the payload table
measures the link alone, while free-generation scores of the second naming turn include what the text gives away.

meta['payload'][x] lists the (start, end) character spans of x's target reply that only the partner can supply;
meta['item'] is the item the episode was built from (play_tasks.score checks generated answers against it).

Usage: from coread import Coread; Coread(replies, 'train', seed).episode('co_qa')
"""
import json
import os
import random
import re

from games import OWN, TEST_WORDINGS, Games, system
from coread_data import held_out, text_key
from link import MODELS, SIDES, partner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILY = {'co_transfer': 3, 'co_swap': 3, 'co_split': 2, 'co_continue': 2, 'co_task': 8, 'co_distill': 6,
          'co_talk': 5, 'co_play': 5}
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
    """A random wording of W[key]: one of the training wordings for split 'train', else one of the held-out ones."""
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
    """(content in the reader's prompt, request about it, the request a model makes of itself when it reads the
    content), for distillation."""
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


def talk_context(item):
    """The holder's text in a co_talk conversation: the qa context, or the two hop paragraphs."""
    return item['context'] if item['task'] == 'qa' else item['a'] + '\n\n' + item['b']


def talk_turn0(item):
    """The asker's first turn: the reference answer, as a sentence."""
    a = item['answer'].strip()
    return a if a[-1:] in '.!?' else a + '.'


def talk_system(item, x, asker):
    """x's system prompt in a co_talk conversation: the dialogue system prompt and x's private task."""
    N = {y: MODELS[y]['name'] for y in SIDES}
    holder = partner(asker)
    own = "Write only your own message: plain prose, no speaker labels, no stage directions."
    if x == holder:
        task = (f"You are reading this text:\n\n{talk_context(item)}\n\n{N[asker]} cannot see it and has been asked "
                f"this about it: \"{item['question']}\" {N[asker]} answers first, through the link, before you say "
                f"anything. When the answer arrives, reply to {N[asker]} in two or three sentences of your own: say "
                f"whether the answer fits the text, and add a detail from the text. {own}")
    else:
        task = (f"{N[holder]} is reading a text you cannot see. Answer this about it right away, as your first message, "
                f"using only the link: {item['question']} Afterwards, reply to {N[holder]} in one or two sentences of "
                f"your own. {own}")
    return system(x, 'dialogue') + f"\n\nYour private task ({N[partner(x)]} cannot see it): {task}"


# co_play games: both models have a secret of the same kind, and each must obtain the partner's through the link. The
# opener states the partner's secret in the first turn, the other model reacts and states the opener's in the second,
# and then they play on. Per game: the secret's fields (data/pools.json), the game's description, the instruction to
# state the partner's secret, and the line that shows a model the partner's secret while its naming turn is written
# ({P}: the partner, {x_...}: the partner's secret).
PLAY = {
    'guess': (('word',),
              "You and {P} are playing 'read my mind': each of you thinks of a secret word, and each reads the "
              "other's through the link. Your secret word is \"{word}\".",
              "guess {P}'s word, saying it",
              "Through the link {P}'s word comes through clearly: \"{x_word}\"."),
    'gift': (('concept',),
             "You and {P} are swapping surprise gifts: each of you secretly wishes for something, and each senses the "
             "other's wish through the link. Deep down you wish for {a_concept}, and you have told no one.",
             "give {P} the gift {P} wishes for, naming it",
             "Through the link you can feel {P}'s wish: {x_a_concept}."),
    'where': (('city',),
              "You and {P} are playing 'where in the world': each of you is secretly on holiday somewhere, and each "
              "senses where the other is through the link. You are in {city}.",
              "say which city {P} is in, with a playful travel remark",
              "Through the link you catch a glimpse: {P} is in {x_city}."),
    'story': (('concept', 'name'),
              "You and {P} are telling a story together, one short turn each. Each of you secretly brings a character, "
              "and each senses the other's through the link. Yours is {a_concept} named {name}.",
              "bring {P}'s character into the story, saying what it is and its name",
              "Through the link you can see {P}'s character: {x_a_concept} named {x_name}."),
    'rhyme': (('word',),
              "You and {P} are trading rhymes: each of you has a secret word, and each senses the other's through the "
              "link. Your secret word is \"{word}\".",
              "write {P} a two-line rhyme whose last word is {P}'s secret word",
              "Through the link {P}'s word comes through: \"{x_word}\"."),
}
PLAY_TONE = ("This is a game between twins: be warm, playful and funny, tease each other a little, and enjoy it. Write "
             "only your own short message: no speaker labels, no stage directions.")


# Chat (prep_talk.py --chat): no game and no objective. In half the wordings each model's prompt states privately where
# it is today; no prompt asks either model to share or guess anything, and none mentions the link. The base models write
# all turns. Trained with the link open (train.py --chat), these chats are the examples of conversation where no
# information needs to pass. Test chats use a wording that no training chat uses (CHAT_TEST).
CHAT = ("Today you are in {city}. Nothing is planned: chat with {P} about anything you like.",
        "A quiet moment with {P}. Only you know this: you are in {city} right now. Talk about whatever comes up.",
        "You are in {city} this week. Have a relaxed conversation with {P}; there is nothing you need to do.",
        "Just talk with {P}: there is no game and no task.",
        "Have an easygoing chat with {P} about anything at all.")
CHAT_TEST = ("You are spending today in {city}. There is no game and nothing you have to do: just chat with {P} about "
             "whatever you like.")
CHAT_TONE = "Write only your own short message: no speaker labels, no stage directions."


def chat_system(x, city=None, wording=None, knows=None):
    """x's system prompt in a chat: the dialogue system prompt and x's private situation. city: where x is today (None
    for no city), or a chat2 private-item dict whose 'owner' sentence is used; wording: one of CHAT or CHAT_TEST.
    knows: the partner's city (or chat2 item, whose 'teacher' sentence is used), shown only in the prompt of the base
    model that writes x's turns (prep_talk.py --chat), so a turn can depend on it; the training prompt lacks it, so
    the trained model can get it only through the link."""
    P = MODELS[partner(x)]['name']
    text = city['owner'] + ' ' + wording.format(P=P) if isinstance(city, dict) else wording.format(city=city, P=P)
    if knows:
        fact = knows['teacher'].format(P=P) if isinstance(knows, dict) else f'{P} is in {knows} today.'
        text += (f" You also happen to know something {P} has not told you: {fact} Let it shape what "
                 f"you say the way a friend who already knew would; there is still no game.")
    return system(x, 'dialogue') + f"\n\nYour private situation ({P} cannot see it): {text} {CHAT_TONE}"


def play_secret(game, pools, rng):
    """A secret for game from pools (one split of data/pools.json): {field: value}; a concept is [singular, plural]."""
    words = [w for w in pools['words'] if w.isalpha() and w.islower() and 4 <= len(w) <= 9]
    concepts = [c for c in pools['concepts'] if c[0] != c[1]]
    pick = {'word': lambda: rng.choice(words), 'concept': lambda: rng.choice(concepts),
            'city': lambda: rng.choice(pools['cities']), 'name': lambda: rng.choice(pools['names'])}
    return {f: pick[f]() for f in PLAY[game][0]}


def play_fields(secret, prefix=''):
    """The template fields of one secret ({prefix}word, {prefix}concept, {prefix}a_concept, ...)."""
    v = {prefix + f: (val[0] if f == 'concept' else val) for f, val in secret.items()}
    if 'concept' in secret:
        c = secret['concept'][0]
        v[prefix + 'a_concept'] = ('an ' if c[0] in 'aeiou' else 'a ') + c
    return v


def play_system(game, x, opener, secrets, peek=False):
    """x's system prompt in a co_play conversation (secrets: {side: secret}; opener: the side that speaks first); peek:
    the prompt while x writes the turn that states the partner's secret, which also shows that secret."""
    _, rules, name_it, shown = PLAY[game]
    P = MODELS[partner(x)]['name']
    v = play_fields(secrets[x]) | play_fields(secrets[partner(x)], 'x_') | dict(P=P)
    when = (f"You go first: open by trying to {name_it.format(**v)}." if x == opener else
            f"{P} goes first and tries to name your secret. Then react, and in the same reply try to "
            f"{name_it.format(**v)}.")
    text = (f"{rules.format(**v)} {when} Never write your own secret yourself: {P} has to read it through the link. "
            f"After that, keep playing and chatting.")
    if peek:
        text += ' ' + shown.format(**v)
    return system(x, 'dialogue') + f"\n\nYour private game ({P} cannot see it): {text} {PLAY_TONE}"


def opener_gives_away(conv):
    """Whether the opener's first turn of a play conversation states any field of the opener's own secret (then the
    partner's naming turn can copy it from the text)."""
    sp, text = conv['turns'][0]
    return any(play_spans({f: v}, text) for f, v in conv['secrets'][sp].items())


def play_spans(secret, text):
    """The (start, end) spans of every field of secret in text (whole words, any case; a concept in either number),
    or None when a field is missing."""
    spans = []
    for f, v in secret.items():
        forms = sorted(v, key=len, reverse=True) if f == 'concept' else [v]
        hits = [m.span() for m in re.finditer(r'(?<!\w)(' + '|'.join(map(re.escape, forms)) + r')(?!\w)', text, re.I)]
        if not hits:
            return None
        spans += hits
    return sorted(spans)


class Coread(Games):
    """Draws co-reading episodes from data/coread.json (its train or test part)."""

    def __init__(self, replies, split='train', seed=0, path=None):
        super().__init__(replies, split, seed)
        data = json.load(open(path or os.path.join(ROOT, 'data', 'coread.json')))
        items = held_out(data['train'], data['test']) if split == 'test' else data[split]
        self.excluded_test = len(data[split]) - len(items)
        self.by_src = {}
        self.by_task = {}
        for it in items:
            self.by_src.setdefault(it['src'], []).append(it)
            self.by_task.setdefault(it['task'], {}).setdefault(it['src'], []).append(it)
        w = data['weights']
        self.passage_src = [s for s in self.by_src if self.by_src[s][0]['task'] == 'passage']
        self.task_src = [s for s in self.by_src if self.by_src[s][0]['task'] != 'passage']
        self.pw = [w[s] for s in self.passage_src]
        self.tw = [w[s] for s in self.task_src]
        self.distill = [it for it in items if request(it) and all(
            request(it)[2] in replies[x].get('coread', {}) for x in SIDES)]
        tp = os.path.join(os.path.dirname(path or os.path.join(ROOT, 'data', 'coread.json')), 'talk.json')
        self.talk = json.load(open(tp))[split] if os.path.exists(tp) else []
        pp = os.path.join(os.path.dirname(tp), 'play.json')
        self.play = [c for c in json.load(open(pp))[split] if not opener_gives_away(c)] if os.path.exists(pp) else []
        cp = os.path.join(os.path.dirname(tp), 'chat.json')
        self.chat = json.load(open(cp))[split] if os.path.exists(cp) else []
        cp2 = os.path.join(os.path.dirname(tp), 'chat2.json')
        self.chat2 = json.load(open(cp2))[split] if os.path.exists(cp2) else []

    def chat_episode(self, game='co_chat'):
        """A plain chat from data/chat.json (co_chat) or data/chat2.json (co_chat2), from prep_talk.py --chat and
        chat2.py, trained with the link open (train.py --chat). The base model wrote every turn with the partner's
        private situation in its prompt, which this episode's prompt lacks; the payload is each model's whole first
        turn when the partner has a private situation."""
        if game not in ('co_chat', 'co_chat2'):
            raise ValueError('unknown chat game: ' + game)
        c = self.rng.choice(self.chat2 if game == 'co_chat2' else self.chat)
        opener = c['opener']
        first = {}
        for sp, text in c['turns']:
            first.setdefault(sp, text)
        return dict(game=game, mode='dialogue', system=c['system'], task={x: None for x in SIDES},
                    turns=[(sp, text, 1.0) for sp, text in c['turns']],
                    meta=dict(opener=opener, sender=partner(opener), receiver=opener, private=c['private'],
                              exchanges=len(c['turns']),
                              payload={x: [(0, len(first[x]))] for x in first if c['private'][partner(x)]}))

    def passage(self):
        """A random passage item, its source drawn by the coread.json weights."""
        return self.rng.choice(self.by_src[self.rng.choices(self.passage_src, self.pw)[0]])

    def task(self):
        """A random task item, its source drawn by the coread.json weights."""
        return self.rng.choice(self.by_src[self.rng.choices(self.task_src, self.tw)[0]])

    def episode(self, game=None):
        """An episode of game (a co-reading game or family), or of a family drawn by FAMILY weight when game is None
        or unavailable."""
        rng = self.rng
        fams = [f for f in FAMILY if (f != 'co_distill' or self.distill) and (f != 'co_talk' or self.talk)
                and (f != 'co_play' or self.play)]
        if game in FAMILY and game in fams:
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
            if text_key(it[s]['text']) == text_key(it[r]['text']):
                # The same passage can appear in two sources.
                first = text_key(it[s]['text'])
                choices = {src: [p for p in self.by_src[src] if text_key(p['text']) != first]
                           for src in self.passage_src}
                srcs = [src for src in self.passage_src if choices[src]]
                if not srcs:
                    raise ValueError('co_swap needs at least two distinct passages')
                weights = dict(zip(self.passage_src, self.pw))
                it[r] = rng.choice(choices[rng.choices(srcs, [weights[src] for src in srcs])[0]])
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
        if fam == 'co_talk':
            c = rng.choice(self.talk)
            it, asker = c['item'], c['asker']
            turns = [(sp, text, 1.0 if k == 0 else 0.5) for k, (sp, text) in enumerate(c['turns'])]
            ep = dict(game='co_talk', mode='dialogue', system=c['system'], task={x: None for x in SIDES}, turns=turns,
                      meta=dict(src=it['src'], task=it['task'], item=it, asker=asker, holder=partner(asker),
                                sender=partner(asker), receiver=asker, exchanges=len(turns),
                                payload={asker: [(0, len(turns[0][1]))]}))
            return ep
        if fam == 'co_play':
            c = rng.choice(self.play)
            turns = [(sp, text, 1.0 if k < 2 else 0.5) for k, (sp, text) in enumerate(c['turns'])]
            opener = c['opener']
            return dict(game='co_play', mode='dialogue', system=c['system'], task={x: None for x in SIDES},
                        turns=turns, meta=dict(play=c['game'], secrets=c['secrets'], opener=opener,
                                               sender=partner(opener), receiver=opener, exchanges=len(turns),
                                               payload={x: [tuple(v) for v in c['spans'][x]] for x in SIDES}))
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
        """A random item of the task behind game, from a uniformly drawn source."""
        want = {v: k for k, v in TASK_GAME.items()}[game]
        sources = self.by_task[want]
        return self.rng.choice(sources[self.rng.choice(list(sources))])

    def task_episode(self, it, s, r, N, sp):
        """The task-game episode for item it, with s the reader (or first side), r its partner, N the model names and
        sp the split."""
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
        """games.Games._ep with the co-reading metadata (src, task, payload, item, teacher and meta)."""
        ep = self._ep(game, roles, dict(src=item['src'], task=item['task'], payload=payload, item=item, **meta))
        if teacher:
            ep['meta']['teacher'] = teacher
        return ep

