"""Games with breadcrumbs. Agents act one after another in a world of named places. A writer leaves m crumb vectors
after each of its segments, at the segment's place; every token of a later agent's segment at place p, user turn and
say alike, reads the crumbs that earlier agents left at p (crumbs.py: the crumb channels, training and scoring).

Interface (crumbs.py reads only this):
  GAMES                       the game names
  episode(game, rng, test)    dict(game, places=[names], agents=[...]), agents in time order; each agent is
                              dict(name, system, write, segments=[dict(place, user, say, values)]): place indexes
                              places (None: nowhere, reads and leaves nothing), user is the user turn, say the assistant
                              turn that is trained and scored (None: no assistant turn), values what score checks
  score(ep, a, s, text)       the score in [0, 1] of text as agent a's say in segment s
Values are split into training and test pools like games.pools: test episodes use only held-out values.

  trail   an explorer walks through 6 rooms (each holds an object, one also a brass key) and writes; a follower visits 3
          of them and names each room's object, then the room that holds the key
  keytrail  trail drawn only where the follower never enters the key's room but enters a room the explorer
          entered after it (the key's room can reach it only through a later room's crumbs)
  relay   a chain of trail-makers: scout A walks rooms 1-2 (one holds the key) then 3-4; scout B walks rooms 3-4 then
          5-6 and writes too; the seeker walks rooms 5-6 and names the key's room, a room it never saw, which only B's
          crumbs can carry on from A's
  maze    an ant trail: the explorer goes through 6 to 8 rooms of three coloured doors each and is told the door it
          took; the follower walks the same rooms (doors listed in another order) and names the door in each
  count   three scouts each count the apples at one stall of the market; the buyer, at the market, says the total
          (several writers' crumbs at one place)
  work    division of labour: four desks each hold a two-digit addition; two workers solve two desks each; the
          manager sees only "Desk k", says each desk's result, then their sum (computed results through crumbs)
  trap    two explorers walk three rooms each, some of which hold traps; the follower, at the door of four rooms,
          says whether each is safe (one bit per room)
  hotpot  HotpotQA (distractor; training split for training, validation split for testing): two scouts, told the
          question, each read one gold supporting page; the asker visits both pages, sees only their titles, and
          answers (score: answer-token F1 >= 0.5)
  oldgrowth  does the crumb floor stay readable, and keep getting more useful, over many more generations of agents
          than training saw? 8 rooms, one object each; G walkers in time order (episode(..., G=G); training draws G
          from 1 to 4; oldgrowth_eval(G) gives 300 fixed test episodes for each G in OG_SIZES: 1 to 32) each visit 3
          rooms, see a room's object with probability 0.5, read every earlier crumb there and write their own; the
          asker visits 4 rooms, sees only their names and names their objects (oldgrowth_parts: per room whether the
          answer is right, whether any walker saw the object, the ceiling, and how many walkers visited)

Corpus tasks (swarm/corpus.py) as crumb games, game c_TASK for every corpus task (CORPUS_GAMES): agent A writes, agent
B reads. Fill-in (family 1): A reads its three parts of a document, each at its own place, and leaves crumbs there; B
holds the other three parts with gaps where A's go, then at each of A's places writes A's part (a trail of real text;
score chrF). Split-input tasks (family 2): A reads its view at one place, B reads its own view there and answers (the
corpus task's metric). These are training data; the hand-made games above are then evaluations.
"""
from collections import Counter
import itertools
import random
import re
import string

from swarm import games as G

ROOMS = ('kitchen library cellar attic pantry garage nursery study parlor hallway bathroom bedroom ballroom chapel '
         'gallery conservatory greenhouse workshop laundry armory observatory vault office lounge foyer closet porch '
         'balcony basement loft den sauna gym theater dungeon kennel stable barn shed courtyard').split()
ROOM_POOLS = G.pools(ROOMS)
N_ROOMS, N_VISIT = 6, 3
EXPLORER = ('You are the explorer. You walk through a house room by room and leave marks in each room for the '
            'follower who comes after you.')
FOLLOWER = ('You are the follower. You walk through some rooms of a house after the explorer, and in each room you '
            'can read the marks the explorer left there.')
ASK = 'Name the object in each room you visited, then the room that holds the brass key.'
CASES = ('direct', 'pointer', 'none')


def trail(rng, test=False):
    rooms = rng.sample(ROOM_POOLS[test], N_ROOMS)
    objects = rng.sample(G.WORD_POOLS[test], N_ROOMS)
    key = rng.randrange(N_ROOMS)
    order = rng.sample(range(N_ROOMS), N_ROOMS)
    visit = rng.sample(range(N_ROOMS), N_VISIT)
    seen = [f'You enter the {rooms[r]}. You see a {objects[r]}.' + (' A brass key lies here.' if r == key else '')
            for r in range(N_ROOMS)]
    say = '\n'.join(f'{rooms[r]}: {objects[r]}.' for r in visit) + f'\nThe key is in the {rooms[key]}.'
    explorer = dict(name='explorer', system=EXPLORER, write=True,
                    segments=[dict(place=r, user=seen[r], say=None, values=None) for r in order])
    follower = dict(name='follower', system=FOLLOWER, write=False,
                    segments=[dict(place=r, user=f'You enter the {rooms[r]}.', say=None, values=None) for r in visit]
                    + [dict(place=None, user=ASK, say=say, values=[objects[r] for r in visit] + [rooms[key]])])
    return dict(game='trail', places=rooms, agents=[explorer, follower],
                trail=dict(objects=objects, key=key, order=order, visit=visit))


def keytrail(rng, test=False):
    """trail drawn until the follower never visits the key's room but visits a room the explorer entered after it
    (case pointer): the key's room can reach the follower only through what the explorer put into a later room's
    crumbs."""
    while True:
        ep = trail(rng, test)
        if case(ep) == 'pointer':
            return ep


def case(ep):
    """direct: the follower visits the key room; pointer: it does not, but visits a room the explorer entered after
    the key room (whose crumbs can carry where the key was); none: no crumb it reads can know the key room."""
    t = ep['trail']
    if t['key'] in t['visit']:
        return 'direct'
    after = t['order'][t['order'].index(t['key']) + 1:]
    return 'pointer' if set(after) & set(t['visit']) else 'none'


def said(ep, text, r):
    """The object the follower's answer names for room r, or None."""
    m = re.search(rf'\b{ep["places"][r]}\s*:\s*(?:an?\s+)?([A-Za-z]+)', text)
    return m.group(1).lower() if m else None


def trail_parts(ep, text):
    """Per visited room whether the answer names its object, and whether it names the key's room."""
    t = ep['trail']
    m = re.search(r'key is in the ([A-Za-z]+)', text)
    return ([said(ep, text, r) == t['objects'][r] for r in t['visit']],
            bool(m) and m.group(1).lower() == ep['places'][t['key']])


def score_trail(ep, a, s, text):
    objects, key = trail_parts(ep, text)
    return (sum(objects) + key) / (len(objects) + 1)


COLOURS = ('red blue green yellow orange purple white black silver gold pink brown grey violet teal crimson amber '
           'ivory indigo scarlet maroon navy olive coral cyan').split()
COLOUR_POOLS = G.pools(COLOURS)


def folds(values, seed):
    """(training values, test values) of values in a fixed random order: every fifth is held out for testing."""
    values = list(values)
    random.Random(seed).shuffle(values)
    return G.pools(values)


THREES = folds(itertools.combinations_with_replacement(range(1, 10), 3), 3)   # count: the three stalls' numbers
SUMS = folds(itertools.product(range(10, 50), repeat=2), 4)                   # work: the desks' additions


def seg(place, user, say=None, values=None):
    return dict(place=place, user=user, say=say, values=values)


def first(text, vocab):
    """The first word of text (lowercased) that is in vocab, or None."""
    return next((w for w in re.findall(r'[a-z]+', text.lower()) if w in vocab), None)


def last_number(text):
    found = re.findall(r'\d+', text)
    return found[-1] if found else None


def score_first(vocab):
    """Scorer: 1 when the first word of the say that is in vocab is the segment's value."""
    vocab = set(vocab)
    return lambda ep, a, s, text: float(first(text, vocab) == ep['agents'][a]['segments'][s]['values'][0])


def score_number(ep, a, s, text):
    """1 when the say's last number is the segment's value."""
    return float(last_number(text) == ep['agents'][a]['segments'][s]['values'][0])


SCOUT_A = ('You are the first scout. You walk through some rooms of a house and leave marks in each room for the '
           'scouts who come after you.')
SCOUT_B = ('You are the second scout. You walk through some rooms of a house after the first scout, read the marks it '
           'left in each room, and leave your own marks for the seeker who comes after you.')
SEEKER = ('You are the seeker. You walk through some rooms of a house after the scouts and read the marks they left '
          'in each room.')


def relay(rng, test=False):
    rooms = rng.sample(ROOM_POOLS[test], 6)
    key = rng.randrange(2)
    enter = [f'You enter the {r}.' for r in rooms]
    a = rng.sample((0, 1), 2) + rng.sample((2, 3), 2)
    b = rng.sample((2, 3), 2) + rng.sample((4, 5), 2)
    c = rng.sample((4, 5), 2)
    agents = [dict(name='scout A', system=SCOUT_A, write=True,
                   segments=[seg(r, enter[r] + (' A brass key lies here.' if r == key else '')) for r in a]),
              dict(name='scout B', system=SCOUT_B, write=True, segments=[seg(r, enter[r]) for r in b]),
              dict(name='seeker', system=SEEKER, write=False,
                   segments=[seg(r, enter[r]) for r in c] +
                   [seg(None, 'Which room holds the brass key?', f'The key is in the {rooms[key]}.', [rooms[key]])])]
    return dict(game='relay', places=rooms, agents=agents, relay=dict(key=key))


MAZE_EXPLORER = ('You are the explorer. You find the way through a maze of rooms and leave marks in each room for the '
                 'follower who comes after you.')
MAZE_FOLLOWER = ('You are the follower. You walk through a maze of rooms after the explorer, read the marks it left in '
                 'each room, and name the door you take in each room.')


def maze(rng, test=False):
    n = rng.choice((6, 7, 8))
    rooms = rng.sample(ROOM_POOLS[test], n)
    doors = [rng.sample(COLOUR_POOLS[test], 3) for _ in range(n)]
    take = [rng.choice(d) for d in doors]

    def there(k, d):
        return f'The {rooms[k]} has {d[0]}, {d[1]} and {d[2]} doors.'
    explorer = [seg(k, f'{there(k, doors[k])} You take the {take[k]} door' + (' to the exit.' if k == n - 1 else '.'))
                for k in range(n)]
    follower = [seg(k, there(k, rng.sample(doors[k], 3)), f'The {take[k]} door.', [take[k]]) for k in range(n)]
    return dict(game='maze', places=rooms, agents=[
        dict(name='explorer', system=MAZE_EXPLORER, write=True, segments=explorer),
        dict(name='follower', system=MAZE_FOLLOWER, write=False, segments=follower)], maze=dict(doors=doors, take=take))


COUNTER = ('You are a scout. You count the apples at one stall of the market and leave marks there for the buyer who '
           'comes after you.')
BUYER = 'You are the buyer. You visit the market after the scouts and read the marks they left there.'


def count(rng, test=False):
    ns = rng.sample(rng.choice(THREES[test]), 3)
    names = rng.sample(G.NAMES, 3)
    scouts = [dict(name=f'scout {i + 1}', system=COUNTER, write=True,
                   segments=[seg(0, f'You visit the market and count the apples at {names[i]}\'s stall: there are '
                                    f'{n}.')]) for i, n in enumerate(ns)]
    buyer = dict(name='buyer', system=BUYER, write=False,
                 segments=[seg(0, 'You visit the market. How many apples are there at the three stalls together?',
                               f'There are {sum(ns)} apples.', [str(sum(ns))])])
    return dict(game='count', places=['market'], agents=scouts + [buyer], count=dict(numbers=ns, stalls=names))


WORKER = ('You are a worker. You solve the task at each desk you sit at and leave marks there for the manager who '
          'comes after you.')
MANAGER = 'You are the manager. You visit every desk after the workers and read the marks they left there.'
N_DESKS = 4


def work(rng, test=False):
    tasks = rng.sample(SUMS[test], N_DESKS)
    res = [a + b for a, b in tasks]
    split = rng.sample(range(N_DESKS), N_DESKS)
    workers = [dict(name=f'worker {i + 1}', system=WORKER, write=True,
                    segments=[seg(k, f'You sit at Desk {k + 1}. The task: {tasks[k][0]} + {tasks[k][1]}.', f'{res[k]}.',
                                  [str(res[k])]) for k in split[2 * i:2 * i + 2]]) for i in range(2)]
    manager = dict(name='manager', system=MANAGER, write=False,
                   segments=[seg(k, f'You sit at Desk {k + 1}.', f'Desk {k + 1}: {res[k]}.', [str(res[k])])
                             for k in range(N_DESKS)] +
                   [seg(None, 'What is the sum of the results at all four desks?',
                        ' + '.join(map(str, res)) + f' = {sum(res)}.', [str(sum(res))])])
    return dict(game='work', places=[f'desk {k + 1}' for k in range(N_DESKS)], agents=workers + [manager],
                work=dict(tasks=tasks))


TRAPPER = ('You are an explorer. You walk through some rooms of a house and leave marks in each room for the follower '
           'who comes after you.')
WARY = ('You are the follower. You stand at the door of some rooms after the explorers, read the marks they left '
        'there, and say whether each room is safe.')


def trap(rng, test=False):
    rooms = rng.sample(ROOM_POOLS[test], 6)
    traps = [rng.random() < 0.5 for _ in rooms]
    order = rng.sample(range(6), 6)
    seen = [f'You enter the {rooms[r]}. ' + ('The floor gives way: you fall into a trap!' if traps[r] else
                                             'Nothing happens.') for r in range(6)]
    explorers = [dict(name=f'explorer {i + 1}', system=TRAPPER, write=True,
                      segments=[seg(r, seen[r]) for r in order[3 * i:3 * i + 3]]) for i in range(2)]
    follower = dict(name='follower', system=WARY, write=False,
                    segments=[seg(r, f'You stand at the door of the {rooms[r]}.', 'Trap.' if traps[r] else 'Safe.',
                                  ['trap' if traps[r] else 'safe']) for r in rng.sample(range(6), 4)])
    return dict(game='trap', places=rooms, agents=explorers + [follower], trap=dict(traps=traps))


HOTPOT = 'hotpotqa/hotpot_qa'
HOTPOT_FILES = (('distractor/train-00000-of-00002.parquet', 'distractor/train-00001-of-00002.parquet'),
                ('distractor/validation-00000-of-00001.parquet',))
PAGE_WORDS = 100                                    # a page: its first and its supporting sentences, at most 100 words
_hotpot = {}
READER = ('You are a scout. Your team must answer a question. You read one page and leave marks on it for the asker '
          'who comes after you.')
ASKER = ('You are the asker. You must answer a question. You open the pages the scouts read and read the marks they '
         'left there.')


def hotpot_rows(test):
    """The HotpotQA examples of one split (training or validation) whose supporting facts come from exactly two pages:
    dict(question, answer, titles, pages), each page its first sentence and its supporting sentences in order, cut to
    PAGE_WORDS words. Read from the Hugging Face cache, downloaded when missing (360 MB)."""
    if test not in _hotpot:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        rows = []
        for f in HOTPOT_FILES[test]:
            try:
                path = hf_hub_download(HOTPOT, f, repo_type='dataset', local_files_only=True)
            except Exception:
                path = hf_hub_download(HOTPOT, f, repo_type='dataset')
            cols = ['question', 'answer', 'supporting_facts', 'context']
            for part in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=cols):
                for r in part.to_pylist():
                    facts = list(zip(r['supporting_facts']['title'], r['supporting_facts']['sent_id']))
                    titles = list(dict.fromkeys(t for t, _ in facts))
                    pages = dict(zip(r['context']['title'], r['context']['sentences']))
                    if len(titles) != 2 or not all(t in pages for t in titles):
                        continue
                    text = []
                    for t in titles:
                        keep = sorted({0} | {i for u, i in facts if u == t and i < len(pages[t])})
                        text.append(' '.join(' '.join(pages[t][i].strip() for i in keep).split()[:PAGE_WORDS]))
                    rows.append(dict(question=r['question'].strip(), answer=r['answer'], titles=titles, pages=text))
        _hotpot[test] = rows
    return _hotpot[test]


def hotpot(rng, test=False):
    r = rng.choice(hotpot_rows(test))
    order = rng.sample((0, 1), 2)
    q = f'Question: {r["question"]}'
    scouts = [dict(name=f'scout {i + 1}', system=READER, write=True,
                   segments=[seg(i, f'{q}\nYou read the page "{r["titles"][i]}":\n{r["pages"][i]}')]) for i in range(2)]
    asker = dict(name='asker', system=ASKER, write=False,
                 segments=[seg(None, q)] + [seg(i, f'You open the page "{r["titles"][i]}".') for i in order] +
                 [seg(None, 'What is the answer?', r['answer'], [r['answer']])])
    return dict(game='hotpot', places=list(r['titles']), agents=scouts + [asker], hotpot=dict(question=r['question']))


def words(s):
    """SQuAD's answer normalization: lowercase, no punctuation, no articles."""
    s = ''.join(ch for ch in s.lower() if ch not in string.punctuation)
    return re.sub(r'\b(a|an|the)\b', ' ', s).split()


def f1(text, gold):
    p, g = words(text), words(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_hotpot(ep, a, s, text):
    return float(f1(text, ep['agents'][a]['segments'][s]['values'][0]) >= 0.5)


WALKER = ('You are a walker. You visit some rooms of a house after earlier walkers, read the marks they left in each '
          'room, and leave your own marks there for those who come after you.')
ELDER = ('You are the asker. You visit some rooms of a house after many walkers and read the marks they left in each '
         'room.')
OG_ROOMS, OG_WALK, OG_ASK, OG_SEE = 8, 3, 4, 0.5
OG_SIZES = (1, 2, 4, 8, 16, 32)                     # the old-growth test: far more walkers than training's 1 to 4
OBJECT_POOLS = G.WORD_POOLS                         # (oldgrowth's argument G hides the games module)


def oldgrowth(rng, test=False, G=None):
    G = G or rng.randint(1, 4)
    rooms = rng.sample(ROOM_POOLS[test], OG_ROOMS)
    objects = rng.sample(OBJECT_POOLS[test], OG_ROOMS)
    seen, visits = [False] * OG_ROOMS, [0] * OG_ROOMS
    walkers = []
    for i in range(G):
        segs = []
        for r in rng.sample(range(OG_ROOMS), OG_WALK):
            sees = rng.random() < OG_SEE
            seen[r] |= sees
            visits[r] += 1
            segs.append(seg(r, f'You are at the {rooms[r]}.' + (f' You see a {objects[r]}.' if sees else '')))
        walkers.append(dict(name=f'walker {i + 1}', system=WALKER, write=True, segments=segs))
    asked = rng.sample(range(OG_ROOMS), OG_ASK)
    asker = dict(name='asker', system=ELDER, write=False,
                 segments=[seg(r, f'You are at the {rooms[r]}.') for r in asked] +
                 [seg(None, 'Name the object in each room you visited.',
                      '\n'.join(f'{rooms[r]}: {objects[r]}.' for r in asked), [objects[r] for r in asked])])
    return dict(game='oldgrowth', places=rooms, agents=walkers + [asker],
                oldgrowth=dict(G=G, objects=objects, seen=seen, visits=visits, asked=asked))


def oldgrowth_parts(ep, text):
    """Per asked room, in the asker's order: dict(room, right (the answer names its object), seen (some walker saw it:
    the ceiling), visits (walkers who visited it))."""
    o = ep['oldgrowth']
    return [dict(room=r, right=said(ep, text, r) == o['objects'][r], seen=o['seen'][r], visits=o['visits'][r])
            for r in o['asked']]


def score_oldgrowth(ep, a, s, text):
    parts = oldgrowth_parts(ep, text)
    return sum(p['right'] for p in parts) / len(parts)


def oldgrowth_eval(G, n=300, seed=31):
    """The old-growth test set for G walkers (OG_SIZES): n fixed test episodes, the same on every call."""
    rng = random.Random(seed * 1000 + G)
    return [oldgrowth(rng, True, G) for _ in range(n)]


EPISODES = dict(trail=trail, keytrail=keytrail, relay=relay, maze=maze, count=count, work=work, trap=trap,
                hotpot=hotpot, oldgrowth=oldgrowth)
SCORERS = dict(trail=score_trail, keytrail=score_trail, relay=score_first(ROOMS), maze=score_first(COLOURS),
               count=score_number, work=score_number, trap=score_first(('safe', 'trap')), hotpot=score_hotpot,
               oldgrowth=score_oldgrowth)
GAMES = tuple(g for g in EPISODES if g != 'keytrail')     # keytrail: a training variant of trail


CORPUS_WRITER = ('You are agent A. You work on a task with agent B, who comes after you. You read your material and '
                 'leave marks for B.')
CORPUS_READER = 'You are agent B. Agent A came before you and left marks for you.'


def corpus_tasks():
    from swarm import corpus as CP
    return tuple(CP.TASKS)


def corpus_episode(task, rng, test=False):
    """Corpus task task's episode (corpus.episode) as a crumb game: agent A writes, agent B reads."""
    from swarm import corpus as CP
    ep = CP.episode(task, rng, test)
    if ep['family'] == 1:
        chunks = [c.strip() for c in ep['golds']['chunks']]
        kind = CP.KIND[task]
        own = list(range(0, len(chunks), 2))
        head = (f'Task: you hold the odd-numbered parts of {kind}. Agent B, who comes after you, holds the '
                f'even-numbered parts and must write out yours.\n\n')
        writer = dict(name='agent A', system=CORPUS_WRITER, write=True,
                      segments=[seg(j, (head if j == 0 else '') + f'({i + 1}) {chunks[i]}') for j, i in enumerate(own)])
        lines = [f'({i + 1}) [A\'s part]' if i in own else f'({i + 1}) {chunks[i]}' for i in range(len(chunks))]
        reader = dict(name='agent B', system=CORPUS_READER, write=False,
                      segments=[seg(None, f'Task: you and agent A hold alternating parts of {kind}. A held the '
                                      f'odd-numbered parts and left marks where it read each. Write out each of A\'s '
                                      f'parts where A read it.\n\n' + '\n'.join(lines))] +
                      [seg(j, f'You are where A read part {i + 1}.', f'({i + 1}) {chunks[i]}', [chunks[i]])
                       for j, i in enumerate(own)])
        places = [f'part {i + 1}' for i in own]
    else:
        writer = dict(name='agent A', system=CORPUS_WRITER, write=True, segments=[seg(0, ep['views']['A'])])
        reader = dict(name='agent B', system=CORPUS_READER, write=False,
                      segments=[seg(0, ep['views']['B'], ep['answers']['B'], list(ep['golds']))])
        places = ['the desk']
    return dict(game=f'c_{task}', places=places, agents=[writer, reader], corpus=ep)


def score_corpus(ep, a, s, text):
    from swarm import corpus as CP
    if ep['corpus']['family'] == 1:
        return CP.chrf(text.strip(), ep['agents'][a]['segments'][s]['say'])
    return CP.score(ep['corpus'], 'B', text)['main']


def episode(game, rng, test=False, **kw):
    """One episode of game; keywords go to the game (oldgrowth: G, the number of walkers). c_TASK: corpus task TASK."""
    if game.startswith('c_'):
        return corpus_episode(game[2:], rng, test)
    return EPISODES[game](rng, test, **kw)


def score(ep, a, s, text):
    if ep['game'].startswith('c_'):
        return score_corpus(ep, a, s, text)
    return SCORERS[ep['game']](ep, a, s, text)


# choices: forced-choice groups per closed answer (choices(ep, a, s)), element 0 the gold say

def choices_trail(ep, a, s):
    """One group per visited room (its object against the episode's 5 other objects) and one for the key (its room
    against the 5 other rooms)."""
    t, rooms = ep['trail'], ep['places']
    seg = ep['agents'][a]['segments'][s]
    if seg['say'] is None:
        return []

    def say(objs, key):
        return '\n'.join(f'{rooms[r]}: {objs[r]}.' for r in t['visit']) + f'\nThe key is in the {rooms[key]}.'

    groups = []
    for r in t['visit']:
        groups.append([say(t['objects'], t['key'])] + [
            say([w if i != r else t['objects'][o] for i, w in enumerate(t['objects'])], t['key'])
            for o in range(N_ROOMS) if o != r])
    groups.append([say(t['objects'], t['key'])] + [say(t['objects'], k) for k in range(N_ROOMS) if k != t['key']])
    return groups


def swaps(say, gold, wrong):
    """[say] plus say with the last whole-word occurrence of gold replaced by each value of wrong."""
    i = [m.start() for m in re.finditer(rf'\b{re.escape(gold)}\b', say)][-1]
    return [say] + [say[:i] + w + say[i + len(gold):] for w in wrong]


def near(n, lo, hi, k):
    """k numbers other than n in [lo, hi], nearest first (ties: above first), as strings."""
    out = sorted((v for v in range(lo, hi + 1) if v != n), key=lambda v: (abs(v - n), v < n))
    return [str(v) for v in out[:k]]


def reader_say(ep, a, s):
    """Segment s of agent a when it is a reader's say, else None (writers' own says need no crumbs)."""
    agent = ep['agents'][a]
    sg = agent['segments'][s]
    return None if agent['write'] or sg['say'] is None else sg


def choices_relay(ep, a, s):
    """The key's room against the episode's 5 other rooms."""
    sg = reader_say(ep, a, s)
    if sg is None:
        return []
    gold = sg['values'][0]
    return [swaps(sg['say'], gold, [r for r in ep['places'] if r != gold])]


def choices_maze(ep, a, s):
    """The door taken against the room's 2 other doors."""
    sg = reader_say(ep, a, s)
    if sg is None:
        return []
    gold = sg['values'][0]
    return [swaps(sg['say'], gold, [c for c in ep['maze']['doors'][sg['place']] if c != gold])]


def choices_trap(ep, a, s):
    """Trap against safe."""
    sg = reader_say(ep, a, s)
    if sg is None:
        return []
    word = sg['say'].rstrip('.')
    return [swaps(sg['say'], word, [dict(Trap='Safe', Safe='Trap')[word]])]


def choices_count(ep, a, s):
    """The total against the 6 nearest other totals of three numbers from 1 to 9."""
    sg = reader_say(ep, a, s)
    if sg is None:
        return []
    gold = sg['values'][0]
    return [swaps(sg['say'], gold, near(int(gold), 3, 27, 6))]


def choices_work(ep, a, s):
    """Per desk, its result against the 5 nearest other results (sums of two numbers from 10 to 49); for the sum, the
    total against the 5 nearest other totals, the four results in the say left as they are."""
    sg = reader_say(ep, a, s)
    if sg is None:
        return []
    gold = sg['values'][0]
    lo, hi = (20, 98) if sg['place'] is not None else (20 * N_DESKS, 98 * N_DESKS)
    return [swaps(sg['say'], gold, near(int(gold), lo, hi, 5))]


def choices_oldgrowth(ep, a, s):
    """Per asked room, its object against the episode's 7 other objects."""
    sg = reader_say(ep, a, s)
    if sg is None:
        return []
    o, rooms, say = ep['oldgrowth'], ep['places'], sg['say']
    lines = say.split('\n')                         # one line per asked room, in the asker's order
    groups = []
    for k, r in enumerate(o['asked']):
        i = sum(len(line) + 1 for line in lines[:k]) + len(rooms[r]) + 2
        gold = o['objects'][r]
        assert say[i:i + len(gold)] == gold
        groups.append([say] + [say[:i] + w + say[i + len(gold):] for w in o['objects'] if w != gold])
    return groups


CHOICES = dict(trail=choices_trail, keytrail=choices_trail, relay=choices_relay, maze=choices_maze,
               count=choices_count, work=choices_work, trap=choices_trap, oldgrowth=choices_oldgrowth)


def choices(ep, a, s):
    f = CHOICES.get(ep['game'])
    return f(ep, a, s) if f else []
