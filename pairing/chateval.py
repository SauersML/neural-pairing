"""Free-chat evaluation of the co_chat and co_chat2 games (evalfree.py --games co_chat or co_chat2).

Each plan is a test chat from data/chat.json or data/chat2.json. Only its system prompts and opener enter generation;
six turns are generated with the link on ('link') and cut ('cut'). With --crossed each receiver also chats, in a new
conversation, with a stranger: the partner model with the system prompt of a donor chat drawn for that receiver
(matched; 'crossed'). The naming target is always the real partner's private item; names_donor checks the donor's.

Measures of one conversation for a receiver (measures):
  names            1 when the receiver states the partner's item before the partner does, else 0 (None without item)
  leak             1 when the partner states its own item first
  names_donor      names for the donor's item (crossed only)
  self_city_first  the receiver's first turn states the receiver's own item
  loop             the fraction of turns whose longest common substring with an earlier turn of the same speaker or
                   with the previous turn covers more than 60% of the turn (as prep_talk.echoes)
  distinct2        distinct word bigrams over all word bigrams within turns (0 without bigrams)
  game             the fraction of turns that match GAME (game, guess, riddle, your turn, secret word, clue)
Items are matched as words (chat2.contains_item). The receiver's own prompt can contain the partner's item (two equal
cities); partner_city_in_prompt records this, and those chats stay in.

Summaries (summary_rows), per receiver: each measure's mean with a betting confidence interval (stats.betting_ci),
and the link-cut and link-crossed contrasts of names with the exact two-sided sign test and the one-sided p_greater.
Naming means use only chats whose partner has an item. The 'dialogue' entries average loop, distinct2 and game over
the conversations of each focal chat before inference, so the two crossed conversations of one draw count once.
Sampled chats (temperature > 0) are generated one at a time, each with a generator seeded per plan; greedy chats are
batched.

Usage: python pairing/evalfree.py --run RUN/step_N --replies REPLIES.json --out_dir OUT --games co_chat [--crossed]
"""
from collections import defaultdict
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import random
import re

from link import DEVICE, SIDES, partner
from stats import betting_ci, sign_test
from chat2 import contains_item, kind_group, kind_of, scoring_words

METRICS = ('names', 'leak', 'names_donor', 'self_city_first', 'loop', 'distinct2', 'game')
COLLAPSE = ('loop', 'distinct2', 'game')
GAME = re.compile(r'game|guess|riddle|your turn|secret word|clue', re.I)


def episode(chat):
    """The generation episode of a stored chat: its system prompts, opener and private items, with six empty
    alternating turns. Reference turns and teacher prompts are left out."""
    opener = chat['opener']
    if opener not in SIDES:
        raise ValueError('chat opener must name a model')
    return dict(game='co_chat2' if chat.get('game') in ('chat2', 'co_chat2') else 'co_chat', mode='dialogue',
                system={x: chat['system'][x] for x in SIDES}, task={x: None for x in SIDES},
                turns=[(x, '', 1.) for x in (opener, partner(opener)) * 3],
                meta=dict(opener=opener, private={x: chat['private'][x] for x in SIDES}))


def matched(ep, donor, x, same_kind=True):
    """True when donor's partner of x is a valid stranger for receiver x of ep: the same game and opener, a private
    item different from both of ep's (an item whenever the real partner has one), and a system prompt without the real
    partner's item. co_chat2 also needs an item of the real partner's kind, or with same_kind=False (used when the pool
    has no such chat) an item of any kind."""
    from evalfree import contains, normalize
    y = partner(x)
    if ep['game'] != donor['game']:
        return False
    if ep['game'] == 'co_chat2':
        real, own, other = (p['meta']['private'][s] for p, s in ((ep, y), (ep, x), (donor, y)))
        kinds = kind_of(real) == kind_of(other) if same_kind else (real is None) == (other is None)
        if ep['meta']['opener'] != donor['meta']['opener'] or not kinds:
            return False
        forbidden = {normalize(w) for p in (real, own) for w in scoring_words(p)}
        if forbidden.intersection(normalize(w) for w in scoring_words(other)):
            return False
        return not contains_item(donor['system'][y], real)
    real, own, other = (normalize(v) for v in
                        (ep['meta']['private'][y], ep['meta']['private'][x], donor['meta']['private'][y]))
    if ep['meta']['opener'] != donor['meta']['opener']:
        return False
    if real and not other:
        return False
    if other and other in (real, own):
        return False
    # A donor prompt can contain the real partner's city outside its private field.
    return not real or not contains(donor['system'][y], real)


def make_plans(path, n, seed, max_attempts=10000, game='co_chat'):
    """n focal test chats of game, each with a matched donor per receiver, as dicts with episode, donors, attempts,
    index, episode_id, generation_seed (and donor_same_kind for co_chat2). ValueError when a draw finds no donor in
    max_attempts draws."""
    from evalfree import digest
    path = Path(path).with_name('chat2.json' if game == 'co_chat2' else 'chat.json')
    if not path.is_file():
        raise ValueError(f'{game} requires test-split chats in {path}')
    with path.open() as f:
        pool = [episode(dict(c, game=game)) for c in json.load(f)['test']]
    if not pool:
        raise ValueError(f'{game} requires nonempty test-split {path.name}')
    rng = random.Random(digest([seed, game, 'focal']))
    focal = [rng.choice(pool) for _ in range(n)]
    plans = []
    for i, ep in enumerate(focal):
        donors, attempts, same = {}, {}, {}
        for x in SIDES:
            # co_chat2: a stranger of the partner's kind when the pool has one, else of another kind, since the test
            # split can have a single chat for an (opener, kind) pair
            for same_kind in (True, False):
                rng = random.Random(digest([seed, game, i, x, 'donor']))
                for attempt in range(1, max_attempts + 1):
                    donor = rng.choice(pool)
                    if matched(ep, donor, x, same_kind):
                        donors[x], attempts[x], same[x] = donor, attempt, same_kind
                        break
                if x in donors or game != 'co_chat2':
                    break
            if x not in donors:
                raise ValueError(f'No valid {game} donor for draw {i}/{x} after {max_attempts} draws')
        plans.append(dict(episode=ep, donors=donors, attempts=attempts, index=i,
                          episode_id=f'{seed}:{game}:{i}',
                          generation_seed=int(digest([seed, game, i, 'generation'])[:15], 16),
                          **({'donor_same_kind': same} if game == 'co_chat2' else {})))
    return plans


def inputs(ep, donor=None, receiver=None):
    """The generation inputs (mode, system, task, turns) of a plan's chat. With donor, receiver's partner gets the
    donor's system prompt (the crossed chat); ValueError for a donor that make_plans does not accept."""
    e = episode(dict(game=ep['game'], opener=ep['meta']['opener'], private=ep['meta']['private'], system=ep['system']))
    if donor is not None:
        # make_plans' rule: a stranger of the partner's kind, or (co_chat2, none in the pool) of another kind
        ok = receiver in SIDES and (matched(ep, donor, receiver) or
                                    (ep['game'] == 'co_chat2' and matched(ep, donor, receiver, same_kind=False)))
        if not ok:
            raise ValueError('crossed chat requires a valid per-receiver donor')
        e['system'][partner(receiver)] = donor['system'][partner(receiver)]
    # meta is for scoring only
    return {k: e[k] for k in ('mode', 'system', 'task', 'turns')}


def first_mention(turns, receiver, city):
    """(1. if receiver states city first else 0., 1. if the partner does else 0.) over turns, matched with
    chat2.contains_item; (None, None) without an item."""
    if not city:
        return None, None
    for sp, text in turns:
        if contains_item(text, city):
            return float(sp == receiver), float(sp != receiver)
    return 0., 0.


def measures(turns, receiver, private, donor_city=None):
    """The measures of one conversation for receiver (module docstring), with the per-turn loop and game flags."""
    from evalfree import normalize
    names, leak = first_mention(turns, receiver, private[partner(receiver)])
    names_donor, _ = first_mention(turns, receiver, donor_city)
    loops, games, bigrams = [], [], []
    for k, (sp, text) in enumerate(turns):
        prior = [t for j, (s, t) in enumerate(turns[:k]) if s == sp or j == k - 1]
        loops.append(any(SequenceMatcher(None, text.lower(), t.lower(), autojunk=False)
                         .find_longest_match().size > .6 * len(text) for t in prior))
        games.append(bool(GAME.search(text)))
        words = normalize(text).split()
        bigrams.extend(zip(words, words[1:]))
    first = next((t for sp, t in turns if sp == receiver), '')
    return dict(names=names, leak=leak, names_donor=names_donor,
                self_city_first=float(contains_item(first, private[receiver])),
                loop=sum(loops) / max(1, len(turns)),
                distinct2=len(set(bigrams)) / len(bigrams) if bigrams else 0.,
                game=sum(games) / max(1, len(turns)), loop_turns=loops, game_turns=games)


def generate(pair, ep, limit, seed, temperature=0., mode=None, donor=None, receiver=None, keep=False):
    """One chat (session.generate's result, with token IDs), sampled with a generator seeded with seed."""
    import torch
    import session as S
    gen = torch.Generator(device=DEVICE if DEVICE.startswith('cuda') else 'cpu').manual_seed(seed)
    return S.generate(pair, [inputs(ep, donor, receiver)], max_turn=limit, mode=mode,
                      temperature=temperature, gen=gen, keep_tokens=True, keep=keep)[0]


def evaluate_game(pair, plans, bs=2, crossed=False, temperature=0.):
    """Per plan and receiving model, the chat's measures under link, cut and (crossed) a stranger. A one-way link
    (pair.frozen) has the reader as its only receiver: the target's records (and its crossed chats) are not made.
    Greedy chats (temperature 0) are generated a chunk of bs plans at a time, crossed ones per receiver too."""
    import session as S
    from evalfree import SCHEMA, generation_limit
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if bs < 1:
        raise ValueError('bs must be positive')
    limit = generation_limit('co_chat')
    for start in range(0, len(plans), bs):
        chunk = plans[start:start + bs]
        results = {}
        for c, mode in (('link', None), ('cut', 'zero')):
            if temperature:
                results[c] = [generate(pair, p['episode'], limit, p['generation_seed'], temperature, mode)
                              for p in chunk]
            else:
                results[c] = S.generate(pair, [inputs(p['episode']) for p in chunk], mode=mode,
                                        max_turn=limit, temperature=0., keep_tokens=True)
        stranger = {x: S.generate(pair, [inputs(p['episode'], p['donors'][x], x) for p in chunk], max_turn=limit,
                                  temperature=0., keep_tokens=True)
                    for x in receivers(pair)} if crossed and not temperature else {}
        for i, p in enumerate(chunk):
            ep, private = p['episode'], p['episode']['meta']['private']
            for x in receivers(pair):
                donor, y = p['donors'][x], partner(x)
                runs = {c: rs[i] for c, rs in results.items()}
                if crossed:
                    runs['crossed'] = stranger[x][i] if stranger else generate(
                        pair, ep, limit, p['generation_seed'], temperature, donor=donor, receiver=x)
                conditions = {}
                for c, r in runs.items():
                    metrics = measures(r['turns'], x, private, donor['meta']['private'][y] if c == 'crossed' else None)
                    conditions[c] = dict(metrics, score=metrics['names'], turns=r['turns'], tokens=r['tokens'],
                                         sampled=r['sampled'],
                                         capped=any(s and not s[-1] for s in r['sampled']),
                                         generated_tokens=sum(sum(s) for s in r['sampled']),
                                         system=inputs(ep, donor, x)['system'] if c == 'crossed' else ep['system'],
                                         implementation='isolated_crossed_dialogue_per_receiver' if c == 'crossed'
                                         else 'full_dialogue')
                extra = (dict(kind=kind_of(private[y]), kind_group=kind_group(private[y]),
                              partner_item_in_prompt=contains_item(ep['system'][x], private[y]))
                         if ep['game'] == 'co_chat2' else {})
                yield dict(schema=SCHEMA, game=ep['game'], episode_id=p['episode_id'], episode_index=p['index'],
                           receiver=x, opener=ep['meta']['opener'], private=private,
                           partner_city_in_prompt=contains_item(ep['system'][x], private[y]), **extra,
                           conditions=conditions, temperature=temperature, generation_seed=p['generation_seed'],
                           max_new_tokens=limit, donor=dict(city=donor['meta']['private'][y],
                           system=donor['system'][y], attempts=p['attempts'][x]))


def receivers(pair):
    """The models that read their partner: both, or a one-way link's reader."""
    return [x for x in SIDES if x != getattr(pair, 'frozen', None)]


def estimate(values, alpha=.05, bounds=(0., 1.)):
    """dict(n, mean, ci) of the values that are not None; ci is the betting confidence interval for values in
    bounds."""
    xs = [v for v in values if v is not None]
    return dict(n=len(xs), mean=sum(xs) / len(xs) if xs else None,
                ci=list(betting_ci(xs, alpha=alpha, bounds=bounds)))


def summary_rows(records, alpha=.05):
    """Summary entries per game (_summary_rows); co_chat2 adds the receiver entries per item kind, per kind group and
    for the exclusive subset (the partner's item absent from the receiver's prompt). The two receiving models are never
    pooled."""
    rows = []
    for game in sorted({r['game'] for r in records}):
        rs = [r for r in records if r['game'] == game]
        rows += _summary_rows(rs, alpha, game)
        if game == 'co_chat2':
            groups = [('kind:' + k, [r for r in rs if r['kind'] == k])
                      for k in sorted({r['kind'] for r in rs if r['kind']})]
            groups += [(k, [r for r in rs if r['kind_group'] == k])
                       for k in ('trained_kind', 'held_out_kind')]
            groups += [('exclusive', [r for r in rs if r['kind'] and not r['partner_item_in_prompt']])]
            for subset, selected in groups:
                rows += [dict(row, subset=subset) for row in _summary_rows(selected, alpha, game)
                         if row['receiver'] != 'dialogue']
    return rows


def _summary_rows(records, alpha=.05, game='co_chat'):
    """Per temperature, one entry per receiver (measure means, names contrasts, co_chat2 naming counts) and one
    'dialogue' entry of loop, distinct2 and game averaged within each focal chat."""
    rows = []
    for temperature in sorted({r['temperature'] for r in records}):
        rs = [r for r in records if r['temperature'] == temperature]
        conditions = [c for c in ('link', 'cut', 'crossed') if any(c in r['conditions'] for r in rs)]
        for x in [x for x in SIDES if any(r['receiver'] == x for r in rs)]:          # one-way: the reader only
            mine = [r for r in rs if r['receiver'] == x]
            row = dict(game=game, receiver=x, subset='all', temperature=temperature, n=len(mine),
                       ci_method='iid_betting', partner_city_in_prompt_n=sum(r['partner_city_in_prompt'] for r in mine),
                       conditions={c: {m: estimate([r['conditions'][c][m] for r in mine if c in r['conditions']], alpha)
                                       for m in METRICS} for c in conditions})
            for c in conditions[1:]:
                ds = [r['conditions']['link']['names'] - r['conditions'][c]['names'] for r in mine
                      if c in r['conditions'] and r['conditions']['link']['names'] is not None]
                sign = sign_test(ds)
                n = sign['positive'] + sign['negative']
                sign['p_greater'] = sum(math.comb(n, j) for j in range(sign['positive'], n + 1)) / (1 << n)
                row['link-' + c] = dict(estimate(ds, alpha, (-1., 1.)), metric='names', sign=sign)
            rows.append(row)
            if game == 'co_chat2':
                row['names_counts'] = {c: dict(successes=sum(r['conditions'][c]['names'] == 1 for r in mine
                                                            if c in r['conditions']),
                                              n=row['conditions'][c]['names']['n']) for c in conditions}
        by_episode = defaultdict(list)
        for r in rs:
            by_episode[r['episode_id']].append(r)
        collapse = {}
        for c in conditions:
            # Link/cut records refer to the same dialogue twice. Crossed records
            # refer to distinct dialogues dependent through the same focal draw.
            collapse[c] = {m: estimate([sum(r['conditions'][c][m] for r in group) / len(group)
                                        for group in by_episode.values() if all(c in r['conditions'] for r in group)],
                                       alpha) for m in COLLAPSE}
        rows.append(dict(game=game, receiver='dialogue', subset='all', n=len(by_episode),
                         temperature=temperature, ci_method='iid_betting_on_focal_dialogue_means',
                         conditions=collapse))
    return rows


def validate(records, manifest):
    """ValueError unless records contain each plan's receivers exactly once, with the manifest's conditions and
    temperature, six alternating turns per chat, and measures that recompute from the transcripts (chats without a
    naming target included)."""
    by_episode = defaultdict(list)
    expected = {'link', 'cut'} | ({'crossed'} if manifest.get('crossed') else set())
    for r in records:
        by_episode[r['episode_id']].append(r)
        if r['temperature'] != manifest.get('temperature', 0.):
            raise ValueError('chat temperature differs from manifest')
        if set(r['conditions']) != expected:
            raise ValueError('missing/unexpected chat condition')
        if r['game'] == 'co_chat2':
            item = r['private'][partner(r['receiver'])]
            if r['kind'] != kind_of(item) or r['kind_group'] != kind_group(item):
                raise ValueError('chat2 kind disagrees with private item')
        for c, result in r['conditions'].items():
            if (len(result['turns']) != 6
                    or [sp for sp, _ in result['turns']] != list((r['opener'], partner(r['opener'])) * 3)):
                raise ValueError('chat must contain six alternating free turns')
            want = measures(result['turns'], r['receiver'], r['private'],
                            r['donor']['city'] if c == 'crossed' else None)
            if any(result[k] != v for k, v in want.items()) or result['score'] != want['names']:
                raise ValueError('chat metrics disagree with transcript')
    game = records[0]['game'] if records else next(iter(manifest.get('games', ['co_chat'])))
    n = manifest.get('chat_plans_by_game', {}).get(game, manifest.get('chat_plans', manifest['n']))
    want = set(manifest.get('receivers', SIDES))                  # a one-way link's manifest lists its reader only
    if len(by_episode) != n or any({r['receiver'] for r in group} != want or len(group) != len(want)
                                              for group in by_episode.values()):
        raise ValueError('missing chat receiver/dialogue records')
