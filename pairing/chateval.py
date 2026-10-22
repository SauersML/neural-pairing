"""No-objective co_chat evaluation, dispatched by evalfree --games co_chat.

Only test chat.json system prompts and the opener enter generation. Six turns
are generated from nothing; teach, pmi and recorded turns never enter a plan.
Link/cut share one dialogue; crossed runs a fresh isolated dialogue per receiver
with an independently rejection-drawn donor partner, as crossed.py does for play.
The real partner's city remains the crossed naming target; names_donor separately
checks the donor city. First mention wins: a later echo after a leak never counts.
These lexical names are not evidence of semantic knowledge or novelty. In
particular, equal focal cities can expose the target in the receiver's prompt;
partner_city_in_prompt records that fact without excluding the requested draws.

loop uses prep_talk.echoes' lowercase character longest match >60% of the current
turn, against any earlier same-speaker turn or the immediately preceding turn.
distinct2 uses casefolded word bigrams within turns, pooled over the dialogue
(no artificial bigrams across turn boundaries); no bigrams means zero.
game is a LEXICAL flag per turn, using game|guess|riddle|your turn|secret word|clue
case-insensitively, summarized as the fraction of turns flagged. It is not a
semantic game classifier. self_city_first is false when the receiver has no city.

Naming summaries condition on a nonempty partner city; collapse summaries include
every chat. Crossed dialogue summaries average the two receiver conversations
WITHIN each focal draw before inference, avoiding two dependent observations.
All CIs use stats.betting_ci; signs include the module's exact two-sided test and
an exact one-sided positive tail. Sampling uses a local per-episode RNG, shared
seed across conditions, singleton generation when temperature > 0. Greedy
link/cut may batch; crossed is always isolated. No existing score is changed.
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
    """Allowlist inputs: never retain reference text, teacher prompts or forcing."""
    opener = chat['opener']
    if opener not in SIDES:
        raise ValueError('chat opener must name a model')
    return dict(game='co_chat2' if chat.get('game') in ('chat2', 'co_chat2') else 'co_chat', mode='dialogue',
                system={x: chat['system'][x] for x in SIDES}, task={x: None for x in SIDES},
                turns=[(x, '', 1.) for x in (opener, partner(opener)) * 3],
                meta=dict(opener=opener, private={x: chat['private'][x] for x in SIDES}))


def matched(ep, donor, x):
    from evalfree import contains, normalize
    y = partner(x)
    if ep['game'] != donor['game']:
        return False
    if ep['game'] == 'co_chat2':
        real, own, other = (p['meta']['private'][s] for p, s in ((ep, y), (ep, x), (donor, y)))
        if ep['meta']['opener'] != donor['meta']['opener'] or kind_of(real) != kind_of(other):
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
    # A stored system could contain a city incidentally. The donor must not
    # already know the original target even when its private field differs.
    return not real or not contains(donor['system'][y], real)


def make_plans(path, n, seed, max_attempts=10000, game='co_chat'):
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
        donors, attempts = {}, {}
        for x in SIDES:
            rng = random.Random(digest([seed, game, i, x, 'donor']))
            for attempt in range(1, max_attempts + 1):
                donor = rng.choice(pool)
                if matched(ep, donor, x):
                    donors[x], attempts[x] = donor, attempt
                    break
            else:
                raise ValueError(f'No valid {game} donor for draw {i}/{x} after {max_attempts} draws')
        plans.append(dict(episode=ep, donors=donors, attempts=attempts, index=i,
                          episode_id=f'{seed}:{game}:{i}',
                          generation_seed=int(digest([seed, game, i, 'generation'])[:15], 16)))
    return plans


def inputs(ep, donor=None, receiver=None):
    """The isolated two-model context; the original partner never runs crossed."""
    e = episode(dict(game=ep['game'], opener=ep['meta']['opener'], private=ep['meta']['private'], system=ep['system']))
    if donor is not None:
        if receiver not in SIDES or not matched(ep, donor, receiver):
            raise ValueError('crossed chat requires a valid per-receiver donor')
        e['system'][partner(receiver)] = donor['system'][partner(receiver)]
    # Scoring-only metadata is not passed to the generator either.
    return {k: e[k] for k in ('mode', 'system', 'task', 'turns')}


def first_mention(turns, receiver, city):
    """First accepted target form: (receiver first, partner first), nulls without an item.

    String cities retain evalfree.contains verbatim; structured items use their fixed words table.
    """
    if not city:
        return None, None
    for sp, text in turns:
        if contains_item(text, city):
            return float(sp == receiver), float(sp != receiver)
    return 0., 0.


def measures(turns, receiver, private, donor_city=None):
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
    import torch
    import session as S
    gen = torch.Generator(device=DEVICE if DEVICE.startswith('cuda') else 'cpu').manual_seed(seed)
    return S.generate(pair, [inputs(ep, donor, receiver)], max_turn=limit, mode=mode,
                      temperature=temperature, gen=gen, keep_tokens=True, keep=keep)[0]


def evaluate_game(pair, plans, bs=2, crossed=False, temperature=0.):
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
        for i, p in enumerate(chunk):
            ep, private = p['episode'], p['episode']['meta']['private']
            for x in SIDES:
                donor, y = p['donors'][x], partner(x)
                runs = {c: rs[i] for c, rs in results.items()}
                if crossed:
                    runs['crossed'] = generate(pair, ep, limit, p['generation_seed'], temperature,
                                                donor=donor, receiver=x)
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


def estimate(values, alpha=.05, bounds=(0., 1.)):
    xs = [v for v in values if v is not None]
    return dict(n=len(xs), mean=sum(xs) / len(xs) if xs else None,
                ci=list(betting_ci(xs, alpha=alpha, bounds=bounds)))


def summary_rows(records, alpha=.05):
    """Legacy aggregate rows plus chat2 receiver strata; never pool dependent twins."""
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
    rows = []
    for temperature in sorted({r['temperature'] for r in records}):
        rs = [r for r in records if r['temperature'] == temperature]
        conditions = [c for c in ('link', 'cut', 'crossed') if any(c in r['conditions'] for r in rs)]
        for x in SIDES:
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
    """Require complete paired chats, including chats with no naming target."""
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
            if len(result['turns']) != 6 or [sp for sp, _ in result['turns']] != list((r['opener'], partner(r['opener'])) * 3):
                raise ValueError('chat must contain six alternating free turns')
            want = measures(result['turns'], r['receiver'], r['private'], r['donor']['city'] if c == 'crossed' else None)
            if any(result[k] != v for k, v in want.items()) or result['score'] != want['names']:
                raise ValueError('chat metrics disagree with transcript')
    game = records[0]['game'] if records else next(iter(manifest.get('games', ['co_chat'])))
    n = manifest.get('chat_plans_by_game', {}).get(game, manifest.get('chat_plans', manifest['n']))
    if len(by_episode) != n or any({r['receiver'] for r in group} != set(SIDES) or len(group) != 2
                                              for group in by_episode.values()):
        raise ValueError('missing chat receiver/dialogue records')
