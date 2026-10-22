"""Honest free-generation evaluation on test-split draws, with independent controls.

Every focal episode runs greedily with link, cut (LoRA stays on), and a matched
shuffle. Each focal gets its OWN independently drawn, unscored donor. Matching
requires the same game and side-specific roles, a different underlying item,
and different private content at every scored receiver's partner. Shuffle is
Pair's native inverse permutation on [focal, donor], i.e. [1,0] both ways, for
duplex. Dialogue crosses holder contexts instead, so streamed listener text
also follows the permuted pair (native shuffle alone can reconnect the original
partner through text feedback). No teacher forcing or scored-donor recycling.
--bs bounds link/cut batches; shuffle always uses two episodes. This sacrifices
shuffle throughput to preserve independent statistical units without editing
link.py. Models load once. Start with --bs 2 on the Mac; NL_MAXNEW is a cap.

co_talk generates ONLY the asker's first turn, before any partner text arrives.
Other scores retain evaluate/play_tasks semantics (including whole-text chrF).
--crossed adds an isolated full-dialogue control for co_talk/co_play, retaining
the legacy rows. co_play is opt-in through --games; its legacy shuffle is native
two-row shuffle and may carry information through text feedback. Crossed runs
each scored receiver separately with the donor partner; see crossed.py.
co_chat is also opt-in: six free turns from test chat.json system prompts, link
and cut, plus isolated per-receiver donors with --crossed. --temperature (default
0) applies only to co_chat; sampled chats use per-episode seeds. Chat scores and
collapse summaries are separate from the legacy payload scores; see chateval.py.
'exclusive' is a lexical absence flag, not proof of absence from pretraining or
of semantic dependence on the partner. Split rules do not promise novel numeric
values, answer vocabularies, or absence from the base models' pretraining.

python pairing/evalfree.py --run RUN/step_N --replies REPLIES.json --out_dir OUT
python pairing/evalfree.py --summarize --out_dir OUT --tag headline
The latter writes results/evalfree_headline.json, without loading any models.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time
import unicodedata

from stats import betting_ci, sign_test

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('link', 'cut', 'shuffle')
SCHEMA = 1


def normalize(text):
    """Audit finding 7: casefolded word tokens, punctuation/whitespace ignored."""
    return ' '.join(re.findall(r'\w+', str(text or '').casefold()))


def contains(text, payload):
    p = normalize(payload)
    return bool(p) and (' ' + p + ' ') in (' ' + normalize(text) + ' ')


def digest(value):
    """Content IDs ignore source labels and normalize strings, not prompt frames."""
    def canonical(v):
        if isinstance(v, str):
            return ' '.join(unicodedata.normalize('NFKC', v).casefold().split())
        if isinstance(v, dict):
            return {k: canonical(w) for k, w in sorted(v.items()) if k != 'src'}
        if isinstance(v, (tuple, list)):
            return [canonical(w) for w in v]
        return v
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target(ep, x):
    if ep['mode'] == 'duplex':
        return ep['reply'][x][0]
    return ' '.join(t for s, t, _ in ep['turns'] if s == x)


def payloads(ep):
    """Use literal task payloads; scripted reference frames are not a contract.

    Games' relay_word frame changed from 'The word is ...' to 'The secret word
    is ...'; historical span parsing missed it. Metadata avoids that coupling.
    """
    from link import SIDES, partner
    g, m = ep['game'], ep['meta']
    if g.startswith('relay'):
        return {m['receiver']: [m['value']]}
    if g == 'merge_sum':
        return {x: [str(m['values'][partner(x)]), str(m['sum'])] for x in SIDES}
    if g == 'merge_fact':
        return {m['has_name']: [m['city']], partner(m['has_name']): [m['name']]}
    if g == 'lookup':
        return {m['asker']: [m['value']]}
    import session as S
    return {x: [target(ep, x)[a:b] for a, b in spans] for x, spans in S.payload(ep).items() if spans}


def identity(ep):
    """Return roles and actual held-content IDs using the generator's role fields.

    The item ID ignores source names. The additional held IDs reject different
    rows that repeat the SAME private passage, operand, name, options, etc.
    QA holders answering the question and holders answering a distractor are
    distinct roles. co_swap metadata has no side assignment: the opposite
    receiver's payload identifies each side's held passage unambiguously.
    """
    import coread as C
    from link import SIDES, partner
    g, m = ep['game'], ep['meta']
    roles, held = {}, {}

    def two(first, a, b):
        return {first: a, partner(first): b}

    if g.startswith('relay'):
        roles = two(m['sender'], 'sender', 'receiver')
        held = {m['sender']: m['value']}
        item = m['value']
    elif g == 'merge_sum':
        roles = {x: 'operand_holder' for x in SIDES}
        held, item = m['values'], m['values']
    elif g == 'merge_fact':
        roles = two(m['has_name'], 'name_holder', 'city_holder')
        held = two(m['has_name'], m['name'], m['city'])
        item = [m['name'], m['city']]
    elif g == 'lookup':
        roles = two(m['asker'], 'asker', 'table_holder')
        held, item = {m['holder']: m['table']}, [m['key'], m['table']]
    elif g == 'co_play':
        roles = two(m['opener'], m['play'] + '_opener', m['play'] + '_responder')
        held, item = m['secrets'], [m['play'], m['secrets']]
    else:
        item = m['item']
        if g == 'co_swap':
            p = payloads(ep)
            roles = {x: 'passage_holder' for x in SIDES}
            held = {x: p[partner(x)] for x in SIDES}
            item = held
        elif g in ('co_split', 'co_hop'):
            roles = two(m['first'], 'first_half' if g == 'co_split' else 'paragraph_a',
                        'second_half' if g == 'co_split' else 'paragraph_b')
            a, b = C.halves(item)[:2] if g == 'co_split' else (item['a'], item['b'])
            held = two(m['first'], a, b)
        elif g == 'co_nli':
            roles = two(m['premise'], 'premise_holder', 'hypothesis_holder')
            held = two(m['premise'], item['premise'], item['hypothesis'])
        elif g == 'co_mcq':
            roles = two(m['question'], 'question_holder', 'options_holder')
            held = two(m['question'], item['question'], item['options'])
        elif g == 'co_talk':
            roles = two(m['asker'], 'asker_first_turn', 'context_holder')
            held = {m['holder']: C.talk_context(item)}
        else:
            reader = m['reader']
            role = 'reader'
            if g == 'co_qa':
                # The loader does not store this branch in meta. Match its
                # exact templates, not the target (a distractor can say 'Yes.').
                asks_answer = ep['task'][reader] in {
                    w.format(t=item['context'], q=item['question']) for w in C.W['ask_both']}
                role += '_answer' if asks_answer else '_distractor'
                content = item['context']
            elif g == 'co_transfer':
                content = item['text']
            else:
                content = C.request(item)[0]
            roles = two(reader, role, 'receiver')
            held = {reader: content}
    return dict(roles=roles, held={x: digest(t) for x, t in held.items()}, item_id=digest([g, item]))


def matched(ep, donor):
    from link import partner
    if ep['game'] != donor['game'] or ep['mode'] != donor['mode']:
        return False
    a, b = identity(ep), identity(donor)
    if a['roles'] != b['roles'] or a['item_id'] == b['item_id']:
        return False
    if ep['mode'] == 'dialogue' and ep['game'] == 'co_play' and \
            [t[0] for t in ep['turns']] != [t[0] for t in donor['turns']]:
        return False
    return all(partner(x) in a['held'] and partner(x) in b['held'] and
               a['held'][partner(x)] != b['held'][partner(x)] for x in payloads(ep))


def choose_donor(ep, source, max_attempts=10000):
    """Independent rejection draw, never fall back to a same-item/wrong-role donor."""
    for attempt in range(1, max_attempts + 1):
        if ep['game'] == 'co_play':
            from crossed import draw_play_donor
            d = draw_play_donor(ep, source)
        else:
            d = source.episode(ep['game'])
        if matched(ep, d):
            return d, attempt
    raise ValueError(f"No valid matched donor found for {ep['game']} after {max_attempts} draws; "
                     f"roles/item={identity(ep)}. Expand the test corpus; do not relax the control.")


def flags(ep, x):
    from link import partner
    spans = payloads(ep)[x]
    own = [contains(ep['task'].get(x), p) or contains(ep['system'][x], p) for p in spans]
    symmetric = [contains(target(ep, partner(x)), p) for p in spans]
    # Multiple spans (merge_sum): conservative exclusion if ANY span is available.
    return dict(exclusive=all(normalize(p) for p in spans) and not any(own),
                symmetric=all(symmetric), payload_in_own=own, payload_in_partner_target=symmetric)


def first_turn(ep):
    """Strip every forcing/replay field; retain only speaker order for dialogue."""
    e = {k: v for k, v in ep.items() if k not in
         ('forced', 'reply_tokens', 'reply_sampled', 'turn_tokens', 'turn_sampled')}
    if ep['mode'] == 'dialogue':
        if ep['game'] == 'co_play':
            return e
        if ep['game'] != 'co_talk' or ep['turns'][0][0] != ep['meta']['asker']:
            raise ValueError('only co_talk with the asker speaking first is supported')
        e['turns'] = ep['turns'][:1]
    return e


def receiver_scores(ep, result):
    """Existing metrics; merge games expose the per-side predicates in score()."""
    import evaluate as E
    import play_tasks as P
    from link import SIDES, partner
    if ep['game'] == 'co_talk':
        x, text = result['turns'][0]
        return {x: P.score_one('co_talk', ep['meta']['item'], text, x, ep)}
    if ep['game'] == 'co_play':
        return E.score(ep, result)['each']
    if ep['game'].startswith('co_'):
        # The local venv's ledger otherwise reserves the parent's entire model
        # budget AGAIN for each tiny Python assertion subprocess, timing out
        # correct code before it starts. Keep that harness in the 1-GiB lane.
        budget = os.environ.get('MPD_MEM_GIB')
        if ep['game'] != 'co_code_write' or budget is None:
            return P.score(ep, result['text'])
        try:
            os.environ['MPD_MEM_GIB'] = '1'
            return P.score(ep, result['text'])
        finally:
            os.environ['MPD_MEM_GIB'] = budget
    scores = E.score(ep, result)
    if ep['game'] == 'merge_sum':
        m = ep['meta']
        return {x: float(E.has(result['text'][x], str(m['sum'])) and
                         E.has(result['text'][x], str(m['values'][partner(x)]))) for x in SIDES}
    if ep['game'] == 'merge_fact':
        m = ep['meta']
        return {x: float(E.has(result['text'][x], m['name']) and E.has(result['text'][x], m['city'])) for x in SIDES}
    return {x: scores['ok'] for x in payloads(ep)}


def generation_limit(game):
    from evaluate import MAXNEW
    cap = int(os.environ.get('NL_MAXNEW', 0))
    if cap < 0:
        raise ValueError('NL_MAXNEW must be nonnegative')
    default = MAXNEW.get(game, 64)
    return min(default, cap) if cap else default


def generate_shuffle(pair, ep, donor, limit, keep=False):
    """Historical duplex/co_talk shuffle, plus native shuffle for new co_play.

    Native shuffle permutes neural states but dialogue's listener still reads
    its ORIGINAL asker's text. That can relay the true partner via the donor
    asker. Crossing holder inputs and running normal linked dialogue makes the
    text feedback follow the crossed pair too. For duplex use native shuffle
    verbatim. co_play also uses native shuffle (with its text-feedback route);
    request the separate crossed condition for isolation over whole dialogues.
    Return only the focal result; donor results are never observations.
    """
    import session as S
    from link import partner
    eps = [first_turn(ep), first_turn(donor)]
    if not matched(ep, donor):
        raise ValueError('invalid matched donor in evaluation plan')
    mode = 'shuffle'
    if ep['mode'] == 'dialogue' and ep['game'] == 'co_talk':
        holder = partner(ep['meta']['asker'])
        eps = [dict(e, system=dict(e['system']), task=dict(e['task'])) for e in eps]
        for field in ('system', 'task'):
            eps[0][field][holder], eps[1][field][holder] = eps[1][field][holder], eps[0][field][holder]
        # Only the focal crossed dialogue is scored. Omit the other independent
        # conversation entirely, including its padding and original holder.
        eps = eps[:1]
        mode = None
    return S.generate(pair, eps, mode=mode, temperature=0.0, max_turn=limit,
                      keep_tokens=True, keep=keep)[0]


def evaluate_game(pair, plans, bs=2, crossed=False, temperature=0.):
    """Yield JSON-serializable receiver records. No gold tokens enter generation."""
    import session as S
    from link import SIDES, partner
    if plans and plans[0]['episode']['game'] == 'co_chat':
        import chateval
        yield from chateval.evaluate_game(pair, plans, bs, crossed, temperature)
        return
    if bs < 2:
        raise ValueError('bs must be at least 2 for native shuffle')
    for start in range(0, len(plans), bs):
        chunk = plans[start:start + bs]
        eps = [first_turn(p['episode']) for p in chunk]
        limit = generation_limit(eps[0]['game'])
        results = {c: S.generate(pair, eps, mode=mode, temperature=0.0, max_turn=limit, keep_tokens=True)
                   for c, mode in (('link', None), ('cut', 'zero'))}
        results['shuffle'] = []
        for ep, p in zip(eps, chunk):
            results['shuffle'].append(generate_shuffle(pair, p['episode'], p['donor'], limit))
        if crossed and eps[0]['mode'] == 'dialogue':
            import crossed as X
            results['crossed'] = [{x: X.generate(pair, p['episode'], p['donor'], x, limit)
                                   for x in payloads(p['episode'])} for p in chunk]
        for i, plan in enumerate(chunk):
            ep = plan['episode']
            a, b = identity(ep), identity(plan['donor'])
            sc = {c: receiver_scores(ep, results[c][i]) for c in CONDITIONS}
            for x, parts in payloads(ep).items():
                cond = {}
                for c in CONDITIONS:
                    r = results[c][i]
                    texts = r['text'] if ep['mode'] == 'duplex' else dict(r['turns'])
                    if ep['game'] == 'co_play':
                        texts = {y: next(t for sp, t in r['turns'] if sp == y) for y in SIDES}
                    k = next((k for k, (sp, _) in enumerate(r.get('turns', [])) if sp == x), 0)
                    sampled = r['sampled'][x] if ep['mode'] == 'duplex' else r['sampled'][k]
                    cond[c] = dict(score=sc[c][x], texts=texts,
                                   capped=bool(sampled and not sampled[-1]),
                                   generated_tokens=sum(sampled))
                    if ep['game'] == 'co_play':
                        cond[c].update(turns=r['turns'], tokens=r['tokens'], sampled=r['sampled'])
                if 'crossed' in results:
                    r = results['crossed'][i][x]
                    k = next(k for k, (sp, _) in enumerate(r['turns']) if sp == x)
                    sampled = r['sampled'][k]
                    cond['crossed'] = dict(score=receiver_scores(ep, r)[x],
                                           texts={x: r['turns'][k][1]}, turns=r['turns'],
                                           tokens=r['tokens'], sampled=r['sampled'],
                                           capped=bool(sampled and not sampled[-1]),
                                           generated_tokens=sum(sampled),
                                           implementation='isolated_crossed_dialogue_per_receiver')
                yield dict(schema=SCHEMA, game=ep['game'], episode_id=plan['episode_id'],
                           episode_index=plan['index'], receiver=x, role=a['roles'][x], item_id=a['item_id'],
                           partner_role=a['roles'][partner(x)], partner_item_id=a['held'][partner(x)],
                           payload=parts, flags=flags(ep, x), system=ep['system'], prompts=ep['task'],
                           targets={y: target(ep, y) for y in SIDES}, conditions=cond,
                           donor=dict(episode_id=plan['episode_id'] + ':donor', item_id=b['item_id'],
                                      partner_role=b['roles'][partner(x)], partner_item_id=b['held'][partner(x)],
                                      attempts=plan['attempts'], permutation=[1, 0],
                                      implementation='native_shuffle' if ep['game'] != 'co_talk' else
                                      'crossed_holder_contexts_with_text_feedback'), max_new_tokens=limit)


def make_plans(replies, n=20, seed=4242, coread_data=None, games=None, max_digits=16, max_attempts=10000):
    """Sample before model loading. Focal draws are unchanged by donor rejection."""
    import coread as C
    import coread_data as CD
    import games as G
    import play_tasks as P
    from payload_ctl import PAYLOAD_GAMES
    if n < 1:
        raise ValueError('n must be positive')
    path = Path(coread_data or ROOT / 'data/coread.json')
    chat_plans = {}
    if games and 'co_chat' in games:
        import chateval
        if len(set(games)) != len(games):
            raise ValueError('games must be distinct')
        chat_plans['co_chat'] = chateval.make_plans(path, n, seed, max_attempts)
        games = [g for g in games if g != 'co_chat']
        if not games:
            return chat_plans, dict(coread_excluded=0, talk_excluded=0)
    co = C.Coread(replies, 'test', seed, str(path))
    talk_path = path.with_name('talk.json')
    # Existing talk.json files can predate prep_talk's decontamination fix.
    talk_excluded = 0
    if talk_path.exists():
        with path.open() as f:
            train = json.load(f)['train']
        with talk_path.open() as f:
            talk_train = [c['item'] for c in json.load(f).get('train', [])]
        forbidden = {k for it in train + talk_train for k in CD.content_keys(it)}
        clean = [c for c in co.talk if not forbidden.intersection(CD.content_keys(c['item']))]
        talk_excluded, co.talk = len(co.talk) - len(clean), clean
    selected = list(games or (*PAYLOAD_GAMES, *P.SCORED, *(('co_talk',) if talk_path.exists() else ())))
    allowed = set(PAYLOAD_GAMES) | set(P.SCORED) | {'co_talk', 'co_play'}
    if len(set(selected)) != len(selected) or set(selected) - allowed:
        raise ValueError('games must be distinct payload/scored co-reading games, co_talk or co_play')
    if 'co_talk' in selected and not co.talk:
        raise ValueError('co_talk requested/present but no uncontaminated test conversations exist')
    if 'co_play' in selected and not co.play:
        raise ValueError('co_play requested but no test conversations exist')
    sources = {False: G.Games(replies, 'test', seed, max_digits), True: co}
    plans = {}
    for game in selected:
        source = sources[game.startswith('co_')]
        source.rng = random.Random(digest([seed, game, 'focal']))
        eps = [source.episode(game) for _ in range(n)]
        if any(ep['game'] != game for ep in eps):
            raise ValueError(f'episode loader substituted another game for {game}')
        plans[game] = []
        for i, ep in enumerate(eps):
            source.rng = random.Random(digest([seed, game, i, 'donor']))
            donor, attempts = choose_donor(ep, source, max_attempts)
            plans[game].append(dict(episode=ep, donor=donor, index=i, attempts=attempts,
                                    episode_id=f'{seed}:{game}:{i}'))
    return plans | chat_plans, dict(coread_excluded=co.excluded_test, talk_excluded=talk_excluded)


def summary_rows(records, games, alpha=0.05):
    """Per-game iid bounds; pooled bounds combine simultaneous stratum bounds.

    Pools weight games by their observed receiver counts, conditional on those
    counts. The pooled sign test is exact under sign balance in EVERY game;
    it does not test a zero pooled mean. No inference pools receiving models.
    """
    from link import SIDES
    from payload_ctl import PAYLOAD_GAMES
    from play_tasks import SCORED
    chat_rows = []
    if 'co_chat' in games:
        import chateval
        chat_rows = chateval.summary_rows([r for r in records if r['game'] == 'co_chat'], alpha)
        games = [g for g in games if g != 'co_chat']
        if not games:
            return chat_rows
    groups = {g: [g] for g in games}
    groups.update(pooled_payload=[g for g in games if g in PAYLOAD_GAMES],
                  pooled_coread=[g for g in games if g in SCORED])
    rows = []
    for group, members in groups.items():
        for receiver in SIDES:
            for subset in ('all', 'exclusive'):
                rs = [r for r in records if r['game'] in members and r['receiver'] == receiver and
                      (subset == 'all' or r['flags']['exclusive'])]
                n = len(rs)
                row = dict(game=group, receiver=receiver, subset=subset, n=n,
                           mean={c: sum(r['conditions'][c]['score'] for r in rs) / n if n else None
                                 for c in CONDITIONS}, symmetric_n=sum(r['flags']['symmetric'] for r in rs),
                           capped_n={c: sum(r['conditions'][c]['capped'] for r in rs) for c in CONDITIONS})
                crossed_rs = [r for r in rs if 'crossed' in r['conditions']]
                if crossed_rs:
                    ds = [r['conditions']['link']['score'] - r['conditions']['crossed']['score'] for r in crossed_rs]
                    row['crossed_n'] = len(crossed_rs)
                    row['mean']['crossed'] = sum(r['conditions']['crossed']['score'] for r in crossed_rs) / len(crossed_rs)
                    row['capped_n']['crossed'] = sum(r['conditions']['crossed']['capped'] for r in crossed_rs)
                    row['link-crossed'] = dict(mean=sum(ds) / len(ds), ci=list(betting_ci(ds, alpha=alpha, bounds=(-1, 1))),
                                               sign=sign_test(ds))
                by_game = defaultdict(list)
                for r in rs:
                    by_game[r['game']].append(r)
                pooled = group.startswith('pooled_')
                if pooled:
                    row['game_n'] = {g: len(by_game[g]) for g in members}
                row['ci_method'] = 'stratified_bonferroni_betting' if pooled else 'iid_betting'
                for control in ('cut', 'shuffle'):
                    ds = [r['conditions']['link']['score'] - r['conditions'][control]['score'] for r in rs]
                    if pooled and n:
                        strata = [v for v in by_game.values() if v]
                        intervals = [(len(v) / n, betting_ci(
                            [r['conditions']['link']['score'] - r['conditions'][control]['score'] for r in v],
                            alpha=alpha / len(strata), bounds=(-1, 1))) for v in strata]
                        ci = [sum(w * interval[j] for w, interval in intervals) for j in (0, 1)]
                    else:
                        ci = betting_ci(ds, alpha=alpha, bounds=(-1, 1))
                    row['link-' + control] = dict(mean=sum(ds) / n if n else None, ci=list(ci),
                                                  sign=sign_test(ds))
                rows.append(row)
    return rows + chat_rows


def summarize(out_dir, tag, results_dir=None):
    """Reject partial, duplicate, or incompatible runs instead of silently pooling."""
    out = Path(out_dir)
    with (out / 'manifest.json').open() as f:
        manifest = json.load(f)
    if manifest['schema'] != SCHEMA or not manifest.get('complete'):
        raise ValueError('evaluation is incomplete or uses an incompatible schema')
    if file_hash(out / 'episodes.jsonl') != manifest['episodes_sha256']:
        raise ValueError('episodes.jsonl does not match the completed manifest')
    with (out / 'episodes.jsonl').open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    keys = [(r['episode_id'], r['receiver']) for r in records]
    if len(set(keys)) != len(keys) or len(records) != manifest['receiver_records']:
        raise ValueError('duplicate or missing receiver records')
    if 'co_chat' in manifest['games']:
        import chateval
        chateval.validate([r for r in records if r['game'] == 'co_chat'], manifest)
    for r in records:
        if r['schema'] != SCHEMA or r['game'] not in manifest['games']:
            raise ValueError('unexpected record schema/game')
        if r['game'] == 'co_chat':
            continue
        if manifest.get('crossed') and r['game'] in ('co_talk', 'co_play') and 'crossed' not in r['conditions']:
            raise ValueError('missing requested crossed condition')
        for c in (*CONDITIONS, *(('crossed',) if 'crossed' in r['conditions'] else ())):
            s = r['conditions'][c]['score']
            if not isinstance(s, (float, int)) or not math.isfinite(s) or not 0 <= s <= 1:
                raise ValueError('scores must be finite and in [0,1]')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        raise ValueError('tag must contain only letters, digits, underscore, dot or hyphen')
    table = dict(schema=SCHEMA, tag=tag, run=manifest['run'], seed=manifest['seed'],
                 tiny=manifest['tiny'], confidence=0.95, temperature=manifest.get('temperature', 0.),
                 notes=["CIs are pointwise across table rows/contrasts, not simultaneous headline discoveries.",
                        "Mean CIs: WSR 2024 terminal hedged capital, predictable bets, fixed n, bounds [-1,1].",
                        "Focal episodes and their unscored rejection-sampled donors are independent across draws.",
                        "Pool CIs combine Bonferroni per-game bounds, weighted by observed receiver counts.",
                        "Sign p tests equal non-tie sign probabilities; pooled null requires this in every game.",
                        "Exclusive means lexical payload absence, not absence of prior knowledge; scores are unchanged."],
                 rows=summary_rows(records, manifest['games']))
    if 'co_chat' in manifest['games']:
        table['notes'] += ["Chat names/leak: first city mention wins; absent-city targets are null and excluded.",
                           "Chat game is the fraction of turns with a lexical game flag, not semantic judging.",
                           "Chat means use betting bounds [0,1]; contrasts [-1,1]. Signs include p_greater.",
                           "Chat collapse includes every focal; crossed averages its two conversations before inference.",
                           "Equal focal cities remain included; partner_city_in_prompt_n reports lexical exposure."]
    dest = Path(results_dir or ROOT / 'results') / f'evalfree_{tag}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(table, indent=2, allow_nan=False) + '\n')
    return dest


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_revision(root=ROOT):
    """Optional provenance: pod source archives have no Git checkout."""
    if not (Path(root) / '.git').exists():
        return None
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run')
    ap.add_argument('--replies')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n', type=int, default=20, help='fixed focal draws per game, before role/subset filtering')
    ap.add_argument('--bs', '--batch_size', type=int, default=2)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--coread_data')
    ap.add_argument('--max_digits', type=int, default=16)
    ap.add_argument('--max_donor_attempts', type=int, default=10000)
    ap.add_argument('--games', nargs='+', help='optional explicit subset; default is every requested scored game')
    ap.add_argument('--crossed', action='store_true', help='add isolated full-dialogue crossed receiver controls')
    ap.add_argument('--temperature', type=float, default=0., help='co_chat only: 0 greedy, >0 per-episode seeded sampling')
    ap.add_argument('--summarize', action='store_true', help='summarize an already complete evaluation')
    ap.add_argument('--tag', default='headline')
    ap.add_argument('--shard', default=None, help="I/K: co_chat plans whose index is I modulo K only, so K processes "
                    "split one evaluation (summarize each shard; chateval.summary_rows over all shards' records)")
    args = ap.parse_args()
    if args.summarize:
        print(summarize(args.out_dir, args.tag))
        return
    if not args.run or not args.replies:
        ap.error('--run and --replies are required for generation')
    if not math.isfinite(args.temperature) or args.temperature < 0 or (args.temperature and
            (not args.games or 'co_chat' not in args.games)):
        ap.error('--temperature must be finite, nonnegative, and requires --games co_chat when nonzero')
    if args.bs < 2 or args.max_donor_attempts < 1 or args.max_digits < 1:
        ap.error('bs >= 2, max_donor_attempts >= 1 and max_digits >= 1 are required')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('episodes.jsonl', 'plans.jsonl', 'manifest.json')):
        raise ValueError('output files already exist; choose a fresh out_dir')
    with open(args.replies) as f:
        plans, excluded = make_plans(json.load(f), args.n, args.seed, args.coread_data, args.games,
                                    args.max_digits, args.max_donor_attempts)
    if args.shard:
        i, k = map(int, args.shard.split('/'))
        if set(plans) != {'co_chat'} or not 0 <= i < k:
            ap.error('--shard I/K requires --games co_chat and 0 <= I < K')
        plans = {'co_chat': [p for p in plans['co_chat'] if p['index'] % k == i]}
    with (out / 'plans.jsonl').open('x') as f:
        for ps in plans.values():
            for p in ps:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
    manifest = dict(schema=SCHEMA, complete=False, run=str(Path(args.run).resolve()), seed=args.seed,
                    n=args.n, bs=args.bs, games=list(plans), tiny=bool(os.environ.get('NL_TINY')),
                    device=os.environ.get('NL_DEVICE', 'auto'), nl_maxnew=os.environ.get('NL_MAXNEW'),
                    sampling='independent focal + independent conditional donor; isolated 2-row swap',
                    excluded=excluded, hashes={str(p): file_hash(p) for p in
                    (Path(args.replies), Path(args.coread_data or ROOT / 'data/coread.json'),
                     Path(args.run) / 'bridge.pt', out / 'plans.jsonl')},
                    git_commit=git_revision(), crossed=args.crossed, temperature=args.temperature, shard=args.shard,
                    chat_plans=len(plans.get('co_chat', [])),
                    code_hashes={p: file_hash(ROOT / 'pairing' / p) for p in
                                 ('evalfree.py', 'chateval.py', 'crossed.py', 'stats.py', 'link.py', 'session.py', 'evaluate.py',
                                  'coread.py', 'coread_data.py', 'games.py', 'play_tasks.py')})
    if 'co_chat' in plans:
        chat_path = Path(args.coread_data or ROOT / 'data/coread.json').with_name('chat.json')
        manifest['hashes'][str(chat_path)] = file_hash(chat_path)
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    from evaluate import load_pair
    import torch
    pair, _ = load_pair(args.run)       # once; cut never disables LoRA
    count, started = 0, time.monotonic()
    with torch.inference_mode(), (out / 'episodes.jsonl').open('x') as f:
        for g, ps in plans.items():
            for rec in evaluate_game(pair, ps, args.bs, crossed=args.crossed, temperature=args.temperature):
                f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()
                count += 1
                print(f"{g} {rec['episode_index'] + 1}/{args.n} {rec['receiver']} "
                      + ' '.join(f"{c}={r['score']:.3f}" if r['score'] is not None else f'{c}=no_city'
                                 for c, r in rec['conditions'].items()), flush=True)
            if os.environ.get('NL_DEVICE') == 'mps':
                torch.mps.empty_cache()
    manifest.update(complete=True, receiver_records=count, elapsed_seconds=time.monotonic() - started,
                    episodes_sha256=file_hash(out / 'episodes.jsonl'))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Wrote', out / 'episodes.jsonl', flush=True)


if __name__ == '__main__':
    main()
