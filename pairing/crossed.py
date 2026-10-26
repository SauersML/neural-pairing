"""The crossed control of free dialogue evaluation (evalfree.py --crossed): a new conversation of the receiver with the
donor's partner.

For receiver x of focal episode A, only x with A's prompts and the donor's partner with the donor's prompts run, with
the link on and a fresh cache, so both the link and the dialogue text come from that pair and A's real partner has no
path to x. An episode with two scored receivers gives two separate conversations. The control changes the text that x
reads as well as its link partner, so it does not estimate the link's effect at fixed text.

Usage: crossed.generate(pair, ep, donor, receiver, max_turn)
"""
import torch

from link import DEVICE, SIDES, partner
import session as S


def draw_play_donor(ep, source):
    """A new co_play donor for ep: the same game, opener and speaker order, with secrets drawn from source's split
    pools and no turn text. Free generation needs only the prompts and the speaker order, and the stored conversations
    can have a single conversation per game and opener. evalfree.matched rejects a draw with the same secrets."""
    from coread import play_secret, play_system
    game, opener = ep['meta']['play'], ep['meta']['opener']
    secrets = {x: play_secret(game, source.P, source.rng) for x in SIDES}
    return dict(game='co_play', mode='dialogue',
                system={x: play_system(game, x, opener, secrets) for x in SIDES},
                task={x: None for x in SIDES}, turns=[(sp, '', 1.0) for sp, *_ in ep['turns']],
                meta=dict(play=game, opener=opener, secrets=secrets, sender=partner(opener), receiver=opener,
                          payload={}, donor_source='fresh_split_pool_secrets', split=source.split))


def episode(ep, donor, receiver):
    """The crossed dialogue for receiver: receiver's prompts from ep, the partner's from donor, empty turns in ep's
    speaker order, and no metadata or reference text. ValueError unless donor is a matched donor of the same
    co_talk or co_play schedule."""
    from evalfree import matched, payloads
    if ep['mode'] != 'dialogue' or ep['game'] not in ('co_talk', 'co_play'):
        raise ValueError('crossed dialogue requires co_talk or co_play')
    if receiver not in payloads(ep) or not matched(ep, donor):
        raise ValueError('crossed dialogue requires a scored receiver and a matched donor')
    if [t[0] for t in ep['turns']] != [t[0] for t in donor['turns']]:
        raise ValueError('crossed donor must have the same speaker schedule')
    y = partner(receiver)
    e = dict(mode='dialogue', system={receiver: ep['system'][receiver], y: donor['system'][y]},
             task={receiver: ep['task'][receiver], y: donor['task'][y]},
             turns=[(sp, '', 1.0) for sp, *_ in ep['turns']])
    for k in ('pause_turns', 'link_off_after'):
        if k in ep:
            e[k] = ep[k]
    return e


@torch.no_grad()
def generate(pair, ep, donor, receiver, max_turn=64, temperature=0.0, seed=0, keep=False):
    """The generated crossed conversation for receiver (session.generate's result, with token IDs). Each conversation
    samples from its own random generator, seeded with seed, so other conversations do not change its draws."""
    e = episode(ep, donor, receiver)
    return S.generate(pair, [e], max_turn=max_turn, temperature=temperature,
                      gen=torch.Generator(device=DEVICE if DEVICE.startswith('cuda') else 'cpu').manual_seed(seed),
                      keep=keep, keep_tokens=True)[0]


def run_episodes(pair, eps, donors, seed=0, temperature=0.0):
    """One result per episode: each scored receiver's crossed conversation and its score against ep's references
    (evalfree.receiver_scores), with their mean as ok. Every plan is checked before any generation."""
    import os
    from evalfree import payloads, receiver_scores
    from evaluate import MAXNEW, mean
    if donors is None or len(donors) != len(eps):
        raise ValueError('crossed requires one explicit matched donor per episode')
    for ep, donor in zip(eps, donors):
        if ep['mode'] != 'dialogue' or ep['game'] not in ('co_talk', 'co_play'):
            raise ValueError('crossed dialogue requires co_talk or co_play')
        receivers = payloads(ep)
        if not receivers:
            raise ValueError('crossed dialogue requires scored receivers')
        for x in receivers:
            episode(ep, donor, x)
    out = []
    for ep, donor in zip(eps, donors):
        limit = int(os.environ.get('NL_MAXNEW', 0)) or MAXNEW.get(ep['game'], 64)
        runs = {x: generate(pair, ep, donor, x, limit, temperature, seed) for x in payloads(ep)}
        scores = {x: receiver_scores(ep, r)[x] for x, r in runs.items()}
        out.append(dict(game=ep['game'], meta=ep['meta'], ok=mean(list(scores.values())), each=scores,
                        receiver_runs=runs, donor=dict(system=donor['system'], task=donor['task'], meta=donor['meta']),
                        implementation='isolated_crossed_dialogue_per_receiver'))
    return out
