"""Isolated mismatched conversations for free dialogue evaluation.

For receiver x in focal episode A, run only (A.x, donor.partner(x)). Both
neural reads and listener text stay in that pair, with a fresh cache. A's real
partner never runs here. For two scored receivers there are TWO conversations;
their turns must not be spliced into a fictional conversation. This removes
the original partner's causal route, including multi-turn text feedback, while
retaining a live linked donor. It changes the text context as well as the link
partner, so is not a fixed-text estimate of a direct link effect.
"""
import torch

from link import DEVICE, SIDES, partner
import session as S


def draw_play_donor(ep, source):
    """Fresh same-role play episode from source's split pools, without gold text.

    Stored play conversations can have just one item per opener/subgame. Free
    generation needs only private prompts and a speaker schedule, so sample
    secrets with the actual game builder instead of recycling that lone item.
    The caller rejects any same-secret draw through evalfree.matched.
    """
    from coread import play_secret, play_system
    game, opener = ep['meta']['play'], ep['meta']['opener']
    secrets = {x: play_secret(game, source.P, source.rng) for x in SIDES}
    return dict(game='co_play', mode='dialogue',
                system={x: play_system(game, x, opener, secrets) for x in SIDES},
                task={x: None for x in SIDES}, turns=[(sp, '', 1.0) for sp, *_ in ep['turns']],
                meta=dict(play=game, opener=opener, secrets=secrets, sender=partner(opener), receiver=opener,
                          payload={}, donor_source='fresh_split_pool_secrets', split=source.split))


def episode(ep, donor, receiver):
    """Construct only model-visible inputs; gold text/metadata cannot be forced."""
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
    """A complete isolated conversation, scored only for receiver, with exact IDs.

    A local random stream per conversation prevents changes in another run's
    length from affecting this run, also when sampling rather than greedy.
    """
    e = episode(ep, donor, receiver)
    return S.generate(pair, [e], max_turn=max_turn, temperature=temperature,
                      gen=torch.Generator(device=DEVICE if DEVICE.startswith('cuda') else 'cpu').manual_seed(seed),
                      keep=keep, keep_tokens=True)[0]


def run_episodes(pair, eps, donors, seed=0, temperature=0.0):
    """Explicit per-receiver traces and original-reference naming scores."""
    import os
    from evalfree import payloads, receiver_scores
    from evaluate import MAXNEW, mean
    if donors is None or len(donors) != len(eps):
        raise ValueError('crossed requires one explicit matched donor per episode')
    # Validate the complete plan before starting expensive generation.
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
