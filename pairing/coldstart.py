"""Opt-in cold-start curriculum and optimizer controls; legacy defaults are unchanged."""
import math

import torch

from link import MODELS, SIDES


def add_arguments(ap):
    ap.add_argument('--gate_norm', action='store_true',
                    help='visibility gates read unit-RMS states; persisted in bridge checkpoints')
    ap.add_argument('--warmup_steps', type=int, default=100,
                    help='LR warmup steps, including after --init (0: no warmup)')
    ap.add_argument('--relay_first', type=int, default=0,
                    help='initial steps of short relay_word transfers only, without competition or send cost')
    ap.add_argument('--relay_games', default='relay_word',
                    help='the games of the --relay_first steps, comma separated, drawn uniformly: relay_* as short '
                         'transfers (the sender unscored), merge_sum, merge_fact and lookup as their own episodes')
    ap.add_argument('--hold', type=int, default=0,
                    help='initial steps in which the visibility gates and alpha do not train, so a link whose writes '
                         'are still noise can only quiet them by learning its reads and writes, not by closing')
    ap.add_argument('--alpha_floor', type=float, default=0.0,
                    help='with --hold, alpha trains and only the gates are held; after every step each write size '
                         'tanh(alpha) is raised to at least this, so the link cannot close (0: --hold holds alpha too)')
    for x in SIDES:
        ap.add_argument('--lr_bridge_' + x, type=float, default=None,
                        help='bridge LR for ' + x + ' (except alpha); default --lr_bridge')


def lr_multiplier(step, steps, warmup_steps=100):
    return min(1.0, step / max(1, warmup_steps)) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))


def bridge_groups(pair, lr, lr_alpha, overrides):
    """Optimizer groups of the bridges' trainable parameters (a one-way link leaves some untrained: link.Pair.freeze)."""
    return [dict(params=[p for n, p in pair.bridge[x].named_parameters() if n != 'alpha' and p.requires_grad],
                 lr=lr if overrides[x] is None else overrides[x]) for x in SIDES] + [
        dict(params=[pair.bridge[x].alpha for x in SIDES if pair.bridge[x].alpha.requires_grad], lr=lr_alpha)]


def hold(pair, alpha=True):
    """Drop the gradients of every visibility gate and (alpha=True) alpha (set to None, so AdamW leaves them exactly as
    they are)."""
    for x in SIDES:
        if alpha:
            pair.bridge[x].alpha.grad = None
        for p in pair.bridge[x].gate.parameters():
            p.grad = None


def floor(pair, f):
    """Raise every trained write size tanh(alpha) below f to f (a projection after the optimizer step; f = 0: nothing)."""
    if f <= 0:
        return
    for x in SIDES:
        a = pair.bridge[x].alpha
        if a.requires_grad:
            with torch.no_grad():
                a.clamp_(min=math.atanh(f))


KIND = {'relay_word': 'word', 'relay_number': 'number', 'relay_phrase': 'two-word phrase',
        'relay_digits': 'string of digits'}
VALUE_GAMES = tuple(KIND) + ('merge_sum', 'merge_fact', 'lookup')


def relay_episode(games, pick=('relay_word',), pause=0):
    """An episode of one of the games pick (drawn uniformly; one game: no draw). A relay game is a short transfer
    with an unscored sender and payload spans recognized by session.payload_labels; merge_sum, merge_fact and lookup
    are their regular episodes, whose values sit in both prompts.

    The regular game's target and all existing metrics stay unchanged. The warm curriculum's receiver wording for a
    word deliberately uses the payload parser's 'The word is ' prefix; the other relays keep the regular answer. No
    answer appears in the receiver's prompt.
    """
    game = pick[0] if len(pick) == 1 else games.rng.choice(pick)
    ep = games.episode(game)
    if not isinstance(pause, int) or isinstance(pause, bool) or pause < 0:
        raise ValueError('pause must be a nonnegative integer')
    if pause:
        ep['pause'] = pause
    if game not in KIND:
        return ep
    s, r, value = (ep['meta'][k] for k in ('sender', 'receiver', 'value'))
    ep['task'][s] = f'Your secret {KIND[game]} is "{value}". Pass it to {MODELS[r]["name"]} through the link.'
    ep['reply'][s] = ('', 0.0)
    if game == 'relay_word':
        ep['reply'][r] = (f'The word is {value}.', 1.0)
    return ep
