"""Opt-in cold-start curriculum and optimizer controls; legacy defaults are unchanged."""
import math

from link import MODELS, SIDES


def add_arguments(ap):
    ap.add_argument('--gate_norm', action='store_true',
                    help='visibility gates read unit-RMS states; persisted in bridge checkpoints')
    ap.add_argument('--warmup_steps', type=int, default=100,
                    help='LR warmup steps, including after --init (0: no warmup)')
    ap.add_argument('--relay_first', type=int, default=0,
                    help='initial steps of short relay_word transfers only, without competition or send cost')
    for x in SIDES:
        ap.add_argument('--lr_bridge_' + x, type=float, default=None,
                        help='bridge LR for ' + x + ' (except alpha); default --lr_bridge')


def lr_multiplier(step, steps, warmup_steps=100):
    return min(1.0, step / max(1, warmup_steps)) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))


def bridge_groups(pair, lr, lr_alpha, overrides):
    return [dict(params=[p for n, p in pair.bridge[x].named_parameters() if n != 'alpha'],
                 lr=lr if overrides[x] is None else overrides[x]) for x in SIDES] + [
        dict(params=[pair.bridge[x].alpha for x in SIDES], lr=lr_alpha)]


def relay_episode(games):
    """A short transfer with an unscored sender and payload spans recognized by session.payload_labels.

    The regular game's target and all existing metrics stay unchanged. The warm curriculum's receiver wording
    deliberately uses the payload parser's 'The word is ' prefix. No answer appears in the receiver's prompt.
    """
    ep = games.episode('relay_word')
    s, r, value = (ep['meta'][k] for k in ('sender', 'receiver', 'value'))
    ep['task'][s] = f'Your secret word is "{value}". Pass it to {MODELS[r]["name"]} through the link.'
    ep['reply'][s] = ('', 0.0)
    ep['reply'][r] = (f'The word is {value}.', 1.0)
    return ep
