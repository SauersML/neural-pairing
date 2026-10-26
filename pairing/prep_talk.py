"""Writes the base-model conversations that coread.py trains on: co_talk (default), co_play (--play) and plain chats
(--chat, --chat2).

co_talk: the holder has a text (a question-answering context); the asker, whose prompt lacks it, answers a question
about it in the first turn of a text conversation, before the holder has written anything, so the answer can come only
through the link. Then the holder reacts to the answer and the asker replies. The base models (no adapters, no bridge)
write these later turns, each from the conversation laid out as in a dialogue episode (session.build): its own turns
as assistant messages, the partner's as user messages, with its private task in its system prompt. Turns are sampled
at temperature 0.7 (top-p 0.95), up to --reply tokens. A conversation is dropped when a generated turn is empty,
contains a speaker label ('Ember:', '**Quill:**': the model wrote both sides of a script), a stage-direction line
('*The link hums...*') or a reference to the link mechanism (META), or when most of the turn repeats the partner's
previous turn. Writes data/talk.json {'train': [...], 'test': [...]}: per conversation the item, the asker and the
turns.

--play: co_play conversations (coread.PLAY), --turns turns long: a game in text in which each model has a secret in
its prompt (a word to guess, a wished-for gift, a holiday city, a story character, a rhyme's last word; test secrets
come from the held-out pools). The opener states the partner's secret in turn 0, the partner reacts and states the
opener's in turn 1, and they play on. Each naming turn is written with the partner's secret in the writer's prompt
(coread.play_system peek). Stage directions and a last sentence cut off by the token limit are removed from every
turn (clean). A conversation is kept only when both naming turns state every field of the partner's secret and the
opener's first turn states no field of its own (so the second naming also needs the link); those spans are the
payload. Writes data/play.json: per conversation the game, the opener, both secrets, the system prompts, the turns and
the payload spans per side.

--chat (data/chat.json) and --chat2 (data/chat2.json, private items from chat2.py): chats with no game. The base model
writing a turn has the partner's private situation in its prompt ('teach'); the training prompt ('system') lacks it.
Each turn of a model whose partner has a private situation gets a score in 'pmi': log p(turn | teach prompt) - log
p(turn | system prompt) under the model that wrote it, in nats.

Usage: python pairing/prep_talk.py [--n_train 1200] [--n_test 120] [--bs 16] [--play | --chat | --chat2] [--turns 6]
"""
import argparse
import json
import os
import random
import re
import time

import torch

import link
import session as S
from coread import (CHAT, PLAY, chat_system, opener_gives_away, play_secret, play_spans, play_system, talk_turn0,
                    talk_system)
from coread_data import held_out
from link import DEVICE, SIDES, partner

BAD = re.compile(r'(^|\n|\*\*)\s*\w*(Quill|Ember)\w*\s*(\*\*)?\s*:|^\s*\*[^*\n]+\*\s*$|<think>', re.M | re.I)


# A turn that writes out the link in text. The dialogue system prompt says that the models exchange vectors from their
# activations; base models, which have no link, then write "[Ember's activation vector]", "*sends a vector*" or numbers
# in brackets. Training on such turns would teach the models to pass the information in text.
META = re.compile(r'activation|vector|neural|\blink(ed|s)?\b|telepath|hidden state|[\[\]]', re.I)
STAGE = re.compile(r'^\s*\*[^*\n]+\*\s*$', re.M)
ACTION = re.compile(r'[ \t]*(?<!\*)\*(?!\*)[^*\n]{1,80}(?<!\*)\*(?!\*)')


def clean(text):
    """A play turn without its stage directions (lines or inline actions in single asterisks: '*sends a vector*',
    '*grins*'; bold stays) and without a last sentence cut off by the token limit (empty when no sentence is
    complete)."""
    t = ACTION.sub('', STAGE.sub('', text))
    t = re.sub(r'\n{3,}', '\n\n', re.sub(r'^[ \t]+', '', t, flags=re.M)).strip()
    if t and t[-1] not in '.!?"\')':
        t = t[:max(t.rfind(c) for c in '.!?') + 1]
    return t.strip()


def echoes(text, prev):
    """A turn that mostly repeats the partner's previous turn (difflib's matching blocks cover 60% of it)."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text.lower(), prev.lower(), autojunk=False).find_longest_match().size > 0.6 * len(text)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--n_train', type=int, default=1200)
ap.add_argument('--n_test', type=int, default=120)
ap.add_argument('--reply', type=int, default=96)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--out', default=None, help='default data/talk.json, with --play data/play.json')
ap.add_argument('--coread', default=os.path.join(ROOT, 'data', 'coread.json'))
ap.add_argument('--play', action='store_true', help='co_play conversations (coread.PLAY) in place of co_talk')
ap.add_argument('--chat', action='store_true',
                help='plain chats (coread.CHAT) with no objective in place of co_talk; the base model writing a turn '
                'also has the partner\'s private city in its prompt (coread.chat_system\'s knows argument), so the '
                'trained model needs the link to match it')
ap.add_argument('--chat2', action='store_true', help='chats about private items of several kinds (chat2.py)')
ap.add_argument('--timing', default=None,
                help='write the generation and scoring seconds and the surviving counts as JSON')
ap.add_argument('--turns', type=int, default=None,
                help='turns per conversation (default 3; with --play, --chat or --chat2 6; at least 2)')
args = ap.parse_args()
if sum((args.play, args.chat, args.chat2)) > 1:
    ap.error('choose only one of --play, --chat, --chat2')
args.turns = args.turns or (6 if args.play or args.chat or args.chat2 else 3)
args.out = args.out or os.path.join(ROOT, 'data', 'chat2.json' if args.chat2 else
                                 'play.json' if args.play else 'chat.json' if args.chat else 'talk.json')
started = time.monotonic()


def context(tok, x, system, turns):
    """x's tokens up to its assistant header for the next turn, as session.build lays out a dialogue for x."""
    end = tok.convert_tokens_to_ids('<|im_end|>')
    t = S.prefix(tok, x, system)
    for sp, text in turns:
        if sp == x:
            t += S.assistant_header(tok, x) + S.ids(tok, text) + [end] + S.ids(tok, '\n')
        else:
            t += S.ids(tok, '<|im_start|>user\n') + S.ids(tok, text) + S.ids(tok, '<|im_end|>\n')
    return t + S.assistant_header(tok, x)


@torch.no_grad()
def generate(tok, model, prompts, seed):
    """The sampled reply (text, up to --reply tokens) of model to each token prompt, in batches of --bs."""
    end = tok.convert_tokens_to_ids('<|im_end|>')
    torch.manual_seed(seed)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for b in range(0, len(order), args.bs):
        idx = order[b:b + args.bs]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.tensor([[tok.pad_token_id] * (L - len(prompts[i])) + prompts[i] for i in idx], device=DEVICE)
        att = torch.zeros_like(ids)
        for k, i in enumerate(idx):
            att[k, L - len(prompts[i]):] = 1
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=args.reply, do_sample=True,
                           temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id, eos_token_id=end,
                           use_cache=True)
        for k, row in enumerate(g[:, L:].tolist()):
            row = row[:row.index(end)] if end in row else row
            out[idx[k]] = tok.decode(row, skip_special_tokens=True).strip()
        print(f'  {b + len(idx)}/{len(prompts)}', flush=True)
    return out


rng = random.Random(0)
convs = {}
if args.play:
    from games import POOLS
    for split, n in (('train', args.n_train), ('test', args.n_test)):
        pools = {k: v[split] for k, v in POOLS.items()}               # test: secrets from the held-out pools
        convs[split] = []
        for i in range(n):
            game, opener = list(PLAY)[i % len(PLAY)], rng.choice(SIDES)
            secrets = {x: play_secret(game, pools, rng) for x in SIDES}
            convs[split].append(dict(game=game, opener=opener, secrets=secrets, asker=opener, holder=partner(opener),
                                     turns=[], system={x: play_system(game, x, opener, secrets) for x in SIDES},
                                     peek=[play_system(game, x, opener, secrets, peek=True)
                                           for x in (opener, partner(opener))]))
elif args.chat2:
    import chat2
    chat2.validate_catalog()
    convs = {split: [chat2.conversation(split, rng) for _ in range(n)]
             for split, n in (('train', args.n_train), ('test', args.n_test))}
elif args.chat:
    from games import POOLS
    for split, n in (('train', args.n_train), ('test', args.n_test)):
        cities = POOLS['cities'][split]                                # test: places from the held-out pool
        convs[split] = []
        for i in range(n):
            opener = rng.choice(SIDES)
            private = {x: rng.choice(cities) if rng.random() < 0.75 else None for x in SIDES}
            wording = {x: rng.choice(CHAT[:3] if private[x] else CHAT[3:]) for x in SIDES}
            convs[split].append(dict(game='chat', opener=opener, private=private, asker=opener, holder=partner(opener),
                                     turns=[], system={x: chat_system(x, private[x], wording[x]) for x in SIDES},
                                     teach={x: chat_system(x, private[x], wording[x], private[partner(x)])
                                            for x in SIDES}))
else:
    data = json.load(open(args.coread))
    # test: no content seen in training
    pool = {'train': data['train'], 'test': held_out(data['train'], data['test'])}
    for split, n in (('train', args.n_train), ('test', args.n_test)):
        items = [it for it in pool[split] if it['task'] in ('qa', 'hop') and len(it['answer']) <= 80]
        convs[split] = [dict(item=it, asker=rng.choice(SIDES)) for it in rng.sample(items, min(n, len(items)))]
        for c in convs[split]:
            c['holder'] = partner(c['asker'])
            c['system'] = {x: talk_system(c['item'], x, c['asker']) for x in SIDES}
            c['turns'] = [(c['asker'], talk_turn0(c['item']))]
allc = convs['train'] + convs['test']
# One model is in memory at a time: Quill's holders, then Ember's holders and askers, then Quill's askers.
# Turn 2 of a conversation needs its turn 1, which the partner (the holder) wrote in an earlier pass.
loaded = {}


def model(x):
    """(tokenizer, base model) of side x; loading it frees the other model."""
    if x not in loaded:
        loaded.clear()
        if DEVICE.startswith('mps'):
            torch.mps.empty_cache()
        tok, m = link.load(x)
        loaded[x] = (tok, m.eval())
    return loaded[x]


def turn(k, who, x):
    """Generates turn k with model x for the conversations whose `who` (asker or holder) is x."""
    tok, m = model(x)
    mine = [c for c in allc if c[who] == x and len(c['turns']) == k]
    # play: the prompts of the naming turns show the partner's secret; chat: every turn's prompt shows the partner's
    # private situation
    sysp = [c['peek'][k] if k < 2 and 'peek' in c else c.get('teach', c['system'])[x] for c in mine]
    texts = generate(tok, m, [context(tok, x, p, c['turns']) for p, c in zip(sysp, mine)], seed=k)
    for c, t in zip(mine, texts):
        c['turns'].append((x, clean(t) if args.play or args.chat or args.chat2 else t))
    if mine:
        print(f'turn {k} ({who} {x}): {len(mine)}; example: {mine[0]["turns"]}', flush=True)


@torch.no_grad()
def turn_nats(tok, m, items):
    """items: [(system, turns, x)]. Per item, the log-probability (nats) of each of x's turns, closing token included,
    in the conversation laid out as context() lays it out for x under system."""
    end = tok.convert_tokens_to_ids('<|im_end|>')
    seqs, spans = [], []
    for system, turns, x in items:
        t, sp = S.prefix(tok, x, system), []
        for who, text in turns:
            if who == x:
                t += S.assistant_header(tok, x)
                sp.append((len(t), len(t) + len(S.ids(tok, text)) + 1))
                t += S.ids(tok, text) + [end] + S.ids(tok, '\n')
            else:
                t += S.ids(tok, '<|im_start|>user\n') + S.ids(tok, text) + S.ids(tok, '<|im_end|>\n')
        seqs.append(t)
        spans.append(sp)
    out, bs = [], max(1, args.bs // 8)               # full-vocabulary logits: a smaller batch than generation
    for b in range(0, len(seqs), bs):
        chunk = seqs[b:b + bs]
        L = max(map(len, chunk))
        ids = torch.tensor([q + [tok.pad_token_id] * (L - len(q)) for q in chunk], device=DEVICE)
        att = torch.tensor([[1] * len(q) + [0] * (L - len(q)) for q in chunk], device=DEVICE)
        lp = torch.log_softmax(m(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
        got = lp.gather(-1, ids[:, 1:, None])[..., 0]
        out += [[float(got[k, a - 1:z - 1].sum()) for a, z in sp] for k, sp in enumerate(spans[b:b + bs])]
    return out


def prune(k):
    """Keeps the conversations whose turn k passes the filters (BAD, META, echoes, and the play payload checks)."""
    global allc
    keep = [c for c in allc if len(c['turns']) == k + 1 and c['turns'][-1][1] and not BAD.search(c['turns'][-1][1])
            and not META.search(c['turns'][-1][1])
            and (k == 0 or not echoes(c['turns'][-1][1], c['turns'][-2][1]))
            and (k > 1 or 'secrets' not in c or play_spans(c['secrets'][partner(c['turns'][k][0])], c['turns'][k][1]))
            and (k > 0 or 'secrets' not in c or not opener_gives_away(c))]
    print(f'turn {k}: kept {len(keep)} of {len(allc)}', flush=True)
    allc = keep


# talk: turn 1 the holder reacts, turn 2 the asker replies. play: turn 0 the opener (the asker) states the partner's
# secret, turn 1 the partner reacts and states the opener's, then they alternate. Each turn runs both models, the one
# loaded last first.
order = list(SIDES)
for k in range(0 if args.play or args.chat or args.chat2 else 1, args.turns):
    for x in order:
        turn(k, 'asker' if k % 2 == 0 else 'holder', x)
    order.reverse()
    prune(k)
generation_done = time.monotonic()
if args.chat or args.chat2:
    # The information each turn has about the partner's private situation: log p(turn | the writer's prompt, which
    # shows it) - log p(turn | the training prompt, which does not), under the model that wrote it. This is the extra
    # loss on the turn of a model without the information; values near 0 everywhere would mean that the turns do not
    # depend on it.
    for x in order:
        tok, m = model(x)
        mine = [c for c in allc if c['private'][partner(x)]]
        shown = turn_nats(tok, m, [(c['teach'][x], c['turns'], x) for c in mine])
        hidden = turn_nats(tok, m, [(c['system'][x], c['turns'], x) for c in mine])
        for c, a, b in zip(mine, shown, hidden):
            c.setdefault('pmi', {})[x] = [round(p - q, 3) for p, q in zip(a, b)]
kept = {id(c) for c in allc}
convs = {split: [c for c in v if id(c) in kept] for split, v in convs.items()}
for split, v in convs.items():
    turns = [(c, x, k, nats) for c in v for x, ns in c.get('pmi', {}).items() for k, nats in enumerate(ns)]
    if turns:
        from chat2 import contains_item
        said = [contains_item([t for sp, t in c['turns'] if sp == x][k], c['private'][partner(x)]) if args.chat2 else
                c['private'][partner(x)].lower() in [t for sp, t in c['turns'] if sp == x][k].lower()
                for c, x, k, _ in turns]
        nats = sorted(n for *_, n in turns)
        target_kind = 'item' if args.chat2 else 'city'
        print(f'{split}: {len(turns)} turns whose writer saw the partner\'s {target_kind}: '
              f'median {nats[len(nats) // 2]:.2f} nats about it, {sum(n > 0 for n in nats) / len(nats):.0%} positive, '
              f'{sum(said) / len(said):.0%} name it',
              flush=True)
for c in allc:
    if args.play:
        c['spans'] = {sp: play_spans(c['secrets'][partner(sp)], text) for sp, text in c['turns'][:2]}
        for k in ('holder', 'asker', 'peek'):
            del c[k]
    if args.chat or args.chat2:
        for k in ('holder', 'asker'):
            del c[k]
json.dump(convs, open(args.out, 'w'), indent=1)
print('wrote', args.out, {s: len(v) for s, v in convs.items()}, flush=True)
if args.timing:
    json.dump(dict(generation_seconds=generation_done - started,
                   pmi_and_write_seconds=time.monotonic() - generation_done,
                   total_seconds=time.monotonic() - started, attempted=dict(train=args.n_train, test=args.n_test),
                   surviving={s: len(v) for s, v in convs.items()}, tiny=bool(os.environ.get('NL_TINY')),
                   device=DEVICE, turns=args.turns, reply=args.reply, bs=args.bs), open(args.timing, 'w'), indent=2)
