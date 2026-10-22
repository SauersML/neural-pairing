"""Conversations for co_talk (coread.py): answer first, then talk.

The holder reads a text (a question-answering context); the asker, who cannot see it, answers a question about it in
the very first turn of a text conversation, before the holder has written a word, so that answer can only come
through the link. Then they talk: the holder reacts to the answer and the asker replies, each in its own words. The
two later turns are written by the original models themselves (no adapters, no bridge), each reading the conversation
exactly as it reads it in a dialogue episode (session.build): its own turns as assistant messages, the partner's as
user messages, with its private task in its system prompt. Sampled at temperature 0.7, up to --reply tokens.

A conversation is dropped when a generated turn is empty, names a speaker ('Ember:', '**Quill:**', 'emember:': the
model wrote both sides of a script) or has a line that is a stage direction ('*The link hums...*'), or when most of the
turn repeats the partner's previous turn verbatim.

Writes data/talk.json {'train': [...], 'test': [...]}: per conversation the item, the asker, the three turns.

With --play, conversations for co_play (coread.PLAY) instead: a game in text while each twin holds a secret the
other reads through the link (a word to guess, a wished-for gift, a holiday city, a story character, a rhyme's last
word; the test part draws secrets from the held-out pools), --turns turns long. The opener names the partner's secret
in turn 0, the partner reacts and names the opener's in turn 1, and they keep playing; every turn is written by the
original models. Each naming turn is written with the partner's secret shown in the writer's prompt
(coread.play_system peek). Stage-direction lines and a cut-off last sentence are trimmed from every play turn (clean),
and a conversation is kept only when both naming turns name every field of the partner's
secret and the opener's first turn names no field of the opener's own (so the second naming also needs the link);
those spans are the payload. Writes data/play.json: per conversation the game, the opener, both secrets, the
system prompts, the turns and the payload spans per side.

Usage: python prep_talk.py [--n_train 1200] [--n_test 120] [--bs 16] [--play [--turns 6]]   (NL_DEVICE=mps on the Mac)
With --mlx the turns come from the 4-bit MLX bases the Mac trains on (mlx_pair.load_base), about 2.5 GB per model.
"""
import argparse
import json
import os
import random
import re

import torch

import link
import session as S
from coread import CHAT, PLAY, chat_system, opener_gives_away, play_secret, play_spans, play_system, talk_turn0, talk_system
from coread_data import held_out
from link import DEVICE, SIDES, partner

BAD = re.compile(r'(^|\n|\*\*)\s*\w*(Quill|Ember)\w*\s*(\*\*)?\s*:|^\s*\*[^*\n]+\*\s*$|<think>', re.M | re.I)


# A turn that acts the link out in text: the dialogue prompt says the twins send vectors from their activations, and
# the original models, which cannot, write "[Ember's activation vector]", "*sends a vector*" or numbers in brackets.
# Training on such a turn teaches a token channel.
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
ap.add_argument('--mlx', action='store_true', help='generate with the 4-bit MLX bases (mlx_pair.load_base)')
ap.add_argument('--mlx_gib', type=float, default=5.0, help='MLX memory limit with --mlx')
ap.add_argument('--play', action='store_true', help='co_play conversations (coread.PLAY) in place of co_talk')
ap.add_argument('--chat', action='store_true', help='plain chats (coread.CHAT) with no objective in place of co_talk; the original model writing a turn also '
                'sees the partner\'s private city (coread.chat_system knows), so the trained model needs the link to match it')
ap.add_argument('--turns', type=int, default=None, help='turns per conversation (default 3; with --play or --chat 6, at least 2)')
args = ap.parse_args()
if args.chat and args.mlx:
    ap.error('--chat scores each turn with torch (turn_nats); drop --mlx')
args.turns = args.turns or (6 if args.play or args.chat else 3)
args.out = args.out or os.path.join(ROOT, 'data', 'play.json' if args.play else 'chat.json' if args.chat else 'talk.json')


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


def generate_mlx(tok, model, prompts, seed):
    import mlx.core as mx
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    end = tok.convert_tokens_to_ids('<|im_end|>')
    mx.set_memory_limit(int(args.mlx_gib * 2**30))
    mx.set_cache_limit(256 * 2**20)        # freed buffers MLX keeps count toward the process footprint
    mx.random.seed(seed)
    out = []
    for b in range(0, len(prompts), 2 * args.bs):
        r = batch_generate(model, TokenizerWrapper(tok, eos_token_ids={end}), prompts[b:b + 2 * args.bs],
                           max_tokens=args.reply, sampler=make_sampler(temp=0.7, top_p=0.95),
                           completion_batch_size=args.bs, prefill_batch_size=4, return_token_ids=True)
        mx.clear_cache()
        out += [tok.decode([t for t in row if t != end], skip_special_tokens=True).strip() for row in r.token_ids]
        print(f'  {len(out)}/{len(prompts)}', flush=True)
    return out


@torch.no_grad()
def generate(tok, model, prompts, seed):
    if args.mlx:
        return generate_mlx(tok, model, prompts, seed)
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
# One model in memory at a time (the Mac): Quill's holders, then Ember's holders and askers, then Quill's askers.
# Turn 2 of a conversation needs its turn 1, which the partner (the holder) wrote in an earlier pass.
loaded = {}


def model(x):
    if x not in loaded:
        loaded.clear()
        if args.mlx:
            import mlx.core as mx
            from transformers import AutoTokenizer
            from mlx_pair import HF, load_base
            mx.clear_cache()
            loaded[x] = (AutoTokenizer.from_pretrained(HF[x], local_files_only=True), load_base(HF[x])[0])
            return loaded[x]
        if DEVICE.startswith('mps'):
            torch.mps.empty_cache()
        tok, m = link.load(x)
        loaded[x] = (tok, m.eval())
    return loaded[x]


def turn(k, who, x):
    tok, m = model(x)
    mine = [c for c in allc if c[who] == x and len(c['turns']) == k]
    # play: the naming turns see the partner's secret; chat: every turn sees the partner's city
    sysp = [c['peek'][k] if k < 2 and 'peek' in c else c.get('teach', c['system'])[x] for c in mine]
    texts = generate(tok, m, [context(tok, x, p, c['turns']) for p, c in zip(sysp, mine)], seed=k)
    for c, t in zip(mine, texts):
        c['turns'].append((x, clean(t) if args.play or args.chat else t))
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
    global allc
    keep = [c for c in allc if len(c['turns']) == k + 1 and c['turns'][-1][1] and not BAD.search(c['turns'][-1][1])
            and not META.search(c['turns'][-1][1])
            and (k == 0 or not echoes(c['turns'][-1][1], c['turns'][-2][1]))
            and (k > 1 or 'secrets' not in c or play_spans(c['secrets'][partner(c['turns'][k][0])], c['turns'][k][1]))
            and (k > 0 or 'secrets' not in c or not opener_gives_away(c))]
    print(f'turn {k}: kept {len(keep)} of {len(allc)}', flush=True)
    allc = keep


# talk: turn 1 the holder reacts, turn 2 the asker replies. play: turn 0 the opener (the asker) names the partner's
# secret, turn 1 the partner reacts and names the opener's, then they alternate. Each turn runs both models, the one
# loaded last first.
order = list(SIDES)
for k in range(0 if args.play or args.chat else 1, args.turns):
    for x in order:
        turn(k, 'asker' if k % 2 == 0 else 'holder', x)
    order.reverse()
    prune(k)
if args.chat:
    # What each turn says about the partner's city: log p(turn | the writer's prompt, which shows that city) - log p(turn
    # | the trained prompt, which does not), under the model that wrote it. A model that ignores the link pays this much
    # more loss on the turn than one that reads it perfectly; ~0 everywhere would mean the data teaches no link use.
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
        said = [c['private'][partner(x)].lower() in [t for sp, t in c['turns'] if sp == x][k].lower()
                for c, x, k, _ in turns]
        nats = sorted(n for *_, n in turns)
        print(f'{split}: {len(turns)} turns whose writer saw the partner\'s city: median {nats[len(nats) // 2]:.2f} nats '
              f'about it, {sum(n > 0 for n in nats) / len(nats):.0%} positive, {sum(said) / len(said):.0%} name it',
              flush=True)
for c in allc:
    if args.play:
        c['spans'] = {sp: play_spans(c['secrets'][partner(sp)], text) for sp, text in c['turns'][:2]}
        for k in ('holder', 'asker', 'peek'):
            del c[k]
    if args.chat:
        for k in ('holder', 'asker'):
            del c[k]
json.dump(convs, open(args.out, 'w'), indent=1)
print('wrote', args.out, {s: len(v) for s, v in convs.items()}, flush=True)
