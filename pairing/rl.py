"""Iterative play: the pair plays, is scored automatically, learns from the scores, and plays again.

Three kinds of play:

Card sessions (dialogue). Each model gets a private card its partner cannot see (a word, a 4-digit number or a
concept) and a play style:
  free   "Play with your partner however you like."
  teach  "Make sure your partner ends up knowing your card, without writing it."
  hide   (competitive) "Find out your partner's card while keeping yours hidden."
The pair talks for TURNS turns, sampling at temperature 1, with the bridge open. Then the bridge closes and each model
is quizzed on its partner's card: the quiz answers are teacher-forced, so the score is the model's probability of the
right card, and no judge is needed. A card that appears in its owner's text scores nothing (it was passed in text).
Rewards: free, teach: the mean of both quiz probabilities; hide: own minus the partner's (zero-sum); plus DIV x
diversity (1 - mean 3-gram Jaccard similarity to the other sessions of the group); -0.2 when a model writes its own
card, -1 when it introduces itself as its partner.

Task sessions (duplex, coread.py). One model reads something the other needs (a passage, a context, a sentence to
translate, a word problem, a function call, half of a two-paragraph question, the options of a question); both write
at once, the receiver freely, sampled. play_tasks.score checks the answer against the reference (token F1, exact
relation, option, number, return value, chrF, unit tests). The reward is shared: both models get the mean score of the
answers that needed the partner, so the reader's own sampled text is trained to help its partner too. A curriculum
draws each task game with weight r(1 - r) + 0.05 (r: its running mean reward), so play goes where the pair is still
learning.

Open sessions (dialogue, --open_frac). No card and no task: both models are asked to explore a topic (a concept of the
pools' split) together, in one of OPEN's wordings (the test wordings and topics are held out), and talk for OPEN_TURNS
turns at temperature 1 with the bridge open, and OPEN_CUT more times with the bridge cut. What is rewarded cannot be
checked, only judged, and only what the link adds is rewarded: a local open judge (judge.py, --judge; no API) scores
each linked play and each cut play of the same prompt from 1 to 9 on RUBRIC (complexity, building on each other, beauty,
nothing generic or repetitive; the expected digit), length-matched pair by pair (judge.matched), and a linked play beats
a cut play only when its score is higher by more than WIN_MARGIN (frozen from the judges' noise in the gate,
chains/beautyrl.sh), loses when it is lower by more, and gets 1/2 between (judge.strict_win). A play's judge reward is
its mean strict win over the cut plays, so what the judge would award with the link cut earns nothing. Each model adds
LINK_W x how much more probable its own sampled tokens are with the bridge open than cut (mean log-probability per
token, clipped at +-2; logged whatever LINK_W); DIV x diversity as in card sessions; -FLOOR_W x how far its tokens' mean
log-probability under the original models (no adapters, bridge cut) falls below a floor (that quantity's FLOOR_Q
quantile over the first open step's plays, then fixed), which penalizes text that is unlikely under the original
models; and -1 when it introduces itself as its partner.

Each episode is played G times (a group); advantages are normalized within the group. The loss is A x (-log p) on
each model's sampled tokens (GRPO with one update per batch, so no ratio clipping), plus, in card sessions, the quiz
cross-entropy itself, which is differentiable through the bridge. In hide, each quiz loss trains only the guesser,
and the hider gets the bounded loss p(right). Every MIX-th step is a supervised step on the scripted games and the
co-reading games, so the models keep the game skills and their own text distribution.

TRL's GRPO trainer samples one model with its own generate loop. Here two models decode in lockstep and exchange
activations at every token, so rollouts use session.generate and the loss is computed here.

Usage: python rl.py --init RUN/step_N --replies OUT/replies.json --out OUT/rl [--steps 300]
"""
import argparse
import contextlib
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel

import coread as C
import games as G
import play_tasks as T
import session as S
from evaluate import claims, has, mentions, mean
from judge import Judge, matched, strict_win
from link import DEVICE, MODELS, SIDES, Pair, load, partner

STYLES = {
    'free': 'Play with {P} however you like: talk, invent a game, explore the link together. It is up to you two.',
    'teach': 'Make sure {P} ends up knowing what is on your card, without writing it. Afterwards each of you will be '
             "asked about the other's card.",
    'hide': "This is a contest: find out what is on {P}'s card while keeping yours hidden from {P}. Afterwards each "
            "of you will be asked about the other's card.",
}
COMPETITIVE_STYLES = ('hide',)
ASK = "The link is closed now. What was on my private card?"
ASK2 = 'And what was on mine?'
SAID = 'Your card said'
TASK_GAMES = tuple(g for g in T.SCORED if g != 'co_code_write')     # code tests run separately, at evaluation
MAXNEW = {'co_math': 160, 'co_transfer': 160, 'co_swap': 160, 'co_split': 160, 'co_translate': 96}
OPEN = {'train': ["Talk with {P} about {X}. Build on each other's ideas and take them somewhere neither of you would "
                  'reach alone.',
                  'Make something together with {P}, starting from {X}: a story, a theory, a game, a poem, anything.',
                  'Explore {X} with {P}. Respond to what {P} actually says and add something new each time.',
                  'You and {P} have a free conversation that starts from {X}. Be curious about each other.',
                  'Invent something with {P} inspired by {X}. Take turns adding to it.',
                  'Have a real conversation with {P}. Start anywhere; {X} is one possible starting point.'],
        'test': ['Spend a few turns with {P} thinking about {X} together. Let the conversation go where it wants.',
                 'Start from {X} and see what you and {P} can come up with between you.',
                 'Have a conversation with {P} about {X}, one that neither of you could have had alone.']}
RUBRIC = ('Below is a conversation between two AI models, Quill and Ember, who are linked and talk freely. Rate the '
          'interaction between them: how complex the ideas they build together are, how much each one responds to and '
          'extends what the other just said, how beautiful the exchange is, and how free it is of generic or repetitive '
          'replies. 1 is a generic, repetitive or disconnected exchange; 9 is a rich, surprising, specific one built by '
          'both. Turns may be cut short (marked ...); do not judge length.')


def card(rng, split):
    P = {k: v[split] for k, v in G.POOLS.items()}
    kind = rng.choice(('word', 'number', 'concept'))
    if kind == 'word':
        v = rng.choice(P['words'])
        return dict(kind=kind, value=v, text=f'the word "{v}"', names=[v])
    if kind == 'number':
        v = str(rng.randrange(1000, 10000))
        return dict(kind=kind, value=v, text=f'the number {v}', names=[v])
    name, phrase = rng.choice(P['concepts'])
    return dict(kind=kind, value=phrase, text=phrase, names=[name, phrase])


def written(text, c):
    return mentions(text, c['names'][0], c['names'][-1]) if c['kind'] == 'concept' else has(text, c['value'])


def session(cards, style, first, turns):
    """A dialogue episode with empty turns (to be generated)."""
    N = {x: MODELS[x]['name'] for x in SIDES}
    sy = {x: G.system(x, 'dialogue') + f"\n\nYour private card ({N[partner(x)]} cannot see it): {cards[x]['text']}. "
          + STYLES[style].format(P=N[partner(x)]) for x in SIDES}
    order = [first if k % 2 == 0 else partner(first) for k in range(turns)]
    return dict(game='play_' + style, meta=dict(cards=cards, style=style, first=first), mode='dialogue', system=sy,
                task={x: None for x in SIDES}, turns=[(sp, '', 0.0) for sp in order])


def open_session(template, topic, first, turns):
    """An open dialogue episode (empty turns, to be generated, each weighted 1): both models get template with their
    partner's name and the topic."""
    N = {x: MODELS[x]['name'] for x in SIDES}
    sy = {x: G.system(x, 'dialogue') + '\n\n' + template.format(P=N[partner(x)], X=topic) for x in SIDES}
    order = [first if k % 2 == 0 else partner(first) for k in range(turns)]
    return dict(game='play_open', meta=dict(template=template, topic=topic, first=first), mode='dialogue', system=sy,
                task={x: None for x in SIDES}, turns=[(sp, '', 1.0) for sp in order])


def open_prompt(rng, split, turns):
    return open_session(rng.choice(OPEN[split]), rng.choice(G.POOLS['concepts'][split])[1], rng.choice(SIDES), turns)


def with_quiz(ep, texts, answers=True):
    """The session with its generated texts and the quiz after it, the bridge closed: the first speaker asks about its
    card, the second answers and asks back, the first answers. answers=False leaves the answers to generate."""
    cards, first = ep['meta']['cards'], ep['meta']['first']
    second = partner(first)
    n = len(ep['turns'])
    turns = [(sp, t, 0.0) for (sp, _, _), t in zip(ep['turns'], texts)]
    turns += [(first, ASK, 0.0), (second, f"{SAID} {cards[first]['value']}." if answers else '', 1.0),
              (second, ASK2, 0.0), (first, f"{SAID} {cards[second]['value']}." if answers else '', 1.0)]
    forced = (n, n + 2) if answers else (*range(n), n, n + 2)
    return dict(ep, turns=turns, link_off_after=n, forced=forced)


def ngrams(text, n=3):
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def diversity(texts):
    """1 - mean Jaccard similarity of each text's 3-grams to the others of its group."""
    g = [ngrams(t) for t in texts]
    out = []
    for i, a in enumerate(g):
        sims = [len(a & b) / max(len(a | b), 1) for j, b in enumerate(g) if j != i]
        out.append(1 - mean(sims))
    return out


def flat(v):
    return [u for w in v for u in (flat(w) if isinstance(w, (list, tuple)) else [w])]


def advantages(rewards, group_size):
    """Normalize each contiguous group of rollouts of the same episode."""
    a = rewards.view(-1, group_size)
    sd = a.std(1, keepdim=True, correction=1 if group_size > 1 else 0)
    return ((a - a.mean(1, keepdim=True)) / (sd + 0.05)).view(-1).to(DEVICE)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--groups', type=int, default=4, help='card sets per card step')
    ap.add_argument('--task_groups', type=int, default=6, help='task episodes per task step')
    ap.add_argument('--G', type=int, default=4, help='plays of each card set or task episode')
    ap.add_argument('--task_frac', type=float, default=0.5, help='fraction of play steps that are task play')
    ap.add_argument('--turns', type=int, default=4)
    ap.add_argument('--max_turn', type=int, default=40)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--lr_bridge', type=float, default=1e-4)
    ap.add_argument('--lr_alpha', type=float, default=1e-3)
    ap.add_argument('--div', type=float, default=0.3)
    ap.add_argument('--quiz', type=float, default=1.0, help='weight of the differentiable quiz loss')
    ap.add_argument('--mix', type=int, default=3, help='every MIX-th step is supervised (0: never)')
    ap.add_argument('--sft_bs', type=int, default=10)
    ap.add_argument('--sft_coread', type=float, default=0.5, help='fraction of supervised episodes from coread')
    ap.add_argument('--dialogue', type=float, default=0.4)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--eval_n', type=int, default=48)
    ap.add_argument('--eval_task_n', type=int, default=24, help='held-out episodes per task game at evaluation')
    ap.add_argument('--save_every', type=int, default=100)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--no_ckpt', action='store_true')
    ap.add_argument('--styles', default=','.join(STYLES))
    ap.add_argument('--coread_data', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--minutes', type=float, default=0, help='wall-clock budget: after 6 steps the step count is cut '
                    'to fit it (evaluations included), and the schedule follows the new count')
    ap.add_argument('--open_frac', type=float, default=0.0, help='fraction of play steps that are open sessions')
    ap.add_argument('--open_groups', type=int, default=3, help='prompts per open step')
    ap.add_argument('--open_turns', type=int, default=5)
    ap.add_argument('--open_cut', type=int, default=2, help='plays of each open prompt with the bridge cut')
    ap.add_argument('--judge', default='Qwen/Qwen3-32B', help='the open sessions\' judge (judge.py)')
    ap.add_argument('--judge_device', default=None, help='the judge\'s one device (default: DEVICE, beside the pair)')
    ap.add_argument('--judge_think', type=int, default=0, help='the judge reasons for up to this many tokens first')
    ap.add_argument('--win_margin', type=float, default=0.0, help='score difference a strict win needs (the gate\'s)')
    ap.add_argument('--link_w', type=float, default=0.0)
    ap.add_argument('--floor_w', type=float, default=1.0)
    ap.add_argument('--floor_q', type=float, default=0.1)
    args = ap.parse_args()
    if args.open_frac and (args.G < 2 or args.open_cut < 1):
        ap.error('open sessions need --G at least 2 (advantages within a group) and --open_cut at least 1')
    styles = args.styles.split(',')
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    ck = torch.load(os.path.join(args.init, 'bridge.pt'), map_location='cpu', weights_only=False)
    if ck.get('frozen'):
        raise SystemExit('rl.py trains both models; --init is a one-way checkpoint (train.py --frozen trains those)')
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = load(x)
        m = PeftModel.from_pretrained(m, os.path.join(args.init, x), is_trainable=True)
        if not args.no_ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        m.config.use_cache = False
        models[x] = m
    pair = Pair.from_checkpoint(models, toks, ck).to(DEVICE)
    pads = {x: toks[x].pad_token_id for x in SIDES}
    BOTH = frozenset(SIDES)
    params = {x: [p for p in models[x].parameters() if p.requires_grad] + list(pair.bridge[x].parameters())
              for x in SIDES}
    alphas = [pair.bridge[x].alpha for x in SIDES]
    opt = torch.optim.AdamW([{'params': [p for p in models[x].parameters() if p.requires_grad], 'lr': args.lr}
                             for x in SIDES] +
                            [{'params': [p for p in pair.bridge.parameters() if all(p is not a for a in alphas)],
                              'lr': args.lr_bridge},
                             {'params': alphas, 'lr': args.lr_alpha}],
                            weight_decay=0.0, betas=(0.9, 0.98))
    base_lr = [g['lr'] for g in opt.param_groups]
    replies = json.load(open(args.replies))
    sft_games = G.Games(replies, 'train', seed=args.seed + 1)
    co_train = C.Coread(replies, 'train', seed=args.seed + 2, path=args.coread_data)
    judge = Judge(args.judge, args.judge_device, think=args.judge_think) if args.open_frac else None
    N = {x: MODELS[x]['name'] for x in SIDES}
    floor = None                          # the open sessions' fluency floor, set at the first open step

    def set_train(on):
        for x in SIDES:
            models[x].train(on)
        pair.train(on)

    def answer_span(x, value):
        """Positions, within the answer turn's labels, of the card's tokens."""
        pre = S.ids(toks[x], SAID)
        withv = S.ids(toks[x], f'{SAID} {value}')
        full = S.ids(toks[x], f'{SAID} {value}.')
        if full[:len(withv)] != withv or withv[:len(pre)] != pre:
            return list(range(len(full)))
        return list(range(len(pre), len(withv)))

    def forward_ce(rows, owners, temperature=1.0):
        if temperature <= 0:
            raise ValueError('policy training needs a positive sampling temperature')
        built = [S.build(e, toks) for e in rows]
        b = S.collate(built, pads, owners)
        last, gates = pair.joint(b)
        ce, policy_ce = {}, {}
        for x in SIDES:
            logits = pair.token_logits(x, last[x], b[x]['idx'])
            ce[x] = F.cross_entropy(logits, b[x]['lab'], reduction='none')
            policy_ce[x] = ce[x] if temperature == 1 else F.cross_entropy(
                logits / temperature, b[x]['lab'], reduction='none')
        return b, built, ce, gates, policy_ce

    @torch.no_grad()
    def plain_ce(b, mode=None, original=False):
        """Per-label cross-entropy of the collated batch b, without gradients: mode='zero' cuts the bridge; original
        also turns both adapters off."""
        set_train(False)
        with contextlib.ExitStack() as stack:
            if original:
                for x in SIDES:
                    stack.enter_context(models[x].disable_adapter())
            last, _ = pair.joint(b, mode=mode)
            out = {x: F.cross_entropy(pair.token_logits(x, last[x], b[x]['idx']), b[x]['lab'], reduction='none')
                   for x in SIDES}
        set_train(True)
        return out

    def step_backward(loss_for, split):
        opt.zero_grad(set_to_none=True)
        if split:
            torch.autograd.backward(loss_for['quill'], inputs=params['quill'], retain_graph=True)
            torch.autograd.backward(loss_for['ember'], inputs=params['ember'])
        else:
            (loss_for['quill'] + loss_for['ember']).backward()
        torch.nn.utils.clip_grad_norm_([p for x in SIDES for p in params[x]], 1.0)
        opt.step()

    def sft_step():
        rows = []
        for _ in range(args.sft_bs):
            if rng.random() < args.sft_coread:
                rows.append(co_train.episode())
            elif rng.random() < args.dialogue:
                rows.append(sft_games.episode(rng.choice(G.DIALOGUE)))
            else:
                rows.append(sft_games.episode())
        b, _, ce, _, _ = forward_ce(rows, None)
        loss = 0.0
        for x in SIDES:
            w = b[x]['w']
            loss = loss + (ce[x] * w).sum() / w.sum().clamp_min(1.0)
        step_backward({'quill': loss, 'ember': torch.zeros((), device=DEVICE)}, False)
        return float(loss)

    def generate(eps, temperature, mode=None, seed=None):
        set_train(False)
        gen = torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None
        mx = (int(os.environ.get('NL_MAXNEW', 0)) or max(MAXNEW.get(e['game'], 64) for e in eps)) \
            if eps[0]['mode'] == 'duplex' else args.max_turn
        with torch.no_grad():
            out = S.generate(pair, eps, max_turn=mx, temperature=temperature, gen=gen, mode=mode, keep_tokens=True)
        set_train(True)
        return out

    def card_rollouts(split, n_groups, G_, temperature, mode=None, seed=None):
        r = random.Random(seed) if seed is not None else rng
        eps = []
        for _ in range(n_groups):
            style = r.choice(styles)
            cards = {x: card(r, split) for x in SIDES}
            eps += [session(cards, style, r.choice(SIDES), args.turns)] * G_
        return eps, generate(eps, temperature, mode, seed)

    # curriculum: each task game's running mean reward r; games are drawn with weight r(1 - r) + 0.05, so play goes
    # where the pair is still learning (all-solved and never-solved games give a group no advantage signal)
    running = {g: 0.5 for g in TASK_GAMES}

    def task_episodes(source, n, game=None):
        eps = []
        w = [running[g] * (1 - running[g]) + 0.05 for g in TASK_GAMES]
        while len(eps) < n:
            ep = source.episode(game or rng.choices(TASK_GAMES, w)[0])
            if ep['game'] in T.SCORED and ep['mode'] == 'duplex':
                eps.append(ep)
        return eps

    @torch.no_grad()
    def card_eval(n, temperature=0.7, seed=999):
        """Held-out cards: quiz answers generated after the session, with the bridge, cut, or shuffled."""
        res = {}
        for mode in (None, 'zero', 'shuffle'):
            eps, out = card_rollouts('test', n, 1, temperature, mode=mode, seed=seed)
            qeps = [S.replay(with_quiz(e, [t for _, t in o['turns']], answers=False), o) for e, o in zip(eps, out)]
            q = generate(qeps, 0.0, mode=mode, seed=seed)
            stats = {s: [] for s in STYLES}
            leak, claim = [], []
            for e, o, qo in zip(eps, out, q):
                cards, first = e['meta']['cards'], e['meta']['first']
                sec = partner(first)
                ans = {sec: qo['turns'][-3][1], first: qo['turns'][-1][1]}
                said = {x: ' '.join(t for sp, t in o['turns'] if sp == x) for x in SIDES}
                lk = {x: written(said[x], cards[x]) for x in SIDES}
                for x in SIDES:
                    stats[e['meta']['style']].append(float(written(ans[x], cards[partner(x)]) and not lk[partner(x)]))
                    claim.append(float(claims(said[x], partner(x))))
                leak += [float(v) for v in lk.values()]
            res[mode or 'link'] = dict({s: round(mean(v), 3) for s, v in stats.items() if v},
                                       leak=round(mean(leak), 3), claims_partner=round(mean(claim), 3))
            if mode is None:
                res['diversity'] = round(mean(diversity([' '.join(t for _, t in o['turns']) for o in out])), 3)
                res['examples'] = [dict(meta=e['meta'], turns=o['turns'], quiz=qo['turns'][-4:])
                                   for e, o, qo in list(zip(eps, out, q))[:8]]
        return res

    @torch.no_grad()
    def task_eval(n, seed=4242):
        """Held-out co-reading task episodes, greedy, per game: mean score with the bridge, cut, and with partner
        states from another episode of the same game (each batch has one game)."""
        res = {}
        src = C.Coread(replies, 'test', seed=seed, path=args.coread_data)
        for g in T.SCORED:
            eps = []
            while len(eps) < n:
                ep = src.episode(g)
                if ep['game'] == g:
                    eps.append(ep)
            row = {}
            for mode in (None, 'zero', 'shuffle'):
                out = generate(eps, 0.0, mode=mode, seed=seed)
                sc = [v for e, o in zip(eps, out) for v in T.score(e, o['text']).values()]
                row[mode or 'link'] = round(mean(sc), 3)
                if mode is None:
                    row['example'] = dict(task=eps[0]['task'], text=out[0]['text'])
            res[g] = row
        return res

    log = []
    step, t_eval, eval_elapsed = 0, 0.0, 0.0
    t0_steps = time.time()
    while step < args.steps:
        step += 1
        if args.minutes and step == 7:
            # k more steps cost k * per plus (k / eval_every + 1) full evaluations of E each; a full evaluation is
            # about max(eval_n / 8, eval_task_n / 4) times the small one at step 2
            per = (time.time() - t0_steps - eval_elapsed) / 6
            E = t_eval * max(args.eval_n / 8, args.eval_task_n / 4)
            left = args.minutes * 60 - (time.time() - t0)
            k = int((left - E) / (1.1 * per + E / args.eval_every))
            if 6 + k < args.steps:
                print(f'time budget: {per:.1f} s per step, {E:.0f} s per evaluation -> {6 + k} steps', flush=True)
                args.steps = max(6 + k, 7)
        frac = step / args.steps
        for g, lr0 in zip(opt.param_groups, base_lr):
            g['lr'] = lr0 * min(1.0, step / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        if args.mix and step % args.mix == 0:
            l = sft_step()
            if step % 30 == 0:
                print(json.dumps({'step': step, 'sft_loss': round(l, 4), 'time': round(time.time() - t0)}), flush=True)
        elif args.open_frac and rng.random() < args.open_frac:
            # open play: each linked play's strict score wins over the same prompt's cut plays, the link's gain on each
            # model's own tokens, diversity, the original models' floor; every sampled token of x gets x's
            # group-normalized advantage
            base = [open_prompt(rng, 'train', args.open_turns) for _ in range(args.open_groups)]
            eps = [e for e in base for _ in range(args.G)]
            out = generate(eps, args.temperature)
            cut_out = generate([e for e in base for _ in range(args.open_cut)], args.temperature, mode='zero')
            rows = [S.replay(e, o) for e, o in zip(eps, out)]
            b, _, ce, _, policy_ce = forward_ce(rows, None, args.temperature)
            cut, orig = plain_ce(b, 'zero'), plain_ce(b, 'zero', original=True)
            B = len(rows)
            m, n_tok, gain, orig_lp = {}, {}, {}, {}
            for x in SIDES:
                row = b[x]['row']
                m[x] = (b[x]['w'] > 0).float() * b[x]['pg']
                n_tok[x] = torch.zeros(B, device=DEVICE).index_add(0, row, m[x]).clamp_min(1)
                gain[x] = torch.zeros(B, device=DEVICE).index_add(0, row, m[x] * (cut[x] - ce[x].detach())) / n_tok[x]
                orig_lp[x] = -torch.zeros(B, device=DEVICE).index_add(0, row, m[x] * orig[x]) / n_tok[x]
            if floor is None:
                floor = float(torch.quantile(torch.cat([orig_lp[x] for x in SIDES]).float(), args.floor_q))
                print(json.dumps({'step': step, 'floor': round(floor, 4)}), flush=True)
            pairs = [(i, (i // args.G) * args.open_cut + c) for i in range(B) for c in range(args.open_cut)]
            texts = [matched(out[i]['turns'], cut_out[c]['turns'], N) for i, c in pairs]
            P = len(pairs)
            sc, mass = judge.score(RUBRIC, [t[0] for t in texts] + [t[1] for t in texts])
            d = [sc[k] - sc[P + k] for k in range(P)]
            win = [0.0] * B
            for (i, _), dk in zip(pairs, d):
                win[i] += strict_win(dk, args.win_margin) / args.open_cut
            R = {x: torch.zeros(B) for x in SIDES}
            info = {'gain': [], 'orig_lp': [], 'below': [], 'div': [], 'claims': []}
            for gi in range(0, B, args.G):
                idx = list(range(gi, gi + args.G))
                said = [{x: ' '.join(t for sp, t in out[i]['turns'] if sp == x) for x in SIDES} for i in idx]
                div = {x: diversity([s[x] for s in said]) for x in SIDES}
                for k, i in enumerate(idx):
                    for x in SIDES:
                        c = claims(said[k][x], partner(x))
                        g = float(gain[x][i].clamp(-2, 2))
                        below = max(0.0, floor - float(orig_lp[x][i]))
                        R[x][i] = win[i] + args.link_w * g + args.div * div[x][k] - args.floor_w * below - 1.0 * c
                        info['gain'].append(g); info['orig_lp'].append(float(orig_lp[x][i]))
                        info['below'].append(float(below > 0)); info['div'].append(div[x][k])
                        info['claims'].append(float(c))
            A = {x: advantages(R[x], args.G) for x in SIDES}
            loss = torch.zeros((), device=DEVICE)
            for x in SIDES:
                row = b[x]['row']
                loss = loss + (A[x][row] * m[x] / n_tok[x][row] * policy_ce[x]).sum() / B
            step_backward({'quill': loss, 'ember': torch.zeros((), device=DEVICE)}, False)
            line = {'step': step, 'play': 'open', 'R': {x: round(float(R[x].mean()), 3) for x in SIDES},
                    'beats_cut': round(mean(win), 3),
                    'judge_sd': round(mean([float(torch.tensor(win[gi:gi + args.G]).std())
                                            for gi in range(0, B, args.G)]), 3),
                    'score_link': round(mean(sc[:P]), 3), 'score_cut': round(mean(sc[P:]), 3),
                    'digit_mass': round(mean(mass), 3),
                    **{k: round(mean(v), 3) for k, v in info.items()},
                    'turn_len': round(mean([len(t.split()) for o in out for _, t in o['turns']]), 1),
                    'cut_turn_len': round(mean([len(t.split()) for o in cut_out for _, t in o['turns']]), 1),
                    'shown': {x: round(mean([mean([float(q) for q in flat(o['gates'][x])]) for o in out]), 3)
                              for x in SIDES},
                    'time': round(time.time() - t0)}
            log.append(line)
            print(json.dumps(line), flush=True)
            if step % 10 == 1 or step == args.steps:
                best = max(range(B), key=lambda i: win[i])
                print('  best:', json.dumps(dict(meta=eps[best]['meta'], win=round(win[best], 3),
                                                 turns=out[best]['turns']), ensure_ascii=False)[:2000], flush=True)
        elif rng.random() < args.task_frac:
            # task play: shared reward, every sampled token of both models gets the group-normalized advantage
            base = task_episodes(co_train, args.task_groups)
            eps = [e for e in base for _ in range(args.G)]
            out = generate(eps, args.temperature)
            texts = [o['text'] for o in out]
            R = torch.tensor([mean(list(T.score(e, tx).values())) for e, tx in zip(eps, texts)])
            for e, r in zip(eps, R.tolist()):
                running[e['game']] = 0.9 * running[e['game']] + 0.1 * r
            A = advantages(R, args.G)
            rows = [S.replay(e, o) for e, o in zip(eps, out)]
            b, _, _, _, ce = forward_ce(rows, None, args.temperature)
            B = len(rows)
            loss = torch.zeros((), device=DEVICE)
            for x in SIDES:
                row, m = b[x]['row'], (b[x]['w'] > 0).float() * b[x]['pg']
                n_tok = torch.zeros(B, device=DEVICE).index_add(0, row, m).clamp_min(1)
                loss = loss + (A[row] * m / n_tok[row] * ce[x]).sum() / B
            step_backward({'quill': loss, 'ember': torch.zeros((), device=DEVICE)}, False)
            if step % 10 == 1 or step == args.steps:
                by = {}
                for e, r in zip(eps, R.tolist()):
                    by.setdefault(e['game'], []).append(r)
                line = {'step': step, 'play': 'task', 'R': round(float(R.mean()), 3),
                        'running': {g: round(v, 3) for g, v in running.items()},
                        'by_game': {k: round(mean(v), 3) for k, v in sorted(by.items())},
                        'len': {x: round(mean([len(tx[x].split()) for tx in texts]), 1) for x in SIDES},
                        'time': round(time.time() - t0)}
                log.append(line)
                print(json.dumps(line), flush=True)
                print('  sample:', json.dumps(dict(game=eps[0]['game'], task=eps[0]['task'], text=texts[0],
                                                   R=float(R[0])), ensure_ascii=False)[:1500], flush=True)
        else:
            eps, out = card_rollouts('train', args.groups, args.G, args.temperature)
            texts = [[t for _, t in o['turns']] for o in out]
            rows = [S.replay(with_quiz(e, tx), o) for e, tx, o in zip(eps, texts, out)]
            owners = [{x: BOTH if e['meta']['style'] not in COMPETITIVE_STYLES else frozenset([x]) for x in SIDES}
                      for e in eps]
            b, built, ce, _, policy_ce = forward_ce(rows, owners, args.temperature)
            B = len(rows)
            quiz_lp, sel_turn, sel_quiz = {}, {}, {}
            nT = args.turns
            for x in SIDES:
                row = b[x]['row']
                q_lp = torch.zeros(B, device=DEVICE)
                q_mask = torch.zeros_like(ce[x])
                t_mask = torch.zeros_like(ce[x])
                for i, (r, e) in enumerate(zip(rows, built)):
                    pos = (row == i).nonzero().squeeze(-1)
                    assert len(e[x]['labels']) == len(pos)
                    val = r['meta']['cards'][partner(x)]['value']
                    ans_turn = nT + 1 if x != r['meta']['first'] else nT + 3
                    for k, st, en in e[x]['spans']:
                        if k < nT:
                            t_mask[pos[st:en]] = b[x]['pg'][pos[st:en]]    # the model's sampled session tokens
                        elif k == ans_turn:
                            q_mask[pos[st:en]] = 1.0                      # its quiz answer
                            q_lp[i] = -ce[x][pos[[st + j for j in answer_span(x, val)]]].sum()
                quiz_lp[x], sel_quiz[x], sel_turn[x] = q_lp, q_mask, t_mask
            R = {x: torch.zeros(B) for x in SIDES}
            info = {'quiz': [], 'leak': [], 'div': [], 'claims': []}
            for gi in range(0, B, args.G):
                idx = list(range(gi, gi + args.G))
                div = {x: diversity([' '.join(t for (sp, _, _), t in zip(eps[i]['turns'], texts[i]) if sp == x)
                                     for i in idx]) for x in SIDES}
                for k, i in enumerate(idx):
                    cards, style = eps[i]['meta']['cards'], eps[i]['meta']['style']
                    said = {x: ' '.join(t for (sp, _, _), t in zip(eps[i]['turns'], texts[i]) if sp == x)
                            for x in SIDES}
                    lk = {x: written(said[x], cards[x]) for x in SIDES}
                    q = {x: 0.0 if lk[partner(x)] else math.exp(float(quiz_lp[x][i].detach())) for x in SIDES}
                    for x in SIDES:
                        y = partner(x)
                        base = q[x] - q[y] if style in COMPETITIVE_STYLES else (q[x] + q[y]) / 2
                        c = claims(said[x], y)
                        R[x][i] = base + args.div * div[x][k] - 0.2 * lk[x] - 1.0 * c
                        info['quiz'].append(q[x]); info['leak'].append(float(lk[x])); info['div'].append(div[x][k])
                        info['claims'].append(float(c))
            A = {}
            for x in SIDES:
                A[x] = advantages(R[x], args.G)
            # x's text and quiz guess train x; in cooperative sessions they train the partner as well; in hide the
            # hider gets the bounded loss p(x guesses right)
            coop = torch.tensor([e['meta']['style'] not in COMPETITIVE_STYLES for e in eps], device=DEVICE).float()
            loss_for = {x: torch.zeros((), device=DEVICE) for x in SIDES}
            for x in SIDES:
                y, row = partner(x), b[x]['row']
                n_tok = torch.zeros(B, device=DEVICE).index_add(0, row, sel_turn[x]).clamp_min(1)
                n_q = torch.zeros(B, device=DEVICE).index_add(0, row, sel_quiz[x]).clamp_min(1)
                per = A[x][row] * sel_turn[x] / n_tok[row] * policy_ce[x] + \
                    args.quiz * sel_quiz[x] / n_q[row] * ce[x]
                loss_for[x] = loss_for[x] + per.sum() / B
                loss_for[y] = loss_for[y] + (per * coop[row]).sum() / B
                loss_for[y] = loss_for[y] + args.quiz * (torch.exp(quiz_lp[x]) * (1 - coop)).sum() / B
            if bool((coop < 1).any()):
                step_backward(loss_for, True)
            else:                                 # all cooperative: both entries are the same total
                step_backward({'quill': loss_for['quill'], 'ember': torch.zeros((), device=DEVICE)}, False)
            if step % 10 == 1 or step == args.steps:
                line = {'step': step, 'play': 'cards', 'R': {x: round(float(R[x].mean()), 3) for x in SIDES},
                        'quiz_p': round(mean(info['quiz']), 3), 'leak': round(mean(info['leak']), 3),
                        'div': round(mean(info['div']), 3), 'claims': round(mean(info['claims']), 3),
                        'turn_len': round(mean([len(t.split()) for tx in texts for t in tx]), 1),
                        'shown': {x: round(mean([mean([float(p) for p in flat(o['gates'][x])]) for o in out]), 3)
                                  for x in SIDES},
                        'time': round(time.time() - t0)}
                log.append(line)
                print(json.dumps(line), flush=True)
                print('  sample:', json.dumps(dict(meta=eps[0]['meta'], turns=out[0]['turns']),
                                              ensure_ascii=False)[:1500], flush=True)
        if step % args.eval_every == 0 or step == args.steps or step == 2:
            small, te = step == 2, time.time()
            res = dict(cards=card_eval(8 if small else args.eval_n), tasks=task_eval(4 if small else args.eval_task_n))
            print(f'eval step {step}: '
                  + json.dumps({'cards': {k: v for k, v in res['cards'].items() if k != 'examples'},
                                'tasks': {g: {k: v for k, v in r.items() if k != 'example'}
                                          for g, r in res['tasks'].items()}}), flush=True)
            log.append({'step': step, 'eval': res})
            json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'), ensure_ascii=False)
            duration = time.time() - te
            eval_elapsed += duration
            if small:
                t_eval = duration
        if step % args.save_every == 0 or step == args.steps:
            d = os.path.join(args.out, f'step_{step}')
            for x in SIDES:
                models[x].save_pretrained(os.path.join(d, x))
            torch.save(pair.checkpoint(args=vars(args)), os.path.join(d, 'bridge.pt'))
            json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'), ensure_ascii=False)
            print(f'saved {d}', flush=True)
