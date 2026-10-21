"""Self-directed play: the pair talks freely and learns from what each model can tell, afterwards, about the other.

Each session gives each model a private card its partner cannot see (a word, a 4-digit number or a concept) and a
play style:
  free   "Play with your partner however you like."
  teach  "Make sure your partner ends up knowing your card, without writing it."
  hide   (competitive) "Find out your partner's card while keeping yours hidden."
The pair talks for TURNS turns, sampling at temperature 1, with the link open. Then the link closes and each model is
quizzed on its partner's card: the quiz answers are teacher-forced, so the score is the model's probability of the
right card, and no judge is needed. A card that appears in its owner's text scores nothing (it crossed in text, not
over the link).

Rewards per model and session:
  free, teach: the mean of both quiz probabilities, plus DIV x diversity;
  hide: own quiz probability minus the partner's (zero-sum), plus DIV x diversity;
  diversity: 1 - the mean 3-gram Jaccard similarity of the model's text to the other sessions of its group;
  -0.2 when a model writes its own card, -1 when it introduces itself as its partner.
Each card set and style is played G times (a group); advantages are normalized within the group. The loss is
A x (-log p) on each model's sampled tokens (GRPO with one update per batch, so no ratio clipping), plus the quiz
cross-entropy itself, which is differentiable through the link and so trains the send heads and receive adapters
directly. In hide, each quiz loss trains only the guesser, and the hider's parameters get the bounded loss p(right).
Every MIX-th step is a supervised step on the scripted games, so the game skills and the text stay.

TRL's GRPO trainer samples one model with its own generate loop. Here two models decode in lockstep and exchange
activations at every token, so rollouts use session.generate and the loss is computed here.

Usage: python rl.py --init RUN/step_N --replies OUT/replies.json --out OUT/rl [--steps 240]
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

import games as G
import session as S
from evaluate import claims, has, mentions, mean
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


def with_quiz(ep, texts, weights=None, answers=True):
    """The session with its generated texts and the quiz after it, the link closed: the first speaker asks about its
    card, the second answers and asks back, the first answers. answers=False leaves the answers to generate."""
    cards, first = ep['meta']['cards'], ep['meta']['first']
    second = partner(first)
    T = len(ep['turns'])
    turns = [(sp, t, 0.0) for (sp, _, _), t in zip(ep['turns'], texts)]
    turns += [(first, ASK, 0.0), (second, f"{SAID} {cards[first]['value']}." if answers else '', 1.0),
              (second, ASK2, 0.0), (first, f"{SAID} {cards[second]['value']}." if answers else '', 1.0)]
    return dict(ep, turns=turns, link_off_after=T, forced=(T, T + 2))


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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=240)
    ap.add_argument('--groups', type=int, default=4, help='card sets per step')
    ap.add_argument('--G', type=int, default=4, help='sessions per card set')
    ap.add_argument('--turns', type=int, default=4)
    ap.add_argument('--max_turn', type=int, default=40)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--lr_link', type=float, default=1e-4)
    ap.add_argument('--div', type=float, default=0.3)
    ap.add_argument('--quiz', type=float, default=1.0, help='weight of the differentiable quiz loss')
    ap.add_argument('--mix', type=int, default=3, help='every MIX-th step is supervised (0: never)')
    ap.add_argument('--sft_bs', type=int, default=10)
    ap.add_argument('--dialogue', type=float, default=0.6)
    ap.add_argument('--eval_every', type=int, default=80)
    ap.add_argument('--eval_n', type=int, default=48)
    ap.add_argument('--save_every', type=int, default=120)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--no_ckpt', action='store_true')
    ap.add_argument('--styles', default=','.join(STYLES))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    styles = args.styles.split(',')
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    ck = torch.load(os.path.join(args.init, 'link.pt'), weights_only=False)
    toks, models = {}, {}
    for x in SIDES:
        toks[x], m = load(x)
        cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        m = get_peft_model(m, cfg)
        set_peft_model_state_dict(m, load_file(os.path.join(args.init, x, 'adapter_model.safetensors')))
        if not args.no_ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        m.config.use_cache = False
        models[x] = m
    pair = Pair(models, toks, ck['lr'], ck['ls'], ck['sizes'], soft=ck.get('soft', False)).to(DEVICE)
    pair.side.load_state_dict(ck['link'])
    pair.inbox = ck.get('inbox', False)
    pads = {x: toks[x].pad_token_id for x in SIDES}
    BOTH = frozenset(SIDES)
    params = {x: [p for p in models[x].parameters() if p.requires_grad] + list(pair.side[x].parameters()) for x in SIDES}
    opt = torch.optim.AdamW([{'params': [p for p in models[x].parameters() if p.requires_grad], 'lr': args.lr}
                             for x in SIDES] + [{'params': list(pair.side.parameters()), 'lr': args.lr_link}],
                            weight_decay=0.0, betas=(0.9, 0.98))
    base_lr = [g['lr'] for g in opt.param_groups]
    replies = json.load(open(args.replies))
    sft_games = G.Games(replies, 'train', seed=args.seed + 1)

    def answer_span(x, value):
        """Positions, within the answer turn's labels, of the card's tokens."""
        pre = S.ids(toks[x], SAID)
        withv = S.ids(toks[x], f'{SAID} {value}')
        full = S.ids(toks[x], f'{SAID} {value}.')
        if full[:len(withv)] != withv or withv[:len(pre)] != pre:
            return list(range(len(full)))
        return list(range(len(pre), len(withv)))

    def forward_ce(rows, owners):
        b = S.collate([S.build(e, toks) for e in rows], pads, owners, soft=pair.soft)
        last, msgs = pair.coupled(b, rounds=3 if random.random() < 0.3 else 2)
        ce = {}
        for x in SIDES:
            logits = pair.token_logits(x, last[x], b[x]['idx'])
            ce[x] = F.cross_entropy(logits, b[x]['lab'], reduction='none')
        return b, ce, msgs

    def step_backward(loss_for, split):
        opt.zero_grad(set_to_none=True)
        if split:
            torch.autograd.backward(loss_for['quill'], inputs=params['quill'], retain_graph=True)
            torch.autograd.backward(loss_for['ember'], inputs=params['ember'])
        else:
            (loss_for['quill'] + loss_for['ember']).backward()
        torch.nn.utils.clip_grad_norm_([p for x in SIDES for p in params[x]], 1.0)
        opt.step()
        pair.clear()

    def sft_step():
        rows = []
        for _ in range(args.sft_bs):
            ep = sft_games.episode(rng.choice(G.DIALOGUE)) if rng.random() < args.dialogue else sft_games.episode()
            rows.append(ep)
        b, ce, msgs = forward_ce(rows, None)
        loss = 0.0
        for x in SIDES:
            w = b[x]['w']
            loss = loss + (ce[x] * w).sum() / w.sum().clamp_min(1.0)
        step_backward({'quill': loss, 'ember': torch.zeros((), device=DEVICE)}, False)
        return float(loss)

    def rollouts(split, n_groups, G_, temperature, mode=None, seed=None):
        r = random.Random(seed) if seed is not None else rng
        eps = []
        for _ in range(n_groups):
            style = r.choice(styles)
            cards = {x: card(r, split) for x in SIDES}
            first = r.choice(SIDES)
            ep = session(cards, style, first, args.turns)
            eps += [ep] * G_
        for x in SIDES:
            models[x].eval()
        with torch.no_grad():
            out = S.generate(pair, eps, max_turn=args.max_turn, temperature=temperature, gen=None, mode=mode)
        for x in SIDES:
            models[x].train()
        return eps, out

    @torch.no_grad()
    def free_play_eval(n, temperature=0.7, seed=999):
        """Held-out cards: quiz answers generated after the session, with the link, cut, or shuffled (all sessions)."""
        res = {}
        for x in SIDES:
            models[x].eval()
        for mode in (None, 'zero', 'shuffle'):
            eps, out = rollouts('test', n, 1, temperature, mode=mode, seed=seed)
            qeps = [with_quiz(e, [t for _, t in o['turns']], answers=False) for e, o in zip(eps, out)]
            torch.manual_seed(seed)
            q = S.generate(pair, qeps, max_turn=16, temperature=0.0, mode=mode)
            stats = {s: [] for s in STYLES}
            leak, claim = [], []
            for e, o, qo in zip(eps, out, q):
                cards, first = e['meta']['cards'], e['meta']['first']
                sec = partner(first)
                ans = {sec: qo['turns'][-3][1], first: qo['turns'][-1][1]}   # each guesser's answer about the other
                said = {x: ' '.join(t for sp, t in o['turns'] if sp == x) for x in SIDES}
                lk = {x: written(said[x], cards[x]) for x in SIDES}
                for x in SIDES:
                    stats[e['meta']['style']].append(float(written(ans[x], cards[partner(x)]) and not lk[partner(x)]))
                    claim.append(float(claims(said[x], partner(x))))
                leak += [float(v) for v in lk.values()]
            res[mode or 'link'] = dict({s: round(mean(v), 3) for s, v in stats.items()},
                                       leak=round(mean(leak), 3), claims_partner=round(mean(claim), 3))
            if mode is None:
                texts = [' '.join(t for _, t in o['turns']) for o in out]
                res['diversity'] = round(mean(diversity(texts)), 3)
                res['examples'] = [dict(meta=e['meta'], turns=o['turns'], quiz=qo['turns'][-4:], gates=o['gates'],
                                        gate_turn=o['gate_turn']) for e, o, qo in list(zip(eps, out, q))[:12]]
        for x in SIDES:
            models[x].train()
        return res

    log = []
    for step in range(1, args.steps + 1):
        frac = step / args.steps
        for g, lr0 in zip(opt.param_groups, base_lr):
            g['lr'] = lr0 * min(1.0, step / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        if args.mix and step % args.mix == 0:
            l = sft_step()
            if step % 30 == 0:
                print(json.dumps({'step': step, 'sft_loss': round(l, 4), 'time': round(time.time() - t0)}), flush=True)
            continue
        eps, out = rollouts('train', args.groups, args.G, args.temperature)
        texts = [[t for _, t in o['turns']] for o in out]
        # rewards need the quiz probabilities: one forward pass gives them and the losses
        rows = [with_quiz(e, tx) for e, tx in zip(eps, texts)]
        owners = [{x: BOTH if e['meta']['style'] not in COMPETITIVE_STYLES else frozenset([x]) for x in SIDES}
                  for e in eps]
        b, ce, msgs = forward_ce(rows, owners)
        B = len(rows)
        quiz_lp, sel_turn, sel_quiz = {}, {}, {}
        built = [S.build(r, toks) for r in rows]
        T = args.turns
        for x in SIDES:
            row = b[x]['row']
            q_lp = torch.zeros(B, device=DEVICE)
            q_mask = torch.zeros_like(ce[x])
            t_mask = torch.zeros_like(ce[x])
            for i, (r, e) in enumerate(zip(rows, built)):
                pos = (row == i).nonzero().squeeze(-1)
                assert len(e[x]['labels']) == len(pos)
                val = r['meta']['cards'][partner(x)]['value']
                ans_turn = T + 1 if x != r['meta']['first'] else T + 3
                for k, st, en in e[x]['spans']:
                    if k < T:
                        t_mask[pos[st:en]] = 1.0                      # the model's sampled session tokens
                    elif k == ans_turn:
                        q_mask[pos[st:en]] = 1.0                      # its quiz answer
                        q_lp[i] = -ce[x][pos[[st + j for j in answer_span(x, val)]]].sum()
            quiz_lp[x], sel_quiz[x], sel_turn[x] = q_lp, q_mask, t_mask
        # rewards
        R = {x: torch.zeros(B) for x in SIDES}
        info = {'quiz': [], 'leak': [], 'div': [], 'claims': []}
        for gi in range(0, B, args.G):
            idx = list(range(gi, gi + args.G))
            div = {x: diversity([' '.join(t for (sp, _, _), t in zip(eps[i]['turns'], texts[i]) if sp == x)
                                 for i in idx]) for x in SIDES}
            for k, i in enumerate(idx):
                cards, style = eps[i]['meta']['cards'], eps[i]['meta']['style']
                said = {x: ' '.join(t for (sp, _, _), t in zip(eps[i]['turns'], texts[i]) if sp == x) for x in SIDES}
                lk = {x: written(said[x], cards[x]) for x in SIDES}
                q = {x: 0.0 if lk[partner(x)] else math.exp(float(quiz_lp[x][i])) for x in SIDES}
                for x in SIDES:
                    y = partner(x)
                    base = q[x] - q[y] if style in COMPETITIVE_STYLES else (q[x] + q[y]) / 2
                    c = claims(said[x], y)
                    R[x][i] = base + args.div * div[x][k] - 0.2 * lk[x] - 1.0 * c
                    info['quiz'].append(q[x]); info['leak'].append(float(lk[x])); info['div'].append(div[x][k])
                    info['claims'].append(float(c))
        A = {}
        for x in SIDES:
            a = R[x].view(-1, args.G)
            A[x] = ((a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 0.05)).view(-1).to(DEVICE)
        # x's text and quiz guess train x; in cooperative sessions they train the partner as well; in hide the hider
        # gets the bounded loss p(x guesses right)
        coop = torch.tensor([e['meta']['style'] not in COMPETITIVE_STYLES for e in eps], device=DEVICE).float()
        loss_for = {x: torch.zeros((), device=DEVICE) for x in SIDES}
        for x in SIDES:
            y, row = partner(x), b[x]['row']
            n_tok = torch.zeros(B, device=DEVICE).index_add(0, row, sel_turn[x]).clamp_min(1)
            n_q = torch.zeros(B, device=DEVICE).index_add(0, row, sel_quiz[x]).clamp_min(1)
            per = (A[x][row] * sel_turn[x] / n_tok[row] + args.quiz * sel_quiz[x] / n_q[row]) * ce[x]
            loss_for[x] = loss_for[x] + per.sum() / B
            loss_for[y] = loss_for[y] + (per * coop[row]).sum() / B
            loss_for[y] = loss_for[y] + args.quiz * (torch.exp(quiz_lp[x]) * (1 - coop)).sum() / B
        if bool((coop < 1).any()):
            step_backward(loss_for, True)
        else:                                     # all cooperative: both entries hold the same total
            step_backward({'quill': loss_for['quill'], 'ember': torch.zeros((), device=DEVICE)}, False)
        if step % 10 == 1 or step == args.steps:
            line = {'step': step, 'R': {x: round(float(R[x].mean()), 3) for x in SIDES},
                    'quiz_p': round(mean(info['quiz']), 3), 'leak': round(mean(info['leak']), 3),
                    'div': round(mean(info['div']), 3), 'claims': round(mean(info['claims']), 3),
                    'turn_len': round(mean([len(t.split()) for tx in texts for t in tx]), 1),
                    'sent': {x: round(mean([mean([p > 0.5 for st in o['gates'][x] for p in st]) for o in out]), 3)
                             for x in SIDES},
                    'time': round(time.time() - t0)}
            log.append(line)
            print(json.dumps(line), flush=True)
            print('  sample:', json.dumps(dict(meta=eps[0]['meta'], turns=out[0]['turns']), ensure_ascii=False)[:1500],
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps or step == 2:
            res = free_play_eval(args.eval_n if step > 2 else 8)
            print(f'eval step {step}: ' + json.dumps({k: v for k, v in res.items() if k != 'examples'}), flush=True)
            log.append({'step': step, 'eval': res})
            json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'), ensure_ascii=False)
        if step % args.save_every == 0 or step == args.steps:
            d = os.path.join(args.out, f'step_{step}')
            for x in SIDES:
                models[x].save_pretrained(os.path.join(d, x))
            torch.save({'link': pair.side.state_dict(), 'lr': ck['lr'], 'ls': ck['ls'], 'sizes': ck['sizes'], 'inbox': pair.inbox, 'soft': pair.soft,
                        'args': vars(args)}, os.path.join(d, 'link.pt'))
            json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'), ensure_ascii=False)
            print(f'saved {d}', flush=True)
