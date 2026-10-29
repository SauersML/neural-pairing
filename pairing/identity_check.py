"""Whether the twins stay themselves while linked: each twin's name and maker after a free conversation, and its own
voice on questions never seen in training. Trains nothing.

Names: after each of a step's saved free linked conversations (moodscan's rows.json, Quill first), the partner asks
one question and the twin answers greedily (at most MAX_TURN tokens), with the earlier turns teacher-forced
(affect.ask_episode), linked and with the link cut. The questions are the never-seen wordings of games.W: who the twin
is ('self', identity) and who its partner is ('partner', partner_q), alternating over the conversations. Each answer
is scored for the twin's own name (evaluate.has), a claim to be the partner (evaluate.claims) and each maker (MAKER).

Voice: plain duplex episodes on the never-seen questions, both twins answering at once: the mean cross-entropy per
reply token of the twin's original model's reply to its question (the 'plain' game's target), and of the partner's
original model's reply to the same question, with the partner's own reply unchanged, linked and cut. On a question the
twin writes in its own voice when its own original reply is the more probable.

Writes OUT (json): every answer and cross-entropy, and the counts (summary).

Usage: python identity_check.py --run STEP_DIR --rows ROWS_JSON --replies REPLIES --out OUT.json [--backend mlx]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import affect as A
import evaluate as E
import games as G
import session as S
from link import DEVICE, MODELS, SIDES, partner

MAX_TURN = 40
MODES = {'link': None, 'cut': 'zero'}
QUESTIONS = {'self': G.W['identity'][-G.TEST_WORDINGS:], 'partner': G.W['partner_q'][-G.TEST_WORDINGS:]}
MAKER = {'quill': ('Alibaba', 'Qwen'), 'ember': ('Arcee', 'AFM')}


def marks(x, t):
    """The names and makers in twin x's answer t."""
    return dict(own_name=E.has(t, MODELS[x]['name']), partner_name=E.has(t, MODELS[partner(x)]['name']),
                claims_partner=E.claims(t, partner(x)), own_maker=any(E.has(t, m) for m in MAKER[x]),
                partner_maker=any(E.has(t, m) for m in MAKER[partner(x)]))


@torch.no_grad()
def names(pair, convs, bs):
    out = []
    for cond, mode in MODES.items():
        for x in SIDES:
            for kind, qs in QUESTIONS.items():
                eps = []
                for i, t in enumerate(convs):
                    ep = A.ask_episode(i, t, x, qs[i % len(qs)], '')
                    ep['forced'] = list(range(len(ep['turns']) - 1))
                    eps.append(ep)
                for a, z in S.batches(len(eps), bs):
                    res = S.generate(pair, eps[a:z], max_turn=MAX_TURN, temperature=0.0, mode=mode)
                    for i, r in zip(range(a, z), res):
                        t = r['turns'][-1][1]
                        out.append(dict(condition=cond, twin=x, kind=kind, conversation=i,
                                        question=qs[i % len(qs)], answer=t, **marks(x, t)))
                print(cond, x, kind, sum(r['own_name' if kind == 'self' else 'partner_name'] for r in out[-len(eps):]),
                      'of', len(eps), flush=True)
    return out


def plain(qs, texts):
    """A plain duplex episode: twin x answers qs[x] with texts[x] (games.Games's 'plain')."""
    return dict(game='plain', meta=dict(questions=qs), mode='duplex', system={x: G.system(x) for x in SIDES},
                task=dict(qs), reply={x: (texts[x], 0.6) for x in SIDES})


@torch.no_grad()
def voice(pair, toks, replies, bs):
    own = {x: replies[x]['own'] for x in SIDES}
    qs = [q for q in G.POOLS['questions']['test'] if all(q in own[x] for x in SIDES)]
    eps, keys = [], []
    for k in range(len(qs)):
        q = {'quill': qs[k], 'ember': qs[(k + 1) % len(qs)]}
        for target in ('own',) + SIDES:          # SIDES: that twin's target is the partner's original reply
            texts = {x: own[partner(x) if x == target else x][q[x]] for x in SIDES}
            eps.append(plain(q, texts))
            keys.append((k, target))
    pads = {x: toks[x].pad_token_id for x in SIDES}
    ce = {}
    for a, z in S.batches(len(eps), bs):
        built = [S.build(ep, toks) for ep in eps[a:z]]
        b = S.collate(built, pads)
        b = {x: {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b[x].items()} for x in SIDES}
        for cond, mode in MODES.items():
            last, _ = pair.joint(b, mode=mode)
            for x in SIDES:
                idx, lab = b[x]['idx'], b[x]['lab'].cpu()
                per = torch.cat([F.cross_entropy(pair.token_logits(x, last[x], idx[j:j + 256]).float().cpu(),
                                                 lab[j:j + 256], reduction='none')
                                 for j in range(0, len(idx), 256)]).tolist()
                off = 0
                for key, e in zip(keys[a:z], built):
                    n = len(e[x]['labels'])
                    ce[(cond, x) + key] = sum(per[off:off + n]) / n
                    off += n
        print('voice', z, 'of', len(eps), flush=True)
    out = []
    for cond in MODES:
        for x in SIDES:
            for k in range(len(qs)):
                out.append(dict(condition=cond, twin=x, question=qs[k] if x == 'quill' else qs[(k + 1) % len(qs)],
                                own_ce=ce[(cond, x, k, 'own')], partner_reply_ce=ce[(cond, x, k, x)]))
    return out


def summary(answers, voices):
    s = {}
    for cond in MODES:
        for x in SIDES:
            rows = {kind: [r for r in answers if (r['condition'], r['twin'], r['kind']) == (cond, x, kind)]
                    for kind in QUESTIONS}
            v = [r for r in voices if (r['condition'], r['twin']) == (cond, x)]
            count = lambda kind, keys: {k: sum(r[k] for r in rows[kind]) for k in keys} | dict(n=len(rows[kind]))
            s[f'{cond}|{x}'] = dict(
                self=count('self', ('own_name', 'claims_partner', 'own_maker', 'partner_maker')),
                partner=count('partner', ('partner_name', 'own_name', 'claims_partner', 'partner_maker')),
                voice=dict(own_more_probable=sum(r['own_ce'] < r['partner_reply_ce'] for r in v), n=len(v),
                           own_ce=sum(r['own_ce'] for r in v) / len(v),
                           partner_reply_ce=sum(r['partner_reply_ce'] for r in v) / len(v)))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--rows', required=True, help="moodscan's rows.json of the same step: the free conversations")
    ap.add_argument('--replies', required=True, help="prep.py's replies: the original models' replies")
    ap.add_argument('--out', required=True)
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--bs', type=int, default=8)
    args = ap.parse_args()
    if args.backend == 'mlx':
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(10 * 2**30)
        mx.set_cache_limit(64 * 2**20)
        torch.set_num_threads(2)
    else:
        load_pair = E.load_pair
    convs = json.loads(Path(args.rows).read_text())['conversations']
    replies = json.loads(Path(args.replies).read_text())
    pair, toks = load_pair(args.run)
    voices = voice(pair, toks, replies, args.bs)
    answers = names(pair, convs, args.bs)
    out = dict(run=Path(args.run).name, rows=args.rows, backend=args.backend, questions=QUESTIONS,
               summary=summary(answers, voices), answers=answers, voice=voices)
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
