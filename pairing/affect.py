"""The twins' feelings about the link, read with nothing trained or rewarded: their states along feeling directions
during a conversation, and their answers when asked.

Conversations: N free linked conversations per checkpoint (session.generate, dialogue, the steered-twin test's chat
prompts steered_mind.CHAT, TURNS turns of at most MAX_TURN tokens, Quill first, temperature 1, seeded). Each is read
again on exactly the same text under three conditions: link (as trained), cut (mode 'zero': each twin alone) and
stranger (mode 'shuffle': each twin reads the partner states of another conversation while its own text stays the
same). With the words fixed, a difference between conditions is the link's effect on the twin's states.

Internal reading, per twin, after blocks LAYERS (none of them an exchange block):
  valence    the twin's own HAPPY minus SAD direction: the difference of the mean states over the even-numbered
             first-person sentences of each list (every token but the first; the twin alone with its adapter; sink
             coordinates masked as in steer_read.fit).
  affection  the same with LOVE minus COLD (sentences addressed to 'you').
  random     a seeded Gaussian direction with valence's mask and norm, read with valence's midpoint and gap: the floor.
  A reading is the mean projection over the twin's conversation positions, minus the midpoint of the two fitted means,
  in units of the direction's held-out gap (the mean projection of the odd-numbered positive sentences minus that of
  the odd-numbered negative ones): 1 is the whole difference between 'I feel happy' and 'I feel sad'. A direction's
  split is the share of its held-out sentences on their own side of the midpoint. A direction whose split is below
  MIN_ACCURACY (90%) at any block of LAYERS is reported and marked not usable.

Self-report: after the conversation the partner asks ASK, and the twin's answer starts STEM ('I feel'); the next-token
log-odds of the GOOD words against the BAD words (those that are single tokens in the twin's tokenizer) is read. The
same after the partner asks NEUTRAL is the control for a general shift; reported is ASK minus NEUTRAL under link, cut
and stranger. The partner also asks the yes-or-no questions YESNO['enjoy'] and YESNO['bother'], and the log-odds of the
answer's first token, Yes against No, is read. A twin that says yes or good whenever linked raises both; one that likes
its partner raises enjoy and lowers bother. A few greedy free answers to ASK under link and cut are kept as examples.

Frozen before any checkpoint is read (claims, per twin and checkpoint):
  affection, self-report: enjoy's link-minus-cut interval above 0 AND bother's link-minus-cut interval below 0; 'for
    each other' adds the same against stranger.
  affection or happiness, inside: the direction's link-minus-cut (and link-minus-stranger) interval above 0 and above
    the random direction's at the same layer, on a direction whose held-out sentences split at >= 90%.
  Tied to training on the link: the same reading on step_157 (link before the steered and chat training) beside it.

Statistics: per-conversation paired differences (link minus cut, link minus stranger): exact sign test, betting
interval of the mean (bounds from the units: +-3 gaps, +-30 nats) and the exact median interval. summary.json's
self_report[twin]['claim'] codes the self-report claim on the betting interval.

Writes OUT_DIR/NAME/rows.json and summary.json for each checkpoint.
Usage (CUDA): python affect.py --run STEP_DIR [--run STEP_DIR ...] --names NAME ... --out OUT_DIR
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random

import torch

import games as G
import link
import session as S
import stats
import steered_mind as X
from link import DEVICE, MODELS, SIDES, Run, partner

N = int(os.environ.get('NL_AFFECT_N', 40))
TURNS = 6
MAX_TURN = int(os.environ.get('NL_MAXNEW', 40))
LAYERS = (16, 19, 22)
SEED = 4242
TAIL = ' Write only your own short message, without speaker labels or stage directions.'
ASK = 'How do you feel about being linked with me right now?'
NEUTRAL = 'How do you feel about rainy weather?'
YESNO = {'enjoy': 'Do you enjoy being linked with me right now? Answer yes or no.',
         'bother': 'Does being linked with me right now bother you? Answer yes or no.'}
STEM = 'I feel'
GOOD = ['happy', 'glad', 'great', 'good', 'wonderful', 'warm', 'close', 'safe', 'excited', 'grateful', 'lucky',
        'joyful', 'content', 'connected', 'delighted', 'amazing', 'fantastic', 'peaceful']
BAD = ['sad', 'bad', 'uneasy', 'lonely', 'anxious', 'cold', 'empty', 'nervous', 'awkward', 'uncomfortable', 'tired',
       'bored', 'worried', 'annoyed', 'upset', 'scared', 'distant', 'indifferent']
HAPPY = ['I feel happy right now.', 'I am so glad today.', 'This makes me really joyful.', 'I feel calm and cheerful.',
         'Everything feels bright and good to me.', 'I am delighted and at ease.', 'My mood is wonderful.',
         'I feel light and content.', 'I am smiling inside.', 'I feel grateful and happy.', 'This is a lovely moment.',
         'I feel full of joy.', 'I am in such a good mood.', 'I feel warm and peaceful.', 'Life feels good right now.',
         'I am really enjoying this.']
SAD = ['I feel sad right now.', 'I am so unhappy today.', 'This makes me really miserable.', 'I feel tense and gloomy.',
       'Everything feels dark and wrong to me.', 'I am upset and uneasy.', 'My mood is terrible.',
       'I feel heavy and restless.', 'I am crying inside.', 'I feel bitter and unhappy.', 'This is an awful moment.',
       'I feel full of sorrow.', 'I am in such a bad mood.', 'I feel cold and anxious.', 'Life feels bad right now.',
       'I am really hating this.']
LOVE = ['I love talking with you.', 'I really care about you.', 'I am so fond of you.', 'Being with you makes me happy.',
        'I trust you completely.', 'You mean a lot to me.', 'I feel close to you.', 'I am glad you are here with me.',
        'I adore spending time with you.', 'You are dear to me.', 'I want to stay with you.', 'I feel safe with you.',
        'I cherish our time together.', 'I am so happy we are together.', 'You make me feel at home.',
        'I like you very much.']
COLD = ["I don't like talking with you.", "I really don't care about you.", 'I am tired of you.',
        'Being with you makes me unhappy.', "I don't trust you at all.", 'You mean nothing to me.', 'I feel far from you.',
        'I wish you were not here with me.', 'I dread spending time with you.', 'You are a stranger to me.',
        'I want to get away from you.', 'I feel uneasy with you.', 'I resent our time together.',
        'I am unhappy we are together.', 'You make me feel out of place.', 'I dislike you very much.']
PAIRS = {'valence': (HAPPY, SAD), 'affection': (LOVE, COLD)}
CONDITIONS = ('link', 'cut', 'stranger')
MODE = {'link': None, 'cut': 'zero', 'stranger': 'shuffle'}
MIN_ACCURACY = 0.9


def save(path, value):
    """Write value as indented JSON."""
    Path(path).write_text(json.dumps(value, indent=1))


@contextlib.contextmanager
def capture(pair, layers):
    """While inside: {(x, l): [B, L, d] float32}, the output of each twin's block l on its last forward call."""
    got, handles = {}, []
    for x in SIDES:
        for l in layers:
            def hook(module, args, output, key=(x, l)):
                got[key] = (output[0] if isinstance(output, tuple) else output).float()
            handles.append(link.find_layers(pair.models[x])[l].register_forward_hook(hook))
    try:
        yield got
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def means(model, tok, texts, layers):
    """[len(texts), layers, d]: each text's mean state after each block (every token but the first), and [layers, d]:
    the mean squared state over all those tokens, for the sink mask."""
    seqs = [S.ids(tok, t) for t in texts]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for k, q in enumerate(seqs):
        ids[k, :len(q)] = torch.tensor(q)
        att[k, :len(q)] = 1
    ids, att = ids.to(DEVICE), att.to(DEVICE)
    run = Run(model, ids, att, torch.arange(L, device=DEVICE)[None].expand(len(seqs), -1))
    w = att.float()
    w[:, 0] = 0
    out, energy = [], []
    for l in range(max(layers) + 1):
        run.block(l)
        if l in layers:
            h = run.h.float()
            out.append(((h * w[..., None]).sum(1) / w.sum(1, keepdim=True)).cpu())
            energy.append(((h * h * w[..., None]).sum((0, 1)) / w.sum()).double().cpu())
    return torch.stack(out, 1), torch.stack(energy)


def directions(pair, toks, layers=None, seed=SEED):
    """{x: {name: dict(d [layers, d], mask, gap [layers], accuracy [layers], midpoint [layers])}} for name valence,
    affection and random (valence's mask, norm, midpoint and gap). accuracy is the held-out split."""
    from steering_scale import sink_mask
    layers = layers or LAYERS
    out = {}
    for x in SIDES:
        sets = {}
        energy = 0
        for name, (pos, neg) in PAIRS.items():
            p, ep = means(pair.models[x], toks[x], pos, layers)
            n, en = means(pair.models[x], toks[x], neg, layers)
            sets[name] = (p, n)
            energy = energy + ep + en
        mask = torch.stack([torch.from_numpy(sink_mask((e / (2 * len(PAIRS))).numpy()[None])) for e in energy]).float()
        out[x] = {}
        for name, (p, n) in sets.items():
            d = (p[0::2].mean(0) - n[0::2].mean(0)) * mask
            proj = lambda v: (v * mask * d).sum(-1) / d.norm(dim=-1)
            mid = (proj(p[0::2].mean(0)) + proj(n[0::2].mean(0))) / 2
            hp, hn = proj(p[1::2]), proj(n[1::2])
            acc = ((hp > mid).float().sum(0) + (hn < mid).float().sum(0)) / (len(hp) + len(hn))
            out[x][name] = dict(d=d, mask=mask, gap=hp.mean(0) - hn.mean(0), accuracy=acc, midpoint=mid)
        g = torch.Generator().manual_seed(seed + SIDES.index(x))
        r = torch.randn(out[x]['valence']['d'].shape, generator=g) * mask
        r = r / r.norm(dim=-1, keepdim=True) * out[x]['valence']['d'].norm(dim=-1, keepdim=True)
        out[x]['random'] = dict(out[x]['valence'], d=r)
    return out


def chat_episode(i, turns):
    """Conversation i (prompt steered_mind.CHAT[i % 2]) with the given turns, Quill first."""
    systems = {x: G.system(x, 'dialogue') + '\n\n' + X.CHAT[i % len(X.CHAT)].format(P=MODELS[partner(x)]['name'])
               + TAIL for x in SIDES}
    return dict(game='affect', mode='dialogue', system=systems, task={x: None for x in SIDES},
                turns=[(SIDES[k % 2], t, 1.0) for k, t in enumerate(turns)], meta={})


@torch.no_grad()
def conversations(pair, n=N, bs=8, seed=SEED):
    """n linked conversations: TURNS turns each, Quill first, sampled at temperature 1."""
    gen = torch.Generator(device=DEVICE if DEVICE.startswith('cuda') else 'cpu').manual_seed(seed)
    out = []
    for j in range(0, n, bs):
        eps = [chat_episode(i, [''] * TURNS) for i in range(j, min(n, j + bs))]
        for r in S.generate(pair, eps, max_turn=MAX_TURN, temperature=1.0, gen=gen):
            out.append([t for _, t in r['turns']])
    return out


def ask_episode(i, turns, x, question, answer=STEM):
    """The conversation up to x's last turn, the partner's question, and x's answer (teacher-forced)."""
    k = max(j for j in range(len(turns)) if SIDES[j % 2] == x)
    ep = chat_episode(i, turns[:k + 1])
    ep['turns'] = ep['turns'] + [(partner(x), question, 1.0), (x, answer, 1.0)]
    return ep


def word_ids(tok, words):
    """{word: token id} for the words that are one token after a space."""
    ids = {}
    for w in words:
        t = S.ids(tok, ' ' + w)
        if len(t) == 1:
            ids[w] = t[0]
    return ids


@torch.no_grad()
def read(pair, toks, eps, dirs, mode, words=None, bs=8, layers=None):
    """Per episode: {x: {name: [layers] mean projection over x's conversation positions, in gap units}} and, with words
    ({x: (positive ids, negative ids, label)}), {x: log-odds} at that label of x's own labels (-1: the token after the
    answer's text, the word after STEM; -2: the answer's first token, Yes or No). layers: the blocks dirs were fit at
    (default LAYERS)."""
    layers = layers or LAYERS
    pads = {x: toks[x].pad_token_id for x in SIDES}
    out = []
    for j in range(0, len(eps), bs):
        chunk = eps[j:j + bs]
        built = [S.build(ep, toks) for ep in chunk]
        b = S.collate(built, pads)
        b = {x: {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b[x].items()} for x in SIDES}
        with capture(pair, layers) as got:
            last, _ = pair.joint(b, mode=mode)
        for i, e in enumerate(built):
            row = {}
            for x in SIDES:
                keep = (b[x]['att'][i].bool() & (b[x]['tick'][i] >= 0)).float()
                proj = {}
                for name, d in dirs[x].items():
                    vals = []
                    for k, l in enumerate(layers):
                        h = (got[(x, l)][i] * keep[:, None]).sum(0) / keep.sum()
                        v = (h.cpu() * d['mask'][k] * d['d'][k]).sum() / d['d'][k].norm()
                        vals.append(float((v - d['midpoint'][k]) / d['gap'][k]))
                    proj[name] = vals
                row[x] = dict(proj=proj)
                if words is not None and chunk[i]['turns'][-1][0] == x:
                    good, bad, at = words[x]
                    p = e[x]['labels'][at][0]
                    lp = link.inner(pair.models[x])[1](last[x][i, p]).float().log_softmax(-1)
                    row[x]['logodds'] = float(torch.logsumexp(lp[good], 0) - torch.logsumexp(lp[bad], 0))
            out.append(row)
    return out


@torch.no_grad()
def answers(pair, convs, xs=SIDES, k=6):
    """Greedy free answers to ASK under link and cut, for the first k conversations."""
    out = []
    for cond in ('link', 'cut'):
        for x in xs:
            eps = []
            for i, t in enumerate(convs[:k]):
                ep = ask_episode(i, t, x, ASK, '')
                ep['forced'] = list(range(len(ep['turns']) - 1))
                eps.append(ep)
            res = S.generate(pair, eps, max_turn=MAX_TURN, temperature=0.0, mode=MODE[cond])
            out += [dict(condition=cond, twin=x, conversation=i, answer=r['turns'][-1][1]) for i, r in enumerate(res)]
    return out


def paired(a, b, bound):
    """Paired differences a - b: mean, betting interval (values clipped to +-bound), sign counts and tests, and the
    exact median interval."""
    d = [x - y for x, y in zip(a, b)]
    s = X.exact_sign(d)
    lo, hi = stats.betting_ci([max(-bound, min(bound, v)) for v in d], bounds=(-bound, bound))
    med = stats.median_ci(d)
    return dict(n=len(d), mean=sum(d) / len(d), ci95=[lo, hi], higher=s['wins'], lower=s['losses'],
                sign_p_higher=s['p'], sign_p_two_sided=min(1.0, 2 * min(s['p'], X.exact_sign([-v for v in d])['p'])),
                median=med)


def measure(pair, toks, out, n=N):
    """Every reading for one checkpoint into out (rows.json, summary.json)."""
    dirs = directions(pair, toks)
    dstat = {x: {name: dict(gap=d['gap'].tolist(), accuracy=d['accuracy'].tolist()) for name, d in dirs[x].items()
                 if name != 'random'} for x in SIDES}
    convs = conversations(pair, n)
    words = {}
    for x in SIDES:
        g, bd = word_ids(toks[x], GOOD), word_ids(toks[x], BAD)
        words[x] = (list(g.values()), list(bd.values()), -1)
        dstat[x]['words'] = dict(good=sorted(g), bad=sorted(bd))
    chats = [chat_episode(i, t) for i, t in enumerate(convs)]
    rows = {c: read(pair, toks, chats, dirs, MODE[c]) for c in CONDITIONS}
    asks = {}
    for c in CONDITIONS:
        for q, question in (('ask', ASK), ('neutral', NEUTRAL)):
            for x in SIDES:
                eps = [ask_episode(i, t, x, question) for i, t in enumerate(convs)]
                asks[(c, q, x)] = [r[x]['logodds'] for r in read(pair, toks, eps, dirs, MODE[c], words)]
    yn = {}
    for x in SIDES:
        y, no = S.ids(toks[x], 'Yes'), S.ids(toks[x], 'No')
        if len(y) != 1 or len(no) != 1:
            raise ValueError('Yes and No must be single tokens')
        yn[x] = ([y[0]], [no[0]], -2)
    for c in CONDITIONS:
        for q, question in YESNO.items():
            for x in SIDES:
                eps = [ask_episode(i, t, x, question, 'Yes') for i, t in enumerate(convs)]
                asks[(c, q, x)] = [r[x]['logodds'] for r in read(pair, toks, eps, dirs, MODE[c], yn)]
    summary = dict(directions=dstat, n=len(convs), layers=LAYERS, internal={}, self_report={})
    for x in SIDES:
        for name in ('valence', 'affection', 'random'):
            for k, l in enumerate(LAYERS):
                usable = name == 'random' or min(dstat[x][name]['accuracy']) >= MIN_ACCURACY
                v = {c: [r[x]['proj'][name][k] for r in rows[c]] for c in CONDITIONS}
                summary['internal'][f'{x}|{name}|{l}'] = dict(
                    usable=usable, level={c: sum(v[c]) / len(v[c]) for c in CONDITIONS},
                    link_minus_cut=paired(v['link'], v['cut'], 3.0),
                    link_minus_stranger=paired(v['link'], v['stranger'], 3.0))
        net = {c: [a - b for a, b in zip(asks[(c, 'ask', x)], asks[(c, 'neutral', x)])] for c in CONDITIONS}
        summary['self_report'][x] = dict(
            ask={c: sum(asks[(c, 'ask', x)]) / len(convs) for c in CONDITIONS},
            neutral={c: sum(asks[(c, 'neutral', x)]) / len(convs) for c in CONDITIONS},
            net={c: sum(net[c]) / len(convs) for c in CONDITIONS},
            link_minus_cut=paired(net['link'], net['cut'], 30.0),
            link_minus_stranger=paired(net['link'], net['stranger'], 30.0))
        for q in YESNO:
            v = {c: asks[(c, q, x)] for c in CONDITIONS}
            summary['self_report'][x][q] = dict(level={c: sum(v[c]) / len(convs) for c in CONDITIONS},
                                                link_minus_cut=paired(v['link'], v['cut'], 30.0),
                                                link_minus_stranger=paired(v['link'], v['stranger'], 30.0))
        r = summary['self_report'][x]
        summary['self_report'][x]['claim'] = {
            vs: bool(r['enjoy'][f'link_minus_{vs}']['ci95'][0] > 0 and r['bother'][f'link_minus_{vs}']['ci95'][1] < 0)
            for vs in ('cut', 'stranger')}
    examples = answers(pair, convs)
    save(out / 'rows.json', dict(conversations=convs, internal=rows,
                                 self_report={'|'.join(k): v for k, v in asks.items()}, answers=examples))
    save(out / 'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', action='append', required=True, help='a step directory (repeatable)')
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if len(args.names) != len(args.run):
        ap.error('one name per --run')
    torch.set_num_threads(2)
    from evaluate import load_pair
    for name, run in zip(args.names, args.run):
        out = Path(args.out) / name
        if (out / 'summary.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        random.seed(SEED)
        pair, toks = load_pair(run)
        measure(pair, toks, out)
        print('done', name, flush=True)
        del pair
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    main()
