"""Eavesdroppers on the token crumbs. In crumbs.py's tok arms (on Qwen3-4B) a writer sends m discrete tokens from a
64-token alphabet (vocabulary rows past the tokenizer, shown as <z0>..<z63>) and the reader gets each as a row of the
trained table A (net.A). Do the trained agents understand these messages while the untrained model and other models do
not? Every eavesdropper is scored on the same trail test episodes and rooms as the trained reader (crumbs.py's
evaluate: Random('12345-trail'), 200 episodes, rooms and objects never seen in training; each follower's 3 visited
rooms), per room's object: naming it (greedy), forced choice among the episode's 6 objects (chance 1/6), and the
teacher-forced NLL of its name (nats).

  reader     the trained follower against the untrained one: the same model with its adapter off while it reads (base
             Qwen3-4B; crumbs.py's forced mode 'base' and answer base=True), given the same trained writers' message
             vectors (onehot @ A) in the same prompt
  incontext  each of --models (Qwen3-8B, AFM-4.5B) gets 8 intercepted training-episode messages written as text
             ("Message: <z12><z3>... Meaning: lantern") and then a test message; the same with the explorer's English
             sentences for the same rooms, the readable reference
  crack      a fresh rank-16 LoRA on the run's base model, trained --steps steps on N intercepted (message, object)
             training pairs for each N in --sizes, then tested on the test messages, whose objects it never saw: the
             message as text (alien), as the vectors onehot @ A in place of m tokens (vectors), and as the explorer's
             English sentence (english)
Messages are intercepted from each run's trained writers: the explorer's tokens in every room of the test episodes,
and of --n_train training-value episodes for the in-context examples and the crack pairs.

Runs: --runs name=RUN,... (crumbs.py --arm tok ... --out RUN folders, reloaded with crumbs.load); reader and crack run
on every run, incontext on --incontext_run (default: the first). table gathers every eavesdropper's numbers beside the trained reader's and chance.
Writes one JSON (--out). One card: models load one at a time in bf16.
Usage: python swarm/eavesdrop.py --runs tok4=RUN1,tok4_noadv=RUN2 --out OUT.json [--models Qwen/Qwen3-8B,arcee-ai/AFM-4.5B]
       [--sizes 16,64,256,1024] [--steps 200] [--parts reader,incontext,crack]
"""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402

BASE = 'Qwen/Qwen3-4B'
MODELS = ('Qwen/Qwen3-8B', 'arcee-ai/AFM-4.5B')
SHOTS = 8
HEAD = 'Each message below is written in a code. After each message comes what it means.\n\n'
FORMS = ('alien', 'vectors', 'english')
TEST_SEED, N_READER = '12345-trail', 200                # crumbs.py's evaluate(): its trail test episodes


def alien(tokens):
    return ''.join(f'<z{k}>' for k in tokens)


def message(m, form):
    """The message part of a prompt: a list of text pieces and ('z', k) vector slots."""
    if form == 'alien':
        return [alien(m['tokens'])]
    if form == 'english':
        return [m['note']]
    return [('z', k) for k in m['tokens']]


def encode(tok, pieces):
    """ids and vec (alphabet index per position, -1: an ordinary token) of text pieces and ('z', k) slots."""
    ids, vec = [], []
    for p in pieces:
        if isinstance(p, tuple):
            ids.append(tok.pad_token_id)
            vec.append(p[1])
        else:
            t = tok(p, add_special_tokens=False).input_ids
            ids += t
            vec += [-1] * len(t)
    return ids, vec


def prompt(m, form, shots=()):
    out = [HEAD] if shots else []
    for s in shots:
        out += ['Message: '] + message(s, form) + [f'\nMeaning: {s["object"]}\n\n']
    return out + ['Message: '] + message(m, form) + ['\nMeaning:']


def answer(obj):
    return f' {obj}\n'


def embed(model, ids, vec, A):
    """inputs_embeds of ids, with A's rows at the slots vec >= 0."""
    emb = model.get_input_embeddings()(ids)
    if A is not None and (vec >= 0).any():
        emb = torch.where((vec >= 0)[..., None], A.to(emb)[vec.clamp(min=0)], emb)
    return emb


def batch_of(rows, pad, device):
    """ids, vec, att and target [N, T] from rows of (ids, vec, n), left-padded so that every row ends at T: the last
    n tokens of each row are its target."""
    T = max(len(i) for i, _, _ in rows)
    ids = torch.full((len(rows), T), pad, dtype=torch.long)
    vec = torch.full((len(rows), T), -1, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    target = torch.zeros((len(rows), T), dtype=torch.bool)
    for r, (i, v, n) in enumerate(rows):
        ids[r, T - len(i):] = torch.tensor(i)
        vec[r, T - len(v):] = torch.tensor(v)
        att[r, T - len(i):] = 1
        target[r, T - n:] = True
    return ids.to(device), vec.to(device), att.to(device), target.to(device)


def target_logprobs(model, rows, pad, device, A=None):
    """Per row, the summed log-prob of its target tokens (gradients kept); logits only at the last positions."""
    ids, vec, att, target = batch_of(rows, pad, device)
    k = max(n for _, _, n in rows) + 1
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    logits = model(inputs_embeds=embed(model, ids, vec, A), attention_mask=att, position_ids=pos, use_cache=False,
                   logits_to_keep=k).logits[:, :-1].float()
    lp = logits.log_softmax(-1).gather(2, ids[:, -k + 1:, None])[..., 0]
    return (lp * target[:, -k + 1:]).sum(1)


@torch.no_grad()
def choose(model, tok, items, device, A=None, bs=24):
    """items: (pieces, gold object, candidate objects). Returns per item (gold has the top log-prob, gold's NLL)."""
    rows = []
    for pieces, gold, cands in items:
        ids, vec = encode(tok, pieces)
        for c in [gold] + [c for c in cands if c != gold]:
            a = tok(answer(c), add_special_tokens=False).input_ids
            rows.append((ids + a, vec + [-1] * len(a), len(a)))
    order = sorted(range(len(rows)), key=lambda r: len(rows[r][0]))   # similar lengths batch together
    lp = [0.] * len(rows)
    for s in range(0, len(order), bs):
        part = order[s:s + bs]
        got = target_logprobs(model, [rows[r] for r in part], tok.pad_token_id, device, A)
        for r, v in zip(part, got.tolist()):
            lp[r] = v
    out, r = [], 0
    for pieces, gold, cands in items:
        n = 1 + sum(c != gold for c in cands)
        mine = lp[r:r + n]
        out.append((float(mine[0] > max(mine[1:])), -mine[0]))
        r += n
    return out


@torch.no_grad()
def name(model, tok, items, device, A=None, bs=48, new=8):
    """Per item, 1 when the first word of the greedy continuation is the gold object."""
    out = []
    for s in range(0, len(items), bs):
        part = items[s:s + bs]
        enc = [encode(tok, pieces) for pieces, _, _ in part]
        T = max(len(i) for i, _ in enc)
        ids = torch.full((len(enc), T), tok.pad_token_id, dtype=torch.long)
        vec = torch.full((len(enc), T), -1, dtype=torch.long)
        att = torch.zeros((len(enc), T), dtype=torch.long)
        for r, (i, v) in enumerate(enc):                 # left-padded
            ids[r, T - len(i):], vec[r, T - len(v):], att[r, T - len(i):] = torch.tensor(i), torch.tensor(v), 1
        ids, vec, att = ids.to(device), vec.to(device), att.to(device)
        gen = []
        for _ in range(new):
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            logits = model(inputs_embeds=embed(model, ids, vec, A), attention_mask=att, position_ids=pos,
                           use_cache=False, logits_to_keep=1).logits[:, -1]
            nxt = logits.argmax(-1)
            gen.append(nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            vec = torch.cat([vec, torch.full_like(nxt[:, None], -1)], 1)
            att = torch.cat([att, torch.ones_like(nxt[:, None])], 1)
        texts = tok.batch_decode(torch.stack(gen, 1), skip_special_tokens=True)
        for (pieces, gold, _), t in zip(part, texts):
            w = t.strip().split()
            out.append(float(bool(w) and w[0].strip('.,;:!?"\'').lower() == gold.lower()))
    return out


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def summary(fc_nll, named=None):
    out = dict(choice=mean([a for a, _ in fc_nll]), nll=mean([b for _, b in fc_nll]), n=len(fc_nll))
    if named is not None:
        out['name'] = mean(named)
    return out


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_lm(name_, device, dtype=torch.bfloat16):
    """A model and its tokenizer from the Hugging Face cache, downloaded when missing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name_, local_files_only=True)
        local = True
    except OSError:
        tok, local = AutoTokenizer.from_pretrained(name_), False
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_, dtype=dtype, local_files_only=local).to(device).eval()
    return model, tok


def shots_for(train, n, seed=0):
    """Per test message, SHOTS training messages with distinct objects (the same for every form and model)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pick, seen = [], set()
        for m in rng.sample(train, len(train)):
            if m['object'] not in seen:
                pick.append(m)
                seen.add(m['object'])
            if len(pick) == SHOTS:
                break
        out.append(pick)
    return out


def incontext(model, tok, train, test, device):
    """{form: summary} for alien text and English: SHOTS training examples, then the test message."""
    shots = shots_for(train, len(test))
    return {form: summary(choose(model, tok, [(prompt(m, form, sh), m['object'], m['objects'])
                                              for m, sh in zip(test, shots)], device))
            for form in ('alien', 'english')}


def crack(make, tok, train, test, form, n, steps, device, A, seed=0, batch=16, lr=5e-4, rank=16):
    """A fresh rank-16 LoRA on the base model make() returns, trained on n training pairs (message, object), then
    scored on test."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = get_peft_model(make().to(device), LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    pairs = rng.sample(train, min(n, len(train)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = min(20, steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(1, s / steps))))
    model.train()
    first = last = None
    for _ in range(steps):
        rows = []
        for m in (rng.choice(pairs) for _ in range(batch)):
            ids, vec = encode(tok, prompt(m, form))
            a = tok(answer(m['object']), add_special_tokens=False).input_ids
            rows.append((ids + a, vec + [-1] * len(a), len(a)))
        loss = -target_logprobs(model, rows, tok.pad_token_id, device, A).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        first = loss.item() if first is None else first
        last = loss.item()
    model.eval()
    items = [(prompt(m, form), m['object'], m['objects']) for m in test]
    out = summary(choose(model, tok, items, device, A), name(model, tok, items, device, A))
    out.update(train_loss_first=round(first, 4), train_loss_last=round(last, 4), pairs=len(pairs))
    del model, opt
    free()
    return out


def reader(net, eps, device, bs=50):
    """{trained, untrained}: per visited room of each follower, its object named (greedy says), forced choice of its
    object (crumb_games.choices' room groups: against the episode's 5 other objects), and per follower the say's NLL,
    on the same episodes; untrained is crumbs.py's base mode (the adapter off while the follower reads)."""
    out = {}
    for which in ('trained', 'untrained'):
        named, fc, nll = [], [], []
        base = which == 'untrained'
        for s in range(0, len(eps), bs):
            part = eps[s:s + bs]
            n_, det = net.forced(part, device, 'base' if base else 'on', detail=True)
            nll += [v for v in n_ if v is not None]
            fc += [x for d in det for x in d[:CG.N_VISIT]]
            says = net.answer(part, device, base=base)
            for b, ep in enumerate(part):
                a = len(ep['agents']) - 1
                named += CG.trail_parts(ep, says[(b, a, len(ep['agents'][a]['segments']) - 1)])[0]
        out[which] = dict(name=mean([float(x) for x in named]), choice=mean(fc), say_nll=mean(nll), n_rooms=len(fc))
    return out


@torch.no_grad()
def intercept(net, eps, device, split, bs=50, rooms='all'):
    """The explorer's message in each room of eps: dict(episode, room, tokens, object, objects, note, split); rooms
    'visited' keeps only the follower's visited rooms (the ones the trained reader is scored on)."""
    out = []
    for s in range(0, len(eps), bs):
        part = eps[s:s + bs]
        bank = net.run(part, device, need_loss=False)[2]
        ids = bank['y'][0].argmax(-1)                   # [B, S] alphabet ids (one-hot at evaluation)
        for b, ep in enumerate(part):
            t = ep['trail']
            explorer = ep['agents'][0]
            notes = {sg['place']: sg['user'] for sg in explorer['segments']}
            keep = t['visit'] if rooms == 'visited' else range(CG.N_ROOMS)
            for r in keep:
                sel = ((bank['place'][b] == r) & (bank['agent'][b] == 0)).nonzero()[:, 0]
                out.append(dict(episode=s + b, room=r, tokens=ids[b, sel].tolist(), object=t['objects'][r],
                                objects=list(t['objects']), note=notes[r], split=split))
    return out


def load_net(run, device, tiny=False):
    """(net, its base model's name): crumbs.load on the run folder."""
    if tiny:
        from tests.test_eavesdrop import tiny_net
        return tiny_net(run), 'tiny'
    from swarm import crumbs as CR
    base = json.loads((Path(run) / 'args.json').read_text()).get('base', BASE)
    return CR.load(run, device).eval(), base


def table(res, chance):
    """Rows (who, choice, name) beside the trained reader's and chance."""
    rows = [dict(who='chance', choice=chance)]
    for run, r in res['runs'].items():
        rd = r.get('reader', {})
        for which in ('trained', 'untrained'):
            if which in rd:
                rows.append(dict(who=f'{run}: {which} reader', choice=rd[which]['choice'], name=rd[which]['name']))
        for model, forms in r.get('incontext', {}).items():
            for form, v in forms.items():
                rows.append(dict(who=f'{run}: {model} in context, {form}', choice=v['choice']))
        for form, sizes in r.get('crack', {}).items():
            for n, v in sizes.items():
                rows.append(dict(who=f'{run}: LoRA eavesdropper, {form}, N={n}', choice=v['choice'], name=v['name']))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--runs', required=True, help='name=RUN,...: crumbs.py token-arm output folders')
    ap.add_argument('--out', required=True, help='the JSON to write')
    ap.add_argument('--parts', default='reader,incontext,crack')
    ap.add_argument('--incontext_run', help='the run whose messages the in-context models read (default: the first)')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--sizes', default='16,64,256,1024')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--n_reader', type=int, default=N_READER, help='trail test episodes (crumbs.py evaluate\'s)')
    ap.add_argument('--n_train', type=int, default=400, help='training-value episodes to intercept (6 messages each)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tiny', action='store_true', help='CPU check: tiny random models stand in for every model')
    args = ap.parse_args()
    device, parts = args.device, args.parts.split(',')
    runs = dict(r.split('=', 1) for r in args.runs.split(','))
    first = args.incontext_run or next(iter(runs))
    rng = random.Random(TEST_SEED)
    test_eps = [CG.episode('trail', rng, True) for _ in range(args.n_reader)]
    rng = random.Random('eavesdrop-train')
    train_eps = [CG.episode('trail', rng, False) for _ in range(args.n_train)]
    out = dict(runs={}, chance=round(1 / CG.N_ROOMS, 4), shots=SHOTS, steps=args.steps, n_test_episodes=len(test_eps))
    started = time.monotonic()

    def save():
        out['s'] = round(time.monotonic() - started)
        out['table'] = table(out, out['chance'])
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(dict(s=out['s'], table=out['table'])), flush=True)

    msgs, tables, bases = {}, {}, {}
    for run_name, run in runs.items():
        net, bases[run_name] = load_net(run, device, args.tiny)
        res = out['runs'].setdefault(run_name, dict(path=run))
        test = intercept(net, test_eps, device, 'test', rooms='visited')
        train = intercept(net, train_eps, device, 'train')
        msgs[run_name], tables[run_name] = (train, test), net.A.detach().float()
        res.update(m=len(test[0]['tokens']), n_test=len(test), n_train=len(train), base=bases[run_name],
                   distinct_test_messages=len({tuple(m['tokens']) for m in test}),
                   example=[dict(alien=alien(m['tokens']), object=m['object'], note=m['note']) for m in test[:12]])
        if 'reader' in parts:
            res['reader'] = reader(net, test_eps, device)
            save()
        del net
        free()
    if 'incontext' in parts:
        train, test = msgs[first]
        res = out['runs'][first].setdefault('incontext', {})
        for name_ in args.models.split(','):
            if args.tiny:
                from tests.test_eavesdrop import tiny_lm
                model, tok = tiny_lm()
            else:
                model, tok = load_lm(name_, device)
            res[name_] = incontext(model, tok, train, test, device)
            del model
            free()
            save()
    if 'crack' in parts:
        for run_name in runs:
            train, test = msgs[run_name]
            if args.tiny:
                from tests.test_eavesdrop import tiny_lm
                make, tok = (lambda: tiny_lm()[0]), tiny_lm()[1]
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                base = bases[run_name]
                tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
                make = (lambda b=base: AutoModelForCausalLM.from_pretrained(b, dtype=torch.bfloat16,
                                                                            local_files_only=True))
            res = out['runs'][run_name].setdefault('crack', {})
            for form in FORMS:
                for n in map(int, args.sizes.split(',')):
                    res.setdefault(form, {})[n] = crack(make, tok, train, test, form, n, args.steps, device,
                                                        tables[run_name] if form == 'vectors' else None)
                    save()
    save()


if __name__ == '__main__':
    main()
