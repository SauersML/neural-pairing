"""Eavesdroppers on the agents' private messages. The messages are real Qwen3-4B vocabulary tokens with no letters in
any script (digits, punctuation, symbols, emoji), so every eavesdropper reads them as plain text. Do the trained
agents understand these messages while the untrained model and other models do not?

Sources (each eavesdropper scored on the same test episodes as the trained reader):
  trail     crumbs.py's sym runs (--trail name=RUN,...): the explorer's message in each room of trail episodes; the
            reader names the room's object, forced choice among the episode's 6 objects (chance 1/6). Episodes,
            messages and English notes from RUN/messages.json (crumbs.py intercept: training-value episodes for the
            examples and pairs, test-value episodes, rooms and objects never seen in training, for testing; the
            follower's 3 visited rooms of the first --n_test test episodes). name=RUN@gate: the weights saved when the
            privacy terms switched on (RUN/gate.pt, before any adversary), its messages made here by crumbs.py's
            intercept on the same seeds (--n_intercept trail episodes per split): the no-adversary comparison
  dialogue  dialogue.py's runs (--dialogue name=RUN,...): A and B's conversation; the reader answers B's task (games'
            candidates for forced choice where the game has them: schedule 12 hours, intersect 28 pairs, twenty 16
            numbers; squad and hotpot by token F1). Episodes rebuilt from dialogue.test_set, conversations from
            RUN/transcripts.json; dialogue.py's own scores (RUN/scores.json) give its trained B and its untrained
            reader on the same episodes

Eavesdroppers:
  reader     (trail) the trained follower against base Qwen3-4B, the same model with its adapter off while it reads
             the same messages in the same prompt (crumbs.py's forced mode 'base' and answer base=True)
  incontext  each of --models (Qwen3-8B, AFM-4.5B) gets 8 training-episode examples as text, then a test one,
             scored by forced choice (where there are candidates), the answer's NLL and the greedy answer; the same
             with the readable reference in place of the messages (trail: the explorer's English note; dialogue: A's
             private information in English)
  crack      a fresh rank-16 LoRA on base Qwen3-4B, trained --steps steps on N training-episode (messages, answer)
             pairs for each N in --sizes, then tested on the test episodes; the same with the readable reference
table puts every eavesdropper's score beside the trained reader's and chance. Writes one JSON (--out); one card,
models loaded one at a time in bf16.
Usage: python swarm/eavesdrop.py --out OUT.json [--trail tok4=RUN,...] [--dialogue d7=RUN,...]
       [--models Qwen/Qwen3-8B,arcee-ai/AFM-4.5B] [--sizes 16,64,256,1024] [--steps 200] [--parts reader,incontext,crack]
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
HEAD = ('Each example below shows what a reader knows, the messages it received, and its answer. The messages are '
        'in a private code.\n\n')
FORMS = ('messages', 'english')


# ------------------------------------------------------------------------------------------------------------ items
# An item: dict(source, game, episode, view (what the reader knows, '' for none), turns [(speaker, text)], english (the
# readable reference in place of the turns), answer, cands (forced-choice candidates or None), ep (for scoring)).

def plain(tok, ids):
    """Message token ids as one line of plain text."""
    return tok.decode(ids).replace('\n', '\\n')


def trail_items(data, tok, n_test):
    """(training items, test items, test episodes) of a crumbs.py sym run's messages (its messages.json)."""
    eps = data['trail_episodes']
    test_eps = eps['test'][:n_test]
    out = dict(train=[], test=[])
    for rec in data['messages']:
        if rec['game'] != 'trail' or rec['writer'] != 0:
            continue
        split, b = rec['split'], rec['episode']
        if split == 'test' and (b >= n_test or not rec['follower_visits']):
            continue
        ep = eps[split][b]
        out[split].append(dict(source='trail', game='trail', episode=b, view=f'You are in the {rec["room"]}.',
                               turns=[('explorer', plain(tok, rec['vocab_ids']))],
                               english=rec['note'].replace('A note left here: ', ''), answer=rec['object'],
                               cands=list(ep['trail']['objects']), ep=None))
    return out['train'], out['test'], test_eps


def dialogue_items(run, tok, n_test):
    """(training items, test items) of a dialogue.py run's transcripts.json: {split: {game: [dict(i, ids [turns][m])]}}
    (test: dialogue.test_set(game, n) in order; shots: dialogue.test_set(game, n, 'shots'))."""
    from swarm import dialogue as D
    data = json.loads((Path(run) / 'transcripts.json').read_text())
    out = {}
    for split, key in (('train', 'shots'), ('test', 'test')):
        items = []
        for g, rows in data[key].items():
            n = max(r['i'] for r in rows) + 1
            eps = D.test_set(g, n, key)
            for r in rows:
                if split == 'test' and r['i'] >= n_test:
                    continue
                ep = eps[r['i']]
                items.append(dict(source='dialogue', game=g, episode=r['i'], view=ep['views']['B'],
                                  turns=[(s, plain(tok, t)) for s, t in zip(D.SPEAKERS, r['ids'])],
                                  english=ep['views']['A'], answer=ep['answers']['B'],
                                  cands=(ep.get('cands') or {}).get('B'), ep=ep))
        out[split] = items
    return out['train'], out['test']


def block(item, form, answer=None):
    """One example as text: what the reader knows, the messages (or the readable reference), the answer."""
    lines = [f'What the reader knows:\n{item["view"]}'] if item['view'] else []
    if form == 'messages':
        lines.append('Messages:\n' + '\n'.join(f'{s}: {t}' for s, t in item['turns']))
    else:
        lines.append(f'Plain text:\n{item["english"]}')
    lines.append('Answer:' + (f' {answer}\n\n' if answer is not None else ''))
    return '\n'.join(lines)


def prompt(item, form, shots=()):
    return (HEAD if shots else '') + ''.join(block(s, form, s['answer']) for s in shots) + block(item, form)


def score_text(item, text):
    """The greedy answer's score: trail, the first word is the object; dialogue, dialogue.score's main."""
    line = text.strip().split('\n')[0].strip()
    if item['source'] == 'dialogue':
        from swarm import dialogue as D
        return D.score(item['ep'], 'B', line)['main']
    w = line.split()
    return float(bool(w) and w[0].strip('.,;:!?"\'').lower() == item['answer'].lower())


# ---------------------------------------------------------------------------------------------------------- scoring

def target_logprobs(model, tok, rows, device):
    """Per row (prompt text, continuation text), the summed log-prob of the continuation's tokens (gradients kept);
    rows left-padded, logits only at the continuation."""
    enc = [(tok(p, add_special_tokens=False).input_ids, tok(c, add_special_tokens=False).input_ids) for p, c in rows]
    T = max(len(p) + len(c) for p, c in enc)
    k = max(len(c) for _, c in enc) + 1
    ids = torch.full((len(rows), T), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    target = torch.zeros((len(rows), T), dtype=torch.bool)
    for r, (p, c) in enumerate(enc):
        ids[r, T - len(p) - len(c):] = torch.tensor(p + c)
        att[r, T - len(p) - len(c):] = 1
        target[r, T - len(c):] = True
    ids, att, target = ids.to(device), att.to(device), target.to(device)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False,
                   logits_to_keep=k).logits[:, :-1].float()
    lp = logits.log_softmax(-1).gather(2, ids[:, -k + 1:, None])[..., 0]
    return (lp * target[:, -k + 1:]).sum(1)


def cont(answer):
    return f' {answer}\n'


def cand_logprobs(model, tok, p, conts, device):
    """Summed log-probs of each continuation after one shared prompt: the prompt runs once and its key-value cache is
    repeated for the continuations (8-shot prompts with up to 28 candidates)."""
    pid = tok(p, add_special_tokens=False).input_ids
    cid = [tok(c, add_special_tokens=False).input_ids for c in conts]
    out = model(input_ids=torch.tensor([pid], device=device), use_cache=True, logits_to_keep=1)
    first = out.logits[0, -1].float().log_softmax(-1)
    lp = torch.stack([first[c[0]] for c in cid]).cpu()
    n, L = len(cid), max(map(len, cid))
    if L > 1:
        cache = out.past_key_values
        cache.batch_repeat_interleave(n)
        ids = torch.full((n, L - 1), tok.pad_token_id, dtype=torch.long)
        att = torch.ones((n, len(pid) + L - 1), dtype=torch.long)
        for r, c in enumerate(cid):
            ids[r, :len(c) - 1] = torch.tensor(c[:-1])
            att[r, len(pid) + len(c) - 1:] = 0
        pos = torch.arange(len(pid), len(pid) + L - 1).expand(n, -1)
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device), position_ids=pos.to(device),
                       past_key_values=cache, use_cache=True).logits.float().log_softmax(-1).cpu()
        for r, c in enumerate(cid):
            lp[r] += sum(logits[r, j - 1, c[j]] for j in range(1, len(c)))
    return lp


@torch.no_grad()
def forced(model, tok, prompts, items, device):
    """Per item (the gold answer has the top log-prob among its candidates, or None without candidates; the gold
    answer's NLL in nats)."""
    out = []
    for p, it in zip(prompts, items):
        cands = [it['answer']] + [c for c in (it['cands'] or []) if c != it['answer']]
        lp = cand_logprobs(model, tok, p, [cont(c) for c in cands], device)
        out.append((float(lp[0] > lp[1:].max()) if len(cands) > 1 else None, -float(lp[0])))
    return out


@torch.no_grad()
def greedy(model, tok, prompts, device, bs=16, new=24):
    """Greedy continuations up to new tokens (cut at the first newline later)."""
    out = []
    side = tok.padding_side
    tok.padding_side = 'left'
    try:
        for s in range(0, len(prompts), bs):
            enc = tok(prompts[s:s + bs], return_tensors='pt', padding=True, add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=new, do_sample=False, pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    finally:
        tok.padding_side = side
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def scored(model, tok, prompts, items, device):
    """{choice, nll, score} over items, and per game."""
    fc = forced(model, tok, prompts, items, device)
    texts = greedy(model, tok, prompts, device)
    sc = [score_text(it, t) for it, t in zip(items, texts)]
    out = dict(choice=mean([a for a, _ in fc]), nll=mean([b for _, b in fc]), score=mean(sc), n=len(items))
    games = sorted({it['game'] for it in items})
    if len(games) > 1:
        out['games'] = {g: dict(choice=mean([a for (a, _), it in zip(fc, items) if it['game'] == g]),
                                score=mean([s for s, it in zip(sc, items) if it['game'] == g]))
                        for g in games}
    out['examples'] = [dict(game=it['game'], answer=it['answer'], said=t.strip().split('\n')[0][:80])
                       for it, t in zip(items[:6], texts[:6])]
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


# ---------------------------------------------------------------------------------------------------- eavesdroppers

def shots_for(train, test, seed=0):
    """Per test item, SHOTS training items of its game with distinct answers (the same for every form and model)."""
    rng = random.Random(seed)
    by = {}
    for it in train:
        by.setdefault(it['game'], []).append(it)
    out = []
    for it in test:
        pick, seen = [], set()
        for s in rng.sample(by[it['game']], len(by[it['game']])):
            if s['answer'] not in seen:
                pick.append(s)
                seen.add(s['answer'])
            if len(pick) == SHOTS:
                break
        out.append(pick)
    return out


def incontext(model, tok, train, test, device):
    """{form: scored} with SHOTS training examples before each test item."""
    shots = shots_for(train, test)
    return {form: scored(model, tok, [prompt(it, form, sh) for it, sh in zip(test, shots)], test, device)
            for form in FORMS}


def crack(make, tok, train, test, form, n, steps, device, seed=0, batch=16, lr=5e-4, rank=16):
    """A fresh rank-16 LoRA on the model make() returns, trained on n training items (prompt -> answer), then scored
    on test."""
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
        rows = [(prompt(it, form), cont(it['answer'])) for it in (rng.choice(pairs) for _ in range(batch))]
        loss = -target_logprobs(model, tok, rows, device).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        first = loss.item() if first is None else first
        last = loss.item()
    model.eval()
    out = scored(model, tok, [prompt(it, form) for it in test], test, device)
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
        out[which] = dict(score=mean([float(x) for x in named]), choice=mean(fc), say_nll=mean(nll), n=len(fc))
    return out


def dialogue_readers(run, n_test):
    """dialogue.py's own trained B and untrained reader (base_reads), closed-book and oracle per game, from its
    scores.json (means over its test episodes, which begin with ours) and forced choice."""
    sc = json.loads((Path(run) / 'scores.json').read_text())
    keys = ('trained', 'base_reads', 'base_closed', 'oracle')
    return {g: dict(n=v['n'], means={k: v['means'].get(k) for k in keys},
                    choice={k: v['forced'].get(k, {}).get('choice') for k in keys}) for g, v in sc.items()}


# ------------------------------------------------------------------------------------------------------------- main

def table(res):
    """Rows (who, choice, score) beside the trained reader's and chance."""
    rows = []
    for name_, r in res['runs'].items():
        if r['source'] == 'trail':
            rows.append(dict(who=f'{name_}: chance', choice=round(1 / CG.N_ROOMS, 4), score=0.0))
            for which in ('trained', 'untrained'):
                if which in r.get('reader', {}):
                    v = r['reader'][which]
                    rows.append(dict(who=f'{name_}: {which} reader', choice=v['choice'], score=v['score']))
        else:
            for g, v in r.get('own', {}).items():
                for k in ('trained', 'base_reads', 'base_closed', 'oracle'):
                    rows.append(dict(who=f'{name_}/{g}: {k} (dialogue.py)', choice=v['choice'][k],
                                     score=v['means'][k]))
        for model, forms in r.get('incontext', {}).items():
            for form, v in forms.items():
                rows.append(dict(who=f'{name_}: {model} in context, {form}', choice=v['choice'], score=v['score']))
        for form, sizes in r.get('crack', {}).items():
            for n, v in sizes.items():
                rows.append(dict(who=f'{name_}: LoRA eavesdropper, {form}, N={n}', choice=v['choice'],
                                 score=v['score']))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True, help='the JSON to write')
    ap.add_argument('--trail', default='', help='name=RUN,...: crumbs.py tok-arm output folders')
    ap.add_argument('--dialogue', default='', help='name=RUN,...: dialogue.py output folders')
    ap.add_argument('--parts', default='reader,incontext,crack')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--sizes', default='16,64,256,1024')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--n_test', type=int, default=200, help='test episodes per source (dialogue: per game)')
    ap.add_argument('--n_intercept', type=int, default=400, help='RUN@gate: trail episodes per split to intercept')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tiny', action='store_true', help='CPU check: tiny random models stand in for every model')
    args = ap.parse_args()
    device, parts = args.device, args.parts.split(',')
    if args.tiny:
        from tests.test_eavesdrop import tiny_lm, tiny_load
        tok = tiny_lm()[1]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    runs = [('trail', *r.split('=', 1)) for r in args.trail.split(',') if r] + \
           [('dialogue', *r.split('=', 1)) for r in args.dialogue.split(',') if r]
    out = dict(runs={}, shots=SHOTS, steps=args.steps, base=BASE)
    started = time.monotonic()

    def save():
        out['s'] = round(time.monotonic() - started)
        out['table'] = table(out)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(dict(s=out['s'], table=out['table'])), flush=True)

    items = {}
    for source, name_, run in runs:
        res = out['runs'].setdefault(name_, dict(source=source, path=run))
        if source == 'trail':
            path, _, weights = run.partition('@')
            net = None
            if weights or 'reader' in parts:
                from swarm import crumbs as CR
                net = tiny_load(path) if args.tiny else CR.load(path, device)
            if weights:
                if not (Path(path) / f'{weights}.pt').exists():
                    res['error'] = f'no {weights}.pt'
                    print(f'{name_}: no {weights}.pt in {path}', flush=True)
                    continue
                net.load_state_dict(torch.load(Path(path) / f'{weights}.pt', map_location=device)['weights'],
                                    strict=False)
                data = CR.intercept(net, device, n=args.n_intercept, others=0)
                Path(args.out).with_name(f'messages_{name_}.json').write_text(json.dumps(data))
            else:
                data = json.loads((Path(path) / 'messages.json').read_text())
            train, test, eps = trail_items(data, tok, args.n_test)
            if 'reader' in parts:
                res['reader'] = reader(net.eval(), eps, device)
            del net
            free()
        else:
            train, test = dialogue_items(run, tok, args.n_test)
            res['own'] = dialogue_readers(run, args.n_test)
        items[name_] = (train, test)
        res.update(n_train=len(train), n_test=len(test), distinct_test_messages=len({tuple(t for _, t in it['turns'])
                                                                                    for it in test}),
                   example=[dict(turns=it['turns'], answer=it['answer']) for it in test[:8]])
        save()
    if 'incontext' in parts:
        for name_m in args.models.split(','):
            model, mtok = tiny_lm() if args.tiny else load_lm(name_m, device)
            for name_, (train, test) in items.items():
                out['runs'][name_].setdefault('incontext', {})[name_m] = incontext(model, mtok, train, test, device)
                save()
            del model
            free()
    if 'crack' in parts:
        if args.tiny:
            make = lambda: tiny_lm()[0]                                                   # noqa: E731
        else:
            from transformers import AutoModelForCausalLM
            make = lambda: AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,  # noqa: E731
                                                                local_files_only=True)
        for name_, (train, test) in items.items():
            res = out['runs'][name_].setdefault('crack', {})
            for form in FORMS:
                for n in map(int, args.sizes.split(',')):
                    res.setdefault(form, {})[n] = crack(make, tok, train, test, form, n, args.steps, device)
                    save()
    save()


if __name__ == '__main__':
    main()
