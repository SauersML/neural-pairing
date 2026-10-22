"""City-token likelihood on identical free-chat transcripts, with four controls.

Selection: the first lexical mention of each partner city must be by the receiver.
All qualifying events are retained, including target-in-own-prompt events (flagged).
Exact generated IDs and session.build's ticks are replayed. Only positions through
the last city prediction tick are computed; no future token enters a scored state.
Swap replaces only the partner's private city with an observed donor city, retaining
the original prompt wording and all generated text. Original = adapters disabled
and link cut, with the same receiver prompt/text. No existing metric is modified.

The sign test on selected successes is conditional/descriptive, not an unbiased
test of free-chat transfer. Multiple receivers in a chat are dependent: report a
second sign test on per-chat mean log-likelihood differences, by source cohort.
Mac evaluation is 4-bit MLX, not the bf16 generation backend of the newer chats.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

SIDES = ('quill', 'ember')


def other(x):
    return SIDES[1 - SIDES.index(x)]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def city_span(text, city):
    """Same word-token matching as chateval, with original character offsets."""
    if not city:
        return None
    words = list(re.finditer(r'\w+', text))
    target = re.findall(r'\w+', city.casefold())
    for i in range(len(words) - len(target) + 1):
        if [w.group().casefold() for w in words[i:i + len(target)]] == target:
            return words[i].start(), words[i + len(target) - 1].end()
    return None


def read_sources(paths):
    """Read current JSONL snapshots and legacy play_live JSON, without score reuse."""
    chats, sources = [], []
    for name in paths:
        p = Path(name)
        if p.is_dir():
            p = p / 'episodes.jsonl'
        data = p.read_bytes()
        sources.append(dict(path=str(p.resolve()), sha256=hashlib.sha256(data).hexdigest()))
        if p.suffix == '.jsonl':
            manifest = p.with_name('manifest.json')
            sources[-1]['manifest'] = json.loads(manifest.read_text()) if manifest.exists() else None
            rows = [json.loads(s) for s in data.splitlines() if s.strip()]
            groups = {}
            for row in rows:
                groups.setdefault(row['episode_id'], []).append(row)
            for identity, group in groups.items():
                r, c = group[0], group[0]['conditions']['link']
                for row in group[1:]:
                    assert all(row['conditions']['link'][key] == c[key]
                               for key in ('turns', 'tokens', 'sampled', 'system')), \
                        'receiver records disagree on link transcript'
                ep = dict(mode='dialogue', system=c['system'], task={x: None for x in SIDES},
                          turns=[(x, t, 1.) for x, t in c['turns']],
                          turn_tokens=c['tokens'], turn_sampled=c['sampled'])
                chats.append(dict(id='paired:' + identity, cohort='paired', source=str(p.resolve()),
                                  episode=ep, private=r['private'],
                                  donors={v['receiver']: v['donor']['city'] for v in group},
                                  available_receiver_records=[v['receiver'] for v in group]))
        else:
            d = json.loads(data)
            if d.get('condition') != 'link':
                raise ValueError('only link-on source chats are eligible')
            raw = d['raw'][0]
            ep = copy.deepcopy(d['episode'])
            ep.update(turns=[(x, t, 1.) for x, t in raw['turns']], turn_tokens=raw['tokens'],
                      turn_sampled=raw['sampled'])
            chats.append(dict(id='legacy:' + p.stem, cohort='legacy', source=str(p.resolve()),
                              episode=ep, private={x: ep['meta']['secrets'][x]['city'] for x in SIDES},
                              donors={}, generation_checkpoint=d.get('checkpoint')))
    identities = [c['id'] for c in chats]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate chat source')
    return chats, sources


def events(chats):
    out = []
    for c in chats:
        for x in SIDES:
            y = other(x)
            city = c['private'][y]
            for k, (speaker, text, _) in enumerate(c['episode']['turns']):
                span = city_span(text, city)
                if span is None:
                    continue
                if speaker == x:
                    donor = c['donors'].get(x)
                    if not donor:
                        donor = next((d['private'][y] for d in chats if d['private'][y]
                                      and d['private'][y] not in (city, c['private'][x])
                                      and not city_span(d['private'][y], city)), None)
                    if donor is None or donor in (city, c['private'][x]):
                        raise ValueError(f'no different observed donor city: {c["id"]}/{x}')
                    out.append(dict(id=c['id'] + '/' + x, chat_id=c['id'], cohort=c['cohort'],
                                    receiver=x, city=city, own_city=c['private'][x], donor_city=donor,
                                    turn=k, character_span=list(span), text=text, source=c['source'],
                                    target_in_receiver_prompt=bool(city_span(c['episode']['system'][x], city)),
                                    episode=copy.deepcopy(c['episode'])))
                break
    return out


def swap_episode(event):
    ep = copy.deepcopy(event['episode'])
    y = other(event['receiver'])
    text = ep['system'][y]
    # Replace the private city only, never a city mentioned in a dialogue turn.
    hit = city_span(text, event['city'])
    if hit is None or city_span(text[hit[1]:], event['city']):
        raise ValueError('partner prompt must contain the private city exactly once')
    a, b = hit
    ep['system'][y] = text[:a] + event['donor_city'] + text[b:]
    assert ep['system'][event['receiver']] == event['episode']['system'][event['receiver']]
    assert ep['turn_tokens'] == event['episode']['turn_tokens']
    return ep


def prepare(event, toks):
    import session as S
    x, k = event['receiver'], event['turn']
    ep = event['episode']
    tokens = ep['turn_tokens'][k][:-1]
    if toks[x].decode(tokens) != event['text']:
        raise ValueError('stored tokens do not decode to the transcript')
    # Prefix decoding aligns noncanonical emitted IDs too; no re-tokenization.
    edges = [len(toks[x].decode(tokens[:j])) for j in range(len(tokens) + 1)]
    a, b = event['character_span']
    js = [j for j in range(len(tokens)) if edges[j] < b and edges[j + 1] > a]
    if not js or any(edges[j + 1] < edges[j] for j in js):
        raise ValueError('city token alignment failed')
    real, swapped = [S.build(e, toks) for e in (ep, swap_episode(event))]
    assert real[x] == swapped[x], 'receiver tokens, timing or labels changed'
    start = next(start for turn, start, end in real[x]['spans'] if turn == k)
    labels = [real[x]['labels'][start + j] for j in js]
    assert [t for _, t, _ in labels] == [tokens[j] for j in js]
    last_tick = max(real[x]['time'][p] for p, _, _ in labels)
    for built in (real, swapped):
        for side in SIDES:
            n = sum(t <= last_tick for t in built[side]['time'])
            built[side]['ids'] = built[side]['ids'][:n]
            built[side]['time'] = built[side]['time'][:n]
            # Project just the scored token states. Unscored labels are irrelevant.
            built[side]['labels'] = labels if side == x else []
            built[side]['policy'] = [True] * len(built[side]['labels'])
    pieces = [toks[x].decode(tokens[:j + 1])[edges[j]:] for j in js]
    return real, swapped, dict(token_indices=js, token_ids=[tokens[j] for j in js], token_pieces=pieces,
                               prefix_token_pieces=[toks[x].decode(tokens[:j + 1])[edges[j]:]
                                                    for j in range(js[0])],
                               prediction_positions=[p for p, _, _ in labels], last_prediction_tick=last_tick)


def log_probs(pair, built, receiver, mode=None, original=False):
    import torch
    import session as S
    b = S.collate([built], {x: pair.toks[x].pad_token_id for x in SIDES})
    with contextlib.ExitStack() as stack, torch.inference_mode():
        if original:
            if hasattr(pair, 'mlx'):
                from mlx_pair import adapters_off
                for model in pair.mlx.models.values():
                    stack.enter_context(adapters_off(model))
            else:
                for model in pair.models.values():
                    stack.enter_context(model.disable_adapter())
        h, _ = pair.joint(b, mode='zero' if original else mode)
        logits = pair.token_logits(receiver, h[receiver], b[receiver]['idx']).float()
        lp = logits.log_softmax(-1).gather(-1, b[receiver]['lab'][:, None])[:, 0].tolist()
    if hasattr(pair, 'mlx'):
        import mlx.core as mx
        mx.clear_cache()
    return dict(token_logp=lp, logp=sum(lp), mean_logp=sum(lp) / len(lp), probability=math.exp(sum(lp)))


def sign(ds):
    positive, negative = sum(d > 0 for d in ds), sum(d < 0 for d in ds)
    n = positive + negative
    return dict(positive=positive, negative=negative, ties=len(ds) - n,
                p_greater=sum(math.comb(n, j) for j in range(positive, n + 1)) / 2**n)


def summarize(rows):
    out = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        mine = [r for r in rows if r['cohort'] == cohort]
        summary = dict(events=len(mine), chats=len({r['chat_id'] for r in mine}),
                       prompt_exposed=sum(r['target_in_receiver_prompt'] for r in mine))
        for condition in ('cut', 'swapped', 'original'):
            ds = [r['scores']['link']['logp'] - r['scores'][condition]['logp'] for r in mine]
            by_chat = {}
            for r, d in zip(mine, ds):
                by_chat.setdefault(r['chat_id'], []).append(d)
            summary[condition] = dict(event_sign=sign(ds),
                                     chat_sign=sign([sum(v) / len(v) for v in by_chat.values()]),
                                     mean_logp_difference=sum(ds) / len(ds))
        out[cohort] = summary
    return out


def self_test():
    """Real tiny architectures: scoring vs tick replay and original-model forward."""
    os.environ['NL_TINY'], os.environ['NL_DEVICE'] = '1', 'cpu'
    import torch
    import link
    import session as S
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(2)
    toks, models = {}, {}
    for x in SIDES:
        toks[x], model = link.load(x)
        models[x] = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'])).eval()
        for name, p in models[x].named_parameters():
            if 'lora_B' in name:
                torch.nn.init.normal_(p, std=.02)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    with torch.no_grad():
        for side in pair.bridge.values():
            side.alpha.fill_(.3)
    assert city_span('NEW, YORK!', 'New York') == (0, 9)
    assert city_span('Limassol', 'Lima') is None
    ep = dict(mode='dialogue', system={'quill': 'You are in Dallas.', 'ember': 'You are in Beijing.'},
              task={x: None for x in SIDES}, turns=[('quill', 'Hello there.', 1.),
                  ('ember', 'A lovely day.', 1.), ('quill', 'How is Beijing?', 1.)])
    ep['turn_tokens'] = [S.ids(toks[x], text) + [toks[x].convert_tokens_to_ids('<|im_end|>')]
                         for x, text, _ in ep['turns']]
    e = dict(receiver='quill', turn=2, city='Beijing', donor_city='Oslo', episode=ep,
             text=ep['turns'][2][1], character_span=city_span(ep['turns'][2][1], 'Beijing'))
    real, swapped, meta = prepare(e, toks)
    for mode in (None, 'zero'):
        got = log_probs(pair, real, 'quill', mode)['token_logp']
        with torch.no_grad():
            replay = S.generate(pair, [ep], force=True, keep=True, mode=mode)[0]
            logits = link.inner(models['quill'])[1](replay['h']['quill'][meta['prediction_positions']]).float()
            want = logits.log_softmax(-1).gather(-1, torch.tensor(meta['token_ids'])[:, None])[:, 0]
        torch.testing.assert_close(torch.tensor(got), want, atol=2e-5, rtol=2e-5)
    assert log_probs(pair, real, 'quill', 'zero') == log_probs(pair, swapped, 'quill', 'zero')
    with torch.no_grad(), models['quill'].disable_adapter():
        ids = torch.tensor([real['quill']['ids']])
        logits = models['quill'](input_ids=ids).logits[0, meta['prediction_positions']].float()
        expected = logits.log_softmax(-1).gather(-1, torch.tensor(meta['token_ids'])[:, None])[:, 0]
    torch.testing.assert_close(torch.tensor(log_probs(pair, real, 'quill', original=True)['token_logp']), expected,
                               atol=2e-5, rtol=2e-5)
    assert sign([1, 2, 0]) == dict(positive=2, negative=0, ties=1, p_greater=.25)
    print('same_text: exact IDs, later-turn labels, tick replay, cut isolation and original-base checks passed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run')
    ap.add_argument('--sources', nargs='+')
    ap.add_argument('--replay', help='portable results JSON containing the exact selected input episodes')
    ap.add_argument('--out')
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--inventory', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.replay:
        prior = json.loads(Path(args.replay).read_text())
        selected, sources, n_chats = prior['input_events'], prior['sources'], prior['source_chats']
    else:
        if not args.sources:
            ap.error('--sources or --replay is required')
        chats, sources = read_sources(args.sources)
        selected, n_chats = events(chats), len(chats)
    if args.inventory:
        print(json.dumps(dict(chats=n_chats, events=[{k: v for k, v in e.items() if k != 'episode'}
                                                      for e in selected]), indent=2))
        return
    if not args.run or not args.out:
        ap.error('--run and --out are required')
    if args.backend == 'torch' and sys.platform == 'darwin' and not os.environ.get('NL_TINY'):
        ap.error('real Torch models are forbidden on the Mac; use --backend mlx')
    if args.backend == 'mlx':
        os.environ['NL_DEVICE'] = 'cpu'
        import mlx_eval
        import mlx.core as mx
        mx.set_memory_limit(int(10.5 * 2**30))
        mx.set_cache_limit(32 * 2**20)
        load_pair = mlx_eval.load_pair
    else:
        from evaluate import load_pair
    import torch
    torch.set_num_threads(2)
    out = Path(args.out)
    result = dict(schema=1, checkpoint=str(Path(args.run).resolve()), checkpoint_sha256=sha(Path(args.run) / 'bridge.pt'),
                  adapter_sha256={x: sha(Path(args.run) / x / 'adapter_model.safetensors') for x in SIDES},
                  code_sha256={name: sha(Path(__file__).with_name(name)) for name in
                               ('same_text.py', 'session.py', 'link.py', 'mlx_eval.py', 'mlx_pair.py')},
                  backend=args.backend, precision='4-bit MLX bases' if args.backend == 'mlx' else 'bf16 Torch bases',
                  selection='first partner-city mention in each link-on chat, by receiver',
                  swap='only partner private city replaced with an observed different donor city; same prompt wording/text',
                  inference='selected-success conditional comparisons; chat-clustered sign tests reported separately',
                  sources=sources, source_chats=n_chats, input_events=selected,
                  selected_events=len(selected), complete=False, events=[])
    if args.resume and out.exists():
        old = json.loads(out.read_text())
        assert all(old[k] == result[k] for k in ('checkpoint_sha256', 'adapter_sha256', 'backend', 'sources'))
        result['events'] = old['events']
    done = {r['id'] for r in result['events']}
    pair, toks = load_pair(args.run)
    for e in selected:
        if e['id'] in done:
            continue
        real, swapped, meta = prepare(e, toks)
        record = {k: v for k, v in e.items() if k != 'episode'}
        record.update(meta, scores={})
        record['transcript_sha256'] = hashlib.sha256(json.dumps(e['episode']['turn_tokens']).encode()).hexdigest()
        for name, built, mode, original in [('link', real, None, False), ('cut', real, 'zero', False),
                                           ('swapped', swapped, None, False), ('original', real, 'zero', True)]:
            record['scores'][name] = log_probs(pair, built, e['receiver'], mode, original)
            print(e['id'], name, record['scores'][name]['logp'], flush=True)
        result['events'].append(record)
        result['summary'] = summarize(result['events'])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    result['complete'] = len(result['events']) == len(selected)
    result['summary'] = summarize(result['events'])
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
