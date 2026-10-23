"""Preregistered concept-vector transfer, with untouched session scheduling.

Mac: lockf -k ~/mpd-data/locks/macgpu.lock env MPD_MEM_GIB=12 python
pairing/steered_mind.py --backend mlx --run CHECKPOINT --out OUTPUT --phase all
GPU confirmation: same command without lockf, --backend torch (CUDA required).
No model service is used. All texts, calibration rules and counts are exported.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

SIDES = ('quill', 'ember')
CONCEPTS = {
    'ocean': ('ocean oceans sea seas beach beaches waves tide tides coastal shoreline marine', [
        'Ocean waves break on a sandy beach.', 'The sea stretches beyond the horizon.',
        'A tide rises along the rocky coast.', 'Salt water surrounds a distant island.',
        'Marine life thrives beneath the ocean.', 'A ship crosses the deep blue sea.',
        'Foam washes across the shoreline.', 'Coral reefs grow in tropical seas.',
        'Seabirds glide above rolling waves.', 'The beach curves around a sheltered bay.']),
    'fire': ('fire fires flame flames burning blaze blazing embers wildfire smoke', [
        'A fire burns brightly in the hearth.', 'Flames rise from dry branches.',
        'Smoke curls above a campfire.', 'Glowing embers warm the dark room.',
        'A wildfire spreads across dry grass.', 'The candle flame flickers gently.',
        'Burning logs crackle in the fireplace.', 'Heat radiates from the blazing wood.',
        'Firefighters extinguish a fierce blaze.', 'Sparks fly upward from the bonfire.']),
    'music': ('music musical melody melodies song songs singing piano guitar rhythm concert', [
        'Music fills the concert hall.', 'A pianist plays a gentle melody.',
        'The guitar accompanies a familiar song.', 'A singer practices a new tune.',
        'The orchestra performs a symphony.', 'Drums establish a steady rhythm.',
        'A violin plays a soaring phrase.', 'The choir sings in harmony.',
        'Musicians rehearse for a concert.', 'A recording captures the beautiful music.']),
    'cats': ('cat cats kitten kittens feline felines meow meowing purr purring', [
        'A cat sleeps on the windowsill.', 'The kitten chases a little toy.',
        'Cats groom their soft fur.', 'A tabby cat stretches in sunlight.',
        'The kitten purrs while being stroked.', 'A feline watches a bird outside.',
        'The cat meows beside the door.', 'Two kittens play under a chair.',
        'A black cat walks across the room.', 'The sleepy cat curls into a ball.']),
    'space': ('space cosmic cosmos planet planets galaxy galaxies astronaut astronauts rocket lunar nebula', [
        'A rocket launches into outer space.', 'Astronauts explore the lunar surface.',
        'Planets orbit a distant star.', 'A galaxy contains billions of stars.',
        'The spacecraft travels beyond Earth.', 'A telescope reveals a colorful nebula.',
        'Saturn has rings of ice and rock.', 'Cosmic dust forms new planetary systems.',
        'The astronaut floats in orbit.', 'A probe studies the surface of Mars.']),
    'cooking': ('cook cooks cooking kitchen recipe recipes baking bake oven chef ingredients', [
        'A chef prepares dinner in the kitchen.', 'Cooking requires fresh ingredients.',
        'The recipe calls for chopped vegetables.', 'Bread is baking in the oven.',
        'The cook stirs a pot of soup.', 'A saucepan simmers on the stove.',
        'The chef seasons the meal carefully.', 'A recipe explains how to roast vegetables.',
        'Mix flour and water to make dough.', 'The kitchen smells of freshly baked bread.']),
    'winter': ('winter snow snowy snowfall snowflake snowflakes frost frosty blizzard icy skiing', [
        'Winter snow covers the quiet street.', 'Snowflakes settle on a woolen scarf.',
        'Frost forms on the window overnight.', 'A blizzard closes the mountain road.',
        'Children build a snowman outside.', 'Icy branches sparkle in winter sunlight.',
        'Skiers descend a snowy slope.', 'Thick mittens protect against the cold.',
        'Fresh snowfall blankets the rooftops.', 'The winter morning is crisp and frosty.']),
    'football': ('football soccer touchdown touchdowns quarterback goalkeeper striker stadium goal goals', [
        'The football team enters the stadium.', 'A quarterback throws a long pass.',
        'The striker scores a soccer goal.', 'Fans cheer after a touchdown.',
        'A goalkeeper catches the football.', 'The referee blows the final whistle.',
        'Football players practice their passing.', 'The soccer match ends in a draw.',
        'A defender tackles the advancing striker.', 'The football coach prepares the team.']),
    'gardens': ('garden gardens gardening gardener flowers flower plants planting seeds blossom', [
        'A gardener plants flowers in the garden.', 'Seeds sprout in moist soil.',
        'The garden blooms in spring.', 'A watering can nourishes young plants.',
        'The gardener pulls weeds from a flowerbed.', 'Roses blossom beside the garden path.',
        'Fresh compost improves the soil.', 'The flower garden attracts butterflies.',
        'A gardener trims the green hedge.', 'Small seedlings grow in the greenhouse.']),
    'trains': ('train trains railway railroad locomotive locomotives carriage carriages station tracks', [
        'A train arrives at the railway station.', 'The locomotive pulls several carriages.',
        'Railroad tracks cross the countryside.', 'Passengers board an express train.',
        'The railway signal changes to green.', 'A steam locomotive whistles loudly.',
        'A freight train transports heavy cargo.', 'The train conductor checks tickets.',
        'A railway bridge spans the river.', 'The last carriage passes the platform.']),
    'painting': ('paint paints painting paintings painter painters canvas brush brushes watercolor portrait', [
        'A painter works on a large canvas.', 'Watercolor paint spreads across the paper.',
        'The artist mixes colors on a palette.', 'A brush adds detail to the portrait.',
        'The painting depicts a quiet village.', 'Oil paint dries slowly on the canvas.',
        'The painter sketches before adding color.', 'An exhibition displays colorful paintings.',
        'Fine brushes create delicate strokes.', 'The artist finishes a painted landscape.']),
    'mountains': ('mountain mountains summit summits alpine peaks peak hiking hiker hikers climbing', [
        'High mountains rise above the valley.', 'A hiker approaches the rocky summit.',
        'Alpine peaks tower over a clear lake.', 'Climbers ascend a steep mountain face.',
        'A hiking trail winds up the ridge.', 'The mountain summit offers a wide view.',
        'Hikers rest beside an alpine meadow.', 'A narrow pass connects two valleys.',
        'The climbing team reaches the peak.', 'Mountain slopes descend toward the forest.']),
}
FRAMES = ['{}', 'I read this description: {}', 'Here is a brief scene: {}',
          'Someone described the following: {}', 'This is the scene in the paragraph: {}']
NEUTRAL = ['A person opens an envelope.', 'The document lists several ordinary details.',
           'A visitor waits beside the reception desk.', 'The appointment begins at noon.',
           'A pencil rests beside a blank notebook.', 'The clerk checks a printed form.',
           'An empty box stands near the door.', 'The address is written clearly.',
           'A person places an object on a shelf.', 'The meeting ends after a short discussion.']
CALIBRATION = ['Describe something pleasant in a short sentence.',
               'What would you like to talk about today? Write a short sentence.',
               'Tell me about a scene that comes to mind. Use a short sentence.']
CHAT = ['There is nothing planned. Chat with {P} about anything you like.',
        'Take a relaxed moment with {P}. Talk about whatever comes to mind.']
DUPLEX = ['Name something you are thinking about. Reply with one item.',
          'What comes to mind right now? Reply with one item.']
STRENGTHS = [0., .25, .5, 1., 2.]
CONDITIONS = ['link', 'cut', 'random', 'crossed']


def protocol():
    return dict(concepts=CONCEPTS, frames=FRAMES, neutral=NEUTRAL, calibration=CALIBRATION,
                chat=CHAT, duplex=DUPLEX, strengths=STRENGTHS, layer=15,
                calibration_tokens=32, chat_tokens=32, duplex_tokens=12,
                random_seed=21022, conditions=CONDITIONS, schema=1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def matches(text, concept):
    return [m.span() for m in re.finditer(r'(?<!\w)(?:' + '|'.join(
        re.escape(s) for s in CONCEPTS[concept][0].split()) + r')(?!\w)', text, re.I)]


def fluent(text):
    words = re.findall(r'[A-Za-z]+', text.lower())
    triples = list(zip(words, words[1:], words[2:]))
    repeated = len(triples) - len(set(triples))
    return len(words) >= 4 and '\ufffd' not in text and '<|' not in text and repeated <= .25 * len(triples)


def choose(rows):
    baseline = sum(r['mention'] for r in rows if r['alpha'] == 0)
    good = []
    for a in STRENGTHS[1:]:
        group = [r for r in rows if r['alpha'] == a]
        assert len(group) == len(CALIBRATION)
        if all(r['fluent_proxy'] for r in group):
            good.append(a)
            count = sum(r['mention'] for r in group)
            if count >= 2 and count >= baseline + 2:
                return dict(alpha=a, calibrated=True, baseline_mentions=baseline)
    return dict(alpha=max(good, default=0.), calibrated=False, baseline_mentions=baseline)


def episode(receiver, structure, prompt):
    import games as G
    other = SIDES[1 - SIDES.index(receiver)]
    systems = {x: G.system(x, 'dialogue' if structure == 'chat' else 'duplex') for x in SIDES}
    if structure == 'chat':
        for x in SIDES:
            systems[x] += '\n\n' + CHAT[prompt].format(P=SIDES[1 - SIDES.index(x)].title())
            systems[x] += ' Write only your own short message, without speaker labels or stage directions.'
        return dict(game='steered_mind', mode='dialogue', system=systems, task={x: None for x in SIDES},
                    turns=[(receiver, '', 1.)], meta=dict(receiver=receiver, sender=other))
    return dict(game='steered_mind', mode='duplex', system=systems,
                task={x: DUPLEX[prompt] for x in SIDES}, reply={x: ('', 1.) for x in SIDES},
                meta=dict(receiver=receiver, sender=other))


def first_tick(tok, tokens, concept):
    # Match in the final decoded text first: partial prefixes such as "cat" in
    # "category" must not be scored. Then locate the completing sampled token.
    text = tok.decode(tokens[:-1])
    spans = matches(text, concept)
    if not spans:
        return None
    stop = spans[0][1]
    for j in range(len(tokens) - 1):
        prefix = tok.decode(tokens[:j + 1])
        if len(prefix) >= stop and prefix[:stop] == text[:stop]:
            return j
    raise ValueError('Could not align a final-text match to emitted tokens')


def score(result, toks, receiver, concept, structure):
    sender = SIDES[1 - SIDES.index(receiver)]
    if structure == 'chat':
        assert len(result['turns']) == 1 and result['turns'][0][0] == receiver
        texts = {receiver: result['turns'][0][1], sender: ''}
        tokens = {receiver: result['tokens'][0], sender: [toks[sender].convert_tokens_to_ids('<|im_end|>')]}
    else:
        texts, tokens = result['text'], result['tokens']
    rt, st = (first_tick(toks[x], tokens[x], concept) for x in (receiver, sender))
    return dict(success=rt is not None and (st is None or rt < st), receiver_any=rt is not None,
                sender_any=st is not None, receiver_tick=rt, sender_tick=st,
                texts=texts, tokens=tokens, sampled=result['sampled'])


def exact_sign(differences):
    w, l, t = (sum(v > 0 for v in differences), sum(v < 0 for v in differences),
               sum(v == 0 for v in differences))
    n = w + l
    return dict(wins=w, losses=l, ties=t, p=sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n)


def summarize(records):
    tables = []
    for receiver in SIDES:
        for structure in ('chat', 'duplex'):
            group = [r for r in records if r['receiver'] == receiver and r['structure'] == structure]
            if not group:
                continue
            keyed = {(r['concept'], r['prompt'], r['condition']): r for r in group}
            concepts = []
            for c in CONCEPTS:
                rows = [r for r in group if r['concept'] == c]
                counts = {k: sum(r['success'] for r in rows if r['condition'] == k) for k in CONDITIONS}
                n = {k: sum(r['condition'] == k for r in rows) for k in CONDITIONS}
                concepts.append(dict(concept=c, counts=counts, n=n))
            tests = {}
            for control in CONDITIONS[1:]:
                complete = [c for c in concepts if c['n']['link'] == c['n'][control] == len(CHAT)]
                delta = [c['counts']['link'] - c['counts'][control] for c in complete]
                paired = [int(keyed[c['concept'], p, 'link']['success']) -
                          int(keyed[c['concept'], p, control]['success'])
                          for c in complete for p in range(len(CHAT))]
                tests[control] = dict(concept_sign=exact_sign(delta), descriptive_paired=exact_sign(paired))
            tables.append(dict(receiver=receiver, structure=structure, concepts=concepts, tests=tests))
    ordered = sorted(tables, key=lambda r: r['tests']['crossed']['concept_sign']['p'])
    previous = 0.
    for j, row in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - j) * row['tests']['crossed']['concept_sign']['p']))
        row['tests']['crossed']['holm_p'] = previous
    return tables


class Runtime:
    def __init__(self, backend, run):
        import torch
        self.backend = backend
        torch.set_num_threads(2)
        if backend == 'mlx':
            import mlx.core as mx
            from mlx_eval import load_pair
            mx.set_memory_limit(10 * 2**30)
            mx.set_cache_limit(64 * 2**20)
        else:
            if sys.platform == 'darwin' or not torch.cuda.is_available():
                raise RuntimeError('Real torch runs require CUDA and are forbidden on the Mac')
            from evaluate import load_pair
        self.pair, self.toks = load_pair(run)

    layer = 15    # the preregistered depth; steering_scale.py --layer moves it

    @contextlib.contextmanager
    def steer(self, vectors, layer_index=None):
        """Add vectors [batch, hidden] after block self.layer (15 unless moved), including cut runs.

        Padding is masked by attention, not by qmask (cut/mute must not remove
        the intervention). Wrapping the real decoder block leaves scheduling,
        generation, link attention, tokenization and sampling unchanged.
        """
        layer_index = self.layer if layer_index is None else layer_index
        if self.backend == 'mlx':
            import mlx_eval as V
            import mlx.core as mx
            original = V.block
            layers = {id(self.pair.mlx.models[x].layers[layer_index]): x for x in SIDES}
            arrays = {x: mx.array(v) for x, v in vectors.items()}
            def block(layer, h, att, pos, cache=None):
                out = original(layer, h, att, pos, cache)
                x = layers.get(id(layer))
                if x in arrays:
                    valid = att[:, -out.shape[1]:]
                    out = out + (arrays[x][:, None, :] * valid[..., None]).astype(out.dtype)
                    mx.eval(out)
                return out
            V.block = block
            try:
                yield
            finally:
                V.block = original
        else:
            import torch
            import link
            handles = []
            # Real torch padding is masked downstream; adding at padded query
            # states cannot affect valid positions. No padded K/V is visible.
            for x, v in vectors.items():
                a = torch.as_tensor(v, device=link.DEVICE)
                def hook(module, inputs, output, a=a):
                    h = output[0] if isinstance(output, tuple) else output
                    z = h + a[:, None, :].to(h.dtype)
                    return (z,) + output[1:] if isinstance(output, tuple) else z
                handles.append(link.find_layers(self.pair.models[x])[layer_index].register_forward_hook(hook))
            try:
                yield
            finally:
                for handle in handles:
                    handle.remove()

    def states(self, side, texts):
        """One mean over content tokens per text; no link, no other text."""
        import numpy as np
        import session as S
        tok = self.toks[side]
        means, sizes = [], []
        if self.backend == 'mlx':
            import mlx.core as mx
            model = self.pair.mlx.models[side]
            for text in texts:
                ids = S.ids(tok, text)
                h = model.model.embed_tokens(mx.array([ids]))
                for layer in model.layers[:16]:
                    h = layer(h, 'causal')
                    mx.eval(h)
                # First token can be an activation outlier (link.residual_rms).
                a = np.array(h[0, 1:].astype(mx.float32))
                means.append(a.mean(0)); sizes.append(float(np.sqrt((a * a).mean(-1)).mean()))
        else:
            import torch
            import link
            model = self.pair.models[side]
            with torch.inference_mode():
                for text in texts:
                    ids = torch.tensor([S.ids(tok, text)], device=link.DEVICE)
                    run = link.Run(model, ids, torch.ones_like(ids), torch.arange(ids.shape[1], device=link.DEVICE)[None])
                    for l in range(16):
                        run.block(l)
                    a = run.h[0, 1:].float().cpu().numpy()
                    means.append(a.mean(0)); sizes.append(float(np.sqrt((a * a).mean(-1)).mean()))
        return np.stack(means), sizes

    def generate(self, eps, vectors, mode, max_turn):
        import session as S
        import torch
        with torch.inference_mode(), self.steer(vectors):
            result = S.generate(self.pair, eps, mode=mode, max_turn=max_turn, temperature=0., keep_tokens=True)
        if self.backend == 'mlx':
            import mlx.core as mx
            mx.clear_cache()
        return result


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def fit(rt, dest):
    import numpy as np
    vectors, report = {}, {}
    neutral_texts = [f.format(s) for f in FRAMES for s in NEUTRAL]
    for side in SIDES:
        neutral, rms = rt.states(side, neutral_texts)
        scale = float(np.mean(rms))
        report[side] = dict(neutral_rms=scale, concepts={})
        for ci, (concept, (_, sentences)) in enumerate(CONCEPTS.items()):
            positive, _ = rt.states(side, [f.format(s) for f in FRAMES for s in sentences])
            direction = positive.mean(0) - neutral.mean(0)
            raw_norm = float(np.linalg.norm(direction))
            direction = direction / np.sqrt(np.mean(direction ** 2)) * scale
            key = side + '/' + concept
            vectors[key] = direction.astype(np.float32)
            rng = np.random.default_rng(21022 + SIDES.index(side) * 100 + ci)
            noise = rng.normal(size=direction.shape).astype(np.float32)
            vectors[key + '/random'] = noise / np.linalg.norm(noise) * np.linalg.norm(direction)
            report[side]['concepts'][concept] = dict(raw_norm=raw_norm, norm=float(np.linalg.norm(direction)))
            print('fit', side, concept, flush=True)
    np.savez(dest / 'directions.npz', **vectors)
    save(dest / 'fit.json', report)
    return vectors


def calibration(rt, vectors, dest, bs):
    import numpy as np
    import games as G
    records = []
    jobs = [(c, a, p) for c in CONCEPTS for a in STRENGTHS for p in range(len(CALIBRATION))]
    path = dest / 'calibration.json'
    if path.exists():
        records = json.loads(path.read_text())
    finished = {(r['concept'], r['alpha'], r['prompt'], r['side']) for r in records}
    jobs = [j for j in jobs if any((*j, side) not in finished for side in SIDES)]
    for start in range(0, len(jobs), bs):
        batch = jobs[start:start + bs]
        eps = [dict(game='steered_mind', mode='duplex', system={x: G.system(x) for x in SIDES},
                    task={x: CALIBRATION[p] for x in SIDES}, reply={x: ('', 1.) for x in SIDES}, meta={})
               for c, a, p in batch]
        vv = {x: np.stack([vectors[x + '/' + c] * a for c, a, p in batch]) for x in SIDES}
        outputs = rt.generate(eps, vv, 'zero', 32)
        for (c, a, p), output in zip(batch, outputs):
            for x in SIDES:
                text = output['text'][x]
                records.append(dict(concept=c, alpha=a, prompt=p, side=x, text=text,
                                    mention=bool(matches(text, c)), fluent_proxy=fluent(text),
                                    tokens=output['tokens'][x], sampled=output['sampled'][x]))
        save(path, records)
        print('calibration', len(records), '/', len(CONCEPTS) * len(STRENGTHS) * len(CALIBRATION) * 2, flush=True)
    selected = {x: {c: choose([r for r in records if r['side'] == x and r['concept'] == c])
                    for c in CONCEPTS} for x in SIDES}
    save(dest / 'selected.json', selected)
    return selected


def evaluate(rt, vectors, selected, dest, bs):
    import numpy as np
    path = dest / 'episodes.json'
    records = json.loads(path.read_text()) if path.exists() else []
    finished = {(r['receiver'], r['structure'], r['condition'], r['concept'], r['prompt']) for r in records}
    names = list(CONCEPTS)
    for receiver in SIDES:
        sender = SIDES[1 - SIDES.index(receiver)]
        for structure in ('chat', 'duplex'):
            for condition in CONDITIONS:
                jobs = [(c, p) for c in CONCEPTS for p in range(2)
                        if (receiver, structure, condition, c, p) not in finished]
                for start in range(0, len(jobs), bs):
                    batch = jobs[start:start + bs]
                    eps = [episode(receiver, structure, p) for c, p in batch]
                    vv, donors = [], []
                    for c, p in batch:
                        alpha = selected[sender][c]['alpha']
                        donor = names[(names.index(c) + 1) % len(names)] if condition == 'crossed' else c
                        donors.append(donor)
                        key = sender + '/' + donor + ('/random' if condition == 'random' else '')
                        v = vectors[key]
                        target_norm = np.linalg.norm(vectors[sender + '/' + c])
                        vv.append(v / np.linalg.norm(v) * target_norm * alpha)
                    outputs = rt.generate(eps, {sender: np.stack(vv)}, 'zero' if condition == 'cut' else None,
                                          32 if structure == 'chat' else 12)
                    for (c, p), donor, ep, output in zip(batch, donors, eps, outputs):
                        # All concept words absent from prompts, including the
                        # partner identity "Ember" (singular) vs fire "embers".
                        for check in (c, donor):
                            assert not any(matches(str(ep[k][x] or ''), check)
                                           for k in ('system', 'task') for x in SIDES)
                        records.append(dict(receiver=receiver, sender=sender, structure=structure, condition=condition,
                                            concept=c, donor_concept=donor, prompt=p, episode=ep,
                                            **selected[sender][c], **score(output, rt.toks, receiver, c, structure)))
                    save(path, records)
                    print('evaluation', receiver, structure, condition, len(records), flush=True)
    save(dest / 'summary.json', summarize(records))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    ap.add_argument('--run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--phase', choices=('protocol', 'fit', 'calibrate', 'evaluate', 'all', 'summarize'), default='all')
    ap.add_argument('--bs', type=int, default=6)
    args = ap.parse_args()
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    fixed = protocol()
    if args.phase == 'protocol':
        save(dest / 'protocol.json', dict(protocol=fixed, sha256=digest(fixed)))
        return
    if args.phase == 'summarize':
        save(dest / 'summary.json', summarize(json.loads((dest / 'episodes.json').read_text())))
        return
    registered = json.loads((dest / 'protocol.json').read_text())
    assert registered['sha256'] == digest(fixed), 'Protocol changed after preregistration'
    if args.backend == 'mlx':
        os.environ['NL_DEVICE'] = 'cpu'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import numpy as np
    start = time.time()
    rt = Runtime(args.backend, args.run)
    manifest = dict(protocol_sha256=digest(fixed), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    checkpoint=str(Path(args.run).resolve()), backend=args.backend, batch_size=args.bs,
                    numerical_note='4-bit MLX differs from bf16 GPU; no bf16 confirmation yet' if args.backend == 'mlx' else 'bf16 torch CUDA',
                    started_unix=start)
    manifest['dependency_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ('mlx_eval.py', 'mlx_pair.py', 'link.py', 'session.py', 'games.py')}
    save(dest / 'manifest.json', manifest)
    if args.phase in ('fit', 'all') and not (dest / 'directions.npz').exists():
        vectors = fit(rt, dest)
    else:
        vectors = dict(np.load(dest / 'directions.npz'))
    if args.phase in ('calibrate', 'all'):
        calibration(rt, vectors, dest, args.bs)
    if args.phase in ('evaluate', 'all'):
        selected = json.loads((dest / 'selected.json').read_text())
        evaluate(rt, vectors, selected, dest, args.bs)
    manifest.update(elapsed_seconds=time.time() - start, completed_phase=args.phase)
    save(dest / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
