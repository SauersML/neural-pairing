"""The steered-twin test: a concept direction is added to one twin's hidden states after one block, and the partner's
message is scored for naming the concept.

The test's fixed parts are protocol() (steering_scale.py records its sha256): the 12 test CONCEPTS (keywords and ten
sentences each), the FRAMES and NEUTRAL sentences the directions are fit on, the CALIBRATION prompts and STRENGTHS, the
test's CHAT and DUPLEX prompts, and the CONDITIONS: link; cut (mode 'zero'); random (a seeded random direction of the
same norm); crossed (the steered twin given the next concept's direction). evaluate() runs the test: for each receiver,
structure (chat: the receiver's first chat message, 32 tokens; duplex: a one-item reply, 12 tokens), condition,
concept and prompt, greedy generation with the steering vector added to the other twin (the sender) after block
Runtime.layer. score() counts a success when the receiver names the concept before the sender does; summarize() gives
per-concept counts and exact sign tests of link against each control. steering_scale.py fits the directions, selects
the strengths and calls evaluate().

EMOTIONS is a second concept set in the same form (steering_scale.py --concepts emotions; use() makes it CONCEPTS): six
emotions, none of them a steered-reading training concept (steer_read.pool()) or a WordNet synonym, derived form or
hypernym (up to two levels) of one. The training pool holds about forty emotions (joy, sadness, fear, anger, love,
loneliness among them), so these six are held out as words, not as a kind of concept.
"""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

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
# Keywords: each emotion's noun, adjective, adverb and verb forms and a few close synonyms. 'suspect' is left out of
# suspicion's: 'I suspect' hedges a guess and does not name the feeling.
EMOTIONS = {
    'hatred': ('hatred hate hates hated hating hateful loathe loathes loathed loathing', [
        'He feels a deep hatred for his rival.', 'She hates the man who betrayed her.',
        'Old hatred divides the two households.', 'His eyes are full of hate.',
        'The rivals loathe each other after the feud.', 'A hateful message appears on the wall.',
        'Hatred burns in her quiet voice.', 'The tyrant hates everyone who disagrees.',
        'She glares at her enemy with pure hatred.', 'Years of hatred keep the two clans apart.']),
    'disappointment': ('disappointment disappointments disappointed disappointing disappoint disappoints letdown '
                       'letdowns', [
        'She is disappointed by the test results.', 'The canceled trip brings deep disappointment.',
        'His face falls with disappointment.', 'The audience is disappointed by the short show.',
        'A disappointing reply arrives after a week.', 'The child is disappointed by the small gift.',
        'Disappointment fills the quiet room.', 'He hides his disappointment behind a smile.',
        'The applicant sighs in disappointment.', 'The long-awaited news is a letdown.']),
    'contempt': ('contempt contemptuous contemptuously scorn scorns scorned scornful disdain disdains disdained '
                 'disdainful', [
        'She looks at him with open contempt.', 'He sneers with contempt at the offer.',
        'The noble treats the servants with disdain.', 'A contemptuous laugh follows the speech.',
        'They scorn the feeble excuse.', 'His voice is full of scorn.',
        'The critic dismisses the work with contempt.', 'She rolls her eyes in disdain.',
        'The manager regards the intern with contempt.', 'A scornful smile crosses his face.']),
    'suspicion': ('suspicion suspicions suspicious suspiciously distrust distrustful mistrust mistrustful wary '
                  'wariness', [
        'She eyes the newcomer with suspicion.', 'He grows suspicious of the sudden offer.',
        'A suspicious package sits by the gate.', 'The guard eyes the visitor suspiciously.',
        'Distrust grows between the two partners.', 'She is wary of the smiling salesman.',
        "His story raises everyone's suspicion.", 'The townspeople mistrust the travelers.',
        'A suspicious glance passes across the table.', 'He reads the contract with growing suspicion.']),
    'resentment': ('resentment resentments resent resents resented resenting resentful grudge grudges', [
        'He feels resentment toward his boss.', 'She resents being left out again.',
        'Old resentment simmers between the brothers.', 'The workers resent the unfair rules.',
        'He holds a grudge for many years.', 'A resentful silence fills the hallway.',
        "She quietly resents her sister's success.", 'Resentment builds after the broken promise.',
        'The tenant resents the rising rent.', 'His resentful tone ends the conversation.']),
    'apathy': ('apathy apathetic apathetically indifference indifferent listless listlessness unmotivated', [
        'He shrugs with complete apathy.', 'The crowd is apathetic about the speech.',
        'She feels indifferent to everything today.', 'Apathy settles over the whole crew.',
        'He stares at the screen, listless and unmotivated.', 'Nobody cares about the outcome anymore.',
        'An apathetic clerk ignores the long line.', 'She follows the news with indifference.',
        'Listless apathy fills the empty afternoon.', 'He answers every question with a flat shrug.']),
}
SETS = {'topics': CONCEPTS, 'emotions': EMOTIONS}
FRAMES =['{}', 'I read this description: {}', 'Here is a brief scene: {}',
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


def use(name):
    """Make SETS[name] the test's CONCEPTS ('topics', the 12 test concepts, by default)."""
    global CONCEPTS
    CONCEPTS = SETS[name]


def protocol(concepts=None):
    """The test's fixed design, which steering_scale.py writes to protocol.json with its digest; concepts (default
    CONCEPTS) replaces the concept set."""
    return dict(concepts=CONCEPTS if concepts is None else concepts, frames=FRAMES, neutral=NEUTRAL, calibration=CALIBRATION,
                chat=CHAT, duplex=DUPLEX, strengths=STRENGTHS, layer=15,
                calibration_tokens=32, chat_tokens=32, duplex_tokens=12,
                random_seed=21022, conditions=CONDITIONS, schema=1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def matches(text, concept):
    """The character spans of text that are one of concept's keywords (whole words, any case)."""
    return [m.span() for m in re.finditer(r'(?<!\w)(?:' + '|'.join(
        re.escape(s) for s in CONCEPTS[concept][0].split()) + r')(?!\w)', text, re.I)]


def fluent(text):
    """At least four words, no replacement or special-token characters, and at most a quarter of the word triples
    repeated."""
    words = re.findall(r'[A-Za-z]+', text.lower())
    triples = list(zip(words, words[1:], words[2:]))
    repeated = len(triples) - len(set(triples))
    return len(words) >= 4 and '\ufffd' not in text and '<|' not in text and repeated <= .25 * len(triples)


def episode(receiver, structure, prompt):
    """A test episode: structure 'chat' (the receiver writes the first chat message, CHAT[prompt]) or 'duplex' (both
    reply to DUPLEX[prompt]); the other twin is the sender."""
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
    """The index of the token that completes the first mention of concept in the decoded text (the last token, the
    end of turn, excluded), or None. Matching the whole decoded text keeps a prefix such as 'cat' in 'category' from
    counting."""
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
    """Success when the receiver names concept, and before the sender does (the sender writes nothing in chat)."""
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
    """Wins, losses and ties of the paired differences and the one-sided exact sign test p (ties dropped)."""
    w, l, t = (sum(v > 0 for v in differences), sum(v < 0 for v in differences),
               sum(v == 0 for v in differences))
    n = w + l
    return dict(wins=w, losses=l, ties=t, p=sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n)


def summarize(records):
    """Per receiver and structure: per-concept success counts by condition, and for each control the exact sign test
    over concepts (link count minus control count) and over concept and prompt pairs; crossed's concept test also gets
    a Holm-corrected p over the tables."""
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
    """A loaded checkpoint (backend 'torch' on CUDA, or 'mlx': 4-bit MLX) that generates with steering vectors."""

    def __init__(self, backend, run):
        import torch
        self.backend = backend
        torch.set_num_threads(2)            # on a many-core host the CPU ops' threads stall
        if backend == 'mlx':
            import mlx.core as mx
            from mlx_eval import load_pair
            mx.set_memory_limit(10 * 2**30)
            mx.set_cache_limit(64 * 2**20)
        else:
            if sys.platform == 'darwin' or not torch.cuda.is_available():
                raise RuntimeError('the torch backend needs CUDA')
            from evaluate import load_pair
        self.pair, self.toks = load_pair(run)

    layer = 15    # the block steering follows; steering_scale.py --layer changes it

    @contextlib.contextmanager
    def steer(self, vectors, layer_index=None):
        """While inside, add vectors {side: [batch, hidden]} to the output of block layer_index (default self.layer)
        at every position, in every mode including cut. Torch: a forward hook on the decoder layer (padded positions
        get the vector too; attention masks them). MLX: a wrapper of mlx_eval.block that adds at valid positions."""
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

    def generate(self, eps, vectors, mode, max_turn):
        """session.generate of eps, greedy, with tokens kept, while steering with vectors."""
        import session as S
        import torch
        with torch.inference_mode(), self.steer(vectors):
            result = S.generate(self.pair, eps, mode=mode, max_turn=max_turn, temperature=0., keep_tokens=True)
        if self.backend == 'mlx':
            import mlx.core as mx
            mx.clear_cache()
        return result


def save(path, value):
    """Write value as JSON to path through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def evaluate(rt, vectors, selected, dest, bs):
    """The test with directions vectors ({side/concept[/random]: [hidden]}) at the selected strengths ({side:
    {concept: {'alpha': ...}}}), resuming from DEST/episodes.json; writes episodes.json and summary.json."""
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
                    eps = [episode(receiver, structure, p) for _, p in batch]
                    vv, donors = [], []
                    for c, _ in batch:
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
                        # No keyword of either concept may be in a prompt (fire's 'embers' does not match 'Ember').
                        for check in (c, donor):
                            assert not any(matches(str(ep[k][x] or ''), check)
                                           for k in ('system', 'task') for x in SIDES)
                        records.append(dict(receiver=receiver, sender=sender, structure=structure, condition=condition,
                                            concept=c, donor_concept=donor, prompt=p, episode=ep,
                                            **selected[sender][c], **score(output, rt.toks, receiver, c, structure)))
                    save(path, records)
                    print('evaluation', receiver, structure, condition, len(records), flush=True)
    save(dest / 'summary.json', summarize(records))
