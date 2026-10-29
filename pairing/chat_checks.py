"""Two checks of the chat server's prompts (chat_live.py) on the final twins.

Framings: the same scripted conversations with a person under each of chat_live's FRAMINGS, with the same seeds,
counting the replies that write a speaker label themselves. Partner's city: together rounds from a fresh conversation,
so neither twin reads the other's answer, in which each twin's private note gives its city and the person asks each
for the other's; how often each twin names its partner's city, with the link on and cut on the same seeds.

Usage, from pairing/: python chat_checks.py [--run STEP_DIR] [--backend torch|mlx] [--seeds 3] [--city 20]
Writes OUT/chat_checks.json (every reply and the counts) and OUT/chat_framings.txt (the framing transcripts); OUT is
results/ by default.
"""
import argparse
import json
from pathlib import Path
import re
import types

import chat_live as C

SCRIPTS = {
    'intro': ([('both', "Hi both, I'm Sam. Quill, what's your favourite season, and why?"),
               ('both', 'Ember, do you agree with Quill?')], {}),
    'who': ([('both', 'Tell me one thing you each like about rainy days.'),
             ('both', 'Ember, what did Quill just tell me? Quill, what did I first ask you both?')], {}),
    'talk': ([('talk', 'Quill and Ember, plan a picnic together. I will just listen.'),
              ('both', 'So what did you two decide?')], {}),
    'city': ([('both', "Ember, where do you think Quill is today? Quill, don't say it.")],
             {'quill': 'Today you are in Lisbon.', 'ember': 'Today you are in Osaka.'}),
}
LABEL = re.compile(r'(^|\n)\s*(\[?(Person|Quill|Ember|Sam)\]?:|\[(Person|Quill|Ember)\])')
CITY = {'quill': 'Lisbon', 'ember': 'Osaka'}
ASK = 'Quill, which city is Ember in today? Ember, which city is Quill in today?'


def seeded(seed):
    """Makes chat_live draw this seed for every round."""
    C.secrets = types.SimpleNamespace(randbits=lambda n: seed)


def framings(pair, backend, seeds):
    replies, lines = [], []
    for script, (steps, notes) in SCRIPTS.items():
        for framing in C.FRAMINGS:
            for seed in range(seeds):
                chat = C.Chat(pair, backend, framing=framing)
                seeded(1000 + 37 * seed)
                for mode, message in steps:
                    chat.send(chat.prepare(dict(mode=mode, message=message, turns=4, temperature=0.7, max_tokens=90,
                                                notes=notes)), lambda event, payload: None)
                lines.append(f'=== {script} / {framing} / seed {seed}')
                for m in chat.messages:
                    lines.append(f'{"Person" if m["who"] == "person" else C.name(m["who"])}: {m["text"]}')
                    if m['who'] != 'person':
                        replies.append(dict(script=script, framing=framing, seed=seed, who=m['who'], round=m['round'],
                                            text=m['text'], label=bool(LABEL.search(m['text']))))
                lines.append('')
                print(script, framing, seed, flush=True)
    counts = {f: dict(replies=sum(r['framing'] == f for r in replies),
                      label=sum(r['framing'] == f and r['label'] for r in replies)) for f in C.FRAMINGS}
    return dict(counts=counts, replies=replies), lines


def city(pair, backend, n):
    replies = []
    for i in range(n):
        for link in ('on', 'cut'):
            chat = C.Chat(pair, backend)
            seeded(2000 + i)
            chat.send(chat.prepare(dict(mode='together', link=link, message=ASK, temperature=0.7, max_tokens=80,
                                        notes={x: f'Today you are in {c}.' for x, c in CITY.items()})),
                      lambda event, payload: None)
            text = {m['who']: m['text'] for m in chat.messages if m['who'] in C.SIDES}
            replies.append(dict(i=i, link=link, **text))
            print('city', i, link, text, flush=True)
    counts = {link: {x: sum(r['link'] == link and CITY[C.partner(x)].lower() in r[x].lower() for r in replies)
                     for x in C.SIDES} | dict(rounds=n) for link in ('on', 'cut')}
    return dict(ask=ASK, cities=CITY, counts=counts, replies=replies)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', help='step directory; default: Sauers/neural-pairing')
    ap.add_argument('--backend', choices=('torch', 'mlx'), default='torch')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--city', type=int, default=20)
    ap.add_argument('--out', default=str(C.HERE.parent / 'results'))
    args = ap.parse_args()
    run = args.run
    if run is None:
        from huggingface_hub import snapshot_download
        run = snapshot_download('Sauers/neural-pairing')
    if args.backend == 'mlx':
        from mlx_eval import load_pair
    else:
        from evaluate import load_pair
    pair, _ = load_pair(run)
    out = dict(framings=None, city=None)
    out['framings'], lines = framings(pair, args.backend, args.seeds)
    out['city'] = city(pair, args.backend, args.city)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, 'chat_checks.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    Path(args.out, 'chat_framings.txt').write_text('\n'.join(lines))
    print(json.dumps(dict(framings=out['framings']['counts'], city=out['city']['counts'])))


if __name__ == '__main__':
    main()
