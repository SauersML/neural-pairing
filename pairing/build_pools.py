"""Builds data/pools.json: the vocabularies the games draw from, each split into training and held-out parts.

Secret words: lowercase dictionary words (/usr/share/dict/words) of 4 to 9 letters that both tokenizers write as one
token with a leading space, so neither model's tokenizer favours a word; a fifth is held out. Questions: the questions
of pairing/tasks.json, split 75/25 (seed 0). Concepts: concepts.CONCEPTS, with concepts.TEST_CONCEPTS held out.
Adjectives, names and cities: the lists below, a quarter of each held out.

Usage: python build_pools.py   (on a machine with /usr/share/dict/words)
"""
import json
import os
import random
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import concepts as K  # noqa: E402
from link import MODELS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = random.Random(0)

toks = [AutoTokenizer.from_pretrained(MODELS[x]['hf']) for x in MODELS]
dict_words = sorted({w.strip() for w in open('/usr/share/dict/words') if w.strip().isalpha() and w.strip().islower()
                     and 4 <= len(w.strip()) <= 9})
one = [w for w in dict_words if all(len(t(' ' + w, add_special_tokens=False).input_ids) == 1 for t in toks)]
rng.shuffle(one)
nt = len(one) // 5

ADJ = ('red orange yellow green blue purple pink black white silver golden tiny giant ancient frozen burning '
       'sleepy angry happy quiet noisy shy brave clever lazy hungry wooden glass velvet striped spotted invisible '
       'flying dancing singing broken shiny rusty soft heavy hollow electric').split()
NAMES = ('Alice Bruno Chloe Diego Elena Farid Grace Hiro Ingrid Jamal Keiko Liam Maya Nikolai Olga Pablo Quinn Rosa '
         'Sven Tara Umar Vera Wei Xena Yusuf Zara Amara Boris Carmen Dmitri Esme Felix Greta Hassan Isla Jonas Kira '
         'Lars Mina Nadia Omar Priya Rafael Sofia Tomas Uma Viktor Wanda Yara Zane Anya Bodhi Celine Darius Elif '
         'Finn Gaia Hugo Ines Joel Kofi Lena Mateo Noor Oskar Paloma Rhea Sami Talia Ulrich Valentina Wren '
         'Yuki').split()
CITIES = ('Paris Tokyo Cairo Lima Oslo Rome Delhi Seoul Lagos Dublin Vienna Prague Madrid Berlin Boston Denver '
          'Sydney Toronto Nairobi Istanbul Bangkok Hanoi Manila Jakarta Havana Bogota Santiago Quito Lisbon Athens '
          'Warsaw Budapest Helsinki Stockholm Copenhagen Brussels Zurich Geneva Milan Naples Venice Florence Munich '
          'Hamburg Krakow Riga Tallinn Vilnius Kyiv Tbilisi Baku Tehran Dubai Doha Muscat Karachi Lahore Dhaka '
          'Kathmandu Colombo Chennai Mumbai Kolkata Shanghai Beijing Osaka Kyoto Busan Taipei Singapore Perth Auckland '
          'Montreal Vancouver Seattle Chicago Phoenix Dallas Miami Atlanta Accra Dakar Tunis Algiers '
          'Casablanca').split()
tasks = json.load(open(os.path.join(os.path.dirname(__file__), 'tasks.json')))
order = list(range(len(tasks)))
random.Random(0).shuffle(order)
q_train = sorted(order[:int(0.75 * len(order))])
q_test = sorted(order[int(0.75 * len(order)):])
test_c = set(K.TEST_CONCEPTS)


def split(xs, frac=0.25, seed=1):
    """{'train': ..., 'test': ...}: a random fraction frac of xs is the test part."""
    xs = list(xs)
    random.Random(seed).shuffle(xs)
    n = int(len(xs) * frac)
    return {'train': sorted(xs[n:]), 'test': sorted(xs[:n])}


pools = {
    'words': {'train': sorted(one[nt:]), 'test': sorted(one[:nt])},
    'adjectives': split(ADJ),
    'names': split(NAMES),
    'cities': split(CITIES),
    'questions': {'train': [tasks[i] for i in q_train], 'test': [tasks[i] for i in q_test]},
    'concepts': {'train': [[n, p] for i, (n, p, _) in enumerate(K.CONCEPTS) if i not in test_c],
                 'test': [[n, p] for i, (n, p, _) in enumerate(K.CONCEPTS) if i in test_c]},
}
os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
json.dump(pools, open(os.path.join(ROOT, 'data', 'pools.json'), 'w'), indent=1)
print({k: {s: len(v[s]) for s in v} for k, v in pools.items()})
print(pools['words']['test'][:30])
