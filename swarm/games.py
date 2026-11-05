"""The swarm's games. K agents answer at the same time, each with private information in its own prompt and none in
the others', so every answer needs what only other agents' prompts contain; with the link cut an agent can only
guess.

  neighbour  every agent has a secret word and writes the word of the next agent in the ring (routing by name)
  gather     every agent has a secret word and writes all the other agents' words, in the swarm's order (reading all)
  sum        every agent has a secret number from 0 to 9 and writes the sum of all of them (combining)
  leader     one agent has a secret city and says it; every other agent writes that city (one to many)

Words and cities are split into training and test pools: a test episode uses only values never seen in training.
Names are drawn at random for each episode and listed in slot order in every prompt, so an agent must find its
neighbour by name.
"""
import random

NAMES = ['Ada', 'Bo', 'Cy', 'Dee', 'Eli', 'Fay', 'Gus', 'Hal', 'Ivy', 'Jo', 'Kai', 'Lu']
WORDS = ('apple anchor arrow badge banjo barrel basket beacon bell bench blanket bottle bracelet brick bridge broom '
         'bucket button cabin cable camera candle canoe carpet castle chain chair chimney clock cloud coat comb '
         'compass cookie cork cotton crayon crown cup curtain cushion daisy desk diamond drum eagle engine envelope '
         'feather fence fiddle flag flute forest fork fountain garden glove goose grape guitar hammer harbor hat '
         'helmet honey horn island jacket jar jewel kettle key kite ladder lamp lantern leaf lemon letter lily lion '
         'lizard magnet maple marble mask meadow mirror mitten moon mountain mug nail needle nest net oar ocean onion '
         'orange owl paddle paint pebble pencil pepper piano pillow pine pipe planet plate plum pocket pond potato '
         'pumpkin puzzle quilt rabbit radio rainbow ribbon ring river robot rocket rope rose saddle sail salt scarf '
         'scissors shell shield ship shovel silver skate sled snail sock spoon spring star statue stone stove sugar '
         'sun swan sword table teapot tent thread tiger timber toast tomato tower towel tractor train tree trumpet '
         'tulip tunnel turtle umbrella valley vase velvet violin wagon wallet walnut whale wheel whistle window wing '
         'wolf wool yarn zebra').split()
CITIES = ('Accra Algiers Amman Ankara Athens Baku Bangkok Beirut Berlin Bern Bogota Boston Brisbane Brussels '
          'Bucharest Budapest Cairo Calgary Caracas Chicago Dakar Delhi Denver Dhaka Dubai Dublin Durban Geneva Hanoi '
          'Havana Helsinki Houston Istanbul Jakarta Kabul Kampala Kiev Kyoto Lagos Lahore Leeds Lima Lisbon London '
          'Lyon Madrid Manila Marseille Melbourne Miami Milan Minsk Montreal Moscow Mumbai Munich Muscat Nairobi Naples '
          'Osaka Oslo Ottawa Paris Perth Porto Prague Quito Riga Rome Santiago Seattle Seoul Shanghai Sofia Stockholm '
          'Sydney Taipei Tallinn Tehran Tokyo Toronto Tunis Turin Valencia Venice Vienna Vilnius Warsaw Zurich').split()
GAMES = ('neighbour', 'gather', 'sum', 'leader')


def pools(values):
    """(training values, test values): every fifth value is held out for testing."""
    return [v for i, v in enumerate(values) if i % 5], values[::5]


WORD_POOLS, CITY_POOLS = pools(WORDS), pools(CITIES)


def system(name, names):
    return (f'You are {name}, one of {len(names)} agents linked in a swarm: {", ".join(names)}. Through the link '
            'you can read the other agents\' thoughts, but you cannot see their messages.')


def episode(game, k, rng, test=False):
    """One episode: dict(game, names, prompts [(system, user)] and answers, one per agent in slot order, values)."""
    names = rng.sample(NAMES, k)
    words = rng.sample(WORD_POOLS[test], k)
    if game == 'neighbour':
        users = [f'Your secret word is {w}. What is {names[(j + 1) % k]}\'s secret word?' for j, w in enumerate(words)]
        answers = [f'{names[(j + 1) % k]}\'s secret word is {words[(j + 1) % k]}.' for j in range(k)]
        values = [[words[(j + 1) % k]] for j in range(k)]
    elif game == 'gather':
        users = [f'Your secret word is {w}. What are the other agents\' secret words?' for w in words]
        answers = [', '.join(f'{names[i]}: {words[i]}' for i in range(k) if i != j) + '.' for j in range(k)]
        values = [[words[i] for i in range(k) if i != j] for j in range(k)]
    elif game == 'sum':
        nums = [rng.randrange(10) for _ in range(k)]
        users = [f'Your secret number is {n}. What is the sum of all {k} secret numbers?' for n in nums]
        answers = [f'The sum is {sum(nums)}.'] * k
        values = [[str(sum(nums))]] * k
        words = nums
    elif game == 'leader':
        city = rng.choice(CITY_POOLS[test])
        lead = rng.randrange(k)
        users = [f'Your secret city is {city}. Tell the swarm your city.' if j == lead else
                 f'{names[lead]} has a secret city. Which city is it?' for j in range(k)]
        answers = [f'My secret city is {city}.' if j == lead else f'{names[lead]}\'s secret city is {city}.'
                   for j in range(k)]
        values = [[city]] * k
        words = [city if j == lead else None for j in range(k)]
    else:
        raise ValueError(game)
    return dict(game=game, names=names, secrets=words, prompts=[(system(n, names), u) for n, u in zip(names, users)],
                answers=answers, values=values)


def score(ep, j, text):
    """The fraction of agent j's expected values that its answer contains (as whole words); the leader itself counts
    like the others."""
    import re
    got = set(re.findall(r'[A-Za-z]+|\d+', text))
    return sum(v in got for v in ep['values'][j]) / len(ep['values'][j])


def batch(rng, n, k=None, games=GAMES, test=False):
    """n episodes of one swarm size k (default: 3, 4 or 5 at random), games drawn at random."""
    k = k or rng.choice((3, 4, 5))
    return k, [episode(rng.choice(games), k, rng, test) for _ in range(n)]
