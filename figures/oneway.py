"""The one-way chat test (chains/nl25.sh): a twin reads a model left exactly as released and says its item.

Two arms, each from nl26's final two-way step: Quill untouched (the original Qwen3-4B, bit for bit) read by Ember, and
Ember untouched (the original AFM-4.5B) read by Quill. Groups: the two-way step read one-way with no one-way training,
and after one-way training. Counts and rows as in figures/chat2_items.py (12-kind chats, link on, link cut, linked to
a partner from another conversation), the reader the only receiver; every chat whose untouched model has an item, both
kind groups together.

Reads the evalfree --shard outputs (episodes.jsonl; no model) and writes figures/oneway.{svg,png,pdf}.
Usage: python figures/oneway.py [RESULTS_DIR]   (default: results/magic-nl26/nl25)
"""
import sys
from pathlib import Path

from chat_names import COOL, LINK, WARM, load
from chat2_items import ART_H, draw

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results/magic-nl26/nl25'
ARMS = (('quill', 'ember', WARM), ('ember', 'quill', COOL))    # (untouched, reader, panel fill)


def art(target, reader):
    """A drawer of the untouched model, a one-way arrow and the reader; it returns the width drawn."""
    def draw_art(s, x, y, uris):
        size, gap = ART_H, 90
        s.image(x, y, size, uris[target])
        s.image(x + size + gap, y, size, uris[reader])
        ya = y + size * 0.62
        x1, x2 = x + size + 2, x + size + gap - 4
        s.line(x1, ya, x2 - 4, ya, LINK, 6)
        s.path(f'M{x2 - 16:.1f},{ya - 12:.1f} L{x2:.1f},{ya:.1f} L{x2 - 16:.1f},{ya + 12:.1f}', LINK, 6)
        return 2 * size + gap
    return draw_art


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    cards = []
    for target, reader, fill in ARMS:
        cols = []
        for name, sub in (('before', 'Before\none-way\ntraining'), ('after', 'After\none-way\ntraining')):
            records = load([root / f'oneway_{target}' / f'chat2_{name}_{k}' for k in range(3)])
            assert {r['receiver'] for r in records} == {reader}
            cols.append((sub, [r for r in records if r['kind_group'] is not None]))
        cards.append((f'{reader.title()} reads untouched {target.title()}', fill, art(target, reader), None, cols))
    draw('A twin says the private item of a model left exactly as released', cards, HERE / 'oneway', W=2000)
    print(HERE / 'oneway.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
