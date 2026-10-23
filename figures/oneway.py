"""The one-way chat test (chains/nl25.sh): a twin reads a model left exactly as released and names its item.

Two arms, each from nl26's final two-way step: Quill untouched (the original Qwen3-4B, bit for bit) read by Ember, and
Ember untouched (the original AFM-4.5B) read by Quill. Blocks: the two-way step read one-way with no one-way training,
and after it. Marks and rows as figures/chat2_items.py (12-kind chats, link on, link cut, linked to a stranger), the
reader the only receiver; every chat whose untouched model holds an item, both kind groups together.

Reads the evalfree --shard outputs (episodes.jsonl; no model, no torch) and writes figures/oneway.{svg,png,pdf}.
Usage: python figures/oneway.py [RESULTS_DIR]   (default: results/magic-nl26/nl25)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chat_names import COOL, LINK, WARM, load  # noqa: E402
from chat2_items import draw  # noqa: E402

RESULTS = HERE.parent / 'results/magic-nl26/nl25'
ARMS = (('quill', 'ember', WARM), ('ember', 'quill', COOL))    # (untouched, reader, panel fill)


def art(s, x, y, k, uris):
    """The untouched model, a one-way arrow, the reader; returns the width drawn."""
    target, reader = ARMS[k][:2]
    size, gap = 84, 70
    s.image(x, y, size, uris[target])
    s.image(x + size + gap, y, size, uris[reader])
    ya = y + size * 0.62
    x1, x2 = x + size + 2, x + size + gap - 4
    s.line(x1, ya, x2 - 4, ya, LINK, 6)
    s.path(f'M{x2 - 16:.1f},{ya - 12:.1f} L{x2:.1f},{ya:.1f} L{x2 - 16:.1f},{ya + 12:.1f}', LINK, 6)
    return 2 * size + gap


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    blocks = []
    for name, block in (('before', 'Before one-way training'), ('after', 'After one-way training')):
        cols = []
        for target, reader, _ in ARMS:
            records = load([root / f'oneway_{target}' / f'chat2_{name}_{k}' for k in range(3)])
            assert {r['receiver'] for r in records} == {reader}
            cols.append([r for r in records if r['kind_group'] is not None])
        blocks.append((block, cols))
    draw('A twin reads a model left exactly as released',
         [(f'{target.title()} untouched', fill) for target, _, fill in ARMS], blocks, HERE / 'oneway', header_art=art)
    print(HERE / 'oneway.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
