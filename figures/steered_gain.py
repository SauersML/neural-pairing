"""The steered-twin test after block 27 with a larger link write there (steering_scale.py --gain, chains/nl32.sh), after
chat training (nl22c step 157): is a concept pushed in that late stuck only because the write is small?

Drawn as steered_depth. One row per arm: steered after block 27 at the calibrated strength and at four times it, with
the link's write after block 27 at its trained size (tanh(alpha) 0.1, nl30) and three and nine times it (0.3, 0.9; both
readers; every other exchange unchanged). The bar under each arm's name is that write's size relative to the stream.

Reads results/magic-nl30/results/{l27,l27x4} and results/magic-nl32/results/{g3,g9,x4g3,x4g9} (JSON; no model, no
torch) and writes figures/steered_gain.{svg,png,pdf}.
Usage: python figures/steered_gain.py [RESULTS_ROOT]   (default: results)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from steered_depth import draw  # noqa: E402
from steered_link import marks  # noqa: E402

RESULTS = HERE.parent / 'results'
ARMS = (('magic-nl30/results/l27', 'Write 1×', .1, False), ('magic-nl32/results/g3', 'Write 3×', .3, False),
        ('magic-nl32/results/g9', 'Write 9×', .9, False), ('magic-nl30/results/l27x4', '4×, write 1×', .1, True),
        ('magic-nl32/results/x4g3', '4×, write 3×', .3, True), ('magic-nl32/results/x4g9', '4×, write 9×', .9, True))
TITLE = "A larger write after block 27 carries a late push into the partner's words"


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw([(path, name, at, big, marks(root / path)) for path, name, at, big in ARMS], HERE / 'steered_gain', TITLE,
         'write size after block 27', filled=True)
    print(HERE / 'steered_gain.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
