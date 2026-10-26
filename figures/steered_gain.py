"""The steered-twin test after layer 27 with a larger link write there (pairing/steering_scale.py --gain, chains/nl32.sh),
after chat training (nl22c step 157): does the partner say a concept steered in that late when the write is larger?

Drawn as figures/steered_depth.py. One row per arm: steered after layer 27 at the calibrated steering coefficient and
at four times it, with the link's write after layer 27 at its trained size (tanh(alpha) 0.1, nl30) and three and nine
times it (0.3, 0.9; both readers; every other exchange unchanged).

Reads results/magic-nl30/results/{l27,l27x4} and results/magic-nl32/results/{g3,g9,x4g3,x4g9} (JSON; no model) and
writes figures/steered_gain.{svg,png,pdf}.
Usage: python figures/steered_gain.py [RESULTS_ROOT]   (default: results)
"""
import sys
from pathlib import Path

from steered_depth import draw, marks

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
ARMS = (('magic-nl30/results/l27', 'Write 1×', False), ('magic-nl32/results/g3', 'Write 3× larger', False),
        ('magic-nl32/results/g9', 'Write 9× larger', False), ('magic-nl30/results/l27x4', '4× stronger,\nwrite 1×', True),
        ('magic-nl32/results/x4g3', '4× stronger,\nwrite 3× larger', True),
        ('magic-nl32/results/x4g9', '4× stronger,\nwrite 9× larger', True))
TITLE = ('Before training, the partner says a concept steered after layer 27',
         'only with a 3× or 9× larger link write')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    draw([(name, big, marks(root / path)) for path, name, big in ARMS], HERE / 'steered_gain', TITLE)
    print(HERE / 'steered_gain.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
