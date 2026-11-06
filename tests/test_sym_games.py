"""swarm/sym_games.py on a tiny random Qwen3 sym net on the CPU: per game the gaps are its episodes' own (the swap gap
over all of them is dev_nll's swap minus on), the notes counted are every writer's, and trail's probe is note_stats'."""
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402
from swarm import sym_games as SG                       # noqa: E402

from test_crumbs import lora_effect, tiny               # noqa: E402


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def test_per_game_gaps_and_notes(tok):
    torch.manual_seed(0)
    net = tiny(tok, 'sym', 2)
    lora_effect(net)
    games, n = ('trail', 'count'), 3
    res = SG.score(net, 'cpu', games, n)
    assert set(res) == {'trail', 'count', 'all', 'trail_probe'}
    rng = random.Random(777)
    eps = [CG.episode(g, rng) for g in games for _ in range(n)]
    with torch.no_grad():
        on, swap = C.dev_nll(net, 'cpu', eps), C.dev_nll(net, 'cpu', eps, mode='swap')
        cut = C.dev_nll(net, 'cpu', eps, mode='off')
        vals = C.dev_values(net, 'cpu', eps)
    assert res['all']['episodes'] == sum(v is not None for v in vals)
    assert res['all']['gap_swap'] == pytest.approx(swap - on, abs=2e-4)
    assert res['all']['gap_cut'] == pytest.approx(cut - on, abs=2e-4)
    lo, hi = res['all']['gap_swap_ci']
    assert lo <= res['all']['gap_swap'] + 1e-4 and res['all']['gap_swap'] <= hi + 1e-4
    for g in games:
        rows = [v for ep, v in zip(eps, vals) if ep['game'] == g and v is not None]
        assert res[g]['episodes'] == len(rows) and res[g]['on'] == pytest.approx(sum(rows) / len(rows), abs=2e-4)
    notes = SG.notes_of(net, 'cpu', eps)
    assert [len(x) for x in notes] == [sum(1 for w in ep['agents'] if w['write'] for s in w['segments']
                                           if s['place'] is not None) for ep in eps]
    assert res['trail']['notes'] == sum(len(x) for ep, x in zip(eps, notes) if ep['game'] == 'trail')
    assert 1 <= res['count']['distinct'] <= res['count']['notes'] and 0 <= res['trail']['letters_text'] <= 1
    probe = C.note_stats(net, 'cpu', [ep for ep in eps if ep['game'] == 'trail'])
    assert res['trail_probe']['examples'] == probe['examples'] and res['trail_probe']['distinct'] == probe['distinct']
