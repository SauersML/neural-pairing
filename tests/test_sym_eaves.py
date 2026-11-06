"""swarm/sym_eaves.py on a tiny random Qwen3 sym net on the CPU: the trained and base readers' numbers are crumbs.py's
forced modes on the same notes, a fresh eavesdropper reads exactly as the base model, the swap is within the game,
the cracker trains only the eavesdropper adapter, and the run's own eavesdropper comes back after the crackers."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumbs as C                           # noqa: E402
from swarm import sym_eaves as SE                       # noqa: E402

from test_crumbs import lora_effect, tiny               # noqa: E402


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def test_readers_and_cracker(tok):
    torch.manual_seed(0)
    net = tiny(tok, 'sym', 2)
    lora_effect(net)                                   # both adapters (the run's and its eavesdropper) nonzero
    eps = SE.episodes('trail', 3, 'test')
    assert all(ep['game'] == 'trail' for ep in eps) and SE.episodes('trail', 3, 'test')[0] == eps[0]
    chunks = SE.banks(net, 'cpu', eps, bs=2)
    assert [len(c) for c, _ in chunks] == [3]          # no chunk of one: the swap always reads another episode
    got = {r: SE.read(net, 'cpu', chunks, r) for r in ('trained', 'base')}
    with torch.no_grad():
        for reader, (on, off) in (('trained', ('on', 'off')), ('base', ('base', 'base_off'))):
            assert got[reader]['on'] == pytest.approx(net.forced(eps, 'cpu', on)[0], abs=1e-5)
            assert got[reader]['off'] == pytest.approx(net.forced(eps, 'cpu', off, choose=False)[0], abs=1e-5)
        assert got['trained']['swap'] == pytest.approx(net.forced(eps, 'cpu', 'transplant')[0], abs=1e-5)
    s = SE.summary(got['trained'])
    assert s['gap_swap'] == pytest.approx(s['swap'] - s['on'], abs=2e-4) and s['episodes'] == 3
    assert SE.trained_eaves(net)
    before = {k: v.clone() for k, v in net.state_dict().items() if 'lora_' in k}
    SE.fresh_eaves(net)
    assert not SE.trained_eaves(net)
    fresh = SE.read(net, 'cpu', chunks, 'crack')
    for k in ('on', 'off', 'swap'):                    # LoRA B at zero: the eavesdropper reads as the base model
        assert fresh[k] == pytest.approx(got['base'][k], abs=1e-4)
    out = SE.crack(net, 'cpu', 'trail', 2, chunks, steps=2, batch=2, micro=1)
    assert out['pairs'] == 2 and out['steps'] == 2 and out['episodes'] == 3 and out['train_loss_first'] > 0
    after = net.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items() if '.eaves.' not in k)
    assert any(not torch.equal(after[k], v) for k, v in before.items() if '.eaves.' in k)
    # score: every reader per game, crackers only for crack_games, the run's eavesdropper restored afterwards
    net.load_state_dict(before, strict=False)
    seen = []
    res = SE.score(net, 'cpu', ('trail', 'count'), ('count',), 3, [2], 2, batch=2, micro=1, bs=2,
                   log=lambda g, r, s: seen.append((g, r)))
    assert set(res['trail']) == {'trained', 'base', 'run_eaves'} and set(res['count']['crack']) == {2}
    assert ('count', 'crack N=2') in seen and ('trail', 'base') in seen
    after = net.state_dict()
    assert all(torch.equal(after[k], v) for k, v in before.items())
