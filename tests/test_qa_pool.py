"""Pooling math, actual Qwen/AFM gradients, causal letters, and train.py smoke.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu python tests/test_qa_pool.py
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
import pytest
import torch
from peft import LoraConfig, get_peft_model

import bench_eval
import games as G
import link
import qa_pool as Q
import session as S
from qa_pool_probe import capture_writes


def test_log_average_and_responsibility_gradient():
    a = torch.tensor([[2., -1., .5, 0.], [-500., 0., 0., 0.]], requires_grad=True)
    b = torch.tensor([[-1., 1., 0., .2], [-400., 0., 0., 0.]], requires_grad=True)
    logits = dict(quill=a, ember=b)
    lp = Q.log_average(logits)
    torch.testing.assert_close(lp.exp(), (a.softmax(-1) + b.softmax(-1)) / 2)
    loss = -lp[:, 0].sum()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(t.grad).all() for t in (a, b))
    # For ordinary logits each head's CE gradient is weighted by its probability
    # of the gold answer divided by the two heads' sum (not by a fixed 1/2).
    pa, pb = a.detach()[0].softmax(-1), b.detach()[0].softmax(-1)
    for t, p in ((a, pa), (b, pb)):
        expected = p.clone(); expected[0] -= 1
        torch.testing.assert_close(t.grad[0], expected * p[0] / (pa[0] + pb[0]))


@pytest.fixture(scope='module')
def pair():
    torch.set_num_threads(2)
    toks, models = {}, {}
    for x in link.SIDES:
        toks[x], model = link.load(x)
        models[x] = get_peft_model(model, LoraConfig(r=4, lora_alpha=8,
                                  target_modules=['q_proj', 'v_proj', 'up_proj', 'down_proj'])).eval()
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
    with torch.no_grad():
        for side in p.bridge.values():
            side.alpha.fill_(0.3)
    return p


def setup(pair):
    items = G.bench('train')[:3]
    rows = [(G.qa_episode(it, 'pair'), {x: frozenset(link.SIDES) for x in link.SIDES}) for it in items]
    built = [S.build(ep, pair.toks) for ep, _ in rows]
    denom = {x: sum(w for e in built for _, _, w in e[x]['labels']) for x in link.SIDES}
    return items, rows, built, denom


def test_real_gradients_microbatches_and_scoring(pair):
    items, rows, built, denom = setup(pair)
    params = {f'{x}.{n}': p for x, m in pair.models.items() for n, p in m.named_parameters() if p.requires_grad}
    params.update({'bridge.' + n: p for n, p in pair.bridge.named_parameters()})
    pads = {x: t.pad_token_id for x, t in pair.toks.items()}

    def run(groups):
        for p in params.values():
            p.grad = None
        loss = 0.
        for ids in groups:
            r, e = [rows[i] for i in ids], [built[i] for i in ids]
            batch = S.collate(e, pads)
            h, _ = pair.joint(batch)
            L = Q.pooled_loss(pair, pair.toks, r, e, batch, h, denom)
            L.backward()
            loss += float(L.detach())
        return loss, {n: p.grad.clone() for n, p in params.items() if p.grad is not None}

    whole, gradients = run([[0, 1, 2]])
    split, separate = run([[2], [0, 1]])
    assert abs(whole - split) < 2e-6
    assert gradients.keys() == separate.keys()
    for n in gradients:
        torch.testing.assert_close(gradients[n], separate[n], atol=2e-6, rtol=2e-4)
    for x in link.SIDES:
        for prefix in (x + '.', 'bridge.' + x + '.q.', 'bridge.' + x + '.k.',
                       'bridge.' + x + '.v.', 'bridge.' + x + '.o.'):
            assert sum(float(v.abs().sum()) for n, v in gradients.items() if n.startswith(prefix)) > 0, prefix
    P = bench_eval.letter_probs(pair, items, 'pair', bs=3)
    gold = torch.tensor([it['answer'] for it in items])
    expected = -2 * ((P['quill'] + P['ember']) / 2)[torch.arange(3), gold].log().mean()
    assert abs(whole - float(expected)) < 2e-6


def test_gold_is_future_and_observer_is_exact(pair):
    items, rows, built, denom = setup(pair)
    _, at, _ = Q.letter_layout(pair.toks, rows, built)
    pads = {x: t.pad_token_id for x, t in pair.toks.items()}
    batch = S.collate(built, pads)
    with torch.no_grad():
        original, _ = pair.joint(batch)
        with capture_writes(pair, at) as got:
            observed, _ = pair.joint(batch)
        for x in link.SIDES:
            assert torch.equal(original[x], observed[x])
            assert set(got[x]) == {0, 1, 2}
        changed = copy.deepcopy(built)
        for x in link.SIDES:
            for i, p in enumerate(at[x]):
                changed[i][x]['ids'][p + 1] = S.ids(pair.toks[x], ' Z')[0]
        altered, _ = pair.joint(S.collate(changed, pads))
        for x in link.SIDES:
            for i, p in enumerate(at[x]):
                torch.testing.assert_close(original[x][i, p], altered[x][i, p], atol=1e-6, rtol=1e-6)
    unsafe = copy.deepcopy(built)
    unsafe[0]['ember']['time'][at['ember'][0] + 1] = -1
    with pytest.raises(ValueError, match='partner gold'):
        Q.letter_layout(pair.toks, rows, unsafe)


@pytest.mark.parametrize('arm', [['--qa', '1'], ['--solo']])
def test_train_cli_smoke(tmp_path, arm):
    # Actual trainer/optimizer/checkpoint, initialized randomly on actual tiny
    # architectures; real replies and benchmark data, no replacement forward.
    # --solo is the control: the same pooled objective, the link cut.
    replies = os.environ.get('COLDSTART_REPLIES', '/Users/user/mpd-data/neural-pairing/replies.json')
    args = [sys.executable, str(ROOT / 'pairing/train.py'), '--replies', replies, '--out', str(tmp_path / 'run'),
            '--steps', '2', '--bs', '2', '--micro', '2', *arm, '--qa_pool', '1', '--anchor', '0',
            '--comp', '0', '--rank', '4', '--heads', '2', '--head_dim', '16', '--xblocks', '1,2,4',
            '--ckpt', 'none', '--first_eval', '999', '--eval_every', '999', '--save_every', '999',
            '--quick_n', '1', '--read_start', '0.1', '--hold', '0']
    proc = subprocess.run(args, cwd=ROOT, env=dict(os.environ, MPD_MEM_GIB='8', OMP_NUM_THREADS='2'),
                          capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ck = torch.load(tmp_path / 'run/step_2/bridge.pt', weights_only=False, map_location='cpu')
    assert ck['args']['qa_pool'] == 1.0
    assert all((tmp_path / f'run/step_2/{x}/adapter_model.safetensors').exists() for x in link.SIDES)
    lines = [json.loads(s) for s in proc.stdout.splitlines() if s.startswith('{')]
    assert len(lines) == 2 and all(torch.isfinite(torch.tensor(s['loss'])) for s in lines)
    print(json.dumps(dict(arm=arm, smoke_steps=2, losses=[s['loss'] for s in lines], checkpoint='step_2')))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-s']))
