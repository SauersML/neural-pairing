"""Real Qwen/AFM tiny architectures: pause timelines, replay, logits and legacy zero path."""
import copy
import torch
import test_link as T
import session as S
from link import SIDES, inner

torch.set_num_threads(2)
for eps in (T.duplex, T.dialogue):
    zero = [dict(ep, pause=0, pause_turns={j: 0 for j in range(len(ep.get('turns', [])))}) for ep in eps]
    assert [S.build(ep, T.toks) for ep in eps] == [S.build(ep, T.toks) for ep in zero]
    a = S.generate(T.pair, eps, max_turn=3, keep=True)
    b = S.generate(T.pair, zero, max_turn=3, keep=True)
    for aa, bb in zip(a, b):
        assert aa['tokens'] == bb['tokens']
        for x in SIDES:
            assert torch.equal(aa['h'][x], bb['h'][x])
    for k in (0, 1, 3):
        paused = [dict(ep, pause=k, pause_turns={j: k for j in range(len(ep.get('turns', [])))}) for ep in eps]
        built = [S.build(ep, T.toks) for ep in paused]
        batch = S.collate(built, T.pads)
        with torch.no_grad():
            h, _ = T.pair.joint(batch)
            out = S.generate(T.pair, paused, force=True, keep=True)
            worst = 0.
            for i, (ep, e, r) in enumerate(zip(paused, built, out)):
                replay = S.build(S.replay(ep, r), T.toks)
                for x in SIDES:
                    assert e[x]['ids'] == replay[x]['ids'] and e[x]['time'] == replay[x]['time']
                    ps = [p for p, _, _ in e[x]['labels']]
                    expected = inner(T.models[x])[1](h[x][i, ps])
                    actual = inner(T.models[x])[1](r['h'][x][ps])
                    worst = max(worst, float((actual - expected).abs().max()))
                    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
                    assert len(e[x]['labels']) == len(S.build(eps[i], T.toks)[x]['labels'])
            print(f'pause {eps[0]["mode"]} k={k}: reply logits max absolute error {worst:.3g}')
    mixed = [dict(ep, pause=i, pause_turns={0: i, 1: 3-i}, link_off_after=1)
             for i, ep in enumerate(eps)]
    T.check(mixed, 'mixed pause counts and cut', mode='shuffle')
for k in (-1, 1.2, True):
    try:
        S.build(dict(T.duplex[0], pause=k), T.toks)
        raise AssertionError('accepted invalid pause')
    except ValueError:
        pass
print('pause exactness and bitwise zero regressions passed')
