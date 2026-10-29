"""defer_mood.py's MLX readings against the torch pair on tiny random models: the feelings log-probabilities equal
affect.read's log-odds, and the exam letter probabilities equal bench_eval.letter_probs on the torch pair, linked and
cut.
"""
import pytest

mx = pytest.importorskip('mlx.core')
import torch

import affect as A
import bench_eval as B
import defer_mood as D
import mlx_eval as V
from test_mlx_eval import pairs  # noqa: F401  (the shared tiny torch and MLX pairs)

CONVS = [['Hello there.', 'Hi! How are you?', 'Fine, thanks.', 'Good to hear.'],
         ['What a day.', 'Tell me more.', 'It rained.', 'I like rain.']]
ITEMS = [dict(src='arc_challenge', q='Which is a gas?', choices=['ice', 'steam', 'rock', 'wood'], answer=1),
         dict(src='mmlu', q='2 + 2 = ?', choices=['3', '4', '5', '22'], answer=1)]


@pytest.mark.parametrize('cond', ['link', 'cut'])
def test_feelings_and_exam_parity(pairs, cond):  # noqa: F811
    pair, _, vp, toks, _, _ = pairs
    mode = D.CONDITIONS[cond]
    for name, (question, answer, words) in D.questions(toks).items():
        for x in A.SIDES:
            eps = [A.ask_episode(i, t, x, question, answer) for i, t in enumerate(CONVS)]
            got = D.logprobs(vp, toks, eps, x, words[x], mode, bs=2)
            want = [r[x]['logodds'] for r in A.read(pair, toks, eps, {y: {} for y in A.SIDES}, mode, {x: words[x]},
                                                     bs=2, layers=(1,))]
            assert [g - b for g, b in zip(got['good'], got['bad'])] == pytest.approx(want, abs=2e-3), (name, x)
    want = B.letter_probs(pair, ITEMS, 'pair', mode=mode, bs=2)
    got = B.letter_probs(vp, ITEMS, 'pair', mode=mode, bs=2)
    for x in A.SIDES:
        torch.testing.assert_close(got[x].float(), want[x].float(), atol=2e-3, rtol=0)
