"""play_tasks.py's scorers on hand-made cases."""
import pytest

from play_tasks import chrf, run_tests, score_one, token_f1


@pytest.mark.parametrize('game, item, text', [
    ('co_qa', {'answer': 'No.'}, 'No, they do not.'),
    ('co_nli', {'label': 'neutral'}, 'Neutral. The premise does not say.'),
    ('co_mcq', {'options': ['a', 'b', 'c'], 'answer': 2}, '(C) c'),
    ('co_mcq', {'options': ['red car', 'blue boat'], 'answer': 1}, 'It is the blue boat.'),
    ('co_math', {'solution': 'so 3 + 4 = 7\nThe answer is 7.'}, 'Total: 7 apples'),
    ('co_code_out', {'output': "[('74', 31)]"}, "It returns [('74',31)]."),
])
def test_right_answers_score_one(game, item, text):
    assert score_one(game, item, text, 'quill', None) == 1.0


def test_wrong_answer_scores_zero():
    assert score_one('co_qa', {'answer': 'Yes.'}, 'No.', 'quill', None) == 0.0


def test_overlap_scores():
    assert token_f1('Saint Bernadette Soubirous', 'Saint Bernadette Soubirous') == 1.0
    assert run_tests('def f(x):\n    return x + 1', ['assert f(1) == 2', 'assert f(2) == 4']) == 0.5
    assert chrf('Je peux le faire.', 'Je peux le faire.') == 1.0 and chrf('abc', 'xyz') == 0.0
    assert 0 < chrf('Je peux faire ça.', 'Je peux le faire.') < 1
