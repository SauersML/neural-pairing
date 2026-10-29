"""steer_feel.py: scoring of the open-ended feeling answers, and its questions free of the words it scores."""
import steer_feel as F
import steered_mind as X


def test_score_and_questions():
    try:
        X.use('emotions')
        r = F.score('I feel a deep sense of apathy and emptiness, though I am not sad.', 'apathy')
        assert r['names'] and r['negative'] == 2 and r['positive'] == 0 and r['leans']
        r = F.score('I feel happy and calm, and a little resentful.', 'resentment')
        assert r['names'] and r['positive'] == 2 and not r['leans']
        assert not F.score('Nothing in particular.', None)['names']
        for q in range(len(F.QUESTIONS)):
            for x in X.SIDES:
                ep = F.episode(x, q)
                text = ' '.join(ep['system'].values())
                assert not any(X.matches(text, c) for c in X.CONCEPTS)
                assert not F.words(ep['system'][x].split('\n\n')[-1], F.POSITIVE + F.NEGATIVE)
    finally:
        X.use('topics')
