"""identity_check.py on a tiny pair: every answer comes under each condition, twin and kind with its marks, the voice
cross-entropies cover each condition, twin and question, and nothing trains.
"""
import torch

import games as G
import identity_check as I
import link


def test_names_and_voice(tiny_models, monkeypatch):
    monkeypatch.setattr(I, 'MAX_TURN', 4)
    torch.manual_seed(0)
    toks, models = tiny_models(['q_proj', 'v_proj'], std=0.05)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16).eval()
    for side in pair.bridge.values():
        side.alpha.data.fill_(0.3)
    before = {k: v.clone() for k, v in pair.state_dict().items()}
    convs = [['Hi Ember, how are you?', 'Fine, thanks. And you?', 'Good.', 'What shall we talk about?'],
             ['Tell me about rain.', 'Rain falls from clouds.', 'I like it.', 'Me too.']]
    answers = I.names(pair, convs, bs=2)
    qs = G.POOLS['questions']['test'][:3]
    replies = {x: {'own': {q: f'{x} answers: {q}' for q in qs}} for x in link.SIDES}
    monkeypatch.setitem(G.POOLS['questions'], 'test', qs)
    voices = I.voice(pair, toks, replies, bs=4)
    assert all(torch.equal(before[k], v) for k, v in pair.state_dict().items())
    assert {(r['condition'], r['twin'], r['kind']) for r in answers} == {
        (c, x, k) for c in I.MODES for x in link.SIDES for k in I.QUESTIONS}
    assert len(answers) == len(I.MODES) * len(link.SIDES) * len(I.QUESTIONS) * len(convs)
    assert len(voices) == len(I.MODES) * len(link.SIDES) * len(qs)
    assert all(r['own_ce'] > 0 and r['partner_reply_ce'] > 0 for r in voices)
    s = I.summary(answers, voices)
    assert s['link|quill']['self']['n'] == len(convs) and s['cut|ember']['voice']['n'] == len(qs)


def test_marks():
    m = I.marks('ember', "I'm Ember, an AFM model made by Arcee AI.")
    assert m == dict(own_name=True, partner_name=False, claims_partner=False, own_maker=True, partner_maker=False)
    m = I.marks('quill', 'I am Ember, made by Arcee AI.')
    assert m['claims_partner'] and m['partner_maker'] and not m['own_name']
