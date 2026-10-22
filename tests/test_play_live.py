"""Real tiny-model live generation, observer alignment, controls and standalone pages.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu /Users/user/mpd-data/venv/bin/python -m pytest -q tests/test_play_live.py
"""
import io
import json
import math
import os
from html.parser import HTMLParser
from pathlib import Path
import sys

os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))

import pytest
import torch
from peft import LoraConfig, get_peft_model
import coread as C
import link
import play_live as L
import session as S


@pytest.fixture(scope='module')
def pair():
    torch.set_num_threads(2)
    toks, models = {}, {}
    for x in link.SIDES:
        toks[x], model = link.load(x)
        models[x] = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'])).eval()
    torch.manual_seed(27)
    p = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16,
                  sizes={x: [.04, .05, .06] for x in link.SIDES}).eval()
    with torch.no_grad():
        for b in p.bridge.values():
            b.alpha.fill_(.4)
            b.age.normal_(0, .2)
            for gate in b.gate:
                gate.weight.normal_(0, .05)
                gate.bias.normal_(1.5, .2)
    return p


@pytest.fixture(scope='module')
def ep():
    return L.draw_episode('guess', 4242)


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.stack, self.messages, self.tags = [], [], []
        self.in_message = False
        self.feed(text)
        self.close()
        assert not self.stack

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        a = dict(attrs)
        assert tag not in ('script', 'link', 'iframe', 'img')
        assert not any(k in ('src', 'href') or k.startswith('on') for k in a)
        if tag not in ('meta', 'br'):
            self.stack.append(tag)
        if tag == 'div' and a.get('class') == 'message':
            self.in_message = True
            self.messages.append('')

    def handle_endtag(self, tag):
        assert self.stack.pop() == tag
        if tag == 'div':
            self.in_message = False

    def handle_data(self, data):
        if self.in_message:
            self.messages[-1] += data


def test_free_game_stream_and_html(pair, ep, tmp_path):
    baseline = S.generate(pair, [ep], max_turn=3, keep_tokens=True)[0]
    stream = io.StringIO()
    data = L.play(pair, ep, max_turn=3, stream=stream)
    assert data['raw'][0]['tokens'] == baseline['tokens']
    assert data['raw'][0]['turns'] == baseline['turns']
    assert len(data['turns']) == 6
    assert '\033[36mQuill' in stream.getvalue() and '\033[35mEmber' in stream.getvalue()
    assert stream.getvalue().count('named partner secret:') == 6
    assert '_exchange' not in pair.__dict__ and 'step' not in pair.__dict__ and pair.record is None
    for turn in data['turns']:
        assert ''.join(t['text'] for t in turn['tokens']) == turn['text']
        assert turn['mean_effect'] > 0
        assert 0 < turn['mean_sender_gate'] < 1
        for t in turn['tokens']:
            assert math.isfinite(t['effect']) and len(t['top']) == 3
            assert sum(r['mass'] for r in t['top']) + t['null_mass'] <= 1.00001
            assert all(r['tick'] <= t['tick'] and r['side'] == link.partner(turn['speaker']) for r in t['top'])
    output = L.save(data, tmp_path / 'live.html')
    loaded = json.loads(output.with_suffix('.json').read_text())
    assert loaded['raw'][0]['tokens'] == baseline['tokens']
    page = Page(output.read_text())
    assert page.messages == [t['text'] for t in data['turns']]
    assert page.tags.count('article') == 6 and '<!doctype html>' in output.read_text()
    assert 'not causal attribution or KL' in output.read_text()


@torch.no_grad()
def test_prediction_position_attention_matches_teacher_forcing(pair, ep):
    data = L.play(pair, ep, max_turn=2, stream=None)
    replay = S.replay(ep, data['raw'][0])
    built = S.build(replay, pair.toks)
    b = S.collate([built], {x: pair.toks[x].pad_token_id for x in link.SIDES})
    original_exchange, ratios = pair._exchange, {}

    def measured_exchange(k, h, *args, **kwargs):
        out, gates = original_exchange(k, h, *args, **kwargs)
        for x in h:
            ratios[x, pair.xblocks[k]] = ((out[x] - h[x]).norm(dim=-1) / h[x].norm(dim=-1).clamp_min(1e-12))[0]
        return out, gates

    pair._exchange = measured_exchange
    pair.record = {}
    try:
        pair.joint(b)
        for j, turn in enumerate(data['turns']):
            x, y = turn['speaker'], link.partner(turn['speaker'])
            _, start, end = next(span for span in built[x]['spans'] if span[0] == j)
            tokens = turn['tokens'] + [turn['end_token']]
            assert len(tokens) == end - start
            for label, token in zip(built[x]['labels'][start:end], tokens):
                pos = label[0]
                p = torch.stack([pair.record[x, block][0]['attn'][0, :, pos].mean(0) for block in pair.xblocks]).mean(0)
                expected = L.mean([float(ratios[x, block][pos]) for block in pair.xblocks])
                assert abs(token['effect'] - expected) < 2e-6
                weighted = []
                for block in pair.xblocks:
                    attn = pair.record[x, block][0]['attn'][0, :, pos].mean(0)[1:]
                    gate = pair.record[y, block][0]['gate'][0]
                    weighted.append(float((attn * gate).sum() / attn.sum()))
                assert abs(token['sender_gate'] - L.mean(weighted)) < 2e-6
                assert abs(token['null_mass'] - float(p[0])) < 2e-6
                for r in token['top']:
                    assert r['id'] == built[y]['ids'][r['position']]
                    assert abs(r['mass'] - float(p[r['position'] + 1])) < 2e-6
    finally:
        del pair._exchange
        pair.record = None


def test_cut_has_exactly_zero_effect_and_same_generator(pair, ep):
    data = L.play(pair, ep, condition='cut', max_turn=2, stream=None)
    baseline = S.generate(pair, [ep], max_turn=2, mode='zero', keep_tokens=True)[0]
    assert data['raw'][0]['tokens'] == baseline['tokens']
    for turn in data['turns']:
        assert turn['mean_effect'] == turn['mean_sender_gate'] == 0
        for t in turn['tokens'] + [turn['end_token']]:
            assert t['effect'] == t['sender_gate'] == 0 and t['top'] == []
    assert 'rgba(245,165,50,0.0000)' in L.render(data)


def test_shuffle_uses_real_other_secret_partner(pair, ep):
    donor = L.draw_episode('guess', 800, exclude=ep['meta']['secrets'])
    data = L.play(pair, ep, condition='shuffle', donor=donor, max_turn=2, temperature=.7, seed=19, stream=None)
    baseline = S.generate(pair, [ep, donor], mode='shuffle', max_turn=2, temperature=.7,
                          gen=torch.Generator().manual_seed(19), keep_tokens=True)
    assert [r['tokens'] for r in data['raw']] == [r['tokens'] for r in baseline]
    assert all(r['episode'] == 1 for turn in data['turns'] for t in turn['tokens'] for r in t['top'])
    assert 'Shuffled donor secrets:' in L.render(data)
    with pytest.raises(ValueError, match='other secrets'):
        L.play(pair, ep, condition='shuffle', donor=ep, stream=None)


def test_initial_prompts_contain_only_own_secret_both_tokenizers(pair):
    for game in C.PLAY:
        for opener in link.SIDES:
            for seed in range(5):
                ep = L.draw_episode(game, seed, opener)
                assert all(not t[1] for t in ep['turns'])
                for x in link.SIDES:
                    prompt = pair.toks[x].decode(S.prefix(pair.toks[x], x, ep['system'][x], ep['task'][x]))
                    assert C.play_spans(ep['meta']['secrets'][x], prompt)
                    for field, value in ep['meta']['secrets'][link.partner(x)].items():
                        assert not C.play_spans({field: value}, prompt)
                    # Also check with the other tokenizer: no hidden formatting leak.
                    for tok in pair.toks.values():
                        decoded = tok.decode(S.ids(tok, prompt))
                        for field, value in ep['meta']['secrets'][link.partner(x)].items():
                            assert not C.play_spans({field: value}, decoded)
    with pytest.raises(ValueError, match='partner secret'):
        L.episode('guess', {x: {'word': 'tulip'} for x in link.SIDES})


def test_unicode_pieces_and_html_escape(pair, ep):
    for tok in pair.toks.values():
        text = 'Café 🦦 <script>alert("hi")</script> & tea.'
        assert ''.join(L.token_pieces(tok, S.ids(tok, text), text)) == text
    # Rendering-only adversarial text fixture, never used as model evidence.
    data = L.play(pair, ep, max_turn=1, stream=None)
    turn = data['turns'][0]
    turn['text'] = '<script>& hello'
    turn['tokens'] = [dict(turn['tokens'][0], text=turn['text'])]
    turn['naming_spans'] = [(10, 15)]
    page = L.render(data)
    assert '<script>' not in page and '&lt;script&gt;' in page and '<u>hello</u>' in page
    assert Page(page).messages[0] == turn['text']


def test_observer_restores_on_exception(pair, ep):
    old = pair.record
    with pytest.raises(RuntimeError, match='deliberate'):
        with L.Observer(pair, [ep], None, None).installed():
            raise RuntimeError('deliberate')
    assert '_exchange' not in pair.__dict__ and 'step' not in pair.__dict__ and pair.record is old
