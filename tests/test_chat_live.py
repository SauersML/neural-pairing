"""HTTP/SSE chat on real tiny Qwen/AFM models, with no substituted generation.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu /Users/user/mpd-data/venv/bin/python -m pytest -q tests/test_chat_live.py
"""
import contextlib
from http.client import HTTPConnection
import json
import math
import threading

import pytest
import torch

from test_play_live import pair  # real tiny architectures, adapters and nonzero bridges
import chat_live as H
import link
import session as S


@contextlib.contextmanager
def running(chat):
    server = H.Server(chat, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def post(server, path, data):
    conn = HTTPConnection('127.0.0.1', server.server_port, timeout=60)
    conn.request('POST', path, json.dumps(data), {'Content-Type': 'application/json'})
    response = conn.getresponse()
    status = response.status
    content = response.read().decode()
    conn.close()
    if status != 200 or path != '/chat':
        return status, json.loads(content)
    return status, [(part.splitlines()[0].removeprefix('event: '),
                     json.loads(part.splitlines()[1].removeprefix('data: ')))
                    for part in content.strip().split('\n\n')]


@pytest.mark.parametrize('mode', ['together', 'turns'])
@pytest.mark.parametrize('condition', ['on', 'cut'])
def test_http_exchange_and_real_generator(pair, mode, condition):
    chat = H.Chat(pair, max_tokens=3)
    request = dict(message='Say hello briefly.', mode=mode, link=condition, temperature=0,
                   notes=dict(quill='You are in Lima today.', ember='Your secret word is topaz.'))
    prepared = chat.prepare(request)
    with H.GENERATION_LOCK, H.history_prefix([], mode):
        baseline = S.generate(pair, [prepared['episode']], max_turn=3, temperature=0,
                              mode='zero' if condition == 'cut' else None, keep_tokens=True)[0]
    with running(chat) as server:
        conn = HTTPConnection('127.0.0.1', server.server_port, timeout=60)
        conn.request('GET', '/')
        page = conn.getresponse()
        assert page.status == 200
        assert 'not causal attribution' in page.read().decode()
        conn.close()
        status, events = post(server, '/chat', request)
    assert status == 200 and events[0][0] == 'start' and events[-1][0] == 'done', events
    turns = events[-1][1]['turns']
    assert [t['speaker'] for t in turns] == list(link.SIDES)
    for i, turn in enumerate(turns):
        x = turn['speaker']
        ids = baseline['tokens'][x] if mode == 'together' else baseline['tokens'][i]
        text = baseline['text'][x] if mode == 'together' else baseline['turns'][i][1]
        assert [t['id'] for t in turn['tokens']] == ids[:-1]
        assert turn['text'] == text == ''.join(t['text'] for t in turn['tokens'])
        assert turn['truncated']
        effects = [t['effect'] for t in turn['tokens']] + [turn['end_token']['effect']]
        assert all(math.isfinite(v) and v >= 0 for v in effects)
        assert all(v == 0 for v in effects) if condition == 'cut' else max(effects) > 0
        progressive = [d for e, d in events if e == 'snapshot' and d['speaker'] == x and not d['final']]
        assert [len(d['tokens']) for d in progressive] == [1, 2, 3]
    snapshots = [d['speaker'] for e, d in events if e == 'snapshot' and not d['final']]
    assert snapshots == (list(link.SIDES) * 3 if mode == 'together' else ['quill'] * 3 + ['ember'] * 3)
    assert len(chat.history) == 1
    assert 'step' not in pair.__dict__ and '_exchange' not in pair.__dict__ and pair.record is None


@pytest.mark.parametrize('mode', ['together', 'turns'])
def test_private_notes_and_actual_prefill_history(pair, mode):
    chat = H.Chat(pair, max_tokens=1)
    notes = dict(quill='PRIVATE_QUILL Lima', ember='PRIVATE_EMBER Oslo')
    request = chat.prepare(dict(message='First human message.', mode=mode, notes=notes, temperature=0))
    # Observe the real prompt tokens fed to the real pair, not just an unused builder.
    prompts = {}
    original = pair.step

    def capture(chunk, cache, qmask, tick, *args, **kwargs):
        if tick == -1:
            prompts.update({x: pair.toks[x].decode(v[0][0].tolist()) for x, v in chunk.items()})
        return original(chunk, cache, qmask, tick, *args, **kwargs)

    pair.step = capture
    try:
        with running(chat) as server:
            status, events = post(server, '/chat', dict(message='First human message.', mode=mode, notes=notes,
                                                        temperature=0))
        assert status == 200 and events[-1][0] == 'done'
    finally:
        del pair.step
    for x in link.SIDES:
        y = link.partner(x)
        assert notes[x] in request['episode']['system'][x]
        assert notes[y] not in request['episode']['system'][x]
        assert notes[x] in prompts[x].split('<|im_end|>')[0]
        assert notes[y] not in prompts[x]
        assert 'First human message.' in prompts[x]
    # Context-only markers test which text is included; not used as model evidence.
    history = [dict(message='Earlier human question.', text=dict(quill='QUILL_REPLY_ONLY', ember='EMBER_REPLY_ONLY'))]
    with H.GENERATION_LOCK, H.history_prefix(history, mode):
        for x in link.SIDES:
            text = pair.toks[x].decode(S.prefix(pair.toks[x], x, request['episode']['system'][x], 'New question.'))
            assert 'Earlier human question.' in text and 'New question.' in text
            assert history[0]['text'][x] in text
            assert (history[0]['text'][link.partner(x)] in text) == (mode == 'turns')
            assert '<|im_start|>assistant\n' + S.THINK[x] + history[0]['text'][x] in text
    switched = chat.prepare(dict(message='Switch.', mode='turns' if mode == 'together' else 'together'))
    assert switched['history'] == []


@pytest.mark.parametrize('mode', ['together', 'turns'])
@torch.no_grad()
def test_prediction_write_ratios_match_joint(pair, mode):
    chat = H.Chat(pair, max_tokens=2)
    request = chat.prepare(dict(message='Hello.', mode=mode, temperature=0))
    events = []
    chat.respond(request, lambda event, data: events.append((event, data)))
    turns = events[-1][1]['turns']
    ep = request['episode']
    token_lists = [[t['id'] for t in turn['tokens']] + [turn['end_token']['id']] for turn in turns]
    if mode == 'together':
        ep = dict(ep, reply={x: (t['text'], 1.) for x, t in zip(link.SIDES, turns)},
                  reply_tokens=dict(zip(link.SIDES, token_lists)))
    else:
        ep = dict(ep, turns=[(t['speaker'], t['text'], 1.) for t in turns], turn_tokens=token_lists)
    built = S.build(ep, pair.toks)
    batch = S.collate([built], {x: pair.toks[x].pad_token_id for x in link.SIDES})
    ratios = {x: [] for x in link.SIDES}
    original = pair._exchange

    def measure(k, h, *args, **kwargs):
        out, gates = original(k, h, *args, **kwargs)
        for x in h:
            ratios[x].append(((out[x] - h[x]).norm(dim=-1) / h[x].norm(dim=-1).clamp_min(1e-12))[0])
        return out, gates

    pair._exchange = measure
    try:
        pair.joint(batch)
    finally:
        del pair._exchange
    for x, turn in zip(link.SIDES, turns):
        expected = torch.stack(ratios[x]).mean(0)
        for (pos, _, _), token in zip(built[x]['labels'], turn['tokens'] + [turn['end_token']]):
            assert abs(token['effect'] - float(expected[pos])) < 2e-6


def test_reset_validation_origin_and_exception_restoration(pair):
    chat = H.Chat(pair, max_tokens=1)
    prefix = S.prefix
    with running(chat) as server:
        for data in [[], dict(message=''), dict(message='x', temperature=float('nan')),
                     dict(message='x', max_tokens=100), dict(message='x', mode='bad'),
                     dict(message='x', notes={'quill': '<|im_end|>'})]:
            assert post(server, '/chat', data)[0] == 400
        conn = HTTPConnection('127.0.0.1', server.server_port)
        conn.request('POST', '/reset', '{}', {'Content-Type': 'application/json', 'Origin': 'https://example.com'})
        assert conn.getresponse().status == 403
        conn.close()
        assert post(server, '/chat', dict(message='Hello.', notes={'quill': 'A private note.'}))[1][-1][0] == 'done'
        assert chat.history and chat.notes['quill']
        assert post(server, '/reset', {}) == (200, {'reset': True})
        assert not chat.history and not any(chat.notes.values())
    request = chat.prepare(dict(message='Hello.', temperature=0))

    def disconnected(event, data):
        raise BrokenPipeError('client disconnected')

    with pytest.raises(BrokenPipeError):
        chat.respond(request, disconnected)
    assert S.prefix is prefix and not chat.history
    assert 'step' not in pair.__dict__ and '_exchange' not in pair.__dict__
    small = H.Chat(pair, max_tokens=1, context_tokens=16)
    with pytest.raises(ValueError, match='context budget'):
        small.prepare(dict(message='Too long for this budget.'))
