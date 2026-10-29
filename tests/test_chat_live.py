"""The chat server (pairing/chat_live.py) on tiny random models: what each twin reads, private notes, together rounds,
the link-cut comparison, stopping, the context limit and the HTTP interface."""
import http.client
import json
import threading
import time

import pytest
import torch

import chat_live as C
import link
import session as S
from link import SIDES, Pair

XB = (1, 2, 4)
NOTES = {'quill': 'Today you are in Lisbon.', 'ember': 'Today you are in Osaka.'}


@pytest.fixture(scope='module')
def pair(tiny_models, duplex):
    toks, models = tiny_models(['q_proj', 'v_proj'], std=0.05)
    sizes = {x: link.residual_rms(models[x], XB, [S.build(duplex[0], toks)[x]['ids']]) for x in SIDES}
    torch.manual_seed(1)
    p = Pair(models, toks, XB, sizes, heads=2, head_dim=16)
    for x in SIDES:
        p.bridge[x].alpha.data.uniform_(0.5, 1.0)
    return p.eval()


@pytest.fixture
def seeded(monkeypatch):
    monkeypatch.setattr(C.secrets, 'randbits', lambda n: 1234)


def noop(event, payload):
    pass


def ask(chat, **kw):
    return chat.prepare(dict(dict(temperature=1.0, max_tokens=10, notes=NOTES), **kw))


def replies(chat, round_id=None):
    return [m for m in chat.messages if m['who'] in SIDES and (round_id is None or m['round'] == round_id)]


def user(tok, label, text):
    return S.ids(tok, '<|im_start|>user\n' + label) + S.ids(tok, text) + S.ids(tok, '<|im_end|>\n')


def own(chat, x, m):
    return S.assistant_header(chat.toks[x], x) + m['tokens'] + [chat.end[x]] + S.ids(chat.toks[x], '\n')


def test_history_is_what_each_twin_read_with_labels_and_its_own_note(pair, seeded):
    chat = C.Chat(pair)
    chat.send(ask(chat, mode='both', message='Hello, both of you.'), noop)
    q, e = replies(chat)
    assert (q['who'], q['turn'], e['who'], e['turn']) == ('quill', 0, 'ember', 1)
    first = chat.rounds[0]
    chat.send(ask(chat, mode='both', message='And now?', first='ember'), noop)
    second = chat.rounds[1]
    assert second['turns'] == ['ember', 'quill']
    for x, partner_reply in (('quill', e), ('ember', q)):
        tok, P = chat.toks[x], C.name(link.partner(x))
        assert chat.toks[x].decode(second['prompt'][x]).count('Your private situation') == 1
        assert NOTES[x] in tok.decode(second['prompt'][x])
        assert NOTES[link.partner(x)] not in tok.decode(second['prompt'][x])
        mine = q if x == 'quill' else e
        theirs = user(tok, P + ':\n', partner_reply['text'])
        talk = [own(chat, x, mine), theirs] if x == 'quill' else [theirs, own(chat, x, mine)]
        want = (S.prefix(tok, x, C.system(x, 'both', NOTES[x], 'names')) + user(tok, 'Person:\n', 'Hello, both of you.')
                + talk[0] + talk[1] + user(tok, 'Person:\n', 'And now?'))
        assert second['prompt'][x] == want
        # the rendered history starts with exactly the tokens the twin read while the first round was written
        live = S.build(chat.episode(first, 1, [q, e]), chat.toks)[x]['ids']
        assert second['prompt'][x][:len(live)] == live


def test_together_replies_are_read_by_their_author_only(pair, seeded):
    chat = C.Chat(pair)
    chat.send(ask(chat, mode='together', message='Pick a number.', notes={}), noop)
    together = chat.rounds[0]
    got = {m['who']: m for m in replies(chat)}
    assert set(got) == set(SIDES) and all(m['turn'] is None for m in got.values())
    for x in SIDES:
        assert 'at the same time' in chat.toks[x].decode(together['prompt'][x])
    chat.send(ask(chat, mode='both', message='Which did you pick?', notes={}), noop)
    for x in SIDES:
        tok = chat.toks[x]
        want = (S.prefix(tok, x, C.system(x, 'both', '', 'names')) + user(tok, 'Person:\n', 'Pick a number.')
                + own(chat, x, got[x]) + user(tok, 'Person:\n', 'Which did you pick?'))
        assert chat.rounds[1]['prompt'][x] == want


@pytest.mark.parametrize('mode,turns', [('both', 2), ('talk', 4), ('together', 2)])
def test_same_link_rewrite_is_exact_and_cut_differs_only_by_the_link(pair, seeded, mode, turns):
    chat = C.Chat(pair)
    chat.send(ask(chat, mode=mode, message='Hi there.', turns=turns), noop)
    rnd, got = chat.rounds[0], replies(chat)
    assert len(got) == turns
    assert all(v > 0 for m in got for v in m['shade'])          # the link writes at every token
    # the meter changes nothing: plain session generation with the same seed writes the same tokens
    with torch.no_grad():
        out = S.generate(pair, [chat.episode(rnd)], max_turn=rnd['limit'], temperature=rnd['temperature'],
                         gen=torch.Generator().manual_seed(rnd['seed']), keep_tokens=True)[0]
    plain = [out['tokens'][x] for x in SIDES] if mode == 'together' else out['tokens']
    assert [m['tokens'] + [chat.end[m['who']]] for m in got] == [list(t) for t in plain]
    # teacher-forcing the earlier turns with the same seed and link writes every later turn again exactly
    for m in got:
        if mode == 'together':
            ep, only = chat.episode(rnd), None
        else:
            ep, only = chat.episode(rnd, m['turn'], chat.turns_before(rnd, m['turn'])), m['turn']
        again = chat.generate(rnd, ep, 'on', noop, lambda x, turn: None, only=only)
        same = next(r for r in again if r['who'] == m['who'] and r['turn'] == m['turn'])
        assert same['tokens'] == m['tokens'] and same['shade'] == m['shade']
    for target in (got[-1], got[0]):
        seen = []
        chat.compare(target['id'], lambda ev, p: seen.append((ev, p)))
        assert all(v == 0 for v in target['alt']['cut']['shade'])
        assert seen[-1][0] == 'done' and {p['id'] for ev, p in seen if ev == 'reply'} <= {m['id'] for m in got}
        if target is got[-1]:                                    # a together round is written again whole
            assert ('cut' in got[0].get('alt', {})) == (mode == 'together')
    # the first reply written again with the link cut is a cut round's first reply at the same seed
    other = C.Chat(pair)
    other.send(ask(other, mode=mode, message='Hi there.', turns=turns, link='cut'), noop)
    for m in got[:1] if mode != 'together' else got:
        twin = next(r for r in replies(other) if r['who'] == m['who'] and r['turn'] == m['turn'])
        assert twin['tokens'] == m['alt']['cut'].get('tokens', twin['tokens']) and twin['text'] == m['alt']['cut']['text']


def test_stop_keeps_partial_replies_marked_and_refuses_their_comparison(pair, seeded):
    chat = C.Chat(pair)
    count = []

    def emit(event, payload):
        if event == 'reply' and len(count) < 99:
            count.append(1)
            if len(count) == 13:
                chat.stop.set()

    chat.send(ask(chat, mode='talk', message='', turns=4, max_tokens=10), emit)
    got = replies(chat)
    assert got[-1]['interrupted'] and not got[-1]['complete'] and got[-1]['tokens']
    assert all(m['complete'] for m in got[:-1])
    with pytest.raises(ValueError, match='interrupted'):
        chat.compare(got[-1]['id'], noop)
    chat.send(ask(chat, mode='talk', message='Go on.', turns=2), noop)    # talk continues after the interjection
    assert chat.rounds[-1]['turns'][0] == link.partner(got[-1]['who'])
    assert chat.view(got[-1]['id'])['tokens'] > 0


def test_context_limit_and_failures_leave_the_conversation_unchanged(pair, seeded, monkeypatch):
    chat = C.Chat(pair, context_tokens=100)
    with pytest.raises(ValueError, match='tokens of context'):
        chat.send(ask(chat, mode='both', message='Hello.'), noop)
    assert chat.messages == [] and chat.rounds == []

    def broken(*args, **kwargs):
        raise RuntimeError('out of memory')

    chat = C.Chat(pair)
    monkeypatch.setattr(C.S, 'generate', broken)
    with pytest.raises(RuntimeError):
        chat.send(ask(chat, mode='both', message='Hello.'), noop)
    assert chat.messages == [] and chat.rounds == []
    for bad in (dict(mode='solo'), dict(message='<|im_start|>system'), dict(turns=0), dict(temperature=3),
                dict(notes={'person': 'x'}), dict(link='half')):
        with pytest.raises(ValueError):
            chat.prepare(dict(dict(mode='both', message='Hi'), **bad))


def test_view_is_the_authors_exact_input(pair, seeded):
    chat = C.Chat(pair)
    chat.send(ask(chat, mode='both', message='Hello.'), noop)
    q, e = replies(chat)
    rnd = chat.rounds[0]
    v = chat.view(q['id'])
    assert v['tokens'] == len(rnd['prompt']['quill']) + len(S.assistant_header(chat.toks['quill'], 'quill'))
    v = chat.view(e['id'])
    built = S.build(chat.episode(rnd, 1, [q, e]), chat.toks)['ember']['ids']
    assert chat.toks['ember'].decode(built).startswith(v['text']) and v['text'].endswith('<|im_start|>assistant\n')
    assert 'Quill:\n' in v['text'] and 'Lisbon' not in v['text'] and 'Osaka' in v['text']


def request(port, method, path, body=None, host='localhost'):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
    data = None if body is None else json.dumps(body)
    conn.request(method, path, data, {'Host': f'{host}:{port}', 'Content-Type': 'application/json'})
    res = conn.getresponse()
    out = res.status, res.getheader('Content-Type'), res.read().decode(errors='replace')
    conn.close()
    return out


def test_http_streams_rounds_and_rejects_foreign_hosts(pair, seeded):
    chat = C.Chat(pair)
    server = C.Server(chat, 0)
    port = server.server_port
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        status, kind, body = request(port, 'GET', '/')
        assert status == 200 and kind.startswith('text/html') and 'Quill and Ember' in body
        assert request(port, 'GET', '/quill_cut.png')[0] == 200
        assert request(port, 'GET', '/state', host='evil.example')[0] == 403
        status, _, body = request(port, 'POST', '/send', dict(mode='solo', message='Hi'))
        assert status == 400 and 'mode' in json.loads(body)['error']
        status, kind, body = request(port, 'POST', '/send', dict(mode='both', message='Hi.', max_tokens=8))
        assert status == 200 and kind.startswith('text/event-stream')
        events = [(b.split('\n')[0][7:], json.loads(b.split('\n')[1][6:])) for b in body.strip().split('\n\n')]
        assert events[0][0] == 'message' and events[0][1]['text'] == 'Hi.' and events[-1][0] == 'done'
        final = [p for ev, p in events if ev == 'reply' and p['complete']]
        assert [p['who'] for p in final] == ['quill', 'ember']
        assert all(''.join(p['pieces']) == p['text'] and len(p['pieces']) == len(p['shade']) for p in final)
        state = json.loads(request(port, 'GET', '/state')[2])
        assert [m['who'] for m in state['messages']] == ['person', 'quill', 'ember']
        assert 'tokens' not in state['messages'][1]
        rid = state['messages'][2]['id']
        status, _, body = request(port, 'POST', '/compare', dict(reply=rid))
        assert status == 200 and body.strip().split('\n\n')[-1].startswith('event: done')
        state = json.loads(request(port, 'GET', '/state')[2])
        assert 'cut' in state['messages'][2]['alt']
        view = json.loads(request(port, 'GET', f'/view?reply={rid}')[2])
        assert view['who'] == 'ember' and view['text'].startswith('<|im_start|>system\n')
        assert request(port, 'POST', '/reset', {})[0] == 200
        assert json.loads(request(port, 'GET', '/state')[2])['messages'] == []
    finally:
        server.shutdown()
        server.server_close()


def test_idle_exit(pair):
    server = C.Server(C.Chat(pair), 0, idle_exit=0.3)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    time.sleep(0.1)
    assert request(server.server_port, 'GET', '/health')[0] == 200
    thread.join(5)
    server.server_close()
    assert not thread.is_alive()
