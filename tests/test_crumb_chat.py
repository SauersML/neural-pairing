"""swarm/crumb_chat.py on a tiny random Qwen3 (the real tokenizer) on the CPU, with a stand-in for swarm/opendata.py's
pieces and teacher rows from the tiny model itself: the whole chat in one pass is the teacher; B reading A's keys and
values from that pass is the teacher again, token for token, whatever order the bank is in; A alone is the teacher up to
A's first turn and not after B has spoken; the framed part spans are the text; a summary pools per stage."""
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_chat as CC                      # noqa: E402
from swarm import crumb_open as CO                      # noqa: E402
from swarm import crumbs as C                           # noqa: E402

from test_crumb_open import record                      # noqa: E402
from test_crumbs import tiny                            # noqa: E402


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


@pytest.fixture
def setup(tok, monkeypatch):
    monkeypatch.setattr(CO, 'OD', types.SimpleNamespace(
        piece=lambda text: tok(text, add_special_tokens=False).input_ids, USER_HEAD='<|im_start|>user\n',
        ASSIST_HEAD='<|im_start|>assistant\n<think>\n\n</think>\n\n', END='<|im_end|>\n',
        IM_END=tok.convert_tokens_to_ids('<|im_end|>')))
    net = tiny(tok, 'kv', 16)
    net.source = 'sentence'
    net.eval()
    rec = record(tok, 0, 'chat_alt')                   # turns A B A B A
    for t in rec['turns']:
        t['teacher_user'] = t['user']
    chat, turn, starts = CC.chat_tokens(rec)
    h, _ = CC.forward(net, 'cpu', chat, list(range(len(chat))))
    rec['topk'] = {}
    for k, start in starts.items():
        n = len(rec['turns'][k]['response_ids'])
        lp, ids = net.model.get_output_embeddings()(h[start - 1:start - 1 + n]).float().log_softmax(-1).topk(8, -1)
        rec['topk'][k] = (ids, lp)
    return net, rec, CO.episode(net, rec, 64, 8)


def test_stages(setup):
    net, rec, ep = setup
    r = CC.row(net, rec, ep, 'cpu', 6144)
    kl = {st: {t: a / b for t, (a, b) in d.items()} for st, d in r.items()}
    assert set(kl['whole_chat']) == {0, 1}             # B's two turns
    # a tiny random model barely depends on its context, so KL is small: the sizes below are relative to it
    for t in (0, 1):
        assert kl['whole_chat'][t] == pytest.approx(0., abs=1e-10)
        assert abs(kl['with_b'][t]) < 1e-3 * kl['cut_plain'][t]
    assert abs(kl['alone'][0]) < 1e-3 * kl['cut_plain'][0]     # before B's first turn only A's first: as in the chat
    assert kl['alone'][1] > 100 * abs(kl['with_b'][1])  # A's second turn was read without B's first
    for st in ('alone_future', 'alone_span', 'alone_system'):  # B's first turn reads more, or the same at other places
        assert kl[st][0] > 100 * abs(kl['alone'][0])
    for st in ('framed', 'ordered', 'no_system', 'true_pos', 'parts', 'cut_agent', 'marked', 'marked_no_system',
               'marked_true_pos', 'chat_walker', 'chat_walker_no_b_system', 'alone_both_system'):
        assert kl[st][0] > 1e-8 and kl[st][1] > 1e-8
    assert kl['framed'] != kl['ordered'] and kl['ordered'] != kl['no_system'] and kl['no_system'] != kl['true_pos']
    got = CC.summary([r, r])
    assert got['alone']['against'] == 'cut_plain' and got['parts']['against'] == 'cut_agent'
    assert got['whole_chat']['closed'] == pytest.approx(1., abs=1e-3)
    cut, on = sum(a for a, _ in r['cut_agent'].values()), sum(a for a, _ in r['parts'].values())
    assert got['parts']['closed'] == pytest.approx((cut - on) / cut, abs=2e-4)
    assert got['alone']['by_turn'][0]['closed'] == pytest.approx(1., abs=1e-3)


def test_streams(setup, tok):
    net, rec, _ = setup
    ids, spans, marks = CC.framed_stream(net, rec, 'A')
    assert [k for k, _, _ in spans] == [0, 2, 4]
    assert all(m[1] < s[1] and s[2] < m[2] for s, m in zip(spans, marks))
    assert tok.decode(ids[marks[1][1]:marks[1][2]]).endswith('<|im_end|>\n')
    assert tok.decode(ids[spans[1][1]:spans[1][2]]).startswith('Part 2:\nA user asked you:\nrequest 2')
    chat, turn, starts = CC.chat_tokens(rec)
    assert all(turn[i] == k for k, s in starts.items() for i in [s])
    reader = {1, 3}
    for system in (False, True):
        got, index, turns, at = CC.reader_stream(net, chat, turn, starts, reader, system)
        assert sum(i < 0 for i in index) == (len(net._ids(f'<|im_start|>system\n{CO.READER}<|im_end|>\n')) if system else 0)
        assert all(got[at[k]] == rec['turns'][k]['response_ids'][0] for k in reader)
        assert all(t in reader for t, i in zip(turns, index) if i >= 0)


def test_split_episode_is_alone(setup, tok):
    """crumb_open's --split, adapter off, is the ladder's alone: the same tokens at the same places read in the same
    order; and the two streams put back by position are the whole chat."""
    net, rec, _ = setup
    ladder = CC.row(net, rec, CO.episode(net, rec, 64, 8), 'cpu', 6144)
    alone, cut = ladder['alone'], ladder['cut_plain']
    net.remember = net.split = True
    try:
        ep = CO.episode(net, rec, 64, 8)
        assert [s['place'] for s in ep['agents'][0]['segments']] == [0, 1]      # A's last turn: no B turn reads it
        assert [s['place'] for s in ep['agents'][1]['segments']] == [0, 1]
        got = CO.kl_pass(net, [ep], 'cpu', ('on',), zero=True)['on'][0]
        a, b = net.stream(ep, 0), net.stream(ep, 1)
    finally:
        net.remember = net.split = False
    for t in (0, 1):
        assert got[t][1] == alone[t][1]
        assert abs(got[t][0] - alone[t][0]) < .01 * cut[t][0]   # float32 rounding of the rest of the vocabulary aside
    chat, turn, _ = CC.chat_tokens(rec)
    placed = dict(zip(a['pos'] + b['pos'], a['ids'] + b['ids']))
    kept = max(placed) + 1
    assert sorted(placed) == list(range(kept)) and [placed[i] for i in range(kept)] == chat[:kept]
    assert len(a['slots']) == len(a['ids']) and not any(a['target'])
