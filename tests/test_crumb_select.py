"""swarm/crumb_select.py on a tiny random Qwen3 on the CPU, with test_crumb_open's stand-in records: keeping every
crumb (shared columns or per key-value head) gives the full copy's KL and keeping none the cut's; each visit keeps its
budget; the attention votes come from the reader's tokens before its first answer only; and the evaluation's full
copy and cuts are crumb_open's own zero_on, zero_cut and cut on the same episodes."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_open as CO                      # noqa: E402
from swarm import crumb_select as CS                    # noqa: E402
from swarm import crumbs as C                           # noqa: E402

from test_crumb_open import attach, record, stand_in    # noqa: E402
from test_crumbs import lora_effect, tiny               # noqa: E402


@pytest.fixture(scope='module')
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(C.BASE, local_files_only=True)


def setup(tok, n=2):
    torch.manual_seed(0)
    net = tiny(tok, 'kv', 2)
    lora_effect(net)
    net.eval()
    recs = [attach(net, tok, record(tok, i, f)) for i, f in enumerate(['doc', 'chat_alt', 'doc'][:n])]
    net.source = 'sentence'
    return net, recs, [CO.episode(net, r, 64, 3) for r in recs]


def flat(got):
    return [v for g in got for _, v in sorted(g.items())]


def test_keep_all_is_full_and_none_is_cut(tok):
    net, _, eps = setup(tok)
    streams = [net.stream(ep, 1) for ep in eps]
    with torch.no_grad():
        bank = net.run(eps, 'cpu', need_loss=False)[2]
        B, S = bank['place'].shape
        kv = bank['k'][0].shape[1]
        full, cut = flat(CS.kl(net, streams, bank, 'cpu')), flat(CS.kl(net, streams, bank, 'cpu', on=False))
        assert any(abs(a[0] - b[0]) > 1e-4 for a, b in zip(full, cut))
        everyone = {i: torch.ones(B, kv, S, dtype=torch.bool) for i in range(len(bank['k']))}
        no_one = {i: torch.zeros(B, kv, S, dtype=torch.bool) for i in range(len(bank['k']))}
        for got, want in ((CS.kl(net, streams, bank, 'cpu', keep=everyone), full),
                          (CS.kl(net, streams, bank, 'cpu', keep=no_one), cut),
                          (CS.kl(net, streams, CS.masked(bank, torch.ones(B, S, dtype=torch.bool)), 'cpu'), full),
                          (CS.kl(net, streams, CS.masked(bank, torch.zeros(B, S, dtype=torch.bool)), 'cpu'), cut)):
            assert [n for _, n in flat(got)] == [n for _, n in want]
            assert [s for s, _ in flat(got)] == pytest.approx([s for s, _ in want], rel=1e-4, abs=1e-5)
        # one head's crumbs hidden changes the KL, but less than hiding them all
        one = {i: torch.ones(B, kv, S, dtype=torch.bool) for i in range(len(bank['k']))}
        one[0][:, 0] = False
        mid = flat(CS.kl(net, streams, bank, 'cpu', keep=one))
        assert any(abs(a[0] - b[0]) > 1e-6 for a, b in zip(mid, full))
    assert not C.CRUMB_KEEP and not C.CRUMB_PROBE


def test_runs_and_budgets(tok):
    net, _, eps = setup(tok)
    runs = CS.runs_of(net, eps[0])
    assert len(runs) == eps[0]['open']['visits'] == 3
    assert runs[0][0] == 0 and all(b > a for a, b in runs) and all(runs[j][1] == runs[j + 1][0] for j in range(2))
    assert runs[-1][1] == len(net.stream(eps[0], 0)['slots'])
    S = runs[-1][1] + 5
    for m in (2, 4, 1000):
        ev = CS.even_keep(runs, S, m)
        assert [int(ev[a:b].sum()) for a, b in runs] == [min(m, b - a) for a, b in runs] and not ev[runs[-1][1]:].any()
        score = torch.randn(3, S)
        top = CS.top_keep(score, runs, m)
        for a, b in runs:
            assert (top[:, a:b].sum(1) == min(m, b - a)).all()
            if b - a > m:
                lo = torch.where(top[:, a:b], score[:, a:b], torch.full_like(score[:, a:b], 9e9)).min(1).values
                hi = torch.where(top[:, a:b], torch.full_like(score[:, a:b], -9e9), score[:, a:b]).max(1).values
                assert (lo >= hi).all()


def test_votes_come_before_the_first_answer(tok):
    net, recs, eps = setup(tok)
    streams = [net.stream(ep, 1) for ep in eps]
    with torch.no_grad():
        bank = net.run(eps, 'cpu', need_loss=False)[2]
        votes = CS.votes_of(net, streams, bank, 'cpu', block=7)
        assert sorted(votes) == list(range(len(bank['k'])))
        B, S = bank['place'].shape
        H = net.model.config.num_attention_heads
        for b, st in enumerate(streams):
            first = st['target'].index(True)
            window = sum(p >= 0 for p in st['place'][:first])
            for v in votes.values():
                assert v.shape == (B, bank['k'][0].shape[1], S) and (v >= 0).all()
                assert 0 < float(v[b].sum()) <= window * H + 1e-3
                assert float(v[b][:, bank['place'][b] < 0].abs().sum()) == 0
        # a later answer changed: the votes stay
        ep = dict(eps[0], agents=[eps[0]['agents'][0], dict(eps[0]['agents'][1], segments=[
            s if i == 0 else dict(s, say_ids=list(reversed(s['say_ids']))) for i, s in
            enumerate(eps[0]['agents'][1]['segments'])])])
        again = CS.votes_of(net, [net.stream(ep, 1), streams[1]], bank, 'cpu')
        for i in votes:
            assert torch.allclose(again[i], votes[i], atol=1e-5)
    assert not C.CRUMB_PROBE


def test_evaluate_select_matches_crumb_open(tok, monkeypatch):
    torch.manual_seed(0)
    net = tiny(tok, 'kv', 2)
    lora_effect(net)
    recs = [attach(net, tok, record(tok, i, f, s)) for i, (f, s) in enumerate([
        ('doc', 'eval_seen'), ('chat_alt', 'eval_seen'), ('doc', 'eval_seen'), ('fill', 'eval_heldout'),
        ('doc', 'eval_heldout')])]
    monkeypatch.setattr(CO, 'OD', stand_in(recs))
    ref = CO.evaluate_open(net, 'cpu', 'unused', 10, 64, 4, bs=3)
    res = CS.evaluate_select(net, 'cpu', 'unused', 10, 64, 4, [2, 1000], bs=3)
    assert net.source == 'slots'
    for split in CO.SPLITS:
        got, want = res[split]['all'], ref[split]
        assert got['episodes'] == want['episodes']
        for mine, theirs in (('zero_full', 'zero_on'), ('zero_cut', 'zero_cut'), ('cut', 'cut')):
            assert got['kl'][mine] == pytest.approx(want[theirs], rel=1e-4)
        assert got['closed']['zero_full'] == pytest.approx(want['closed_zero'], rel=1e-3, abs=1e-5)
        # a budget over every visit's length keeps the full copy, by every rule
        for k in ('even', 'snap', 'snaphead'):
            assert got['kl'][f'zero_{k}1000'] == pytest.approx(got['kl']['zero_full'], rel=1e-4)
        for k in ('even', 'snap'):
            assert got['kl'][f'{k}1000'] == pytest.approx(got['kl']['full'], rel=1e-4)
        assert got['kept'][2] <= 2 * 4 and got['kept'][1000] == got['positions']
        assert set(got['by_turn']) and all('closed' in t for t in got['by_turn'].values())
    assert set(res['eval_seen']) == {'all', 'real', 'simulated', 'by_family'}
    assert set(res['eval_heldout']) == {'all', 'real', 'simulated', 'by_family'}
    # per family, as crumb_open gives it: real chats (chat_alt) apart from fill-in, which eval_heldout's real is alone
    for split, fams in (('eval_seen', {'doc', 'chat_alt'}), ('eval_heldout', {'fill', 'doc'})):
        got = res[split]['by_family']
        assert set(got) == fams and sum(r['episodes'] for r in got.values()) == res[split]['all']['episodes']
        for f in fams:
            assert got[f]['kl']['zero_full'] == pytest.approx(ref[split]['by_family'][f]['zero_on'], rel=1e-4)
    assert res['eval_heldout']['real']['kl'] == pytest.approx(res['eval_heldout']['by_family']['fill']['kl'])
