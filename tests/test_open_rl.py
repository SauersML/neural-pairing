"""rl.py's open sessions and their judges on tiny models: a judge's scores are expected digits from 1 to 9, the same
alone and inside a padded batch, also after reasoning; a strict win needs the margin; the compared dialogues are
length-matched; the gate's degraded transcripts break what they should and keep the speakers; the held-out wordings
and topics are never trained; an open play step (linked plays scored against cut plays) trains and saves; and
beauty_eval.py's gate (with its reasoning arm), generation and judging run.

MPD_MEM_GIB=8 NL_TINY=1 NL_DEVICE=cpu python -m pytest -q tests/test_open_rl.py
"""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(NL_TINY='1', NL_DEVICE='cpu', HF_HUB_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))
import torch
from peft import LoraConfig, get_peft_model

import beauty_eval as E
import games
import judge as J
import link
import rl

JUDGE, HELD_OUT = 'Qwen/Qwen3-4B', 'arcee-ai/AFM-4.5B'      # NL_TINY: random models of their architectures
COREAD = os.environ.get('COREAD_CHECK', str(ROOT / 'data/coread_check.json'))


def test_judge_scores_are_digits_and_padding_free():
    j = J.Judge(JUDGE)
    texts = ['Quill: hi\nEmber: hello there', 'Quill: a poem about the sea and the long light of evening, ' * 3, 'Ember: x']
    s, mass = j.score(rl.RUBRIC, texts)
    assert len(s) == 3 and all(1 <= v <= 9 for v in s) and all(0 < m <= 1 for m in mass)
    alone = [j.score(rl.RUBRIC, [t])[0][0] for t in texts]
    gap, _, top = j.check(rl.RUBRIC, texts)
    assert gap < 1e-4 and len(top) == 5
    assert (J.strict_win(.3, .2), J.strict_win(-.3, .2), J.strict_win(.1, .2)) == (1.0, 0.0, 0.5)
    prompt = j.prompt('Q?', 'x')
    assert prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n') and 'Conversation:\nx\n' in prompt
    t = J.Judge(JUDGE, think=3)
    assert t.label == JUDGE + '+think3' and t.prompt('Q?', 'x').endswith('<|im_start|>assistant\n')
    seqs = t.ids(rl.RUBRIC, texts)
    done = t.reason(seqs)
    close = t.tok('</think>', add_special_tokens=False).input_ids
    for a, b in zip(seqs, done):
        assert b[:len(a)] == a and 0 < len(b) - len(a) <= 3 + 4 and close[0] in b[len(a):]
    ts, _ = t.score(rl.RUBRIC, texts)
    assert all(1 <= v <= 9 for v in ts)


def test_degraded_transcripts():
    q, e = link.SIDES
    d = [[(q if (p + t) % 2 == 0 else e, f's{p}t{t}') for t in range(3)] for p in range(10)]
    sh = E.degrade(d, 0, 'shuffled')
    assert [x for x, _ in sh] == [x for x, _ in d[0]]
    src = [int(t.split('t')[0][1:]) for _, t in sh]
    assert 0 not in src and len(set(src)) == 3 and all(int(t.split('t')[1]) == k for k, (_, t) in enumerate(sh))
    assert E.degrade(d, 3, 'generic') == [(d[3][0][0], 's3t0'), (d[3][1][0], E.GENERIC), (d[3][2][0], 's3t2')]
    assert E.degrade(d, 2, 'loop') == [(x, 's2t0') for x, _ in d[2]]


def test_matched_cuts_each_turn_to_the_shorter():
    names = {x: link.MODELS[x]['name'] for x in link.SIDES}
    a = [('quill', 'one two three four'), ('ember', 'yes')]
    b = [('quill', 'alpha beta'), ('ember', 'so many words here')]
    ta, tb = J.matched(a, b, names)
    assert ta == 'Quill: one two ...\nEmber: yes' and tb == 'Quill: alpha beta\nEmber: so ...'
    try:
        J.matched(a, b[::-1], names)
        assert False
    except ValueError:
        pass


def test_open_sessions_hold_out_wordings_and_topics():
    assert not set(rl.OPEN['train']) & set(rl.OPEN['test'])
    train = {c[1] for c in games.POOLS['concepts']['train']}
    eps = E.prompts(9, 3)
    assert len(eps) == 9 and len({e['meta']['topic'] for e in eps}) == 3
    for e in eps:
        assert e['meta']['template'] in rl.OPEN['test'] and e['meta']['topic'] not in train
        f = e['meta']['first']
        assert [sp for sp, _, _ in e['turns']] == [f, link.partner(f), f]
        for x in link.SIDES:
            assert link.MODELS[link.partner(x)]['name'] in e['system'][x].split('\n\n')[-1]


def tiny_init(d):
    toks, models = {}, {}
    for x in link.SIDES:
        toks[x], m = link.load(x)
        m = get_peft_model(m, LoraConfig(r=4, lora_alpha=12, target_modules=['q_proj', 'v_proj']))
        for name, p in m.named_parameters():
            if 'lora_B' in name:
                torch.nn.init.normal_(p, std=0.02)
        models[x] = m.eval()
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
    for x in link.SIDES:
        pair.bridge[x].alpha.data.fill_(0.5)
        models[x].save_pretrained(str(d / x))
    torch.save(pair.checkpoint(), d / 'bridge.pt')
    return d


def run(cmd, **env):
    proc = subprocess.run([sys.executable] + cmd, cwd=ROOT / 'pairing', env=dict(os.environ, **env),
                          capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, '\n'.join(l for l in proc.stderr.splitlines() if 'Warning' not in l)[-4000:]
    return proc.stdout


def test_open_step_and_eval(tmp_path):
    init = tiny_init(tmp_path / 'init')
    questions = sum(games.POOLS['questions'].values(), [])
    concept = games.POOLS['concepts']['train'][0][0]
    replies = {x: dict(own={q: 'Hello.' for q in questions}, steer=[(q, concept, 'Hello.') for q in questions])
               for x in link.SIDES}
    (tmp_path / 'replies.json').write_text(json.dumps(replies))
    out = tmp_path / 'rl'
    stdout = run(['rl.py', '--init', str(init), '--out', str(out), '--replies', str(tmp_path / 'replies.json'),
                  '--steps', '1', '--mix', '0', '--open_frac', '1', '--open_groups', '1', '--G', '3',
                  '--open_turns', '2', '--open_cut', '2', '--max_turn', '3', '--judge', JUDGE, '--win_margin', '0.05',
                  '--eval_n', '2', '--eval_task_n', '2', '--no_ckpt', '--temperature', '1', '--coread_data', COREAD],
                 NL_MAXNEW='1')
    line = next(json.loads(l) for l in stdout.splitlines() if l.startswith('{') and '"play": "open"' in l)
    assert set(line['R']) == set(link.SIDES) and 0 <= line['below'] <= 1 and 0 <= line['beats_cut'] <= 1
    assert 1 <= line['score_link'] <= 9 and 0 <= line['digit_mass'] <= 1
    assert (out / 'step_1/bridge.pt').exists()
    ev = tmp_path / 'eval'
    run(['beauty_eval.py', '--phase', 'gate', '--runs', f'init={init}', '--judges', JUDGE, HELD_OUT, '--out', str(ev),
         '--pairs', '8', '--turns', '2', '--max_turn', '2', '--think', '3'])
    g = json.loads((ev / 'gate.json').read_text())
    assert g['pairs'] == 8 and isinstance(g['pass'], bool) and g['bars']['sanity_n'] == 8
    assert set(g['judges']) >= {JUDGE, HELD_OUT} and all(len(v['pairs']) == 8 for v in g['judges'].values())
    assert set(g['judges'][JUDGE]['sanity']) == set(E.KINDS) and -1 <= g['spearman'] <= 1
    if not g['judges'][JUDGE]['sane']:
        assert JUDGE + '+think3' in g['judges']             # the reasoning arm ran
    assert g['pass'] == (g['judges'][HELD_OUT]['sane'] and g['train_arm'] is not None and g['primary']['agree'] >= 0.7)
    assert set(g['margins']) == {g['primary']['arm'], HELD_OUT} and all(m >= 0 for m in g['margins'].values())
    run(['beauty_eval.py', '--phase', 'gen', '--runs', f'init={init}', f'rl={out / "step_1"}', '--out', str(ev),
         '--n', '3', '--samples', '2', '--turns', '2', '--max_turn', '2'])
    run(['beauty_eval.py', '--phase', 'judge', '--judges', HELD_OUT, '--out', str(ev)])
    res = json.loads((ev / 'judged.json').read_text())
    s = res['judges'][HELD_OUT]['eval']['strict']['summary']
    assert set(s) == {'rl>init', 'init:link>cut', 'rl:link>cut', 'rl-init:link>cut'}
    assert s['rl>init']['n'] == 3 and s['rl>init']['ci95'][0] <= s['rl>init']['mean'] <= s['rl>init']['ci95'][1]
    assert len(res['judges'][HELD_OUT]['train']['diff']['per_prompt']['rl>init']) == 3
    assert res['judges'][HELD_OUT]['margin'] == g['margins'][HELD_OUT]
