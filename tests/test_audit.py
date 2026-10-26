"""Regressions of the scorers, batching, RL helpers and checkpoints on real tokenizers and tiny random models: answer
scoring on real benchmark items; shuffle batches never hold a single episode; RL advantages, curricula, quiz replays
and step timing; forced, sampled and length-limited generation replays exactly; labels, weights and payload tokens;
checkpoints round trip, also from CUDA; train.py's and rl.py's own loading, timing and distillation code; and rl.py runs
a step of each kind, saves and evaluates.
"""
import ast
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import peft
from peft import LoraConfig, get_peft_model
import pytest
import safetensors.torch
import torch

import evaluate
import games
import link
import play_tasks
import rl
import session as S

ROOT = Path(__file__).resolve().parents[1]

# Real test items of data/coread.json (squad, commonsenseqa, cruxeval, mbpp, hotpotqa, mnli, gsm8k, opus-fr, paws).
QA = {'src': 'squad', 'task': 'qa', 'context': "Because of the religious affiliation of the university, all residence halls "
      "are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as "
      "parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are "
      "allowed to visit other students' dorm rooms; however, all residence halls have 24-hour social spaces for students "
      "regardless of gender. Many residence halls have at least one nun and/or priest as a resident. There are no "
      "traditional social fraternities or sororities at the university, but a majority of students live in the same "
      "residence hall for all four years.", 'question': 'How many dorms for males are on the Notre Dame campus?',
      'answer': '15'}
MCQ = [{'src': 'commonsenseqa', 'task': 'mcq', 'question': 'The sanctions against the school were a punishing blow, and '
        'they seemed to what the efforts the school had made to change?',
        'options': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid'], 'answer': 0},
       {'src': 'commonsenseqa', 'task': 'mcq', 'question': 'Where is a bald eagle safe?',
        'options': ['pine tree', 'open country', 'in washington', 'wildlife refuge', 'sky'], 'answer': 3},
       {'src': 'commonsenseqa', 'task': 'mcq', 'question': "He was receiving workman's compensation, he had tripped over "
        'a metal rod while building where?',
        'options': ['factory', 'construction site', 'engine', "insides of skyscraper's walls", 'shops'], 'answer': 1}]
CODE_OUT = {'src': 'cruxeval', 'task': 'code_out', 'code': "def f(s):\n    return ''.join(reversed(s.rstrip()))",
            'call': "f('ab        ')", 'output': "'ba'"}
CODE_WRITE = {'src': 'mbpp', 'task': 'code_write', 'problem': 'Write a function to find the nth nonagonal number.',
              'code': 'def is_nonagonal(n): \n    return int(n * (7 * n - 5) / 2) ', 'name': 'is_nonagonal'}
REFERENCES = [
    QA, CODE_OUT,
    {'src': 'hotpotqa', 'task': 'hop', 'a': "Arthur's Magazine: Arthur's Magazine (1844–1846) was an American literary "
     'periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, '
     'J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey\'s Lady\'s '
     'Book".', 'b': "First for Women: First for Women is a woman's magazine published by Bauer Media Group in the USA. "
     'The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the '
     'magazine was 1,310,696 copies.', 'question': "Which magazine was started first Arthur's Magazine or First for "
     'Women?', 'answer': "Arthur's Magazine"},
    {'src': 'mnli', 'task': 'nli', 'premise': 'Each of the men wore leather armor and dressed in the style of heavy '
     'riders.', 'hypothesis': 'The men were naked.', 'label': 'contradiction'},
    {'src': 'gsm8k', 'task': 'math', 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold '
     'half as many clips in May. How many clips did Natalia sell altogether in April and May?',
     'solution': 'Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\nThe '
     'answer is 72.'},
    {'src': 'opus-fr', 'task': 'translate', 'source_lang': 'English', 'source': 'I can do that.',
     'target_lang': 'French', 'target': 'Je peux le faire.'},
    {'src': 'paws', 'task': 'paraphrase', 'a': 'The music was composed by Darsan Raman and the lyrics written by '
     'Mariamma Philip.', 'b': 'The music was composed by Darsan Raman and lyrics was written by Mariamma Philip.'}]
FIELDS = dict(qa='answer', hop='answer', nli='label', math='solution', code_out='output', translate='target',
              paraphrase='b')


def score(task, item, text):
    return play_tasks.score_one('co_' + task, item, text, 'quill', None)


def script_nodes(name):
    """The syntax tree of a pairing/ script, read without running its command line."""
    return ast.parse((ROOT / 'pairing' / name).read_text())


def definitions(name, names, env):
    nodes = [n for n in ast.walk(script_nodes(name)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), env)
    return env


def test_qa_number_boundaries():
    assert score('qa', QA, '15') == 1
    assert score('qa', QA, '150') == 0


@pytest.mark.parametrize('it', MCQ, ids=lambda it: 'ABCDE'[it['answer']])
def test_mcq_answer_letters(it):
    right = 'ABCDE'[it['answer']]
    wrong = 'ABCDE'[(it['answer'] + 1) % len(it['options'])]
    for text in (right, f'({right})', f'The answer is {right}.', f'Option {right.lower()}.'):
        assert score('mcq', it, text) == 1, text
    assert score('mcq', it, f'The answer is {wrong}.') == 0
    assert score('mcq', it, f'({wrong}) or ({right})') == 0


def test_mcq_article_is_not_an_answer_letter():
    assert score('mcq', MCQ[1], 'A wildlife refuge.') == 1


def test_python_values():
    for text in ("'ba'", '"ba"', 'It returns "ba".'):
        assert score('code_out', CODE_OUT, text) == 1
    for text in ("'b a'", "['ba']", "('ba',)"):
        assert score('code_out', CODE_OUT, text) == 0


def test_references_score_one(data):
    made_up = json.loads((data / 'coread.json').read_text())['test']
    for it in REFERENCES + made_up:
        if it['task'] in FIELDS:
            assert score(it['task'], it, it[FIELDS[it['task']]]) == 1, it
            assert score(it['task'], it, '') == 0, it


def test_code_writing_with_hand_written_checks():
    # The check corpus has no MBPP tests for this item; these are computed from its problem.
    it = dict(CODE_WRITE, tests=['assert is_nonagonal(1) == 1', 'assert is_nonagonal(2) == 9',
                                 'assert is_nonagonal(3) == 24'])
    assert score('code_write', it, it['code']) == 1
    assert score('code_write', it, 'def is_nonagonal(n):\n    return n') == 1 / 3


def test_no_singleton_shuffle_batches():
    for n in range(2, 40):
        for bs in range(3, 12):
            parts = S.batches(n, bs)
            assert [i for a, b in parts for i in range(a, b)] == list(range(n))
            assert all(2 <= b - a <= bs for a, b in parts), (n, bs, parts)
    with pytest.raises(ValueError):
        S.batches(3, 2)
    assert S.batches(0, 8) == []


def test_quiz_replays_the_scored_transcript():
    cards = {x: rl.card(random.Random(i), 'test') for i, x in enumerate(link.SIDES)}
    ep = rl.session(cards, 'teach', 'quill', 2)
    quiz = rl.with_quiz(ep, ['The sampled first turn.', 'The sampled second turn.'], answers=False)
    assert set(quiz['forced']) == {0, 1, 2, 4}


def test_advantages_sign_grouping_and_singleton():
    a = rl.advantages(torch.tensor([0., 1., 8., 8.]), 2)
    assert a[0] < 0 and a[1] > 0
    torch.testing.assert_close(a[2:], torch.zeros(2))
    # A group of one has no relative reward: its advantage is zero.
    torch.testing.assert_close(rl.advantages(torch.tensor([0.2, 0.8]), 1), torch.zeros(2))


@pytest.mark.parametrize('name, elapsed', [('train.py', 209), ('rl.py', 206)])
def test_step_time_excludes_every_calibration_evaluation(name, elapsed):
    node = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'per' for t in n.targets)
                and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == 'time'
                        for c in ast.walk(n.value)))
    start = time.time() - elapsed
    env = dict(time=time, t1=start, t0_steps=start, s1=2, step=11, E=100, args=SimpleNamespace(first_eval=10),
               t_eval={5: 100, 10: 100} if name == 'train.py' else 100, eval_elapsed=200)
    if name == 'train.py':
        measured = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'measured_eval' for t in n.targets))
        exec(compile(ast.Module(body=[measured], type_ignores=[]), name, 'exec'), env)
    per = eval(compile(ast.Expression(node.value), name, 'eval'), env)
    assert per == pytest.approx(1, abs=0.1)


def test_curriculum_weights():
    task = next(n for n in ast.walk(script_nodes('rl.py')) if isinstance(n, ast.FunctionDef)
                and n.name == 'task_episodes')
    weight = next(n.value for n in task.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'w')
    values = eval(compile(ast.Expression(weight), 'rl.py', 'eval'),
                  dict(running={'never': 0., 'learning': 0.5, 'solved': 1.}, TASK_GAMES=('never', 'learning', 'solved')))
    assert values == [0.05, 0.3, 0.05]


@pytest.fixture(scope='module')
def tiny(tiny_models):
    toks, models = tiny_models(['q_proj', 'v_proj'], lora_alpha=12, std=0.02)
    pair = link.Pair(models, toks, (1, 2, 4), heads=2, head_dim=16)
    for x in link.SIDES:
        pair.bridge[x].alpha.data.fill_(0.5)
    pads = {x: toks[x].pad_token_id for x in link.SIDES}
    ep = dict(mode='duplex', system={x: f'You are {x}.' for x in link.SIDES}, task={x: 'Say hello.' for x in link.SIDES},
              reply={x: ('Hello, good morning!', 0.3 if x == 'quill' else 1.0) for x in link.SIDES})
    return SimpleNamespace(toks=toks, models=models, pair=pair, pads=pads, ep=ep)


def test_singleton_shuffle_rejected(tiny):
    b = S.collate([S.build(tiny.ep, tiny.toks)], tiny.pads)
    with pytest.raises(ValueError):
        tiny.pair.joint(b, mode='shuffle')
    with pytest.raises(ValueError):
        S.generate(tiny.pair, [tiny.ep], mode='shuffle', max_turn=1)


def test_forced_generation_is_not_truncated(tiny):
    b = S.collate([S.build(tiny.ep, tiny.toks)], tiny.pads)
    with torch.no_grad():
        h, _ = tiny.pair.joint(b)
        out = S.generate(tiny.pair, [tiny.ep], force=True, max_turn=1, keep=True)[0]
    for x in link.SIDES:
        assert out['text'][x] == tiny.ep['reply'][x][0]
        torch.testing.assert_close(h[x][0], out['h'][x], atol=3e-6, rtol=1e-5)


@pytest.mark.parametrize('mode', ['duplex', 'dialogue'])
def test_sampled_rollout_replay_is_exact(tiny, mode):
    ep = tiny.ep if mode == 'duplex' else dict(tiny.ep, mode='dialogue', turns=[('quill', '', 1.0), ('ember', '', 1.0)])
    out = S.generate(tiny.pair, [ep], max_turn=12, temperature=1., gen=torch.Generator().manual_seed(3), keep=True)[0]
    b = S.collate([S.build(S.replay(ep, out), tiny.toks)], tiny.pads)
    with torch.no_grad():
        h, _ = tiny.pair.joint(b)
    for x in link.SIDES:
        torch.testing.assert_close(h[x][0], out['h'][x], atol=4e-6, rtol=1e-5)


def test_length_limit_is_not_a_sampled_action(tiny):
    out = S.generate(tiny.pair, [tiny.ep], max_turn=0, keep_tokens=True)[0]
    b = S.collate([S.build(S.replay(tiny.ep, out), tiny.toks)], tiny.pads)
    for x in link.SIDES:
        assert out['tokens'][x] == [tiny.toks[x].eos_token_id]
        assert b[x]['pg'].tolist() == [0.0]


def test_policy_logprob_uses_sampling_temperature(tiny):
    env = definitions('rl.py', {'forward_ce'}, dict(S=S, pair=tiny.pair, toks=tiny.toks, pads=tiny.pads,
                                                    SIDES=link.SIDES, F=torch.nn.functional))
    # forward_ce returns the untempered quiz/SFT cross-entropy and the sampled policy's separately.
    b, _, ce, _, policy_ce = env['forward_ce']([tiny.ep], None, temperature=0.5)
    h, _ = tiny.pair.joint(b)
    for x in link.SIDES:
        logits = tiny.pair.token_logits(x, h[x], b[x]['idx'])
        torch.testing.assert_close(ce[x], torch.nn.functional.cross_entropy(logits, b[x]['lab'], reduction='none'))
        torch.testing.assert_close(policy_ce[x], torch.nn.functional.cross_entropy(logits / 0.5, b[x]['lab'],
                                                                                   reduction='none'))


def test_labels_weights_and_prompt_mask(tiny):
    toks, ep = tiny.toks, tiny.ep
    built = S.build(ep, toks)
    for x in link.SIDES:
        text, weight = ep['reply'][x]
        expected = S.ids(toks[x], text) + [toks[x].convert_tokens_to_ids('<|im_end|>')]
        p = S.prefix(toks[x], x, ep['system'][x], ep['task'][x]) + S.assistant_header(toks[x], x)
        assert built[x]['labels'] == [(len(p) - 1 + j, token, weight) for j, token in enumerate(expected)]
        assert built[x]['ids'] == p + expected[:-1]
        assert toks[x].eos_token_id == toks[x].convert_tokens_to_ids('<|im_end|>')
    assert toks['quill'].decode(S.assistant_header(toks['quill'], 'quill')) == \
        '<|im_start|>assistant\n<think>\n\n</think>\n\n'


def test_payload_word_named_word(tiny):
    assert 'word' in games.POOLS['words']['train']
    ep = dict(tiny.ep, game='relay_word', meta=dict(receiver='quill', value='word'),
              reply=dict(tiny.ep['reply'], quill=('The word is word.', 1.0)))
    assert S.payload(ep)['quill'] == [(12, 16)]
    enc = tiny.toks['quill'](ep['reply']['quill'][0], add_special_tokens=False, return_offsets_mapping=True)
    assert S.payload_labels(ep, tiny.toks)['quill'] == [i for i, (a, b) in enumerate(enc.offset_mapping)
                                                        if a < 16 and b > 12]


def save_adapters(tiny, d):
    for x in link.SIDES:
        tiny.models[x].save_pretrained(str(Path(d) / x))


def test_checkpoint_round_trip(tiny, tmp_path):
    b = S.collate([S.build(tiny.ep, tiny.toks)], tiny.pads)
    with torch.no_grad():
        expected, _ = tiny.pair.joint(b)
    save_adapters(tiny, tmp_path)
    torch.save(tiny.pair.checkpoint(), tmp_path / 'bridge.pt')
    restored, _ = evaluate.load_pair(str(tmp_path))
    with torch.no_grad():
        actual, _ = restored.joint(b)
    for x in link.SIDES:
        assert torch.equal(expected[x], actual[x])


def test_cuda_tagged_checkpoint_loads_on_cpu(tiny, tmp_path):
    save_adapters(tiny, tmp_path)
    raw = io.BytesIO()
    torch.save(tiny.pair.checkpoint(), raw)
    # The storages are device-neutral bytes; write the pickle's device tag as a CUDA run would, and load that archive
    # through torch.load and PEFT without a GPU.
    changed = 0
    with zipfile.ZipFile(raw) as src, zipfile.ZipFile(tmp_path / 'bridge.pt', 'w') as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith('/data.pkl'):
                changed += data.count(b'X\x03\x00\x00\x00cpu')
                data = data.replace(b'X\x03\x00\x00\x00cpu', b'X\x06\x00\x00\x00cuda:0')
            dst.writestr(item, data)
    assert changed > 0
    restored, _ = evaluate.load_pair(str(tmp_path))
    for key, value in tiny.pair.bridge.state_dict().items():
        assert torch.equal(restored.bridge.state_dict()[key], value)
    for name in ('train.py', 'rl.py'):
        assignment = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'ck' for t in n.targets))
        env = dict(torch=torch, os=os, args=SimpleNamespace(init=str(tmp_path)), DEVICE='cpu')
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), name, 'exec'), env)
        assert all(t.device.type == 'cpu' for t in env['ck']['bridge'].values()), name


@pytest.mark.parametrize('name', ['train.py', 'rl.py'])
def test_training_checkpoint_uses_saved_adapter_config(tiny, tmp_path, name):
    save_adapters(tiny, tmp_path)
    # The script's own model-loading loop, with disk IO and PEFT loading.
    loop = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'x'
                and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == 'load'
                        for c in ast.walk(n)))
    env = dict(os=os, args=SimpleNamespace(init=str(tmp_path), rank=32, ckpt='none', no_ckpt=True),
               SIDES=link.SIDES, MODELS=link.MODELS, load=link.load, models={}, toks={}, FROZEN=None,
               LoraConfig=LoraConfig, get_peft_model=get_peft_model, PeftModel=peft.PeftModel,
               set_peft_model_state_dict=peft.set_peft_model_state_dict, load_file=safetensors.torch.load_file)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), name, 'exec'), env)
    for x in link.SIDES:
        cfg = env['models'][x].peft_config['default']
        assert (cfg.r, cfg.lora_alpha, cfg.target_modules) == (4, 12, {'q_proj', 'v_proj'})
        inp = torch.tensor([S.ids(tiny.toks[x], 'Hello!')])
        with torch.no_grad():
            torch.testing.assert_close(tiny.models[x](inp).logits, env['models'][x](inp).logits, atol=0, rtol=0)


def test_distillation_teacher_alignment(tiny):
    env = definitions('train.py', {'teacher', 'distill'}, dict(
        torch=torch, S=S, MODELS=link.MODELS, SIDES=link.SIDES, TRAINED=link.SIDES, toks=tiny.toks,
        models=tiny.models, PADS=tiny.pads, DEVICE='cpu', inner=link.inner,
        END={x: tiny.toks[x].eos_token_id for x in link.SIDES}))
    eps = [dict(tiny.ep, meta={}), dict(tiny.ep, meta={'teacher': {'quill': 'Read this full passage.'}}),
           dict(tiny.ep, meta={'teacher': {x: 'A longer and different teacher prompt.' for x in link.SIDES}})]
    built = [S.build(e, tiny.toks) for e in eps]
    for ep, b in zip(eps, built):
        for x, t in env['teacher'](ep).items():
            b[x]['teacher'] = t
    result = env['distill'](built)
    for x in link.SIDES:
        offset, indices, expected = 0, [], []
        for b in built:
            n = len(b[x]['labels'])
            if 'teacher' in b[x]:
                seq, start = b[x]['teacher']
                assert seq[start + 1:] == [token for _, token, _ in b[x]['labels'][:-1]]
                assert len(seq) - start == n
                with torch.no_grad(), tiny.models[x].disable_adapter():
                    expected.append(tiny.models[x](torch.tensor([seq])).logits[0, start:].float().log_softmax(-1))
                indices.extend(range(offset, offset + n))
            offset += n
        assert result[x][0].tolist() == indices
        torch.testing.assert_close(result[x][1], torch.cat(expected), atol=2e-6, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('kind, extra', [('sft', []),
                                         ('task', ['--mix', '0', '--task_frac', '1', '--task_groups', '1', '--G', '2']),
                                         ('cards', ['--mix', '0', '--task_frac', '0', '--groups', '1', '--G', '2'])])
def test_rl_final_step_saves_and_evaluates(tiny, data, tmp_path, kind, extra):
    init = tmp_path / 'init'
    save_adapters(tiny, init)
    torch.save(tiny.pair.checkpoint(), init / 'bridge.pt')
    out = tmp_path / kind
    cmd = [sys.executable, str(ROOT / 'pairing/rl.py'), '--init', str(init), '--replies', str(data / 'replies.json'),
           '--rank', '4', '--steps', '1', '--mix', '1', '--sft_bs', '2', '--sft_coread', '0', '--dialogue', '0',
           '--styles', 'free', '--turns', '1', '--max_turn', '1', '--eval_n', '2', '--eval_task_n', '2', '--no_ckpt',
           '--coread_data', str(data / 'coread.json'), *extra, '--out', str(out), '--temperature', '0.5']
    proc = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, NL_MAXNEW='1', MPD_MEM_GIB='8'), capture_output=True,
                          text=True, timeout=600, stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert (out / 'step_1/bridge.pt').exists(), proc.stdout[-2000:]
    log = json.loads((out / 'log.json').read_text())
    assert any(row.get('step') == 1 and 'eval' in row for row in log)
