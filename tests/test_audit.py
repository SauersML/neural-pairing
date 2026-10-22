"""Pipeline regression checks; real tokenizers/models, no mocked forward paths.

Run with MPD_MEM_GIB=8 OMP_NUM_THREADS=4 NL_TINY=1 NL_DEVICE=cpu python tests/test_audit.py.
"""
import ast
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import zipfile

os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))

import torch
import peft
import safetensors.torch
from peft import LoraConfig, get_peft_model

import coread
import evaluate
import games
import link
import play_tasks
import rl
import session as S

DATA = json.loads((ROOT / 'data/coread_check.json').read_text())


def script_nodes(name):
    """Read actual script definitions without launching its CLI/training loop."""
    return ast.parse((ROOT / 'pairing' / name).read_text())


def definitions(name, names, env):
    tree = script_nodes(name)
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), env)
    return env


class Scores(unittest.TestCase):
    def score(self, task, item, text):
        return play_tasks.score_one('co_' + task, item, text, 'quill', None)

    def test_qa_number_boundaries(self):
        it = next(i for i in DATA['test'] if i['task'] == 'qa' and i['answer'] == '15')
        self.assertEqual(self.score('qa', it, '15'), 1)
        self.assertEqual(self.score('qa', it, '150'), 0)

    def test_mcq_answer_letters(self):
        for it in (i for i in DATA['test'] if i['task'] == 'mcq'):
            right = 'ABCDE'[it['answer']]
            wrong = 'ABCDE'[(it['answer'] + 1) % len(it['options'])]
            for text in (right, f'({right})', f'The answer is {right}.', f'Option {right.lower()}.'):
                with self.subTest(text=text):
                    self.assertEqual(self.score('mcq', it, text), 1)
            self.assertEqual(self.score('mcq', it, f'The answer is {wrong}.'), 0)
            self.assertEqual(self.score('mcq', it, f'({wrong}) or ({right})'), 0)

    def test_mcq_article_is_not_an_answer_letter(self):
        it = next(i for i in DATA['test'] if i['task'] == 'mcq' and i['answer'] == 3)
        self.assertEqual(self.score('mcq', it, 'A wildlife refuge.'), 1)

    def test_python_values(self):
        it = next(i for i in DATA['test'] if i['task'] == 'code_out' and i['output'] == "'ba'")
        for text in ("'ba'", '"ba"', 'It returns "ba".'):
            self.assertEqual(self.score('code_out', it, text), 1)
        for text in ("'b a'", "['ba']", "('ba',)"):
            self.assertEqual(self.score('code_out', it, text), 0)

    def test_real_references(self):
        fields = dict(qa='answer', hop='answer', nli='label', math='solution', code_out='output',
                      translate='target', paraphrase='b')
        for it in DATA['test']:
            task = it['task']
            if task in fields:
                with self.subTest(task=task, reference=it[fields[task]]):
                    self.assertEqual(self.score(task, it, it[fields[task]]), 1)
                    self.assertEqual(self.score(task, it, ''), 0)

    def test_real_code_writing_item_with_hand_written_checks(self):
        it = next(i for i in DATA['test'] if i['task'] == 'code_write' and i['name'] == 'is_nonagonal')
        # The small check corpus lacks MBPP tests; these are independently calculated checks of its real problem.
        it = dict(it, tests=['assert is_nonagonal(1) == 1', 'assert is_nonagonal(2) == 9',
                             'assert is_nonagonal(3) == 24'])
        self.assertEqual(self.score('code_write', it, it['code']), 1)
        self.assertEqual(self.score('code_write', it, 'def is_nonagonal(n):\n    return n'), 1 / 3)


class BatchChecks(unittest.TestCase):
    def test_no_singleton_shuffle_batches(self):
        for n in range(2, 40):
            for bs in range(3, 12):
                parts = S.batches(n, bs)
                self.assertEqual([i for a, b in parts for i in range(a, b)], list(range(n)))
                self.assertTrue(all(2 <= b - a <= bs for a, b in parts), (n, bs, parts))
        with self.assertRaises(ValueError):
            S.batches(3, 2)
        self.assertEqual(S.batches(0, 8), [])

    def test_quiz_replays_the_scored_transcript(self):
        cards = {x: rl.card(random.Random(i), 'test') for i, x in enumerate(link.SIDES)}
        ep = rl.session(cards, 'teach', 'quill', 2)
        quiz = rl.with_quiz(ep, ['The sampled first turn.', 'The sampled second turn.'], answers=False)
        self.assertEqual(set(quiz['forced']), {0, 1, 2, 4})

    def test_advantages_sign_grouping_and_singleton(self):
        a = rl.advantages(torch.tensor([0., 1., 8., 8.]), 2)
        self.assertLess(a[0], 0)
        self.assertGreater(a[1], 0)
        torch.testing.assert_close(a[2:], torch.zeros(2))
        # A singleton has no relative reward signal, not an undefined update.
        torch.testing.assert_close(rl.advantages(torch.tensor([0.2, 0.8]), 1), torch.zeros(2))

    def test_minutes_excludes_all_calibration_evaluations(self):
        for name, elapsed in (('train.py', 209), ('rl.py', 206)):
            node = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'per' for t in n.targets)
                        and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                                and c.func.attr == 'time' for c in ast.walk(n.value)))
            start = time.time() - elapsed
            env = dict(time=time, t1=start, t0_steps=start, s1=2, step=11, E=100, args=SimpleNamespace(first_eval=10),
                       t_eval={5: 100, 10: 100} if name == 'train.py' else 100, eval_elapsed=200)
            if name == 'train.py':
                measured = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == 'measured_eval' for t in n.targets))
                exec(compile(ast.Module(body=[measured], type_ignores=[]), name, 'exec'), env)
            per = eval(compile(ast.Expression(node.value), name, 'eval'), env)
            self.assertAlmostEqual(per, 1, delta=0.1, msg=name)

    def test_curriculum_weights(self):
        task = next(n for n in ast.walk(script_nodes('rl.py')) if isinstance(n, ast.FunctionDef)
                    and n.name == 'task_episodes')
        weight = next(n.value for n in task.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'w')
        values = eval(compile(ast.Expression(weight), 'rl.py', 'eval'),
                      dict(running={'never': 0., 'learning': 0.5, 'solved': 1.},
                           TASK_GAMES=('never', 'learning', 'solved')))
        self.assertEqual(values, [0.05, 0.3, 0.05])


class TinyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toks, cls.models = {}, {}
        for x in link.SIDES:
            cls.toks[x], m = link.load(x)
            m = get_peft_model(m, LoraConfig(r=4, lora_alpha=12, target_modules=['q_proj', 'v_proj']))
            for name, p in m.named_parameters():
                if 'lora_B' in name:
                    torch.nn.init.normal_(p, std=0.02)
            cls.models[x] = m.eval()
        cls.pair = link.Pair(cls.models, cls.toks, (1, 2, 4), heads=2, head_dim=16)
        for x in link.SIDES:
            cls.pair.bridge[x].alpha.data.fill_(0.5)
        cls.pads = {x: cls.toks[x].pad_token_id for x in link.SIDES}
        cls.ep = dict(mode='duplex', system={x: f'You are {x}.' for x in link.SIDES},
                      task={x: 'Say hello.' for x in link.SIDES},
                      reply={x: ('Hello, good morning!', 0.3 if x == 'quill' else 1.0) for x in link.SIDES})

    def test_singleton_shuffle_rejected(self):
        b = S.collate([S.build(self.ep, self.toks)], self.pads)
        with self.assertRaises(ValueError):
            self.pair.joint(b, mode='shuffle')
        with self.assertRaises(ValueError):
            S.generate(self.pair, [self.ep], mode='shuffle', max_turn=1)

    def test_forced_generation_is_not_truncated(self):
        built = S.build(self.ep, self.toks)
        b = S.collate([built], self.pads)
        with torch.no_grad():
            h, _ = self.pair.joint(b)
            out = S.generate(self.pair, [self.ep], force=True, max_turn=1, keep=True)[0]
        for x in link.SIDES:
            self.assertEqual(out['text'][x], self.ep['reply'][x][0])
            torch.testing.assert_close(h[x][0], out['h'][x], atol=3e-6, rtol=1e-5)

    def test_sampled_rollout_replay_is_exact(self):
        for mode in ('duplex', 'dialogue'):
            ep = self.ep if mode == 'duplex' else dict(self.ep, mode='dialogue',
                                                     turns=[('quill', '', 1.0), ('ember', '', 1.0)])
            out = S.generate(self.pair, [ep], max_turn=12, temperature=1.,
                             gen=torch.Generator().manual_seed(3), keep=True)[0]
            row = S.replay(ep, out)
            b = S.collate([S.build(row, self.toks)], self.pads)
            with torch.no_grad():
                h, _ = self.pair.joint(b)
            for x in link.SIDES:
                with self.subTest(mode=mode, side=x):
                    torch.testing.assert_close(h[x][0], out['h'][x], atol=4e-6, rtol=1e-5)

    def test_length_limit_is_not_a_sampled_action(self):
        out = S.generate(self.pair, [self.ep], max_turn=0, keep_tokens=True)[0]
        b = S.collate([S.build(S.replay(self.ep, out), self.toks)], self.pads)
        for x in link.SIDES:
            self.assertEqual(out['tokens'][x], [self.toks[x].eos_token_id])
            self.assertEqual(b[x]['pg'].tolist(), [0.0])

    def test_policy_logprob_uses_sampling_temperature(self):
        env = definitions('rl.py', {'forward_ce'}, dict(
            S=S, pair=self.pair, toks=self.toks, pads=self.pads, SIDES=link.SIDES, F=torch.nn.functional))
        # The real helper's API returns the untempered quiz/SFT CE and the sampled policy CE separately.
        b, _, ce, _, policy_ce = env['forward_ce']([self.ep], None, temperature=0.5)
        h, _ = self.pair.joint(b)
        for x in link.SIDES:
            logits = self.pair.token_logits(x, h[x], b[x]['idx'])
            torch.testing.assert_close(ce[x], torch.nn.functional.cross_entropy(logits, b[x]['lab'], reduction='none'))
            torch.testing.assert_close(policy_ce[x], torch.nn.functional.cross_entropy(
                logits / 0.5, b[x]['lab'], reduction='none'))

    def test_labels_weights_and_prompt_mask(self):
        built = S.build(self.ep, self.toks)
        for x in link.SIDES:
            text, weight = self.ep['reply'][x]
            expected = S.ids(self.toks[x], text) + [self.toks[x].convert_tokens_to_ids('<|im_end|>')]
            p = S.prefix(self.toks[x], x, self.ep['system'][x], self.ep['task'][x]) + S.assistant_header(self.toks[x], x)
            self.assertEqual(built[x]['labels'], [(len(p) - 1 + j, token, weight) for j, token in enumerate(expected)])
            self.assertEqual(built[x]['ids'], p + expected[:-1])
        self.assertEqual(self.toks['quill'].decode(S.assistant_header(self.toks['quill'], 'quill')),
                         '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        for x in link.SIDES:
            self.assertEqual(self.toks[x].eos_token_id, self.toks[x].convert_tokens_to_ids('<|im_end|>'))

    def test_payload_word_named_word(self):
        self.assertIn('word', games.POOLS['words']['train'])
        ep = dict(self.ep, game='relay_word', meta=dict(receiver='quill', value='word'),
                  reply=dict(self.ep['reply'], quill=('The word is word.', 1.0)))
        self.assertEqual(S.payload(ep)['quill'], [(12, 16)])
        enc = self.toks['quill'](ep['reply']['quill'][0], add_special_tokens=False, return_offsets_mapping=True)
        expected = [i for i, (a, b) in enumerate(enc.offset_mapping) if a < 16 and b > 12]
        self.assertEqual(S.payload_labels(ep, self.toks)['quill'], expected)

    def test_checkpoint_round_trip(self):
        b = S.collate([S.build(self.ep, self.toks)], self.pads)
        with torch.no_grad():
            expected, _ = self.pair.joint(b)
        with tempfile.TemporaryDirectory() as d:
            for x in link.SIDES:
                self.models[x].save_pretrained(str(Path(d) / x))
            torch.save(self.pair.checkpoint(), Path(d) / 'bridge.pt')
            restored, _ = evaluate.load_pair(d)
            with torch.no_grad():
                actual, _ = restored.joint(b)
            for x in link.SIDES:
                self.assertTrue(torch.equal(expected[x], actual[x]))
            del restored

    def test_cuda_tagged_checkpoint_loads_on_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            for x in link.SIDES:
                self.models[x].save_pretrained(str(Path(d) / x))
            raw = io.BytesIO()
            torch.save(self.pair.checkpoint(), raw)
            # Serialized storages are device-neutral bytes; use the exact pickle device tag written on CUDA.
            # This exercises the real torch.load/PEFT path on a CUDA-origin archive without allocating a GPU.
            changed = 0
            with zipfile.ZipFile(raw) as src, zipfile.ZipFile(Path(d) / 'bridge.pt', 'w') as dst:
                for item in src.infolist():
                    data = src.read(item.filename)
                    if item.filename.endswith('/data.pkl'):
                        changed += data.count(b'X\x03\x00\x00\x00cpu')
                        data = data.replace(b'X\x03\x00\x00\x00cpu', b'X\x06\x00\x00\x00cuda:0')
                    dst.writestr(item, data)
            self.assertGreater(changed, 0)
            restored, _ = evaluate.load_pair(d)
            for key, value in self.pair.bridge.state_dict().items():
                self.assertTrue(torch.equal(restored.bridge.state_dict()[key], value))
            for name in ('train.py', 'rl.py'):
                assignment = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == 'ck' for t in n.targets))
                env = dict(torch=torch, os=os, args=SimpleNamespace(init=d), DEVICE='cpu')
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), name, 'exec'), env)
                self.assertTrue(all(t.device.type == 'cpu' for t in env['ck']['bridge'].values()))

    def test_training_checkpoint_uses_saved_adapter_config(self):
        with tempfile.TemporaryDirectory() as d:
            for x in link.SIDES:
                self.models[x].save_pretrained(str(Path(d) / x))
            for name in ('train.py', 'rl.py'):
                with self.subTest(script=name):
                    # Execute the script's real model-loading loop, including actual disk IO and PEFT loading.
                    loop = next(n for n in ast.walk(script_nodes(name)) if isinstance(n, ast.For)
                                and isinstance(n.target, ast.Name) and n.target.id == 'x'
                                and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                                        and c.func.id == 'load' for c in ast.walk(n)))
                    env = dict(os=os, args=SimpleNamespace(init=d, rank=32, ckpt='none', no_ckpt=True),
                               SIDES=link.SIDES, MODELS=link.MODELS, load=link.load, models={}, toks={}, FROZEN=None,
                               LoraConfig=LoraConfig, get_peft_model=get_peft_model, PeftModel=peft.PeftModel,
                               set_peft_model_state_dict=peft.set_peft_model_state_dict,
                               load_file=safetensors.torch.load_file)
                    exec(compile(ast.Module(body=[loop], type_ignores=[]), name, 'exec'), env)
                    for x in link.SIDES:
                        cfg = env['models'][x].peft_config['default']
                        self.assertEqual((cfg.r, cfg.lora_alpha, cfg.target_modules), (4, 12, {'q_proj', 'v_proj'}))
                        inp = torch.tensor([S.ids(self.toks[x], 'Hello!')])
                        with torch.no_grad():
                            torch.testing.assert_close(self.models[x](inp).logits, env['models'][x](inp).logits,
                                                       atol=0, rtol=0)
                    del env

    def test_distillation_teacher_alignment(self):
        env = definitions('train.py', {'teacher', 'distill'}, dict(
            torch=torch, S=S, MODELS=link.MODELS, SIDES=link.SIDES, TRAINED=link.SIDES, toks=self.toks,
            models=self.models, PADS=self.pads, DEVICE='cpu', inner=link.inner,
            END={x: self.toks[x].eos_token_id for x in link.SIDES}))
        eps = [dict(self.ep, meta={}), dict(self.ep, meta={'teacher': {'quill': 'Read this full passage.'}}),
               dict(self.ep, meta={'teacher': {x: 'A longer and different teacher prompt.' for x in link.SIDES}})]
        built = [S.build(e, self.toks) for e in eps]
        for ep, b in zip(eps, built):
            for x, t in env['teacher'](ep).items():
                b[x]['teacher'] = t
        result = env['distill'](built)
        for x in link.SIDES:
            offset, indices, expected = 0, [], []
            for ep, b in zip(eps, built):
                n = len(b[x]['labels'])
                if 'teacher' in b[x]:
                    seq, start = b[x]['teacher']
                    self.assertEqual(seq[start + 1:], [token for _, token, _ in b[x]['labels'][:-1]])
                    self.assertEqual(len(seq) - start, n)
                    with torch.no_grad(), self.models[x].disable_adapter():
                        expected.append(self.models[x](torch.tensor([seq])).logits[0, start:].float().log_softmax(-1))
                    indices.extend(range(offset, offset + n))
                offset += n
            self.assertEqual(result[x][0].tolist(), indices)
            torch.testing.assert_close(result[x][1], torch.cat(expected), atol=2e-6, rtol=1e-5)

    def test_rl_sft_final_step_saves_and_evaluates(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            init = d / 'init'
            for x in link.SIDES:
                self.models[x].save_pretrained(str(init / x))
            torch.save(self.pair.checkpoint(), init / 'bridge.pt')
            questions = sum(games.POOLS['questions'].values(), [])
            concept = games.POOLS['concepts']['train'][0][0]
            replies = {x: dict(own={q: 'Hello.' for q in questions},
                               steer=[(q, concept, 'Hello.') for q in questions]) for x in link.SIDES}
            (d / 'replies.json').write_text(json.dumps(replies))
            cmd = [sys.executable, str(ROOT / 'pairing/rl.py'), '--init', str(init), '--out', str(d / 'out'),
                   '--replies', str(d / 'replies.json'), '--rank', '4', '--steps', '1', '--mix', '1',
                   '--sft_bs', '2', '--sft_coread', '0', '--dialogue', '0', '--styles', 'free',
                   '--turns', '1', '--max_turn', '1', '--eval_n', '2', '--eval_task_n', '2', '--no_ckpt',
                   '--coread_data', str(ROOT / 'data/coread_check.json')]
            cases = {'sft': [], 'task': ['--mix', '0', '--task_frac', '1', '--task_groups', '1', '--G', '2'],
                     'cards': ['--mix', '0', '--task_frac', '0', '--groups', '1', '--G', '2']}
            for name, extra in cases.items():
                with self.subTest(kind=name):
                    output = d / name
                    proc = subprocess.run(cmd + extra + ['--out', str(output), '--temperature', '0.5'], cwd=ROOT,
                                          env=dict(os.environ, NL_MAXNEW='1'), capture_output=True, text=True, timeout=600)
                    self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
                    self.assertTrue((output / 'step_1/bridge.pt').exists(), proc.stdout[-2000:])
                    log = json.loads((output / 'log.json').read_text())
                    self.assertTrue(any(row.get('step') == 1 and 'eval' in row for row in log))


if __name__ == '__main__':
    unittest.main(verbosity=2)
