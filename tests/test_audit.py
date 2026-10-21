"""Pipeline regression checks; real tokenizers/models, no mocked forward paths.

Run with MPD_MEM_GIB=8 OMP_NUM_THREADS=4 NL_TINY=1 NL_DEVICE=cpu python tests/test_audit.py.
"""
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pairing'))

import torch
from peft import LoraConfig, get_peft_model

import coread
import evaluate
import games
import link
import play_tasks
import rl
import session as S

DATA = json.loads((ROOT / 'data/coread_check.json').read_text())


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
