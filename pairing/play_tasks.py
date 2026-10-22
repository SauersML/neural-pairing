"""Scores of freely generated co-reading answers (coread.py task games), for self-play rewards and evaluation.

score(ep, texts) -> {x: score in [0, 1]} for every model whose reply needs the partner (meta['payload'] non-empty).
Each game has an automatic check against its reference, so no judge is needed:
  co_qa, co_hop     SQuAD token F1 against the reference answer (yes/no answers: the first yes or no must match)
  co_nli            the first relation word (entailment, contradiction, neutral) must match
  co_mcq            the option named first (by letter, else by its full text) must be the right one
  co_math           the last number written must equal the answer
  co_code_out       the returned Python literal must equal the expected value
  co_code_write     the fraction of the problem's tests that pass (run in a subprocess, 10 s limit)
  co_translate, co_paraphrase, co_transfer, co_swap, co_split
                    chrF (character 6-gram F-score, beta 2) against the reference text the partner holds
co_continue, co_instruct and co_distill have no automatic check (score() leaves them out); they stay supervised.
"""
import ast
import collections
import json
import os
import re
import secrets
import string
import subprocess
import sys
import tempfile

SCORED = ('co_qa', 'co_hop', 'co_nli', 'co_mcq', 'co_math', 'co_code_out', 'co_code_write', 'co_translate',
          'co_paraphrase', 'co_transfer', 'co_swap', 'co_split')
RELATIONS = ('entailment', 'contradiction', 'neutral')


def normalize(s):
    s = s.lower()
    s = ''.join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def token_f1(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = sum((collections.Counter(p) & collections.Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def chrf(pred, ref, n=6, beta=2.0):
    """chrF: mean character n-gram precision and recall (n = 1..6, spaces removed), combined with recall weighted beta."""
    p, r = re.sub(r'\s+', '', pred), re.sub(r'\s+', '', ref)
    precs, recs = [], []
    for k in range(1, n + 1):
        a = collections.Counter(p[i:i + k] for i in range(len(p) - k + 1))
        b = collections.Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not a or not b:
            continue
        m = sum((a & b).values())
        precs.append(m / sum(a.values()))
        recs.append(m / sum(b.values()))
    if not precs:
        return 0.0
    P, R = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if P + R == 0 else (1 + beta ** 2) * P * R / (beta ** 2 * P + R)


def first_of(text, words):
    t = text.lower()
    hits = [(m.start(), w) for w in words for m in [re.search(r'\b' + re.escape(w) + r'\b', t)] if m]
    return min(hits)[1] if hits else None


def last_number(text):
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(',', '').rstrip('.'))
    except ValueError:
        return None


def run_tests(code, tests, timeout=10):
    """Fraction of `tests` (assert statements) that pass after `code`, each in a fresh subprocess."""
    m = re.search(r'```(?:python)?\n(.*?)```', code, re.S)
    code = m.group(1) if m else code
    passed = 0
    for t in tests:
        completed = secrets.token_hex(24)
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(code + '\n\n' + t + '\n\n' + f'print({completed!r})\n')
        try:
            result = subprocess.run([sys.executable, '-I', f.name], capture_output=True, timeout=timeout)
            # An exit(0) before the assertion is not a passing test. Require its completion receipt.
            passed += result.returncode == 0 and completed.encode() in result.stdout.splitlines()
        except subprocess.TimeoutExpired:
            pass
        finally:
            os.unlink(f.name)
    return passed / max(len(tests), 1)


def score_one(game, it, text, x, ep):
    """Score of model x's generated text in an episode of `game` built from item `it`."""
    if game in ('co_qa', 'co_hop', 'co_talk'):
        ref = it['answer']
        yn = normalize(ref) in ('yes', 'no')
        if yn:
            return float(first_of(text, ('yes', 'no')) == normalize(ref))
        # Match whole normalized tokens: e.g. reference 15 must not match 150.
        contains = (' ' + normalize(ref) + ' ') in (' ' + normalize(text) + ' ')
        return max(token_f1(text, ref), float(contains and len(text) < 4 * len(ref) + 40))
    if game == 'co_nli':
        return float(first_of(text, RELATIONS) == it['label'])
    if game == 'co_mcq':
        m = re.search(r'\(([A-E])\)|\b([A-E])\)|^([A-E])(?=[.:]|\s*$)|'
                      r'\b(?:answer\s*(?:is|:)|option|choice)\s*\(?([A-E])\b', text.strip(), re.I)
        if m:
            return float('ABCDE'.index(next(g for g in m.groups() if g).upper()) == it['answer'])
        hits = [i for i, o in enumerate(it['options']) if normalize(o) and
                (' ' + normalize(o) + ' ') in (' ' + normalize(text) + ' ')]
        return float(hits == [it['answer']])
    if game == 'co_math':
        want, got = last_number(it['solution']), last_number(text)
        return float(want is not None and got is not None and abs(want - got) < 1e-6)
    if game == 'co_code_out':
        value = text.strip()
        value = re.sub(r'^(?:it\s+returns?|the\s+(?:return\s+value|output|result)\s+is|'
                       r'(?:output|result|returns?)\s*:)\s*', '', value, flags=re.I)
        fenced = re.fullmatch(r'```(?:python)?\s*\n?(.*?)\s*```\.?', value, re.S)
        value = fenced.group(1) if fenced else value
        try:
            want = ast.literal_eval(it['output'])
            got = ast.literal_eval(value.removesuffix('.'))
        except (ValueError, SyntaxError):
            return 0.0
        return float(type(got) is type(want) and got == want)
    if game == 'co_code_write':
        return run_tests(text, it.get('tests', [])) if it.get('tests') else 0.0
    if game == 'co_translate':
        return chrf(text, it['target'])
    if game == 'co_paraphrase':
        return chrf(text, it['b'])
    if game in ('co_transfer', 'co_swap', 'co_split'):
        ref = ep['reply'][x][0]
        return chrf(text, ref)
    raise ValueError(game)


def score(ep, texts):
    """{x: score} for the models whose reply needs the partner (texts: {x: generated reply})."""
    game = ep['game']
    if game not in SCORED:
        return {}
    return {x: score_one(game, ep['meta']['item'], texts[x], x, ep) for x, spans in ep['meta']['payload'].items()
            if spans}


if __name__ == '__main__':                      # checks of the scorers on hand-made cases
    assert token_f1('Saint Bernadette Soubirous', 'Saint Bernadette Soubirous') == 1.0
    assert score_one('co_qa', {'answer': 'No.'}, 'No, they do not.', 'quill', None) == 1.0
    assert score_one('co_qa', {'answer': 'Yes.'}, 'No.', 'quill', None) == 0.0
    assert score_one('co_nli', {'label': 'neutral'}, 'Neutral. The premise does not say.', 'quill', None) == 1.0
    assert score_one('co_mcq', {'options': ['a', 'b', 'c'], 'answer': 2}, '(C) c', 'quill', None) == 1.0
    assert score_one('co_mcq', {'options': ['red car', 'blue boat'], 'answer': 1}, 'It is the blue boat.', 'q', None) == 1.0
    assert score_one('co_math', {'solution': 'so 3 + 4 = 7\nThe answer is 7.'}, 'Total: 7 apples', 'q', None) == 1.0
    assert score_one('co_code_out', {'output': "[('74', 31)]"}, "It returns [('74',31)].", 'q', None) == 1.0
    assert run_tests('def f(x):\n    return x + 1', ['assert f(1) == 2', 'assert f(2) == 4']) == 0.5
    assert chrf('Je peux le faire.', 'Je peux le faire.') == 1.0 and chrf('abc', 'xyz') == 0.0
    print('scorer checks passed;', json.dumps({'chrf_partial': round(chrf('Je peux faire ça.', 'Je peux le faire.'), 3)}))
