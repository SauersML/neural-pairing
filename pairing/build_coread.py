"""Items for the co-reading games (coread.py), as data/coread.json: {'weights': {src: w}, 'train': [...], 'test': [...]}.

Two kinds of item, both from public datasets streamed in their fixed order:
- passages {'src', 'task': 'passage', 'kind', 'text'}: web and encyclopedic prose in nine languages, news, science,
  law, fiction and poetry, reviews, forums, code and math, chat and email, and structured text (tables, server logs,
  chess games, weather reports, chemical names, conlangs). Prose passages are runs of whole sentences, code and
  structured passages runs of whole lines, 8 to 80 words (at most 520 characters).
- tasks, each with a reference answer: reading questions (qa: SQuAD, BoolQ, SciQ, Dolly), two-paragraph questions
  (hop: HotpotQA, one supporting paragraph each), translation (OPUS-100, six languages, both directions),
  paraphrase (PAWS), entailment (MultiNLI), math word problems (GSM8K), code output prediction (CRUXEval), code
  writing (MBPP), instructions (Alpaca, Dolly), and multiple choice split into question and options (CommonsenseQA,
  OpenBookQA, HellaSwag).
Every 40th row of a source goes to the held-out split, so no row gives items to both splits. Sources that fail to
load are reported and skipped. The merge removes test items whose complete content components recur in training;
shared labels and short answers are allowed, and near-duplicates still need review.

Usage: python build_coread.py [--train 5000] [--test 100] [--procs 10] [--only a,b] [--redo]   (per source)
Each source is built in its own process into data/coread_parts/NAME.json; data/coread.json merges the parts.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import zlib

from datasets import load_dataset
from coread_data import held_out

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT = re.compile(r'(?<=[.!?。！？])\s+')
MAX_CHARS = 520                                  # passages
MAX_CONTEXT = 700                                # a task's reading
LANGS = {'en': 'English', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'zh': 'Chinese', 'ja': 'Japanese',
         'ru': 'Russian'}


def chat(msgs, user='User', bot='Assistant'):
    return '\n'.join(f"{user if m.get('role', m.get('from')) in ('user', 'human') else bot}: "
                     f"{m.get('content', m.get('value', ''))}" for m in msgs)


def table(r):
    rows = [[c['text'] for c in row] for row in r['tableHeaders'] + r['tableData']]
    return f"{r['pgTitle']}: {r['tableCaption']}\n" + '\n'.join(' | '.join(row) for row in rows)


def untag(html):
    return re.sub(r'[ \t]+', ' ', re.sub(r'<[^>]+>', ' ', html))


def detok(s):
    """Undo PTB-style spacing (PAWS): 'a , b ( c )' -> 'a, b (c)'."""
    s = re.sub(r' ([,.;:!?%)\]])', r'\1', s)
    return re.sub(r'([(\[$]) ', r'\1', s).replace(" 's", "'s").replace(" n't", "n't").strip()


# Passage sources: (name, hub id, config, split, kind, weight, text of a row, rows joined into one document).
# kind: prose (sentence runs) or lines (line runs).
PASSAGES = [
    ('fineweb-edu', 'HuggingFaceFW/fineweb-edu', 'sample-10BT', 'train', 'prose', 4, lambda r: r['text'], 1),
    ('fineweb', 'HuggingFaceFW/fineweb', 'sample-10BT', 'train', 'prose', 4, lambda r: r['text'], 1),
    ('wikipedia-en', 'wikimedia/wikipedia', '20231101.en', 'train', 'prose', 4, lambda r: r['text'], 1),
    *[(f'wikipedia-{c}', 'wikimedia/wikipedia', f'20231101.{c}', 'train', 'prose', 1, lambda r: r['text'], 1)
      for c in ('fr', 'de', 'es', 'zh', 'ja', 'ru', 'ar')],
    ('cosmopedia-stories', 'HuggingFaceTB/cosmopedia', 'stories', 'train', 'prose', 1, lambda r: r['text'], 1),
    ('cosmopedia-wikihow', 'HuggingFaceTB/cosmopedia', 'wikihow', 'train', 'prose', 1, lambda r: r['text'], 1),
    ('tinystories', 'roneneldan/TinyStories', None, 'train', 'prose', 1, lambda r: r['text'], 1),
    ('writingprompts', 'euclaise/writingprompts', None, 'train', 'prose', 1, lambda r: r['story'], 1),
    ('gutenberg-poetry', 'biglam/gutenberg-poetry-corpus', None, 'train', 'lines', 1, lambda r: r['line'], 24),
    ('cc-news', 'vblagoje/cc_news', None, 'train', 'prose', 2, lambda r: r['text'], 1),
    ('arxiv-abstracts', 'gfissore/arxiv-abstracts-2021', None, 'train', 'prose', 2, lambda r: r['abstract'], 1),
    ('pubmed', 'ccdv/pubmed-summarization', 'document', 'train', 'prose', 1, lambda r: r['article'], 1),
    ('terms-of-service', 'joelniklaus/online_terms_of_service', None, 'train', 'prose', 1, lambda r: r['sentence'], 6),
    ('imdb', 'stanfordnlp/imdb', None, 'train', 'prose', 1, lambda r: r['text'].replace('<br />', ' '), 1),
    ('yelp', 'Yelp/yelp_review_full', None, 'train', 'prose', 1, lambda r: r['text'], 1),
    ('askscience', 'HuggingFaceGECLM/REDDIT_comments', None, 'askscience', 'prose', 1, lambda r: r['body'], 1),
    ('explainlikeimfive', 'HuggingFaceGECLM/REDDIT_comments', None, 'explainlikeimfive', 'prose', 1,
     lambda r: r['body'], 1),
    *[(f'stackexchange-{s}', 'HuggingFaceTB/stackexchange_2025_md', f'{s}.stackexchange.com', 'train', 'lines', 1,
       lambda r: r['ThreadText'], 1) for s in ('math', 'stats', 'physics', 'cooking', 'travel', 'unix', 'english')],
    ('ultrachat', 'HuggingFaceH4/ultrachat_200k', None, 'train_sft', 'lines', 2, lambda r: chat(r['messages']), 1),
    ('wildchat', 'allenai/WildChat-1M', None, 'train', 'lines', 1,
     lambda r: None if r['toxic'] or r['redacted'] else chat(r['conversation']), 1),
    ('enron-emails', 'corbt/enron-emails', None, 'train', 'lines', 1, lambda r: r['body'], 1),
    ('sms', 'ucirvine/sms_spam', None, 'train', 'prose', 1, lambda r: None if r['label'] else r['sms'], 1),
    ('open-web-math', 'open-web-math/open-web-math', None, 'train', 'lines', 2, lambda r: r['text'], 1),
    ('python', 'codeparrot/codeparrot-clean-valid', None, 'train', 'lines', 2, lambda r: r['content'], 1),
    ('rosetta-code', 'christopher/rosetta-code', None, 'train', 'lines', 1, lambda r: r['code'], 1),
    ('linux-man', 'memorylost731/linux_man_pages_library', None, 'train', 'lines', 1, lambda r: r['text'], 1),
    ('recipes', 'corbt/all-recipes', None, 'train', 'lines', 1, lambda r: r['input'], 1),
    ('wikitables', 'penfever/wikitables', None, 'train', 'lines', 1, table, 1),
    ('loghub', 'bolu61/loghub_2', None, 'train', 'lines', 1, lambda r: r['text'], 24),
    ('chess', 'Lichess/standard-chess-games', None, 'train', 'prose', 1, lambda r: r['movetext'], 1),
    ('metar', 'consgicody/asos-metar-archive', None, 'train', 'lines', 1, lambda r: r['metar'], 12),
    ('chemistry', 'hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC', None, 'train', 'lines', 1,
     lambda r: f"{r['iupac']}: {r['formula']}, {r['SMILES']}", 6),
    ('scp-wiki', 'AMead10/scp-wiki-dataset', None, 'train', 'prose', 1, lambda r: untag(r['content']), 1),
    ('brainteasers', 'tasksource/brainteasers', 'SP', 'train', 'prose', 1, lambda r: r['question'], 1),
    ('tatoeba-tokipona', 'NetherQuartz/tatoeba-tokipona', None, 'train', 'lines', 1, lambda r: r['tok'], 6),
    ('lojban', 'damfle/english-lojban-pairs', None, 'train', 'lines', 1, lambda r: r['jbo'], 6),
    ('latin', 'PleIAs/Latin-PD', None, 'train', 'prose', 1, lambda r: r['text'], 1),
]


def window(text, must=None, limit=MAX_CONTEXT):
    """A run of whole sentences of at most `limit` characters; it contains the character offset `must` if given."""
    sents, pos = [], 0
    for s in SENT.split(text):
        i = text.find(s, pos)
        sents.append((i, s))
        pos = i + len(s)
    if must is None:
        out = ''
        for _, s in sents:
            if len(out) + len(s) + 1 > limit:
                break
            out = (out + ' ' + s).strip()
        return out or None
    k = max([j for j, (i, _) in enumerate(sents) if i <= must] or [0])
    lo, hi = k, k + 1
    while True:
        grew = False
        for a, b in ((lo - 1, hi), (lo, hi + 1)):
            if 0 <= a and b <= len(sents) and sum(len(s) + 1 for _, s in sents[a:b]) <= limit:
                lo, hi, grew = a, b, True
        if not grew:
            break
    out = ' '.join(s for _, s in sents[lo:hi])
    return out if len(out) <= limit else None


def squad(r):
    a = r['answers']
    ctx = window(r['context'], a['answer_start'][0])
    return ctx and a['text'][0] in ctx and dict(task='qa', context=ctx, question=r['question'], answer=a['text'][0])


def boolq(r):
    ctx = window(r['passage'])
    q = r['question'][0].upper() + r['question'][1:] + '?'
    return ctx and dict(task='qa', context=ctx, question=q, answer='Yes.' if r['answer'] else 'No.')


def sciq(r):
    ctx = window(r['support'])
    return ctx and dict(task='qa', context=ctx, question=r['question'], answer=r['correct_answer'][0].upper() +
                        r['correct_answer'][1:] + '.')


def dolly(r):
    if len(r['response']) > 500:
        return None
    if r['context']:
        ctx = r['context'] if len(r['context']) <= MAX_CONTEXT else None
        return ctx and dict(task='qa', context=ctx, question=r['instruction'], answer=r['response'])
    return dict(task='instruct', instruction=r['instruction'], response=r['response'])


def alpaca(r):
    ins = r['instruction'] + ('\n\n' + r['input'] if r['input'] else '')
    return len(r['output']) <= 500 and len(ins) <= MAX_CONTEXT and dict(task='instruct', instruction=ins,
                                                                         response=r['output'])


def hotpot(r):
    titles = list(dict.fromkeys(r['supporting_facts']['title']))
    if len(titles) != 2:
        return None
    paras = dict(zip(r['context']['title'], r['context']['sentences']))
    out = []
    for t in titles:
        sents = paras.get(t)
        if not sents:
            return None
        need = [i for ti, i in zip(r['supporting_facts']['title'], r['supporting_facts']['sent_id']) if ti == t]
        p = window(''.join(sents), sum(len(s) for s in sents[:min(need)]) if need else None, 450)
        if not p or any(i < len(sents) and sents[i].strip() not in p for i in need):
            return None
        out.append(f'{t}: {p}')
    return dict(task='hop', a=out[0], b=out[1], question=r['question'], answer=r['answer'])


def opus(lang):
    def f(r):
        t = r['translation']
        en, xx = t['en'].strip(), t[lang].strip()
        short = (lambda s: len(s) < 12) if lang in ('zh', 'ja') else (lambda s: len(s.split()) < 4)
        if en == xx or short(xx) or len(en.split()) < 4 or max(len(en), len(xx)) > 300:
            return None
        a, b = (('en', en), (lang, xx)) if zlib.crc32(en.encode()) % 2 else ((lang, xx), ('en', en))
        return dict(task='translate', source_lang=LANGS[a[0]], source=a[1], target_lang=LANGS[b[0]], target=b[1])
    return f


def paws(r):
    return r['label'] == 1 and dict(task='paraphrase', a=detok(r['sentence1']), b=detok(r['sentence2']))


def mnli(r):
    lab = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}.get(r['label'])
    return lab and len(r['premise']) <= 400 and dict(task='nli', premise=r['premise'], hypothesis=r['hypothesis'],
                                                     label=lab)


def gsm8k(r):
    body, ans = r['answer'].split('####')
    body = re.sub(r'<<[^>]*>>', '', body).strip()
    return dict(task='math', question=r['question'], solution=f'{body}\nThe answer is {ans.strip()}.')


def cruxeval(r):
    return len(r['code']) <= 600 and dict(task='code_out', code=r['code'], call=f"f({r['input']})", output=r['output'])


def mbpp(r):
    code = r['code'].replace('\r\n', '\n').replace('\t', '    ')
    tests = [(r['test_setup_code'] + '\n' if r['test_setup_code'] else '') + t for t in r['test_list']]
    return len(code) <= 600 and dict(task='code_write', problem=f"{r['text']} It must pass: {r['test_list'][0]}",
                                     code=code, tests=tests)


def mcq(get):
    def f(r):
        q, opts, k = get(r)
        if not 2 <= len(opts) <= 5 or not 0 <= k < len(opts):
            return None
        return dict(task='mcq', question=q, options=opts, answer=k)
    return f


TASKS = [
    ('squad', 'rajpurkar/squad', None, 'train', 3, squad),
    ('boolq', 'google/boolq', None, 'train', 1, boolq),
    ('sciq', 'allenai/sciq', None, 'train', 1, sciq),
    ('dolly', 'databricks/databricks-dolly-15k', None, 'train', 2, dolly),
    ('alpaca', 'tatsu-lab/alpaca', None, 'train', 2, alpaca),
    ('hotpotqa', 'hotpotqa/hotpot_qa', 'distractor', 'train', 3, hotpot),
    *[(f'opus-{c}', 'Helsinki-NLP/opus-100', '-'.join(sorted(('en', c))), 'train', 1, opus(c))
      for c in ('fr', 'de', 'es', 'zh', 'ja', 'ru')],
    ('paws', 'google-research-datasets/paws', 'labeled_final', 'train', 2, paws),
    ('mnli', 'nyu-mll/multi_nli', None, 'train', 2, mnli),
    ('gsm8k', 'openai/gsm8k', 'main', 'train', 2, gsm8k),
    ('cruxeval', 'cruxeval-org/cruxeval', None, 'test', 1, cruxeval),
    *[(f'mbpp-{sp}', 'google-research-datasets/mbpp', 'full', sp, 1, mbpp) for sp in ('train', 'test', 'validation', 'prompt')],
    ('commonsenseqa', 'tau/commonsense_qa', None, 'train', 1,
     mcq(lambda r: (r['question'], r['choices']['text'], r['choices']['label'].index(r['answerKey'])))),
    ('openbookqa', 'allenai/openbookqa', 'main', 'train', 1,
     mcq(lambda r: (r['question_stem'], r['choices']['text'], r['choices']['label'].index(r['answerKey'])))),
    ('hellaswag', 'Rowan/hellaswag', None, 'train', 1,
     mcq(lambda r: (r['ctx'], [e.strip() for e in r['endings']], int(r['label']) if r['label'] != '' else -1))),
]


def words(s):
    return len(s.split())


def runs(units, joiner, rng, cap):
    """Non-overlapping runs of consecutive units with a random target length of 8-80 words, at most `cap` runs."""
    out, i = [], 0
    while i < len(units) and len(out) < cap:
        target = rng.choice((rng.randint(8, 30), rng.randint(20, 80)))
        j, n = i, 0
        while j < len(units) and n < target:
            n += max(1, words(units[j]))
            j += 1
        text = joiner.join(units[i:j]).strip()
        if 8 <= words(text) and len(text) <= MAX_CHARS and '\ufffd' not in text and \
                not (joiner == ' ' and text[0].islower()):
            out.append(text)
            i = j
        else:
            i += 1
    return out


def passages(text, kind, rng, cap=2):
    if kind == 'prose':
        return runs([s for s in SENT.split(' '.join(text.split())) if s], ' ', rng, cap)
    lines = [ln.rstrip() for ln in text.replace('\r', '').split('\n')]
    lines = [ln[:MAX_CHARS] for ln in lines if ln.strip()]
    return runs(lines, '\n', rng, cap)


def stream(hub, config, split):
    return load_dataset(hub, config, split=split, streaming=True) if config else \
        load_dataset(hub, split=split, streaming=True)


def build(spec, n_train, n_test, max_rows=80000):
    """Items of one source: {'train': [...], 'test': [...]}, or None if it fails to load."""
    if len(spec) == 8:
        name, hub, config, split, kind, weight, get, group = spec
    else:
        (name, hub, config, split, weight, convert), group = spec, 1
    rng = random.Random(name)
    out = {'train': [], 'test': []}
    buf = []
    try:
        for k, row in enumerate(stream(hub, config, split)):
            if k >= max_rows or (len(out['train']) >= n_train and len(out['test']) >= n_test):
                break
            part = 'test' if (k // group) % 40 == 0 else 'train'
            if len(out[part]) >= (n_test if part == 'test' else n_train):
                continue
            if len(spec) == 8:
                text = get(row)
                if text:
                    buf.append(text)
                if len(buf) < group and (k + 1) % group:
                    continue
                doc, buf = '\n'.join(buf), []
                items = [dict(task='passage', kind=kind, text=p) for p in passages(doc, kind, rng)] if doc else []
            else:
                item = convert(row)
                items = [item] if item else []
            for it in items[:(n_test if part == 'test' else n_train) - len(out[part])]:
                out[part].append(dict(src=name, **it))
    except Exception as e:                       # a failing source keeps what it gave so far, and is reported
        print(f'{name}: FAILED after {len(out["train"])} train items: {type(e).__name__}: {str(e)[:200]}', flush=True)
        if not out['train']:
            return name, weight, None
    print(f'{name}: {len(out["train"])} train, {len(out["test"])} test', flush=True)
    return name, weight, out


def merge(names, specs):
    """data/coread.json from the part files of the named sources."""
    res = {'weights': {}, 'train': [], 'test': []}
    weight = {s[0]: s[5] if len(s) == 8 else s[4] for s in specs}
    for name in names:
        part = os.path.join(PARTS, f'{name}.json')
        if not os.path.exists(part):
            continue
        out = json.load(open(part))
        if out['train']:
            res['weights'][name] = weight[name]
            res['train'] += out['train']
            res['test'] += out['test']
    res['test'] = held_out(res['train'], res['test'])
    return res


PARTS = os.path.join(ROOT, 'data', 'coread_parts')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=5000)
    ap.add_argument('--test', type=int, default=100)
    ap.add_argument('--procs', type=int, default=10, help='sources built at once, one process each')
    ap.add_argument('--only', default=None, help='comma-separated source names')
    ap.add_argument('--one', default=None, help='build this one source into its part file (used by the main run)')
    ap.add_argument('--redo', action='store_true', help='rebuild sources whose part file exists')
    args = ap.parse_args()
    specs = PASSAGES + TASKS
    if args.one:
        name, weight, out = build(next(s for s in specs if s[0] == args.one), args.train, args.test)
        if out:
            os.makedirs(PARTS, exist_ok=True)
            json.dump(out, open(os.path.join(PARTS, f'{name}.json') + '.tmp', 'w'))
            os.replace(os.path.join(PARTS, f'{name}.json') + '.tmp', os.path.join(PARTS, f'{name}.json'))
        sys.stdout.flush()
        os._exit(0)                                   # streaming threads can hang a normal interpreter exit
    # each source in its own process, so a crash in one (a native error in a stream) loses only that source
    names = [s[0] for s in specs if not args.only or s[0] in args.only.split(',')]
    todo = [n for n in names if args.redo or not os.path.exists(os.path.join(PARTS, f'{n}.json'))]
    running = []
    while todo or running:
        while todo and len(running) < args.procs:
            n = todo.pop(0)
            running.append((n, time.time(), subprocess.Popen([sys.executable, os.path.abspath(__file__), '--one', n, '--train',
                                                 str(args.train), '--test', str(args.test)], stdin=subprocess.DEVNULL)))
        time.sleep(2)
        for n, t0, pr in list(running):
            if pr.poll() is None and time.time() - t0 > 1200:
                pr.kill()                             # a stalled stream: the source is reported missing
            if pr.poll() is not None:
                running.remove((n, t0, pr))
                if pr.returncode:
                    print(f'{n}: process ended with code {pr.returncode}', flush=True)
    res = merge([s[0] for s in specs], specs)
    failed = [n for n in names if not os.path.exists(os.path.join(PARTS, f'{n}.json'))]
    json.dump(res, open(os.path.join(ROOT, 'data', 'coread.json'), 'w'))
    print(f'{len(res["weights"])} sources, {len(res["train"])} train, {len(res["test"])} test items -> data/coread.json; '
          f'missing: {failed}', file=sys.stderr)
