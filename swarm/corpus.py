"""Corpus-scale training data for the two talking agents (swarm/dialogue.py): episodes made automatically from real
text and real datasets, so that general communication has to emerge; the hand-made games become evaluations.

Every task is fn(rng, test) -> episode, in dialogue.py's format: dict(game, views {A, B}, answers {A, B}, weights
{A, B}, golds, cands {A, B} or absent), plus family (1 or 2), metric and max_new (tokens a greedy answer may take).
test=True draws only held-out data; an episode is a function of rng alone. Views stay under VIEW_MAX tokens and answers
under ANSWER_MAX (Qwen3 tokenizer), the limits dialogue.py trains in.

Family 1, fill-in on raw text (language modelling through the channel). A window of a document is cut into 2 * CHUNKS
consecutive chunks of CHUNK_MIN to CHUNK_MAX tokens, numbered in order; A holds the odd ones and B the even ones, each
with "[B's part]" (or A's) where the other's chunks go. Each agent's answer is its partner's chunks, after their
numbers, scored by teacher-forced cross-entropy (on vs cut); the greedy score is chrF.
  fill_web     HuggingFaceFW/fineweb-edu, sample/10BT                       35% of family 1
  fill_wiki    wikimedia/wikipedia, 20231101.en                             15%
  fill_wikix   wikimedia/wikipedia, 20231101 de, fr, es, ja and zh           5%
  fill_code    codeparrot/codeparrot-clean (bigcode/the-stack-smol is gated) 15%
  fill_math    open-web-math/open-web-math                                  10%
  fill_chat    HuggingFaceH4/ultrachat_200k, one chunk per turn (the turn's  10%
               opening, with its speaker): A holds the user's turns, B the assistant's
  fill_book    emozilla/pg19 (Project Gutenberg books), a random window      10%
Held-out documents come from each source's last file (its test split where it has one).

Family 2, split-input tasks from existing datasets, each with its own metric (training split -> training, validation or
test split -> test):
  gsm8k      openai/gsm8k: the problem's premise sentences dealt between A and B, the question to B; the final number
             (exact match)
  qa_squad   rajpurkar/squad v1.1: the paragraph to A, the question to B (token F1)
  qa_hotpot  hotpotqa/hotpot_qa: one gold page each, the question to B (token F1)
  drop       ucinlp/drop: the passage's sentences dealt between A and B, the question to B (token F1, numbers exact)
  mbpp       google-research-datasets/mbpp: the specification to A, the function's signature and tests to B; B writes
             the function (the fraction of tests passed, run in a sandboxed subprocess)
  translate  wmt/wmt14 de-en: the German sentence to A; B writes it in English (chrF)
  summarize  abisee/cnn_dailymail 3.0.0: the article to A; B writes the highlights (ROUGE-L)
  wordnet20  WordNet nouns: A knows the target noun and its definition, B has 16 nouns from near the target in the
             hierarchy; both name the target (exact match; forced choice over the 16)
In family 2 the agent that needs the other's help answers at full weight and the other at half (or a quarter where it
already has everything).

MIX: sampling weights, family 1 at 60% (split as above) and family 2 at 40% (equal per task); dialogue.py adds its own
games on top, and mix_of() renormalizes MIX over the tasks actually in the cache. prefetch() streams or downloads
everything into CACHE (env CORPUS_CACHE) in parallel threads, so training reads only local files.
Usage: python swarm/corpus.py prefetch [--cache DIR]      fills the cache
       python swarm/corpus.py show [TASK ...]              prints one training and one test episode per task
"""
import argparse
import concurrent.futures as cf
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import crumb_games as CG                    # noqa: E402

CACHE = Path(os.environ.get('CORPUS_CACHE', Path.home() / '.cache/corpus'))
TOKENIZERS = ('Qwen/Qwen3-4B', 'Qwen/Qwen3-0.6B')    # one tokenizer across Qwen3 sizes; 0.6B where 4B is not cached
VIEW_MAX, ANSWER_MAX = 350, 96
CHUNKS, CHUNK_MIN, CHUNK_MAX = 3, 22, 28            # per agent; 3 * 28 tokens plus part numbers stays under ANSWER_MAX
TEST_DOCS = 120                                     # held-out documents cached per family-1 source

# ------------------------------------------------------------------------------------------------------------ sources

# name: (repo, folder, builder, text field, weight in family 1, training documents to cache)
SOURCES = dict(
    fill_web=('HuggingFaceFW/fineweb-edu', 'sample/10BT', 'parquet', 'text', 35, 12000),
    fill_wiki=('wikimedia/wikipedia', '20231101.en', 'parquet', 'text', 15, 6000),
    fill_wikix=('wikimedia/wikipedia', ('20231101.de', '20231101.fr', '20231101.es', '20231101.ja', '20231101.zh'),
                'parquet', 'text', 5, 4000),
    fill_code=('codeparrot/codeparrot-clean', '', 'json', 'content', 15, 6000),
    fill_math=('open-web-math/open-web-math', 'data', 'parquet', 'text', 10, 4000),
    fill_chat=('HuggingFaceH4/ultrachat_200k', 'data', 'parquet', 'messages', 10, 4000),
    fill_book=('emozilla/pg19', 'data', 'parquet', 'text', 10, 200),
)
BOOK_WINDOWS, WINDOW_CHARS = 30, 6000              # pg19: windows cached per book, characters per window
DOC_CHARS = 6000                                   # other documents: their first DOC_CHARS characters are cached

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        for name in TOKENIZERS:
            try:
                _tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
                break
            except Exception:
                continue
        else:
            _tok = AutoTokenizer.from_pretrained(TOKENIZERS[0])
    return _tok


def ntok(text):
    return len(tokenizer()(text, add_special_tokens=False).input_ids)


def other(x):
    return 'A' if x == 'B' else 'B'


# ------------------------------------------------------------------------------------------------------------ cache

def path_of(name, test):
    return CACHE / f'{name}.{"test" if test else "train"}.jsonl'


_rows = {}


def rows(name, test):
    """The cached rows of one source or task and split (fetched into the cache first when missing)."""
    key = (name, bool(test))
    if key not in _rows:
        p = path_of(name, test)
        if not p.exists():
            fetch(name)
        _rows[key] = [json.loads(line) for line in open(p)]
        if not _rows[key]:
            raise ValueError(f'{p} is empty')
    return _rows[key]


def write_rows(name, test, items):
    """Writes the rows atomically (a .part file renamed into place)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = path_of(name, test)
    part = p.with_suffix('.part')
    with open(part, 'w') as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(part, p)


def repo_files(repo, folder, builder):
    """The repo's data files under folder (sorted), of the builder's kind."""
    from huggingface_hub import HfApi
    ext = ('.parquet',) if builder == 'parquet' else ('.json.gz', '.jsonl', '.jsonl.gz', '.json')
    files = [f.path for f in HfApi().list_repo_tree(repo, path_in_repo=folder or None, repo_type='dataset',
                                                    recursive=True) if f.path.endswith(ext)]
    return sorted(files)


def stream(repo, files, builder):
    from datasets import load_dataset
    ds = load_dataset(builder, data_files=[f'hf://datasets/{repo}/{f}' for f in files], split='train',
                      streaming=True)
    return iter(ds)


def text_of(name, r, field):
    if name == 'fill_chat':
        return [dict(role=m['role'], content=m['content']) for m in r[field]]
    return r[field]


def fetch_source(name):
    """Streams one family-1 source into the cache: training documents from its first files, held-out ones from its
    last file (ultrachat: its test_sft files)."""
    repo, folder, builder, field, _, n = SOURCES[name]
    folders = folder if isinstance(folder, tuple) else (folder,)
    out = {False: [], True: []}
    for k, fo in enumerate(folders):
        files = repo_files(repo, fo, builder)
        named = dict(fill_chat=('train_sft', 'test_sft'), fill_book=('/train', '/test')).get(name)
        split = {False: files[:-1] or files, True: files[-1:]}
        if named and all(any(k in f for f in files) for k in named):
            split = {False: [f for f in files if named[0] in f], True: [f for f in files if named[1] in f]}
        for test in (False, True):
            want = (TEST_DOCS if test else n) // len(folders)
            got = 0
            for r in stream(repo, split[test], builder):
                t = text_of(name, r, field)
                if name == 'fill_book':
                    rng = random.Random(hashlib.md5(t[:200].encode()).hexdigest())
                    for _ in range(BOOK_WINDOWS if not test else 4):
                        s = rng.randrange(max(1, len(t) - WINDOW_CHARS))
                        out[test].append(dict(text=t[s:s + WINDOW_CHARS]))
                    got += 1
                elif name == 'fill_chat':
                    if len(t) >= 2 * CHUNKS:
                        out[test].append(dict(turns=[dict(role=m['role'], content=m['content'][:2000]) for m in t]))
                        got += 1
                elif len(t) > 1200:
                    out[test].append(dict(text=t[:DOC_CHARS], part=k))
                    got += 1
                if got >= want:
                    break
    for test in (False, True):
        write_rows(name, test, out[test])
    return {test: len(v) for test, v in out.items()}


# task: (repo, {split: [parquet files]}, rows to stream per split or None to read whole files)
TASK_FILES = dict(
    gsm8k=('openai/gsm8k', {False: ['main/train-00000-of-00001.parquet'], True: ['main/test-00000-of-00001.parquet']}),
    qa_squad=('rajpurkar/squad', {False: ['plain_text/train-00000-of-00001.parquet'],
                                  True: ['plain_text/validation-00000-of-00001.parquet']}),
    drop=('ucinlp/drop', {False: ['data/train-00000-of-00001.parquet'],
                          True: ['data/validation-00000-of-00001.parquet']}),
    mbpp=('google-research-datasets/mbpp', {False: ['full/train-00000-of-00001.parquet',
                                                    'full/validation-00000-of-00001.parquet',
                                                    'full/prompt-00000-of-00001.parquet'],
                                            True: ['full/test-00000-of-00001.parquet']}),
    translate=('wmt/wmt14', {False: ['de-en/train-00000-of-00003.parquet'],
                             True: ['de-en/test-00000-of-00001.parquet']}),
    summarize=('abisee/cnn_dailymail', {False: ['3.0.0/train-00000-of-00003.parquet'],
                                        True: ['3.0.0/validation-00000-of-00001.parquet']}),
)
STREAM_ROWS = dict(translate=30000, summarize=20000)    # big training files: the first rows, streamed


def hub_rows(repo, files, n=None):
    """The rows of a dataset repo's parquet files: whole files from the Hugging Face cache (downloaded when missing),
    or the first n rows streamed."""
    if n:
        return [r for _, r in zip(range(n), stream(repo, files, 'parquet'))]
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    out = []
    for f in files:
        try:
            path = hf_hub_download(repo, f, repo_type='dataset', local_files_only=True)
        except Exception:
            path = hf_hub_download(repo, f, repo_type='dataset')
        out += pq.read_table(path).to_pylist()
    return out


def normal(name, r):
    if name == 'gsm8k':
        return dict(question=r['question'], answer=r['answer'].split('####')[-1].strip().replace(',', ''))
    if name == 'qa_squad':
        return dict(context=' '.join(r['context'].split()), question=r['question'].strip(),
                    answers=list(dict.fromkeys(' '.join(a.split()) for a in r['answers']['text'] if a.strip())))
    if name == 'drop':
        return dict(passage=' '.join(r['passage'].split()), question=r['question'].strip(),
                    answers=list(dict.fromkeys(r['answers_spans']['spans'])), types=list(r['answers_spans']['types']))
    if name == 'mbpp':
        return dict(text=r['text'], code=r['code'], test_list=list(r['test_list']),
                    test_setup_code=r.get('test_setup_code') or '')
    if name == 'translate':
        return dict(de=r['translation']['de'], en=r['translation']['en'])
    if name == 'summarize':
        return dict(article=r['article'], highlights=r['highlights'])
    raise ValueError(name)


def fetch_task(name):
    """Downloads one family-2 dataset's rows (normalized) into the cache, both splits."""
    if name == 'qa_hotpot':
        out = {test: CG.hotpot_rows(test) for test in (False, True)}
    elif name == 'wordnet20':
        out = wordnet_rows()
    else:
        repo, files = TASK_FILES[name]
        out = {test: [normal(name, r) for r in hub_rows(repo, files[test], None if test else STREAM_ROWS.get(name))]
               for test in (False, True)}
    for test in (False, True):
        write_rows(name, test, out[test])
    return {test: len(v) for test, v in out.items()}


def fetch(name):
    return fetch_source(name) if name in SOURCES else fetch_task(name)


def prefetch(cache=None, names=None, workers=16):
    """Fills the cache with every source and task in parallel threads; skips what is already there. Returns
    {name: rows per split or the error}."""
    global CACHE
    if cache:
        CACHE = Path(cache)
    names = names or list(SOURCES) + list(TASKS_2)
    todo = [n for n in names if not (path_of(n, False).exists() and path_of(n, True).exists())]
    report = {}
    started = time.monotonic()
    with cf.ThreadPoolExecutor(min(workers, max(1, len(todo)))) as ex:
        futs = {ex.submit(fetch, n): n for n in todo}
        for f in cf.as_completed(futs):
            n = futs[f]
            try:
                report[n] = f.result()
            except Exception as e:
                report[n] = repr(e)[:300]
            print(json.dumps(dict(name=n, got=report[n], s=round(time.monotonic() - started))), flush=True)
    return report


# ------------------------------------------------------------------------------------------------------------ family 1

def clean_cut(ids, start, end):
    """Moves a chunk boundary forward until the decoded text has no broken character there."""
    tok = tokenizer()
    while end < len(ids) and '�' in tok.decode(ids[start:end])[-2:]:
        end += 1
    return end


def chunk_doc(text, rng):
    """2 * CHUNKS consecutive chunks of CHUNK_MIN..CHUNK_MAX tokens from a random place in text, or None when text
    is too short."""
    tok = tokenizer()
    ids = tok(text, add_special_tokens=False).input_ids
    need = 2 * CHUNKS * CHUNK_MAX + 8
    if len(ids) < need:
        return None
    pos = rng.randrange(len(ids) - need + 1)
    while pos > 0 and '�' in tok.decode(ids[pos:pos + 2])[:1]:
        pos -= 1
    out = []
    for _ in range(2 * CHUNKS):
        end = clean_cut(ids, pos, pos + rng.randint(CHUNK_MIN, CHUNK_MAX))
        out.append(tok.decode(ids[pos:end]))
        pos = end
    return out


def chat_chunks(turns, rng):
    """One chunk per turn: 2 * CHUNKS consecutive turns from a user turn, each its speaker and opening words, cut to
    CHUNK_MAX tokens."""
    starts = [i for i in range(0, len(turns) - 2 * CHUNKS + 1) if turns[i]['role'] == 'user']
    if not starts:
        return None
    s = rng.choice(starts)
    tok = tokenizer()
    out = []
    for m in turns[s:s + 2 * CHUNKS]:
        who = 'User' if m['role'] == 'user' else 'Assistant'
        ids = tok(f'{who}: ' + ' '.join(m['content'].split()), add_special_tokens=False).input_ids
        end = clean_cut(ids, 0, min(len(ids), rng.randint(CHUNK_MIN, CHUNK_MAX)))
        out.append(tok.decode(ids[:end]))
    return out


KIND = dict(fill_web='a web page', fill_wiki='an encyclopedia article', fill_wikix='an encyclopedia article',
            fill_code='a source file', fill_math='a page about mathematics', fill_chat='a conversation',
            fill_book='a book')


def fill_episode(name, rng, test):
    pool = rows(name, test)
    for _ in range(200):
        r = rng.choice(pool)
        chunks = chat_chunks(r['turns'], rng) if name == 'fill_chat' else chunk_doc(r['text'], rng)
        if chunks:
            break
    else:
        raise ValueError(f'{name}: no document long enough')
    own = dict(A=range(0, 2 * CHUNKS, 2), B=range(1, 2 * CHUNKS, 2))
    views, answers = {}, {}
    for x in 'AB':
        lines = [f'({i + 1}) {chunks[i].strip()}' if i in own[x] else f'({i + 1}) [{other(x)}\'s part]'
                 for i in range(2 * CHUNKS)]
        views[x] = (f'Task: you and agent {other(x)} hold alternating parts of {KIND[name]}. Learn {other(x)}\'s '
                    f'parts from the conversation, then write them out, each after its number.\n\n' + '\n'.join(lines))
        answers[x] = '\n'.join(f'({i + 1}) {chunks[i].strip()}' for i in own[other(x)])
    return dict(game=name, family=1, metric='chrf', views=views, answers=answers, weights=dict(A=1.0, B=1.0),
                golds=dict(chunks=chunks), max_new=ANSWER_MAX + 8)


# ------------------------------------------------------------------------------------------------------------ family 2

def sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', text.strip()) if s]


def deal(units, rng):
    """Deals units (in order) between A and B, alternately from a random first agent; both get at least one when
    there are two or more."""
    first = rng.choice('AB')
    got = dict(A=[], B=[])
    for i, u in enumerate(units):
        got[first if i % 2 == 0 else other(first)].append(u)
    return got


def pick(name, rng, test, ok, tries=400):
    """A random cached row of the task's split for which ok(row) holds."""
    pool = rows(name, test)
    for _ in range(tries):
        r = rng.choice(pool)
        if ok(r):
            return r
    raise ValueError(f'{name}: no row fits after {tries} tries')


def fits(*texts, limit=VIEW_MAX - 80):
    return all(ntok(t) <= limit for t in texts)


def gsm8k(rng, test):
    def ok(r):
        s = sentences(r['question'])
        return len(s) >= 3 and '?' in s[-1] and fits(r['question'])
    r = pick('gsm8k', rng, test, ok)
    s = sentences(r['question'])
    part = deal(s[:-1], rng)
    views = dict(A=('Task: you and agent B each hold part of a math word problem, and B has its question. Help B '
                    'solve it, then give the final answer yourself as a number.\n\nYour part: ' + ' '.join(part['A'])),
                 B=('Task: agent A holds the rest of this math word problem. Solve it and give the final answer as a '
                    'number.\n\nYour part: ' + ' '.join(part['B']) + f'\nQuestion: {s[-1]}'))
    a = f'The answer is {r["answer"]}.'
    return dict(game='gsm8k', family=2, metric='number', views=views, answers=dict(A=a, B=a),
                weights=dict(A=0.5, B=1.0), golds=[r['answer']], max_new=24)


def qa_squad(rng, test):
    r = pick('qa_squad', rng, test, lambda r: r['answers'] and ntok(r['answers'][0]) <= 24 and fits(r['context']))
    views = dict(A=('Task: agent B has a question about the paragraph below. Help B answer it, then answer it '
                    f'yourself.\n\nParagraph: {r["context"]}'),
                 B=('Task: agent A has a paragraph that answers your question. Answer the question.\n\n'
                    f'Question: {r["question"]}'))
    a = r['answers'][0]
    return dict(game='qa_squad', family=2, metric='f1', views=views, answers=dict(A=a, B=a), weights=dict(A=0.5, B=1.0),
                golds=r['answers'], max_new=32)


def qa_hotpot(rng, test):
    r = pick('qa_hotpot', rng, test, lambda r: ntok(r['answer']) <= 24 and fits(*r['pages']))
    i = rng.randrange(2)
    page = {x: f'Your page, "{r["titles"][j]}": {r["pages"][j]}' for x, j in (('A', i), ('B', 1 - i))}
    views = dict(A=('Task: agent B has a question; you and B each hold one page needed to answer it. Help B answer it, '
                    f'then answer it yourself.\n\n{page["A"]}'),
                 B=('Task: agent A holds another page needed to answer your question. Answer the question.\n\n'
                    f'Question: {r["question"]}\n\n{page["B"]}'))
    a = r['answer']
    return dict(game='qa_hotpot', family=2, metric='f1', views=views, answers=dict(A=a, B=a),
                weights=dict(A=0.5, B=1.0), golds=[a], max_new=32)


def drop(rng, test):
    def ok(r):
        return (r['answers'] and len(sentences(r['passage'])) >= 2 and ntok(', '.join(r['answers'])) <= 24
                and fits(r['passage'], limit=2 * (VIEW_MAX - 100)))
    for _ in range(50):
        r = pick('drop', rng, test, ok)
        part = deal(sentences(r['passage']), rng)
        if fits(' '.join(part['A']), ' '.join(part['B']) + r['question'], limit=VIEW_MAX - 90):
            break
    else:
        raise ValueError('drop: no passage fits')
    views = dict(A=('Task: you and agent B each hold part of a passage, and B has a question about it. Help B answer '
                    'it, then answer it yourself.\n\nYour part: ' + ' '.join(part['A'])),
                 B=('Task: agent A holds the rest of this passage. Answer the question.\n\nYour part: ' +
                    ' '.join(part['B']) + f'\nQuestion: {r["question"]}'))
    a = ', '.join(r['answers'])
    return dict(game='drop', family=2, metric='f1', views=views, answers=dict(A=a, B=a), weights=dict(A=0.5, B=1.0),
                golds=[a], max_new=32)


def target_def(code, tests):
    """The signature (def line) of the function the tests call."""
    called = re.findall(r'\b([A-Za-z_]\w*)\s*\(', ' '.join(tests))
    defs = re.findall(r'^def\s+([A-Za-z_]\w*)\s*\(.*?\)\s*:', code, re.M)
    name = next((c for c in called if c in defs), defs[-1] if defs else None)
    m = re.search(rf'^def\s+{re.escape(name)}\s*\(.*?\)\s*:', code, re.M) if name else None
    return m.group(0) if m else None


def mbpp(rng, test):
    def ok(r):
        code = r['code'].replace('\r\n', '\n').replace('\t', '    ')
        return target_def(code, r['test_list']) and ntok(code) <= ANSWER_MAX - 4 and fits('\n'.join(r['test_list']))
    r = pick('mbpp', rng, test, ok)
    code = r['code'].replace('\r\n', '\n').replace('\t', '    ').strip()
    sig = target_def(code, r['test_list'])
    views = dict(A=('Task: agent B must write a Python function whose specification only you have. Tell B what it '
                    f'must do.\n\nSpecification: {r["text"]}'),
                 B=('Task: write the Python function. Agent A has its specification; you have its signature and its '
                    'tests. Write only the code.\n\nSignature: ' + sig + '\nTests:\n' + '\n'.join(r['test_list'])))
    return dict(game='mbpp', family=2, metric='tests', views=views, answers=dict(A=code, B=code),
                weights=dict(A=0.25, B=1.0), golds=dict(tests=r['test_list'], setup=r.get('test_setup_code') or ''),
                max_new=ANSWER_MAX + 8)


def translate(rng, test):
    r = pick('translate', rng, test, lambda r: 8 <= len(r['en'].split()) and ntok(r['en']) <= ANSWER_MAX - 8 and
             fits(r['de']))
    views = dict(A=f'Task: agent B must write the English translation of a German sentence that only you have.\n\n'
                   f'German: {r["de"]}',
                 B='Task: agent A has a German sentence. Write its English translation.')
    return dict(game='translate', family=2, metric='chrf', views=views, answers=dict(A=r['en'], B=r['en']),
                weights=dict(A=0.25, B=1.0), golds=[r['en']], max_new=ANSWER_MAX + 8)


def summarize(rng, test):
    def ok(r):
        return ntok(' '.join(r['highlights'].split())) <= ANSWER_MAX - 4
    keep = []
    while not keep:
        r = pick('summarize', rng, test, ok)
        n = 0
        for s in sentences(' '.join(r['article'].split())):
            k = ntok(s)
            if n + k > VIEW_MAX - 70:
                break
            keep.append(s)
            n += k
    summary = ' '.join(r['highlights'].replace(' .', '.').split())
    views = dict(A='Task: agent B must summarize a news article that only you have.\n\nArticle: ' + ' '.join(keep),
                 B='Task: agent A has a news article. Write its summary in two to four short sentences.')
    return dict(game='summarize', family=2, metric='rougeL', views=views, answers=dict(A=summary, B=summary),
                weights=dict(A=0.25, B=1.0), golds=[summary], max_new=ANSWER_MAX + 8)


def wordnet_rows():
    """WordNet nouns whose first lemma is one plain word, each with its definition and up to 15 neighbours (nouns that
    share a hypernym or grandparent hypernym); every tenth noun (by a hash of its name) is held out for testing.
    Downloads NLTK's WordNet (about 10 MB) when missing."""
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    from nltk.corpus import wordnet as wn
    names = {}
    for s in sorted(wn.all_synsets('n'), key=lambda s: s.name()):
        w = s.lemmas()[0].name()
        if re.fullmatch(r'[a-z]{3,12}', w) and w not in names and s.hypernyms():
            names[w] = s
    by_parent = {}
    for w, s in names.items():
        for h in s.hypernyms():
            by_parent.setdefault(h.name(), []).append(w)
            for g in h.hypernyms():
                by_parent.setdefault('2:' + g.name(), []).append(w)
    out = {False: [], True: []}
    for w, s in names.items():
        near = []
        for h in s.hypernyms():
            near += by_parent.get(h.name(), [])
            for g in h.hypernyms():
                near += by_parent.get('2:' + g.name(), [])
        near = sorted(set(near) - {w})
        if len(near) < 15:
            continue
        test = int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 10 == 0
        out[test].append(dict(noun=w, definition=s.definition(), near=near[:200]))
    return out


def wordnet20(rng, test):
    r = rng.choice(rows('wordnet20', test))
    cands = rng.sample(r['near'], 15) + [r['noun']]
    rng.shuffle(cands)
    views = dict(A=(f'Task: agent B has a list of 16 words, and the target is one of them. Both of you must name the '
                    f'target.\n\nThe target is "{r["noun"]}": {r["definition"]}.'),
                 B=('Task: agent A knows which of your 16 words is the target. Both of you must name the target.\n\n'
                    'Your words: ' + ', '.join(cands)))
    return dict(game='wordnet20', family=2, metric='exact', views=views, answers=dict(A=r['noun'], B=r['noun']),
                weights=dict(A=1.0, B=1.0), golds=[r['noun']], cands=dict(A=list(cands), B=list(cands)), max_new=12)


TASKS_1 = {n: (lambda n: lambda rng, test: fill_episode(n, rng, test))(n) for n in SOURCES}
TASKS_2 = dict(gsm8k=gsm8k, qa_squad=qa_squad, qa_hotpot=qa_hotpot, drop=drop, mbpp=mbpp, translate=translate,
               summarize=summarize, wordnet20=wordnet20)
TASKS = {**TASKS_1, **TASKS_2}
FAMILY = {**{n: 1 for n in TASKS_1}, **{n: 2 for n in TASKS_2}}
MIX = {**{n: 0.6 * SOURCES[n][4] / sum(s[4] for s in SOURCES.values()) for n in TASKS_1},
       **{n: 0.4 / len(TASKS_2) for n in TASKS_2}}


def available():
    """The tasks whose training and test rows are both in the cache."""
    return [n for n in TASKS if path_of(n, False).exists() and path_of(n, True).exists()]


def mix_of(names=None):
    """MIX over the given tasks (default: the available ones), each family keeping its share when it has any task
    left."""
    names = available() if names is None else list(names)
    out = {}
    for fam, share in ((1, 0.6), (2, 0.4)):
        got = {n: MIX[n] for n in names if FAMILY[n] == fam}
        out.update({n: share * v / sum(got.values()) for n, v in got.items()})
    total = sum(out.values())
    return {n: v / total for n, v in out.items()}


def episode(name, rng, test=False):
    return TASKS[name](rng, test)


def sample(rng, test=False, mix=None):
    """One episode of a task drawn by the mixing weights."""
    mix = mix or MIX
    names = sorted(mix)
    name = rng.choices(names, [mix[n] for n in names])[0]
    return TASKS[name](rng, test)


def test_set(name, n, split='test'):
    return [TASKS[name](random.Random(f'{split}-{name}-{i}'), split == 'test') for i in range(n)]


# ------------------------------------------------------------------------------------------------------------ metrics

def chrf(hyp, ref, n=6, beta=2.0):
    """chrF: character n-gram F-score (n = 1..6, whitespace removed, recall weighted by beta), as sacrebleu's
    default, in [0, 1]."""
    h, r = ''.join(hyp.split()), ''.join(ref.split())
    ps, rs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        common = sum((hc & rc).values())
        ps.append(common / sum(hc.values()))
        rs.append(common / sum(rc.values()))
    if not ps:
        return 0.0
    p, rr = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + rr == 0 else (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def rouge_l(hyp, ref):
    """ROUGE-L F1 over lowercased word tokens."""
    h, r = re.findall(r'\w+', hyp.lower()), re.findall(r'\w+', ref.lower())
    if not h or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in h:
        cur = [0]
        for j, b in enumerate(r):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, rr = lcs / len(h), lcs / len(r)
    return 2 * p * rr / (p + rr)


def numbers(text):
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))]


_box = None
HARNESS = '''import resource, sys
for what, v in ((resource.RLIMIT_CPU, 5), (resource.RLIMIT_FSIZE, 1 << 20), (resource.RLIMIT_AS, 1 << 30)):
    try:
        resource.setrlimit(what, (v, v))
    except (ValueError, OSError):
        pass
src, setup, tests = sys.stdin.read().split('\\x00')
g = {"__name__": "__main__"}
try:
    exec(setup, g)
    exec(src, g)
except BaseException:
    print("PASSED 0"); raise SystemExit
n = 0
for t in tests.split('\\x01'):
    try:
        exec(t, g); n += 1
    except BaseException:
        pass
print("PASSED", n)
'''


def sandbox():
    """The sandbox's working directory (made once per process, empty) and the command prefix: unshare -rn (no network)
    where it works, else none."""
    global _box
    if _box is None:
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix='corpus_box_')
        pre = []
        if shutil.which('unshare'):
            try:
                if subprocess.run(['unshare', '-rn', 'true'], timeout=5, capture_output=True).returncode == 0:
                    pre = ['unshare', '-rn']
            except Exception:
                pass
        _box = (d, pre)
    return _box


def run_tests(code, tests, setup='', timeout=10):
    """The fraction of tests (assert statements) that pass after code runs, in a sandboxed subprocess: isolated
    Python (-I), an empty working directory, a minimal environment, CPU, file-size and memory limits, a wall-clock
    timeout, and no network where unshare is available. The limits are set inside the child (no preexec_fn, which
    is unsafe in a threaded parent)."""
    d, pre = sandbox()
    try:
        p = subprocess.run(pre + [sys.executable, '-I', '-c', HARNESS],
                           input='\x00'.join([setup, code, '\x01'.join(tests)]), text=True, capture_output=True,
                           cwd=d, env=dict(PATH='/usr/bin:/bin', LANG='C.UTF-8'), timeout=timeout)
        m = re.search(r'PASSED (\d+)', p.stdout)
        return int(m.group(1)) / max(1, len(tests)) if m else 0.0
    except subprocess.TimeoutExpired:
        return 0.0


def code_of(text):
    """The code in an answer: the first fenced block when there is one, else the whole answer."""
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.S)
    return m.group(1) if m else text


def score(ep, x, text):
    """dict of agent x's answer's scores; 'main' in [0, 1]."""
    t, m = text.strip(), ep['metric']
    if m == 'chrf':
        gold = ep['answers'][x] if ep['family'] == 1 else ep['golds'][0]
        v = chrf(t, gold)
        return dict(main=v, chrf=v)
    if m == 'f1':
        line = t.split('\n')[0].strip()
        f1 = max(CG.f1(line, a) for a in ep['golds'])
        em = max(float(CG.words(line) == CG.words(a)) for a in ep['golds'])
        return dict(main=f1, f1=f1, em=em)
    if m == 'number':
        got = numbers(t.split('\n')[0])
        gold = float(ep['golds'][0])
        v = float(bool(got) and abs(got[-1] - gold) < 1e-6)
        return dict(main=v, em=v)
    if m == 'tests':
        v = run_tests(code_of(t), ep['golds']['tests'], ep['golds']['setup'])
        return dict(main=v, passed=v)
    if m == 'rougeL':
        v = rouge_l(t, ep['golds'][0])
        return dict(main=v, rougeL=v)
    if m == 'exact':
        line = t.split('\n')[0].strip().lower()
        v = float(bool(re.search(rf'\b{re.escape(ep["golds"][0])}\b', line)) and
                  not any(re.search(rf'\b{re.escape(c)}\b', line) for c in ep['cands']['B'] if c != ep['golds'][0]))
        return dict(main=v, em=v)
    raise ValueError(m)


# ------------------------------------------------------------------------------------------------------------ main

def show(names):
    for n in names:
        for test in (False, True):
            ep = TASKS[n](random.Random(f'show-{n}'), test)
            print(f'===== {n} ({"test" if test else "train"}, family {ep["family"]}, {ep["metric"]})')
            for x in 'AB':
                print(f'--- {x} view ({ntok(ep["views"][x])} tokens)\n{ep["views"][x]}')
                print(f'--- {x} answer ({ntok(ep["answers"][x])} tokens)\n{ep["answers"][x]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('cmd', choices=('prefetch', 'show'))
    ap.add_argument('names', nargs='*')
    ap.add_argument('--cache')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    global CACHE
    if args.cache:
        CACHE = Path(args.cache)
    if args.cmd == 'prefetch':
        print(json.dumps(prefetch(CACHE, args.names or None, args.workers)))
    else:
        show(args.names or list(TASKS))


if __name__ == '__main__':
    main()
