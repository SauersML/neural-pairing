"""Open-ended, multi-turn training data for agents that talk (swarm/opentalk.py, swarm/crumbs.py): real prompts whose
information is split between two agents, A and B, so that answering needs both halves. A frozen Qwen3 (--model, the
agents' own base: Qwen3-1.7B for development, Qwen3-4B to confirm; thinking off) that sees the whole unsplit prompt
writes every answer; the agents are trained to match it, token by token.

An episode is up to MAX_TURNS task turns. At each task turn one agent (who) gets a user message (user) and answers
it. Before each task turn the two agents exchange a few messages (talk, 2 to 4 message turns). The teacher sees the
same request with everything both agents hold (teacher_user), and the earlier task turns with its own answers. A
record (one JSON line in OUT/shards/SPLIT-NNNN.jsonl.gz; OUT is per teacher model, since the teacher writes the
answers):
  id, source, family, split (train, eval_seen or eval_heldout), kind (the source item's: chat, doc, code, pair, plan
  or tests), views {A, B} (each agent's private material, shown at the start), teacher (False: fill-in, whose
  targets are real text), turns: [{who, user, teacher_user, talk, response, response_ids}]. response_ids ends with
  <|im_end|> when the teacher ended its answer (the answer's last token, also a target), not when it was cut at
  RESP_TOK.
The teacher's top-32 next-token log-probabilities at every answer token are stored beside each shard, from the vLLM
pass that generated the answer (raw log-probabilities: temperature 1, no top-p), so training never runs the teacher:
OUT/teacher/SPLIT-NNNN.parquet, one row per (record, teacher turn): id (string), turn (int16), n (int32, the
answer's token count), ids (binary: int32 little-endian [n, 32] row-major), lp (binary: float16 little-endian
[n, 32]). Row j is the prediction of response_ids[j] after teacher_ids(record, turn)[0] + response_ids[:j], sorted by
descending log-probability. teacher_topk_rows(path) reads a file; Episodes(OUT, split, teacher=True) attaches
record['topk'] = {turn: (ids [n, 32] int32 tensor, lp [n, 32] float16 tensor)}. kl_topk(student_logits, ids, lp) is
the per-token KL. teacher_targets(model, tok, record, k) recomputes the same top-k in one forward pass (checks; a
peft adapter is switched off).

Families (the split is drawn per episode among those the item allows):
  chat_alt    a real chat with 3+ user turns; A holds the odd user turns and B the even ones (or the reverse), and
              the holder answers each: every turn builds on turns the other agent saw
  chat_doc    a chat whose first message pastes material (a long passage, code or a table) and asks about it: A holds
              the material, B the requests (the real later turns, then simulated follow-ups)
  doc         a document: A holds it; B asks requests that need its specifics without quoting it (written by the
              frozen model playing the user), so B has to draw the details out of A
  pair        two documents of one source: A holds one, B the other, and both ask requests that need both
  code        a source file: A holds it; B asks as someone who uses the code (bug reports, changes)
  debug       an MBPP function with a mutation that fails its tests: A holds the code, B the real failing output
  plan        a planning problem the frozen model invents from a seed topic, its constraints split between A and B
  fill        corpus fill-in: a document window cut into chunks of 8 to 192 tokens (log-uniform), dealt
              alternately; each turn asks for one of the partner's chunks (targets: the real text)
User turns are real wherever the source has them (chats); elsewhere a simulator writes them (--sim_model: Qwen3-8B,
thinking off, stronger than the teacher; --model when unset), playing a person who holds only the asker's half (it sees
the whole material and the conversation so far, and is told to ask for something new each turn, never to restate the
assistant and not to quote the other half). A written request sharing a 12-word run with the other agent's material
leaks; one that is degenerate (degenerate(): it shares a 12-word run with an earlier answer, repeats an earlier user
turn, starts with "Assistant" or has under 6 distinct words) is no request. Either is written again, up to RESAMPLES
tries, and then the episode ends before that turn; a degenerate real turn ends it too. Episodes() cuts every record at
its first degenerate user turn when it loads a shard, so shards written before the filter train the same way.

Sources: wildchat (allenai/WildChat-1M, non-toxic), lmsys (lmsys/lmsys-chat-1m, gated: used when the account has
access), ultrachat (HuggingFaceH4/ultrachat_200k), fineweb_edu, wikipedia, fineweb2 (de, fr, es, ja, zh, ru),
openwebmath, stackexchange (HuggingFaceH4/stack-exchange-preferences: the question as a real first turn), github
(codeparrot/codeparrot-clean), mbpp (debug), and two held out entirely: arxiv (ccdv/arxiv-summarization) and books
(emozilla/pg19). Every item and every written turn is scanned for credentials and personal paths (swarm/secret_scan.py);
a hit drops it. Splits: train from each kept source's training files; eval_seen from its last file (unseen items);
eval_heldout from the held-out sources only.

Usage (a GPU pod with vLLM for build):
  python swarm/opendata.py fetch --cache DIR                 items per source and split into DIR/items
  python swarm/opendata.py build --cache DIR --out DIR --model Qwen/Qwen3-1.7B [--sim_model Qwen/Qwen3-8B --train 50000
      --eval_seen 600 --eval_heldout 400 --max_minutes M --skip S --tag T]   (the build stops at M minutes, its shards
      written so far kept; a second generator skips the first S train skeletons and names its shards SPLIT-TNNNN)
  python swarm/opendata.py show --out DIR [--family F]       prints one record per family
"""
import argparse
import ast
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import html

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm import corpus as CP                         # noqa: E402  (hub helpers, tokenizer, sandbox)
from swarm import secret_scan as SEC                   # noqa: E402

TEACHER = 'Qwen/Qwen3-1.7B'
TOPK = 32
MAX_TURNS, MIN_TURNS = 6, 3
TALK = (2, 4)                     # message turns before each task turn
VIEW_TOK, USER_TOK, RESP_TOK = 800, 400, 256
IM_END, EOT = 151645, 151643      # <|im_end|>, <|endoftext|>
LEAK_WORDS = 12
HELD_OUT = ('arxiv', 'books')

# name: repo, folders, builder, kind, field, training items to cache, test items to cache
SOURCES = dict(
    wildchat=('allenai/WildChat-1M', ('data',), 'parquet', 'chat', 'conversation', 40000, 400),
    lmsys=('lmsys/lmsys-chat-1m', ('data',), 'parquet', 'chat', 'conversation', 20000, 200),
    ultrachat=('HuggingFaceH4/ultrachat_200k', ('data',), 'parquet', 'chat', 'messages', 12000, 200),
    fineweb_edu=('HuggingFaceFW/fineweb-edu', ('sample/10BT',), 'parquet', 'doc', 'text', 12000, 200),
    wikipedia=('wikimedia/wikipedia', ('20231101.en',), 'parquet', 'doc', 'text', 8000, 200),
    fineweb2=('HuggingFaceFW/fineweb-2', tuple(f'data/{x}/train' for x in ('deu_Latn', 'fra_Latn', 'spa_Latn',
                                                                         'jpn_Jpan', 'cmn_Hani', 'rus_Cyrl')),
              'parquet', 'doc', 'text', 6000, 180),
    openwebmath=('open-web-math/open-web-math', ('data',), 'parquet', 'doc', 'text', 5000, 150),
    stackexchange=('HuggingFaceH4/stack-exchange-preferences', ('data',), 'parquet', 'chat', 'question', 8000, 200),
    github=('codeparrot/codeparrot-clean', ('',), 'json', 'code', 'content', 8000, 200),
    mbpp=('google-research-datasets/mbpp', ('full',), 'parquet', 'tests', None, 0, 0),
    arxiv=('ccdv/arxiv-summarization', ('document',), 'parquet', 'doc', 'article', 0, 300),
    books=('emozilla/pg19', ('data',), 'parquet', 'doc', 'text', 0, 120),
)
# family weights for train (each episode draws its family, then an item that allows it)
MIX = dict(chat_alt=0.22, chat_doc=0.12, doc=0.18, pair=0.10, code=0.10, debug=0.05, plan=0.08, fill=0.15)
FAMILY_SOURCES = dict(
    chat_alt=('wildchat', 'lmsys', 'ultrachat'), chat_doc=('wildchat', 'lmsys', 'stackexchange'),
    doc=('fineweb_edu', 'wikipedia', 'fineweb2', 'openwebmath', 'arxiv', 'books'),
    pair=('fineweb_edu', 'wikipedia', 'fineweb2', 'openwebmath', 'arxiv', 'books'),
    code=('github',), debug=('mbpp',), plan=('fineweb_edu', 'wikipedia'),
    fill=('fineweb_edu', 'wikipedia', 'fineweb2', 'openwebmath', 'github', 'arxiv', 'books'))

CACHE = Path(os.environ.get('OPENDATA_CACHE', Path.home() / '.cache/opendata'))


def other(x):
    return 'A' if x == 'B' else 'B'


def tok():
    return CP.tokenizer()


def ntok(text):
    return len(tok()(text, add_special_tokens=False).input_ids)


def cut(text, n):
    """text cut to at most n tokens (at a whole character)."""
    ids = tok()(text, add_special_tokens=False).input_ids
    return text if len(ids) <= n else tok().decode(ids[:n]).rstrip('�')


def clean_secret(*texts):
    """True when no text holds a credential or a personal path."""
    return not any(SEC.flags(t) for t in texts if t)


# ------------------------------------------------------------------------------------------------------------ items

def item_path(name, split):
    return CACHE / 'items' / f'{name}.{split}.jsonl'


def items(name, split):
    """The cached items of a source and split ('train' or 'test'): dicts {id, source, kind, text | turns | code,
    tests}. A held-out source has only 'test'."""
    p = item_path(name, split)
    if not p.exists():
        return []
    return [json.loads(line) for line in open(p)]


def chat_turns(name, r):
    """A chat row's turns [{role, content}] (user and assistant only), or None when it is toxic or flagged."""
    if name == 'wildchat':
        if r.get('toxic') or any(m.get('toxic') for m in r['conversation']):
            return None
        conv = r['conversation']
    elif name == 'lmsys':
        if any((m or {}).get('flagged') for m in (r.get('openai_moderation') or [])):
            return None
        conv = r['conversation']
    elif name == 'ultrachat':
        conv = r['messages']
    else:                                                 # stackexchange: the question as the first user turn
        conv = [dict(role='user', content=se_text(r['question']))]
    turns = [dict(role=m['role'], content=(m['content'] or '')[:8000]) for m in conv
             if m.get('role') in ('user', 'assistant')]
    return turns if turns and turns[0]['role'] == 'user' else None


def se_text(h):
    """A Stack Exchange post's HTML as plain text, its code blocks fenced."""
    h = re.sub(r'<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>', lambda m: '\n```\n' + m.group(1) + '\n```\n', h, flags=re.S)
    h = re.sub(r'<br\s*/?>|</p>|</li>', '\n', h)
    return html.unescape(re.sub(r'<[^>]+>', '', h)).strip()


def useful_chat(turns):
    """A chat some family can use: 3+ user turns, or a first message with material to split off."""
    users = [m['content'] for m in turns if m['role'] == 'user']
    return len(users) >= MIN_TURNS or split_material(users[0]) is not None


def to_item(name, r, k):
    """One source row as an item, or None (too short, toxic, or holding a credential)."""
    kind, field = SOURCES[name][3], SOURCES[name][4]
    rid = hashlib.md5(f'{name}-{k}-{json.dumps(r, default=str)[:400]}'.encode()).hexdigest()[:16]
    if kind == 'chat':
        turns = chat_turns(name, r)
        if not turns or not useful_chat(turns) or not clean_secret(*[m['content'] for m in turns]):
            return None
        return dict(id=rid, source=name, kind='chat', turns=turns)
    if kind == 'tests':
        code = r['code'].replace('\r\n', '\n')
        if not clean_secret(code):
            return None
        return dict(id=rid, source=name, kind='tests', code=code, tests=list(r['test_list']),
                    setup=r.get('test_setup_code') or '', text=r['text'])
    text = r[field] or ''
    if kind == 'code':                                    # the leading comment block (a license header) goes
        lines = text.split('\n')
        i = 0
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(('#', '//', '/*', '*'))):
            i += 1
        text = '\n'.join(lines[i:])
    if name == 'books':
        rng = random.Random(rid)
        s = rng.randrange(max(1, len(text) - 9000))
        p = text.find('\n\n', s, s + 3000)                  # from a paragraph's start
        s = p + 2 if p >= 0 else s
        text = text[s:s + 9000]
    if len(text) < 1500 or not clean_secret(text):
        return None
    return dict(id=rid, source=name, kind=kind, text=text[:12000])


def fetch(name):
    """Streams one source into the item cache (training items from its first files, test items from its last)."""
    repo, folders, builder, kind = SOURCES[name][:4]
    n_train, n_test = SOURCES[name][5:7]
    out = dict(train=[], test=[])
    if name == 'mbpp':
        files = dict(train=['full/train-00000-of-00001.parquet', 'full/validation-00000-of-00001.parquet',
                            'full/prompt-00000-of-00001.parquet'], test=['full/test-00000-of-00001.parquet'])
        for split in out:
            out[split] = [x for k, r in enumerate(CP.hub_rows(repo, files[split])) if (x := to_item(name, r, k))]
    else:
        for fo in folders:
            files = CP.repo_files(repo, fo, builder)
            if name == 'ultrachat':
                split = dict(train=[f for f in files if 'train_sft' in f], test=[f for f in files if 'test_sft' in f])
            elif name == 'books':
                split = dict(train=[], test=[f for f in files if '/test' in f] or files[-1:])
            else:
                split = dict(train=files[:-1] or files, test=files[-1:])
            for s, want in (('train', n_train), ('test', n_test)):
                want //= len(folders)
                if not want or not split[s]:
                    continue
                got = 0
                for k, r in enumerate(CP.stream(repo, split[s], builder)):
                    x = to_item(name, r, k)
                    if x:
                        out[s].append(x)
                        got += 1
                    if got >= want:
                        break
    (CACHE / 'items').mkdir(parents=True, exist_ok=True)
    for s, xs in out.items():
        p = item_path(name, s)
        with open(p.with_suffix('.part'), 'w') as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
        os.replace(p.with_suffix('.part'), p)
    return {s: len(v) for s, v in out.items()}


def fetch_all(names=None, workers=12):
    """Fills the item cache, one subprocess per source (the filters are CPU-bound: threads shared one core, 11-05; a
    subprocess's crash at exit, which the hub's threads cause, costs nothing); a source that fails (lmsys without
    access) is reported and skipped."""
    names = names or list(SOURCES)
    todo = [n for n in names if not (item_path(n, 'train').exists() and item_path(n, 'test').exists())]
    report, t0, procs = {}, time.monotonic(), {}
    workers = max(1, min(workers, os.cpu_count() or 1))
    while todo or procs:
        while todo and len(procs) < workers:
            n = todo.pop(0)
            log = CACHE / 'fetch_logs' / f'{n}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, 'w') as f:                                 # a file, not a pipe: no full-buffer stall
                procs[n] = (subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'fetch_one', '--cache',
                                              str(CACHE), '--sources', n], stdout=f, stderr=subprocess.STDOUT), log)
        for n, (p, log) in list(procs.items()):
            if p.poll() is None:
                continue
            lines = [x for x in log.read_text(errors='replace').splitlines() if x.startswith('{"source"')]
            report[n] = json.loads(lines[-1])['got'] if lines else f'exit {p.returncode}, no result'
            del procs[n]
            print(json.dumps(dict(source=n, got=report[n], s=round(time.monotonic() - t0))), flush=True)
        time.sleep(1)
    return report


def fetch_one(name):
    """One source's fetch in its own process (fetch_all's worker): prints its result and leaves at once."""
    try:
        got = fetch(name)
    except Exception as e:
        got = repr(e)[:300]
    print(json.dumps(dict(source=name, got=got)), flush=True)
    os._exit(0)


# ------------------------------------------------------------------------------------------------------ materials

FENCE = re.compile(r'```[^\n]*\n.*?```', re.S)


def table_lines(text):
    """The longest run of 3+ consecutive table-like lines (markdown pipes, or CSV/TSV with a steady column count)."""
    best, run = [], []
    lines = text.split('\n')

    def cols(s):
        if s.count('|') >= 2:
            return ('|', s.count('|'))
        for sep in ('\t', ','):
            if s.count(sep) >= 2:
                return (sep, s.count(sep))
        return None
    prev = None
    for i, line in enumerate(lines):
        c = cols(line.strip())
        if c and (prev is None or c[0] == prev[0] and abs(c[1] - prev[1]) <= 1):
            run.append(i)
        else:
            run = [i] if c else []
        prev = c
        if len(run) > len(best):
            best = list(run)
    return best if len(best) >= 3 else []


def split_material(text):
    """A first chat message cut into (kind, material, request): fenced code, a table, or a long passage with a short
    request before or after it; None when it has no such material or the request part is empty."""
    blocks = FENCE.findall(text)
    if blocks and sum(len(b) for b in blocks) >= 200:
        rest = FENCE.sub(' ', text).strip()
        if len(rest) >= 15:
            return 'code', '\n\n'.join(blocks), rest
    rows = table_lines(text)
    if rows:
        lines = text.split('\n')
        table = '\n'.join(lines[i] for i in rows)
        rest = '\n'.join(x for i, x in enumerate(lines) if i not in set(rows)).strip()
        if len(table) >= 120 and len(rest) >= 15:
            return 'table', table, rest
    paras = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(text) >= 1200 and len(paras) >= 2:                   # the shorter end paragraph, if clearly short
        i = min((0, len(paras) - 1), key=lambda j: len(paras[j]))
        ask = paras[i].strip()
        rest = [p for j, p in enumerate(paras) if j != i]
        body = '\n\n'.join(rest).strip()
        if 10 <= len(ask) <= 500 and len(body) >= 800 and len(ask) < 0.6 * len(body) / len(rest) and asks(ask):
            return 'passage', body, ask
    return None


ASK_WORDS = ('summarize', 'summarise', 'explain', 'rewrite', 'translate', 'write', 'list', 'give', 'tell', 'describe',
             'what', 'how', 'why', 'which', 'who', 'when', 'where', 'can', 'could', 'would', 'please', 'make', 'create',
             'fix', 'help', 'analyze', 'analyse', 'compare', 'identify', 'extract', 'paraphrase', 'continue', 'answer',
             'generate', 'is', 'are', 'do', 'does', 'find', 'improve', 'edit', 'correct', 'convert', 'based')


def asks(text):
    """A paragraph that reads as a request: a question mark, or a first word that opens one (a heading does not)."""
    w = re.findall(r'[a-z]+', text.lower()[:40])
    return '?' in text or bool(w) and w[0] in ASK_WORDS


def leaks(request, hidden, n=LEAK_WORDS):
    """True when the request shares a run of n words with the hidden material."""
    w = re.findall(r'\w+', hidden.lower())
    grams = {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}
    r = re.findall(r'\w+', request.lower())
    return any(' '.join(r[i:i + n]) in grams for i in range(len(r) - n + 1))


def degenerate(user, prior_users=(), prior_responses=(), fill=False):
    """Why a user turn is no real request ('' when it is one): it shares a run of LEAK_WORDS words with an earlier
    answer (it pastes or paraphrases the assistant: 'echo'), repeats an earlier user turn (exactly or nearly: the Dice
    overlap of the two word sets 0.8 or more, 'repeat'), starts with "Assistant" ('assistant'), or has fewer than 6
    distinct words ('short'; fill-in requests such as "[3]" are exempt from the last two)."""
    w = set(re.findall(r'\w+', user.lower()))
    if not fill:
        if re.match(r'\W*assistant\b', user, re.I):
            return 'assistant'
        if len(w) < 6:
            return 'short'
    if any(leaks(user, r) for r in prior_responses):
        return 'echo'
    for q in prior_users:
        q = set(re.findall(r'\w+', q.lower()))
        if w and q and 2 * len(w & q) / (len(w) + len(q)) >= 0.8:
            return 'repeat'
    return ''


def first_degenerate(rec):
    """(k, why): the first task turn whose user turn is degenerate (degenerate()), or (len(turns), '')."""
    fill = rec['family'] == 'fill'
    for k, t in enumerate(rec['turns']):
        why = degenerate(t['user'], [u['user'] for u in rec['turns'][:k]],
                         [u['response'] for u in rec['turns'][:k]], fill)
        if why:
            return k, why
    return len(rec['turns']), ''


# ------------------------------------------------------------------------------------------------------ mutations

class _Flip(ast.NodeTransformer):
    """Replaces the k-th operator or small integer constant it meets (one mutation per call)."""
    SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Lt: ast.LtE,
            ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.And: ast.Or,
            ast.Or: ast.And}

    def __init__(self, k):
        self.k, self.seen, self.done = k, 0, False

    def _hit(self):
        hit = self.seen == self.k and not self.done
        self.seen += 1
        self.done |= hit
        return hit

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.SWAP and self._hit():
            node.op = self.SWAP[type(node.op)]()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.SWAP and self._hit():
            node.op = self.SWAP[type(node.op)]()
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if node.ops and type(node.ops[0]) in self.SWAP and self._hit():
            node.ops[0] = self.SWAP[type(node.ops[0])]()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool) and abs(node.value) <= 10 and self._hit():
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node


def mutants(code, rng, n=6):
    """Up to n distinct one-site mutations of code (operator swaps, constants off by one)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    probe = _Flip(10 ** 9)
    probe.visit(ast.parse(code))
    sites = list(range(probe.seen))
    rng.shuffle(sites)
    out = []
    for k in sites:
        t = _Flip(k).visit(ast.parse(code))
        try:
            src = ast.unparse(ast.fix_missing_locations(t))
        except Exception:
            continue
        if src != ast.unparse(tree) and src not in out:
            out.append(src)
        if len(out) >= n:
            break
    return out


HARNESS = r'''
import resource, sys, traceback
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
try:
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
except Exception:
    pass
setup, code, tests = sys.stdin.read().split('\x00')
g = {}
exec(setup, g)
exec(code, g)
for t in tests.split('\x01'):
    try:
        exec(t, g)
        print('ok:', t)
    except Exception:
        print('FAILED:', t)
        traceback.print_exc(limit=2, file=sys.stdout)
'''


def failing_output(code, tests, setup='', timeout=10):
    """The test run's output when some test fails (each test's line, tracebacks for the failures), else None."""
    d, pre = CP.sandbox()
    try:
        p = subprocess.run(pre + [sys.executable, '-I', '-c', HARNESS], input='\x00'.join([setup, code, '\x01'.join(
            tests)]), text=True, capture_output=True, cwd=d, env=dict(PATH='/usr/bin:/bin', LANG='C.UTF-8'),
            timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = (p.stdout + p.stderr).replace(d, '.')
    return out[:2500] if 'FAILED:' in out else None


# ------------------------------------------------------------------------------------------------------ skeletons

def talk(rng):
    return rng.randint(*TALK)


def chunks_of(text, rng, k):
    """k consecutive chunks of 8 to 192 tokens (log-uniform lengths) from a random place in text, or None."""
    ids = tok()(text, add_special_tokens=False).input_ids
    lens = [int(round(math.exp(rng.uniform(math.log(8), math.log(192))))) for _ in range(k)]
    if len(ids) < sum(lens) + 4:
        return None
    pos = rng.randrange(len(ids) - sum(lens) + 1)
    out = []
    for n in lens:
        end = CP.clean_cut(ids, pos, pos + n)
        out.append(tok().decode(ids[pos:end]))
        pos = end
    return out


def skeleton(family, item, rng, partner=None):
    """An episode before generation: dict(family, source, kind, views, turns, material) where each turn has who,
    talk, and either user + teacher_user (real) or ask (the asker's role for the simulated user, filled by
    generate); material is the text the teacher sees before the first request. None when the item does not fit."""
    first = rng.choice('AB')
    sk = dict(family=family, source=item['source'], kind=item['kind'], item=item['id'], teacher=True)
    if family == 'chat_alt':
        users = [m['content'] for m in item['turns'] if m['role'] == 'user']
        if len(users) < MIN_TURNS:
            return None
        users = [cut(u, USER_TOK) for u in users[:MAX_TURNS]]
        owners = [first if i % 2 == 0 else other(first) for i in range(len(users))]
        sk.update(views=dict(A='', B=''), material='',
                  turns=[dict(who=w, user=u, teacher_user=u, talk=talk(rng)) for u, w in zip(users, owners)])
        return sk
    if family == 'chat_doc':
        users = [m['content'] for m in item['turns'] if m['role'] == 'user']
        got = split_material(users[0])
        if not got:
            return None
        kind, mat, ask = got
        mat = cut(mat, VIEW_TOK)
        later = [cut(u, USER_TOK) for u in users[1:MAX_TURNS]]
        if any(leaks(u, mat) for u in [ask] + later):
            return None
        n = rng.randint(max(MIN_TURNS, 1 + len(later)), MAX_TURNS)
        turns = [dict(who='B', user=cut(ask, USER_TOK), teacher_user=mat + '\n\n' + cut(ask, USER_TOK),
                      talk=talk(rng))]
        turns += [dict(who='B', user=u, teacher_user=u, talk=talk(rng)) for u in later]
        turns += [dict(who='B', ask='B', talk=talk(rng)) for _ in range(n - len(turns))]
        sk.update(views=dict(A=mat, B=''), material='', split=kind, turns=turns)
        return sk
    if family in ('doc', 'code', 'plan'):
        mat = cut(item['text'], VIEW_TOK)
        n = rng.randint(MIN_TURNS, MAX_TURNS)
        if family == 'plan':                                         # generate() writes the scenario first
            sk.update(views=dict(A='', B=''), material='', seed=item['text'][:300],
                      turns=[dict(who=first if i % 2 == 0 else other(first), ask=None, talk=talk(rng))
                             for i in range(n)])
            return sk
        sk.update(views=dict(A=mat, B=''), material=mat, turns=[dict(who='B', ask='B', talk=talk(rng))
                                                                  for _ in range(n)])
        return sk
    if family == 'pair':
        if partner is None:
            return None
        d1, d2 = cut(item['text'], VIEW_TOK // 2 + 250), cut(partner['text'], VIEW_TOK // 2 + 250)
        n = rng.randint(MIN_TURNS, MAX_TURNS)
        sk.update(views=dict(A=d1, B=d2), material=f'Document 1:\n{d1}\n\nDocument 2:\n{d2}', item2=partner['id'],
                  turns=[dict(who=first if i % 2 == 0 else other(first), ask=None, talk=talk(rng)) for i in range(n)])
        for t in sk['turns']:
            t['ask'] = t['who']
        return sk
    if family == 'debug':
        for bad in mutants(item['code'], rng):
            out = failing_output(bad, item['tests'], item['setup'])
            if out:
                n = rng.randint(MIN_TURNS, MAX_TURNS)
                sk.update(variant=hashlib.md5(bad.encode()).hexdigest()[:8], views=dict(A=bad, B=out), material=f'The code:\n```python\n{bad}\n```\n\nThe test output:\n'
                          f'```\n{out}\n```', turns=[dict(who='B', ask='B', talk=talk(rng)) for _ in range(n)])
                return sk
        return None
    if family == 'fill':                                               # turn i: the non-owner writes chunk i
        parts = chunks_of(item['text'], rng, rng.randint(MIN_TURNS, MAX_TURNS))
        if not parts:
            return None
        owner = [first if i % 2 == 0 else other(first) for i in range(len(parts))]
        views = {x: '\n'.join(f'[{i + 1}] ' + (p if owner[i] == x else '...') for i, p in enumerate(parts))
                 for x in 'AB'}
        sk.update(views=views, material='', teacher=False,
                  turns=[dict(who=other(owner[i]), user=f'[{i + 1}]', response=p, talk=talk(rng))
                         for i, p in enumerate(parts)])
        return sk
    raise ValueError(family)


# ------------------------------------------------------------------------------------------------------ generation

SIM_SYSTEM = ('You are role-playing a person who is talking with an AI assistant. You hold only your own half of the '
              'information. Write only the person\'s next message to the assistant, in their own voice, and nothing '
              'else. Each message asks for something new that needs specifics not yet discussed. Never restate, '
              'summarize or quote the assistant\'s replies, never write as the assistant, and never repeat an earlier '
              'message.')
RESAMPLES = 3                     # tries per simulated user turn before the episode ends there
MIN_KEEP = 1                      # an episode keeps at least this many task turns
PLAN_SYSTEM = 'You write realistic planning problems. Reply with JSON only.'


def sim_prompt(sk, k, asker):
    """The simulated user's prompt for turn k: the whole material, what the asker holds, the conversation so far."""
    hold = sk['views'][asker]
    hidden = sk['views'][other(asker)]
    lines = []
    if sk['family'] == 'pair':
        lines.append(f'Here are two documents. You have read only document {1 if asker == "A" else 2}:\n\n'
                     f'{sk["material"]}')
        lines.append('Your message must need both documents to answer well. Do not quote the document you have not '
                     'read.' + ' Name no specific detail (a name, number, term or fact) that only the part you have not seen shows and the conversation has not mentioned yet.' + (f' This first message must name specific details from your own document (document '
                                f'{1 if asker == "A" else 2}) and ask something that also needs the other one.'
                                if k == 0 else ''))
    elif sk['family'] == 'plan':
        lines.append(f'The situation: {sk["situation"]}\n\nWhat you know:\n{hold}\n\nWhat the others know (you do '
                     f'not):\n{hidden}')
        lines.append('Your message must move the plan forward and need what the others know. Do not restate what '
                     'you do not know.' + ' Name no specific detail (a name, number, term or fact) that only the part you have not seen shows and the conversation has not mentioned yet.')
    else:
        what = dict(doc='a document', code='a source file', debug='some code and its failing test output',
                    chat_doc='some material').get(sk['family'], 'some material')
        lines.append(f'The assistant has {what} that you have not seen:\n\n{sk["material"] or hidden}')
        if hold:
            lines.append(f'What you have:\n{hold}')
        lines.append('Your message must need specific details from it to answer well, without quoting them or '
                     'showing that you have seen it.' + ' Name no specific detail (a name, number, term or fact) that only the part you have not seen shows and the conversation has not mentioned yet.')
    conv = []
    for t in sk['turns'][max(0, k - 3):k]:                          # the last three turns, answers shortened
        a = t['response'] if len(t['response']) <= 700 else t['response'][:700] + ' [...]'
        conv.append(f'You: {t["user"]}\n\nAssistant: {a}')
    if k > 3:
        conv.insert(0, 'Your earlier messages: ' + ' | '.join(t['user'][:200] for t in sk['turns'][:k - 3]))
    lines.append(('The conversation so far' + (' (its latest turns)' if k > 3 else '') + ':\n\n' + '\n\n'.join(conv)
                  + '\n\nWrite your next message: ask for something new that this conversation has not covered.')
                 if conv else 'This is your first message.')
    return [dict(role='system', content=SIM_SYSTEM), dict(role='user', content='\n\n'.join(lines))]


JUDGE_Q = ('Stop role-playing now and check your last message. Does it mention any specific detail (a name, number, '
           'term or fact) that appears in the part you have not seen, but neither in what you have nor in the '
           'conversation so far? Answer with one word: yes or no.')
JUDGE_K = 20                                         # top log-probabilities read from the judge's first token


def judge_ids(sk, k, asker, request):
    """The judge's prompt as token ids: the simulator's own prompt for this turn (sim_prompt, token for token, so a
    prefix cache serves it), the request as the simulator's answer, then JUDGE_Q as a new user turn. It asks whether
    the request names a specific detail found only in the half the asker has not seen (not in the asker's own half,
    not in the conversation so far)."""
    return chat_ids(sim_prompt(sk, k, asker)) + piece(request) + piece(END + USER_HEAD + JUDGE_Q + END + ASSIST_HEAD)


def judge_yes(top):
    """The judge's probability of yes against no, from its first token's top log-probabilities (ids [1, K], lp)."""
    import numpy as np
    if top is None:
        return 0.0
    ids, lp = top
    words = {}
    for i, v in zip(ids[0].tolist(), lp[0].astype(np.float32).tolist()):
        w = tok().decode([i]).strip().lower()
        if w in ('yes', 'no'):
            words[w] = words.get(w, 0.0) + math.exp(v)
    y, n = words.get('yes', 0.0), words.get('no', 0.0)
    return y / (y + n) if y + n else 0.0


def plan_prompt(seed):
    return [dict(role='system', content=PLAN_SYSTEM), dict(role='user', content=(
        f'Topic for inspiration: "{seed}"\n\nInvent a realistic planning problem related to this topic that two people '
        'must solve together, where each person knows constraints the other does not. Reply with JSON: '
        '{"situation": "...", "constraints_A": ["..."], "constraints_B": ["..."]}'))]


def teacher_chat(rec, upto=None):
    """The teacher's conversation as messages: every task turn's teacher_user and the teacher's answer (through turn
    upto, whose answer is left out)."""
    msgs = []
    for k, t in enumerate(rec['turns']):
        msgs.append(dict(role='user', content=t['teacher_user']))
        if upto is not None and k == upto:
            break
        msgs.append(dict(role='assistant', content=t['response']))
    return msgs


USER_HEAD, ASSIST_HEAD, END = '<|im_start|>user\n', '<|im_start|>assistant\n<think>\n\n</think>\n\n', '<|im_end|>\n'


def piece(text):
    return tok()(text, add_special_tokens=False).input_ids


def teacher_ids(rec, upto=None):
    """The teacher's token ids: per turn the user message, the assistant header with an empty thinking block (the
    template's thinking-off prompt), the answer's own generated ids and the end of turn; and per turn the position
    its answer starts at. Through turn upto: its answer is left out, so these are the ids vLLM generated it from.
    Every answer sits after exactly the context it was generated in, so one forward pass scores every turn."""
    ids, starts = [], []
    for k, t in enumerate(rec['turns']):
        ids += piece(USER_HEAD + t['teacher_user'] + END + ASSIST_HEAD)
        starts.append(len(ids))
        if upto is not None and k == upto:
            break
        r = list(t['response_ids'])
        ids += r + (piece('\n') if r and r[-1] == IM_END else piece(END))
    return ids, starts


def chat_ids(msgs, generation=True):
    """Token ids of a chat in the Qwen3 template, thinking off (rendered as text, then tokenized: transformers 5
    returns a dict from tokenize=True)."""
    text = tok().apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation, enable_thinking=False)
    return tok()(text, add_special_tokens=False).input_ids


def clean_user(text):
    t = text.strip().strip('"').strip()
    t = re.sub(r'^(You|User|Person|Me)\s*:\s*', '', t, flags=re.I)
    return t


def generate(sks, gen, rng, log=print, sim=None, stats=None, judge=None, samples=None):
    """Fills the skeletons' simulated user turns and every teacher answer, turn by turn across the batch. gen(prompts,
    max_tokens, temperature, topk=0) -> [(text, ids, top)] for prompts that are messages (a chat) or token ids; top:
    None, or with topk the (ids int32 [n, topk], lp float16 [n, topk]) arrays of the raw next-token top-k at each
    generated token. Teacher turns keep it as turn['_top'] (written beside the shard, not into the record). Returns
    the finished records. sim (default gen) writes the user turns and the plans: a stronger model than the teacher.
    A written user turn that leaks the other half, holds a credential or is degenerate (degenerate(): it echoes an
    answer, repeats a turn, plays the assistant or says too little) is written again, up to RESAMPLES tries; then, or
    at a degenerate real turn, the episode ends before that turn; under MIN_KEEP turns it is dropped. stats (a dict)
    gathers per family the user turns asked, the first tries that failed and the turns that ended an episode.
    judge (a gen; the simulator when one is given) reads each written request that passed those checks beside both
    halves and the conversation and says whether it names a detail only the hidden half holds (judge_prompt; yes when
    p(yes) > p(no)): a yes is written again too, within the same RESAMPLES tries. samples (a list) gathers judged
    requests with their verdicts, for checks by hand."""
    judge = judge if judge is not None else sim
    sim = sim or gen
    stats = {} if stats is None else stats
    live = []
    plans = [s for s in sks if s['family'] == 'plan']
    if plans:
        for s, got in zip(plans, sim([plan_prompt(s['seed']) for s in plans], 600, 0.9)):
            text = got[0]
            try:
                j = json.loads(text[text.index('{'):text.rindex('}') + 1])
                s['situation'] = str(j['situation'])
                s['views'] = dict(A='\n'.join(f'- {c}' for c in j['constraints_A']),
                                  B='\n'.join(f'- {c}' for c in j['constraints_B']))
                s['material'] = f'{s["situation"]}\n\nPerson A knows:\n{s["views"]["A"]}\n\nPerson B knows:\n' \
                                f'{s["views"]["B"]}'
                for t in s['turns']:
                    t['ask'] = t['who']
                if s['views']['A'] and s['views']['B'] and clean_secret(s['material']):
                    live.append(s)
            except Exception:
                continue
    live += [s for s in sks if s['family'] != 'plan']
    for s in live:
        s['done'] = 0
    def count(s, key):
        st = stats.setdefault(s['family'], dict(asked=0, first_failed=0, ended=0, real_ended=0, judged=0,
                                                 judge_yes=0))
        st[key] += 1

    for k in range(MAX_TURNS):
        for s in live:                                   # a real user turn that is degenerate ends the episode there
            if s['done'] == k and k < len(s['turns']) and 'user' in s['turns'][k]:
                if degenerate(s['turns'][k]['user'], [t['user'] for t in s['turns'][:k]],
                              [t['response'] for t in s['turns'][:k]], s['family'] == 'fill'):
                    count(s, 'real_ended')
                    s['turns'] = s['turns'][:k]
        need = [s for s in live if s['done'] == k and k < len(s['turns']) and s['turns'][k].get('ask')]
        asked = len(need)
        for s in need:
            count(s, 'asked')
        for attempt in range(RESAMPLES):
            if not need:
                break
            outs = sim([sim_prompt(s, k, s['turns'][k]['ask']) for s in need], USER_TOK, 0.8)
            again, cand = [], []
            for s, got in zip(need, outs):
                t, u = s['turns'][k], cut(clean_user(got[0]), USER_TOK)
                hidden = s['views'][other(t['ask'])]
                if not u or leaks(u, hidden) or not clean_secret(u) or degenerate(
                        u, [x['user'] for x in s['turns'][:k]], [x['response'] for x in s['turns'][:k]]):
                    if attempt == 0:
                        count(s, 'first_failed')
                    again.append(s)
                    continue
                cand.append((s, u))
            if judge is not None and cand:
                verdicts = judge([judge_ids(s, k, s['turns'][k]['ask'], u) for s, u in cand], 1, 0.0,
                                 topk=JUDGE_K)
                kept = []
                for (s, u), got in zip(cand, verdicts):
                    py = judge_yes(got[2] if len(got) > 2 else None)
                    count(s, 'judged')
                    if samples is not None and len(samples) < 200:
                        samples.append(dict(id=s['id'], family=s['family'], turn=k, request=u, p_yes=round(py, 3),
                                            hidden=s['views'][other(s['turns'][k]['ask'])][:1500],
                                            own=s['views'][s['turns'][k]['ask']][:600]))
                    if py > 0.5:
                        count(s, 'judge_yes')
                        if attempt == 0:
                            count(s, 'first_failed')
                        again.append(s)
                    else:
                        kept.append((s, u))
                cand = kept
            for s, u in cand:
                t = s['turns'][k]
                t['user'] = u
                t['teacher_user'] = (s['material'] + '\n\n' + u) if k == 0 and s['material'] else u
                del t['ask']
            need = again
        for s in need:                                   # RESAMPLES tries failed: the episode ends before turn k
            count(s, 'ended')
            s['turns'] = s['turns'][:k]
        ready = [s for s in live if s['done'] == k and k < len(s['turns'])]
        if not ready:
            continue
        teach = [s for s in ready if s['teacher']]
        if teach:
            outs = gen([teacher_ids(s, k)[0] for s in teach], RESP_TOK, 0.7, topk=TOPK)
            for s, got in zip(teach, outs):
                text, ids = got[0], list(got[1])
                top = got[2] if len(got) > 2 else None
                if ids and ids[-1] == EOT:                     # end of text: the turn ends there, as at <|im_end|>
                    ids[-1] = IM_END
                if not ids or not text.strip() or not clean_secret(text):
                    s['turns'] = s['turns'][:k]
                    continue
                s['turns'][k].update(response=text, response_ids=ids)
                if top is not None:
                    s['turns'][k]['_top'] = top
                s['done'] = k + 1
        for s in ready:
            if not s['teacher']:
                s['turns'][k]['response_ids'] = tok()(s['turns'][k]['response'], add_special_tokens=False).input_ids
                s['done'] = k + 1
        log(json.dumps(dict(turn=k, asked=asked, ended=len(need), answered=len(ready))))
    out = []
    for s in live:
        s['turns'] = s['turns'][:s['done']]
        if len(s['turns']) >= MIN_KEEP:
            rec = {k: v for k, v in s.items() if k not in ('done', 'seed')}
            for t in rec['turns']:
                t.pop('ask', None)
            out.append(rec)
    return out


def top_arrays(lps, n, k):
    """vLLM's per-token log-probabilities (a list of {token: Logprob} dicts, or FlatLogprobs) as (ids int32 [n, k],
    lp float16 [n, k]): entry r - 1 holds the token of rank r (the sampled token, also listed, is placed by its rank
    only when that is within k)."""
    import numpy as np
    ids, lp = np.zeros((n, k), np.int32), np.full((n, k), -1e4, np.float32)
    if hasattr(lps, 'token_ids'):                                     # FlatLogprobs: flat lists with spans
        s, e = np.asarray(lps.start_indices[:n]), np.asarray(lps.end_indices[:n])
        pos = np.repeat(np.arange(len(s)), e - s)
        sel = np.concatenate([np.arange(a, b) for a, b in zip(s, e)]) if len(s) else np.zeros(0, np.int64)
        tok_, lp_, rk = (np.asarray(lps.token_ids)[sel], np.asarray(lps.logprobs, np.float32)[sel],
                         np.array(lps.ranks, dtype=np.float64)[sel])          # a None rank reads as nan
        m = (rk >= 1) & (rk <= k)
        r = rk[m].astype(np.int64) - 1
        ids[pos[m], r], lp[pos[m], r] = tok_[m], lp_[m]
    else:
        for i, d in enumerate(lps[:n]):
            for t, L in d.items():
                if L.rank is not None and 1 <= L.rank <= k:
                    ids[i, L.rank - 1], lp[i, L.rank - 1] = t, L.logprob
    return ids, lp.astype(np.float16)


def vllm_gen(model=TEACHER, gpu_memory_utilization=0.9, max_model_len=8192, topk=True):
    """gen() on vLLM: prompts are chats (the Qwen3 template, thinking off) or token ids; returns (text, token ids,
    top) per prompt, top the raw top-k arrays (top_arrays) when topk is asked for, else None."""
    from vllm import LLM, SamplingParams
    kw = dict(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              enable_prefix_caching=True, seed=0)
    if not topk:                                                      # the simulated user (and judge)
        llm = LLM(max_logprobs=JUDGE_K, **kw)
    else:
        try:
            llm = LLM(logprobs_mode='raw_logprobs', max_logprobs=TOPK, **kw)  # the model's own distribution
        except TypeError:
            llm = LLM(max_logprobs=TOPK, **kw)
    flat = 'flat_logprobs' in getattr(SamplingParams, '__struct_fields__', ()) or \
        'flat_logprobs' in getattr(SamplingParams, '__annotations__', {})

    def gen(prompts, max_tokens, temperature, topk=0):
        ids = [p if isinstance(p[0], int) else chat_ids(p) for p in prompts]
        keep = [i for i, x in enumerate(ids) if len(x) + max_tokens <= max_model_len]
        sp = dict(max_tokens=max_tokens, temperature=temperature, top_p=0.95)
        if topk:
            sp.update(logprobs=topk, detokenize=False, **(dict(flat_logprobs=True) if flat else {}))
        outs = llm.generate([dict(prompt_token_ids=ids[i]) for i in keep], SamplingParams(**sp), use_tqdm=False)
        got = {}
        for i, o in zip(keep, outs):
            c = o.outputs[0]
            t = list(c.token_ids)
            text = tok().decode(t, skip_special_tokens=True) if topk else c.text
            got[i] = (text, t, top_arrays(c.logprobs, len(t), topk) if topk else None)
        return [got.get(i, ('', [], None)) for i in range(len(prompts))]
    return gen


# ---------------------------------------------------------------------------------------------------------- build

def pools(split):
    """{source: items} for one split: train items of the kept sources, or test items (eval_seen: kept sources;
    eval_heldout: the held-out ones)."""
    src = [n for n in SOURCES if (n in HELD_OUT) == (split == 'eval_heldout')]
    which = 'train' if split == 'train' else 'test'
    return {n: xs for n in src if (xs := items(n, which))}


def skeletons(split, n, seed):
    """n skeletons for a split, each family drawn by MIX among those with items, each item used at most once per
    family (so no episode repeats)."""
    rng = random.Random(f'{seed}-{split}')
    pool = pools(split)
    fams = {f: [s for s in FAMILY_SOURCES[f] if s in pool] for f in MIX}
    fams = {f: s for f, s in fams.items() if s}
    order = {(f, s): rng.sample(range(len(pool[s])), len(pool[s])) * (6 if f == 'debug' else 1)
             for f, ss in fams.items() for s in ss}                   # debug: up to 6 mutants per function
    out, tries, ids = [], 0, set()
    while len(out) < n and tries < 20 * n and fams:
        tries += 1
        f = rng.choices(list(fams), weights=[MIX[x] for x in fams])[0]
        s = rng.choice(fams[f])
        if not order[(f, s)]:
            fams[f].remove(s)
            if not fams[f]:
                del fams[f]
            continue
        it = pool[s][order[(f, s)].pop()]
        partner = pool[s][order[(f, s)].pop()] if f == 'pair' and order[(f, s)] else None
        sk = skeleton(f, it, rng, partner)
        if sk:
            sk['split'] = split
            sk['id'] = hashlib.md5(f'{split}-{f}-{it["id"]}-{(partner or {}).get("id")}-{sk.get("variant")}'.encode()
                                   ).hexdigest()[:16]
            if sk['id'] not in ids:
                ids.add(sk['id'])
                out.append(sk)
    return out


def write_teacher(rows, path):
    """rows: (id, turn, (ids [n, k] int32, lp [n, k] float16)) as one parquet file (the layout in the docstring)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    tab = pa.table(dict(
        id=pa.array([r[0] for r in rows], pa.string()), turn=pa.array([r[1] for r in rows], pa.int16()),
        n=pa.array([len(r[2][0]) for r in rows], pa.int32()),
        ids=pa.array([np.ascontiguousarray(r[2][0], '<i4').tobytes() for r in rows], pa.binary()),
        lp=pa.array([np.ascontiguousarray(r[2][1], '<f2').tobytes() for r in rows], pa.binary())))
    part = Path(str(path) + '.part')
    pq.write_table(tab, part, compression='zstd')
    os.replace(part, path)


def teacher_topk_rows(path):
    """Yields (id, turn, ids [n, k] int32 tensor, lp [n, k] float16 tensor) from a teacher parquet file."""
    import numpy as np
    import pyarrow.parquet as pq
    tab = pq.read_table(path).to_pydict()
    for i, t, n, a, b in zip(tab['id'], tab['turn'], tab['n'], tab['ids'], tab['lp']):
        ids = np.frombuffer(a, '<i4').reshape(n, -1)
        lp = np.frombuffer(b, '<f2').reshape(n, -1)
        yield i, t, torch.from_numpy(ids.copy()), torch.from_numpy(lp.copy())


def write_shards(recs, out, split, first=0, size=2000, tag=''):
    """Writes recs as shards first, first + 1, ...: the records (without their teacher top-k) in shards/ and the
    top-k of their teacher turns in teacher/ (when they have it). Returns the next shard number."""
    for d in ('shards', 'teacher'):
        (Path(out) / d).mkdir(parents=True, exist_ok=True)
    for i in range(0, len(recs), size):
        p = Path(out) / 'shards' / f'{split}-{tag}{first:04d}.jsonl.gz'
        rows = []
        with gzip.open(p.with_suffix('.part'), 'wt') as f:
            for r in recs[i:i + size]:
                turns = []
                for k, t in enumerate(r['turns']):
                    if '_top' in t:
                        rows.append((r['id'], k, t['_top']))
                    turns.append({a: b for a, b in t.items() if a != '_top'})
                f.write(json.dumps(dict(r, turns=turns), ensure_ascii=False) + '\n')
        if rows:
            write_teacher(rows, Path(out) / 'teacher' / f'{split}-{tag}{first:04d}.parquet')
        os.replace(p.with_suffix('.part'), p)
        first += 1
    return first


def build(out, counts, gen, seed=0, batch=1000, log=print, max_minutes=None, sim=None, skip=0, tag='', cohort=20):
    """Skeletons per split, generated in batches (each batch's shards written as it finishes, named SPLIT-TAGNNNN);
    past max_minutes no new batch starts (the shards so far are kept). skip: the train split starts at that skeleton
    (a second generator takes skeletons the first never reaches; record ids are the skeletons', so a run that reads
    both never sees one twice). Each batch logs per family the share of simulated user turns whose first try was
    degenerate or leaked (first_failed) and the share of user turns that ended an episode. A batch goes through
    generate() in cohorts of `cohort` episodes, each through all its turns before the next starts: a cohort's prompts
    fit the models' key-value caches, so with prefix caching each turn pays only for its new tokens (the material,
    the earlier turns and, for the judge, the simulator's whole prompt are cached). Returns {split: records}."""
    got, t0 = {}, time.monotonic()
    for split, n in counts.items():
        if n <= 0:
            continue
        off = skip if split == 'train' else 0
        sks = skeletons(split, off + int(n * 1.6) + 8, seed)[off:]   # some end early or leak
        done, shard = 0, 0
        for i in range(0, len(sks), batch):
            if max_minutes and (time.monotonic() - t0) / 60 > max_minutes:
                log(json.dumps(dict(split=split, stopped='max_minutes', records=done)))
                break
            stats, samples, recs, t1 = {}, [], [], time.monotonic()
            part = sks[i:i + batch]
            for c in range(0, len(part), cohort):
                recs += generate(part[c:c + cohort], gen, random.Random(off + i + c), lambda *_: None, sim=sim,
                                 stats=stats, samples=samples)
            if samples:                                  # judged requests beside the shards, for checks by hand
                (Path(out) / 'judge').mkdir(parents=True, exist_ok=True)
                with open(Path(out) / 'judge' / f'{split}-{tag}{shard:04d}.jsonl', 'w') as f:
                    for x in samples:
                        f.write(json.dumps(x, ensure_ascii=False) + '\n')
            recs = recs[:max(0, n - done)]
            shard = write_shards(recs, out, split, shard, tag=tag)
            done += len(recs)
            rates = {f: dict(asked=v['asked'], first_failed=round(v['first_failed'] / max(1, v['asked']), 3),
                             ended=v['ended'], real_ended=v['real_ended'], judged=v['judged'],
                             judge_yes=round(v['judge_yes'] / max(1, v['judged']), 3)) for f, v in sorted(stats.items())}
            log(json.dumps(dict(split=split, records=done, of=n, s=round(time.monotonic() - t0),
                                per_min=round(len(recs) / max(1e-9, (time.monotonic() - t1) / 60), 1), user_turns=rates)))
            if done >= n:
                break
        got[split] = done
    return got


# ------------------------------------------------------------------------------------------------------- training

class Episodes:
    """The records of a split, in a seeded order, each yielded once (a run never repeats an episode); len() is the
    count. families: only those families. teacher: attach record['topk'] = {turn: (ids, lp)} from the teacher files
    (a teacher record whose top-k is missing is left out)."""
    def __init__(self, path, split, seed=0, families=None, teacher=False, wait_minutes=0):
        self.path, self.split, self.families, self.teacher = Path(path), split, families, teacher
        self.wait_minutes, self.seed, self.files, self.ids, self.cuts = wait_minutes, seed, set(), set(), {}
        self.recs, self.pos = [], 0
        self.refresh()

    def load(self, p):
        """One shard's records, each cut at its first degenerate user turn (first_degenerate; counts in self.cuts),
        those left with no turns dropped, with their top-k when teacher."""
        with gzip.open(p, 'rt') as f:
            recs = [r for line in f if (r := json.loads(line)) and (not self.families or r['family'] in self.families)]
        for r in recs:
            k, why = first_degenerate(r)
            if why:
                self.cuts[why] = self.cuts.get(why, 0) + 1
            r['turns'] = r['turns'][:k]
        recs = [r for r in recs if r['turns']]
        if self.teacher:
            tp = self.path / 'teacher' / p.name.replace('.jsonl.gz', '.parquet')
            top = {}
            if tp.exists():
                for i, t, ids, lp in teacher_topk_rows(tp):
                    top.setdefault(i, {})[t] = (ids, lp)
            for r in recs:
                r['topk'] = top.get(r['id'], {})
            recs = [r for r in recs if not r['teacher'] or all(k in r['topk'] for k in range(len(r['turns'])))]
        return recs

    def refresh(self):
        """Loads the shards not read yet (another generator's, copied in while a run trains), each record id once;
        the new records go after the ones not yet yielded, in a seeded order. Returns how many came."""
        new = []
        for p in sorted((self.path / 'shards').glob(f'{self.split}-*.jsonl.gz')):
            if p.name in self.files:
                continue
            self.files.add(p.name)
            for r in self.load(p):
                if r['id'] not in self.ids:
                    self.ids.add(r['id'])
                    new.append(r)
        random.Random(f'{self.seed}-{len(self.files)}').shuffle(new)
        self.recs += new
        return len(new)

    def __len__(self):
        return len(self.recs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.recs):
            t0 = time.monotonic()
            while not self.refresh():                                  # wait_minutes for new shards, then stop
                if time.monotonic() - t0 >= 60 * self.wait_minutes:
                    raise StopIteration
                time.sleep(30)
        self.pos += 1
        return self.recs[self.pos - 1]


def _off(model):
    return model.disable_adapter() if hasattr(model, 'disable_adapter') else _Null()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def teacher_targets(model, tk, rec, k=32):
    """Per task turn (ids [T, k], lp [T, k]): the frozen teacher's top-k next-token log-probabilities at every answer
    token, from one forward pass over teacher_ids(rec) (adapter off); k=None gives the full log-softmax [T, V]."""
    full, starts = teacher_ids(rec)
    dev = next(model.parameters()).device
    with _off(model):
        logits = model(input_ids=torch.tensor([full], device=dev)).logits[0].float()
    out = []
    for start, t in zip(starts, rec['turns']):
        z = logits[start - 1:start - 1 + len(t['response_ids'])].log_softmax(-1)
        out.append(z if k is None else tuple(reversed(z.topk(k, -1))))
    return out


@torch.no_grad()
def teacher_topk(model, tk, msgs, response_ids, k=32):
    """The frozen teacher's top-k log-probabilities (ids [T, k], lp [T, k]) at each token of response_ids after the
    chat msgs (Qwen3 template, thinking off; adapter off)."""
    pre = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    dev = next(model.parameters()).device
    with _off(model):
        logits = model(input_ids=torch.tensor([pre + list(response_ids)], device=dev)).logits[0].float()
    z = logits[len(pre) - 1:len(pre) - 1 + len(response_ids)].log_softmax(-1)
    lp, ids = z.topk(k, -1)
    return ids, lp


def kl_topk(student_logits, ids, lp):
    """Per-token KL(teacher || student) over the teacher's top-k tokens plus one bucket for every other token (exact
    on that partition: 0 when the student matches the teacher, and mass the student puts outside the top-k costs):
    student_logits [T, V], ids and lp [T, k] (the teacher's log-probabilities, not renormalized)."""
    t = lp.float()
    s = student_logits.float().log_softmax(-1).gather(-1, ids)
    kl = (t.exp() * (t - s)).sum(-1)
    t_rest = (1 - t.exp().sum(-1)).clamp_min(1e-12)
    s_rest = (1 - s.exp().sum(-1)).clamp_min(1e-12)
    return kl + t_rest * (t_rest.log() - s_rest.log())


def closed_share(kl_none, kl_on):
    """The share of the no-message KL that the messages close."""
    return (kl_none - kl_on) / kl_none if kl_none > 0 else float('nan')


# ------------------------------------------------------------------------------------------------------------ CLI

def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('fetch', 'fetch_one', 'build', 'show'))
    ap.add_argument('--cache', default=None)
    ap.add_argument('--out', default='opendata_out')
    ap.add_argument('--sources', default='')
    ap.add_argument('--train', type=int, default=50000)
    ap.add_argument('--eval_seen', type=int, default=600)
    ap.add_argument('--eval_heldout', type=int, default=400)
    ap.add_argument('--family', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default=TEACHER)
    ap.add_argument('--max_minutes', type=float, default=None)
    ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--sim_model', default='', help='the simulated user (default: --model)')
    ap.add_argument('--sim_mem', type=float, default=0.55, help="the simulator's share of GPU memory")
    ap.add_argument('--teacher_mem', type=float, default=0.33, help="the teacher's share beside a simulator")
    ap.add_argument('--skip', type=int, default=0, help='train skeletons to skip (a second generator)')
    ap.add_argument('--tag', default='', help='shard name tag: SPLIT-TAGNNNN')
    ap.add_argument('--cohort', type=int, default=20, help='episodes generated together through all their turns')
    args = ap.parse_args()
    if args.cache:
        CACHE = Path(args.cache)
        os.environ.setdefault('CORPUS_CACHE', str(CACHE / 'corpus'))
    if args.cmd == 'fetch':
        print(json.dumps(fetch_all([s for s in args.sources.split(',') if s] or None)))
    elif args.cmd == 'fetch_one':
        fetch_one(args.sources)
    elif args.cmd == 'build':
        sim = None
        if args.sim_model and args.sim_model != args.model:
            sim = vllm_gen(args.sim_model, gpu_memory_utilization=args.sim_mem, topk=False)
            gen = vllm_gen(args.model, gpu_memory_utilization=args.teacher_mem)
        else:
            gen = vllm_gen(args.model)
        counts = dict(eval_seen=args.eval_seen, eval_heldout=args.eval_heldout, train=args.train)
        print(json.dumps(dict(model=args.model, sim=args.sim_model or args.model, built=build(
            args.out, counts, gen, args.seed, args.batch, max_minutes=args.max_minutes, sim=sim, skip=args.skip,
            tag=args.tag, cohort=args.cohort))), flush=True)
    else:
        seen = set()
        for split in ('train', 'eval_seen', 'eval_heldout'):
            for r in Episodes(args.out, split):
                if r['family'] in seen or (args.family and r['family'] != args.family):
                    continue
                seen.add(r['family'])
                print(json.dumps({k: v for k, v in r.items() if k != 'turns'}, ensure_ascii=False)[:1500])
                for t in r['turns']:
                    print(f"  [{t['who']}] {t['user'][:300]!r}\n      -> {t['response'][:300]!r}")


if __name__ == '__main__':
    main()
