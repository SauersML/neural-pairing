"""Talk with Quill and Ember, the linked twins, in a web page served from this machine.

Usage, from pairing/:
  python chat_live.py [--run STEP_DIR] [--port 8765] [--idle-exit SECONDS]
      The exact checkpoint in bf16 on a CUDA GPU (NL_TINY=1 NL_DEVICE=cpu: tiny random models, for tests). --run
      defaults to a download of huggingface.co/Sauers/neural-pairing, the final twins.
  python chat_live.py --backend mlx [--run STEP_DIR]
      Apple silicon: 4-bit MLX copies of both base models with the same adapters and link (not the exact weights).
Then open http://localhost:8765. The server answers on 127.0.0.1 only; reach a remote GPU through an SSH port forward.

Who reads what. A conversation is a list of messages from the person, Quill and Ember, written in rounds:
  both:     the person writes; Quill and Ember answer in turn (Quill first unless first='ember'), each reading the
            other's reply as text while it is written, as in training (session's dialogue mode).
  talk:     the twins write N turns to each other, after a message from the person if one is given.
  together: the person writes; both answer at once and neither reads the other's reply (session's duplex mode), so the
            link is the only connection between them.
In each round, twin x's prompt is its system prompt (the dialogue system prompt, the sentence FRAMINGS gives for the
person and the labels, x's private note, which no other prompt contains, and the chat-training tone line), then every
earlier message x has read, in ChatML: x's own replies as assistant messages and every other message as a user message
opened by its writer's label. A reply from a together round is read by its author only. Each round reads the prompts
at its first tick, as session reads every prompt, so with the link on each twin's prompt tokens read the other's.

The link is on (as trained) or cut (session mode 'zero': no exchange runs, so each twin is its LoRA-adapted model
alone). Any twin reply can be written again with the other link state from the same prompts, the same earlier turns of
its round (teacher-forced to their exact tokens) and the same random seed, so the two versions differ only by the link.
A token's shade is the link's write size at the position that predicted it, RMS(h_after - h_before) / RMS(h_before)
averaged over the exchange layers: how much the link changed that hidden state, not what it carried.

HTTP (127.0.0.1, same origin): GET / (the page), /state, /view?reply=ID (the author's exact input before that reply),
/health; POST /send {mode, message, turns, first, link, temperature, max_tokens, notes} and POST /compare {reply}
stream server-sent events; POST /stop ends the running round after the current token (finished and partial replies are
kept, partial ones marked interrupted); POST /reset clears the conversation.
"""
import argparse
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import secrets
import threading
import time
import traceback
from urllib.parse import parse_qs, urlparse

import torch

from coread import CHAT_TONE
import games as G
from link import MODELS, SIDES, partner
import play_live as L
import session as S

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / 'figures' / 'assets'
IMAGES = ('quill.png', 'ember.png', 'quill_cut.png', 'ember_cut.png')
PERSON = 'Person'
# How each twin's input names the person and the partner: the label that opens each message they write (newline-ended,
# so a label never shares a token with the text) and the sentence in the system prompt that explains it. 'plain' keeps
# the partner unlabeled, as in training. chat_checks.py compares the three on the final twins; 'names' is the default.
FRAMINGS = {
    'plain': dict(person=f'{PERSON}:\n', partner=lambda p: '',
                  intro=lambda p: (f"A person has joined your conversation with {p}. The person's messages begin with "
                                   f"'{PERSON}:'; {p}'s messages have no label.")),
    'names': dict(person=f'{PERSON}:\n', partner=lambda p: f'{p}:\n',
                  intro=lambda p: (f"A person has joined your conversation with {p}. Each message you read begins "
                                   f"with its writer's name: '{PERSON}:' for the person and '{p}:' for {p}.")),
    'tags': dict(person=f'[{PERSON}]\n', partner=lambda p: f'[{p}]\n',
                 intro=lambda p: (f"A person has joined your conversation with {p}. Each message you read begins "
                                  f"with its writer in brackets: [{PERSON}] for the person and [{p}] for {p}.")),
}
TOGETHER = "You and {p} now answer the person's latest message at the same time, so you will not read {p}'s answer."
MODES = ('both', 'talk', 'together')


class Stopped(Exception):
    pass


def name(x):
    return MODELS[x]['name']


def system(x, mode, note, framing):
    """x's system prompt in a round of mode, with its private note in the chat-training wording."""
    p = name(partner(x))
    s = G.system(x, 'dialogue') + ' ' + FRAMINGS[framing]['intro'](p)
    if mode == 'together':
        s += ' ' + TOGETHER.format(p=p)
    if note:
        return s + f'\n\nYour private situation ({p} cannot see it): {note} {CHAT_TONE}'
    return s + '\n\n' + CHAT_TONE


def checked_text(value, what, limit, nonempty=False):
    if not isinstance(value, str) or len(value) > limit or (nonempty and not value.strip()):
        raise ValueError(f'{what} must be {"nonempty " if nonempty else ""}text of at most {limit} characters')
    if '<|' in value or '|>' in value:
        raise ValueError(f'{what} cannot contain chat control markers')
    return value.strip()


class Meter:
    """Wraps pair.step and pair._exchange to record, at each tick, the link's write size at the position that predicts
    each side's next token, averaged over the exchanges. It changes no state and no logit."""

    def __init__(self, pair):
        self.pair, self.effect, self.last, self.ratios = pair, {}, {}, {x: [] for x in SIDES}

    def step(self, chunk, cache, qmask, tick, *args, **kwargs):
        # the chunk's last real position predicts the next token (session.generate reads its logits there)
        self.last = {x: int(v[1][0, -v[0].shape[1]:].sum()) - 1 for x, v in chunk.items()}
        self.ratios = {x: [] for x in SIDES}
        out = self.original_step(chunk, cache, qmask, tick, *args, **kwargs)
        self.effect = {x: (float(torch.stack(r).mean()) if r and isinstance(r[0], torch.Tensor) else L.mean(r))
                       for x, r in self.ratios.items()}
        return out

    def exchange(self, k, h, *args, **kwargs):
        out, gates = self.original_exchange(k, h, *args, **kwargs)
        for x in h:
            q = self.last.get(x, -1)
            if q < 0:
                continue
            before, after = h[x][0, q], out[x][0, q]
            if isinstance(before, torch.Tensor):
                before, after = before.float(), after.float()
                self.ratios[x].append((after - before).norm() / before.norm().clamp_min(1e-12))
            else:
                before, after = L.cpu(before), L.cpu(after)
                self.ratios[x].append(float((after - before).norm() / before.norm().clamp_min(1e-12)))
        return out, gates

    @contextlib.contextmanager
    def installed(self):
        self.original_step, self.original_exchange = self.pair.step, self.pair._exchange
        self.pair.step, self.pair._exchange = self.step, self.exchange
        try:
            yield self
        finally:
            del self.pair.step, self.pair._exchange


def public(m):
    keep = ('id', 'who', 'text', 'round', 'mode', 'turn', 'link', 'pieces', 'shade', 'complete', 'truncated',
            'interrupted', 'alt')
    return {k: m[k] for k in keep if k in m}


class Chat:
    """One conversation with the pair: its messages and rounds, and the generation that writes them."""

    def __init__(self, pair, backend='torch', about='', framing='names', max_tokens=200, context_tokens=8192):
        self.pair, self.toks, self.backend, self.about, self.framing = pair, pair.toks, backend, about, framing
        self.max_tokens, self.context_tokens = max_tokens, context_tokens
        self.lock = threading.Lock()             # the conversation's data, held briefly
        self.busy = threading.Lock()             # one round or comparison at a time (held by the server)
        self.stop = threading.Event()
        self.end = {x: self.toks[x].convert_tokens_to_ids('<|im_end|>') for x in SIDES}
        self.notes, self.next_id, self.epoch = {x: '' for x in SIDES}, 1, 0
        self.messages, self.rounds = [], []

    def reset(self):
        self.stop.set()
        with self.lock:
            self.epoch += 1
            self.messages, self.rounds = [], []

    def new_id(self):
        with self.lock:
            self.next_id += 1
            return self.next_id - 1

    # ---- what each twin reads

    def label(self, who):
        f = FRAMINGS[self.framing]
        return f['person'] if who == 'person' else f['partner'](name(who))

    @staticmethod
    def reads(x, m):
        """Whether twin x has read message m: its own, the person's, and the partner's from rounds where it read them."""
        return m['who'] in (x, 'person') or m['mode'] != 'together'

    def render(self, x, mode, notes, messages):
        """x's prompt token IDs for a round of mode: its system prompt, then every message of messages x has read."""
        tok = self.toks[x]
        out = S.prefix(tok, x, system(x, mode, notes[x], self.framing))
        for m in messages:
            if not self.reads(x, m):
                continue
            if m['who'] == x:
                out += S.assistant_header(tok, x) + m['tokens'] + [self.end[x]] + S.ids(tok, '\n')
            else:
                out += S.ids(tok, '<|im_start|>user\n' + self.label(m['who'])) + S.ids(tok, m['text'])
                out += S.ids(tok, '<|im_end|>\n')
        return out

    def episode(self, rnd, upto=None, earlier=()):
        """Round rnd as a session episode: a duplex episode, or a dialogue of its turns up to index upto (all when
        None) whose first len(earlier) turns are teacher-forced to the exact tokens of the replies earlier."""
        if rnd['mode'] == 'together':
            return dict(mode='duplex', system={x: '' for x in SIDES}, task={x: None for x in SIDES},
                        prompt=rnd['prompt'], reply={x: ('', 1.0) for x in SIDES})
        turns = rnd['turns'] if upto is None else rnd['turns'][:upto + 1]
        ep = dict(mode='dialogue', system={x: '' for x in SIDES}, task={x: None for x in SIDES},
                  labels={x: self.label(x) for x in SIDES}, prompt=rnd['prompt'], turns=[(x, '', 1.0) for x in turns])
        if earlier:
            ep['forced'] = list(range(len(earlier)))
            ep['turn_tokens'] = ([r['tokens'] + [self.end[r['who']]] for r in earlier]
                                 + [None] * (len(turns) - len(earlier)))
        return ep

    def turns_before(self, rnd, j):
        """The replies of round rnd's turns 0 .. j - 1 (a dialogue round)."""
        out = [next((m for m in self.messages if m['round'] == rnd['id'] and m.get('turn') == k), None)
               for k in range(j)]
        if None in out:
            raise ValueError('an earlier turn of this round is missing')
        return out

    # ---- requests

    def prepare(self, data):
        if not isinstance(data, dict):
            raise ValueError('send a JSON object')
        mode = data.get('mode', 'both')
        if mode not in MODES:
            raise ValueError('mode must be both, talk or together')
        message = checked_text(data.get('message', ''), 'The message', 4000, nonempty=mode != 'talk')
        link = data.get('link', 'on')
        if link not in ('on', 'cut'):
            raise ValueError('link must be on or cut')
        first = data.get('first', 'quill')
        if first not in SIDES:
            raise ValueError('first must be quill or ember')
        turns = data.get('turns', 4)
        if type(turns) is not int or not 1 <= turns <= 12:
            raise ValueError('turns must be a whole number from 1 to 12')
        temp = data.get('temperature', 0.7)
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not math.isfinite(temp) or not 0 <= temp <= 2:
            raise ValueError('temperature must be from 0 to 2')
        limit = data.get('max_tokens', 120)
        if type(limit) is not int or not 8 <= limit <= self.max_tokens:
            raise ValueError(f'max_tokens must be a whole number from 8 to {self.max_tokens}')
        notes = data.get('notes', self.notes)
        if not isinstance(notes, dict) or set(notes) - set(SIDES):
            raise ValueError('notes may name only quill and ember')
        notes = {x: checked_text(notes.get(x, ''), name(x) + "'s note", 1000) for x in SIDES}
        return dict(mode=mode, message=message, link=link, first=first, turns=turns, temperature=float(temp),
                    limit=limit, notes=notes)

    def send(self, req, emit):
        """One round: the person's message (if any), then the twins' replies, streamed and then kept."""
        self.stop.clear()
        with self.lock:
            epoch, old = self.epoch, list(self.messages)
        history = list(old)
        if req['message']:
            history.append(dict(id=self.new_id(), who='person', text=req['message'], mode=req['mode']))
        last = next((m['who'] for m in reversed(history) if m['who'] in SIDES), None)
        if req['mode'] == 'both':
            order = [req['first'], partner(req['first'])]
        elif req['mode'] == 'talk':
            order = [partner(last) if last else 'quill']
            while len(order) < req['turns']:
                order.append(partner(order[-1]))
        else:
            order = list(SIDES)
        rnd = dict(id=self.new_id(), mode=req['mode'], link=req['link'], seed=secrets.randbits(31),
                   temperature=req['temperature'], limit=req['limit'], turns=order,
                   prompt={x: self.render(x, req['mode'], req['notes'], history) for x in SIDES})
        need = max(map(len, rnd['prompt'].values())) + len(order) * (req['limit'] + 24)
        if need > self.context_tokens:
            raise ValueError(f'This round needs up to {need} tokens of context, more than the {self.context_tokens} '
                             f'this server allows. Reset the conversation, or ask for fewer turns or shorter replies.')
        with self.lock:
            if self.epoch != epoch:
                raise ValueError('The conversation was reset. Send again.')
            self.notes = req['notes']
            if req['message']:
                history[-1]['round'] = rnd['id']
            self.messages = history
            self.rounds.append(rnd)
        if req['message']:
            emit('message', public(history[-1]))
        try:
            replies = self.generate(rnd, self.episode(rnd), rnd['link'], emit, lambda x, turn: self.new_id())
        except Exception:
            with self.lock:
                if self.epoch == epoch:
                    self.messages = old
                    self.rounds.remove(rnd)
            raise
        with self.lock:
            if self.epoch == epoch:
                self.messages = self.messages + replies
        emit('done', dict(round=rnd['id'], stopped=self.stop.is_set()))

    def compare(self, reply_id, emit):
        """Writes twin reply reply_id again with the other link state, from the same prompts, earlier turns and seed.
        A together round is written again whole, so both of its replies get the other version."""
        self.stop.clear()
        with self.lock:
            epoch = self.epoch
            m = next((m for m in self.messages if m['id'] == reply_id and m['who'] in SIDES), None)
            if m is None:
                raise ValueError('There is no such twin reply.')
            if m.get('interrupted'):
                raise ValueError('This reply was interrupted, so there is no whole reply to compare.')
            rnd = next(r for r in self.rounds if r['id'] == m['round'])
            link = 'cut' if rnd['link'] == 'on' else 'on'
            if rnd['mode'] == 'together':
                mine = {r['who']: r for r in self.messages if r['round'] == rnd['id'] and r['who'] in SIDES}
                ep, only = self.episode(rnd), None
            else:
                mine = {m['turn']: m}
                ep, only = self.episode(rnd, m['turn'], self.turns_before(rnd, m['turn'])), m['turn']

        def ids(x, turn):
            r = mine.get(x if turn is None else turn)
            return None if r is None else r['id']

        got = self.generate(rnd, ep, link, emit, ids, only=only, alt=True)
        with self.lock:
            if self.epoch == epoch and not self.stop.is_set():
                for r in got:
                    orig = mine.get(r['who'] if r['turn'] is None else r['turn'])
                    if orig is not None:
                        orig.setdefault('alt', {})[link] = {k: r[k] for k in ('text', 'pieces', 'shade', 'truncated')}
        emit('done', dict(round=rnd['id'], compare=reply_id, link=link, stopped=self.stop.is_set()))

    def generate(self, rnd, ep, link, emit, ids, only=None, alt=False):
        """Writes the episode of round rnd with the link on or cut, streaming each reply as it grows (as an alternative
        version when alt); returns the replies, partial ones marked interrupted."""
        live = {}
        meter = Meter(self.pair)

        def on_token(i, x, k, turn):
            if self.stop.is_set():
                raise Stopped
            if only is not None and turn != only:
                return                           # a teacher-forced earlier turn
            key = x if turn is None else turn
            if key not in live:
                live[key] = dict(id=ids(x, turn), who=x, round=rnd['id'], mode=rnd['mode'], turn=turn, link=link,
                                 tokens=[], pieces=[], shade=[], text='', complete=False)
            r, tok = live[key], self.toks[x]
            if k == self.end[x]:
                r['complete'], r['truncated'] = True, len(r['tokens']) >= rnd['limit']
                r['pieces'] = L.token_pieces(tok, r['tokens'], r['text'])
            else:
                r['tokens'].append(k)
                r['shade'].append(round(meter.effect.get(x, 0.0), 5))
                r['text'] = tok.decode(r['tokens'])
                shown, before = r['text'].rstrip('�'), ''.join(r['pieces'])
                if shown.startswith(before):
                    r['pieces'].append(shown[len(before):])
                else:
                    r['pieces'] = L.token_pieces(tok, r['tokens'], r['text'])
            if r['id'] is not None:
                emit('reply', dict(public(r), alt=link if alt else None))

        gen = torch.Generator(device=S.DEVICE).manual_seed(rnd['seed'])
        try:
            with torch.no_grad(), meter.installed():
                S.generate(self.pair, [ep], max_turn=rnd['limit'], temperature=rnd['temperature'], gen=gen,
                           mode='zero' if link == 'cut' else None, on_token=on_token)
        except Stopped:
            pass
        out = []
        for key in sorted(live, key=lambda k: SIDES.index(k) if isinstance(k, str) else k):
            r = live[key]
            if not r['complete']:
                if not r['tokens']:
                    continue
                r['interrupted'], r['truncated'] = True, False
                r['pieces'] = L.token_pieces(self.toks[r['who']], r['tokens'], r['text'])
                if r['id'] is not None:
                    emit('reply', dict(public(r), alt=link if alt else None))
            out.append(r)
        return out

    # ---- views

    def view(self, reply_id):
        """The author's exact input, special tokens included, up to the position that predicts the reply's first
        token."""
        with self.lock:
            m = next((m for m in self.messages if m['id'] == reply_id and m['who'] in SIDES), None)
            if m is None:
                raise ValueError('There is no such twin reply.')
            rnd = next(r for r in self.rounds if r['id'] == m['round'])
            x = m['who']
            if rnd['mode'] == 'together':
                ids = rnd['prompt'][x] + S.assistant_header(self.toks[x], x)
            else:
                j = m['turn']
                built = S.build(self.episode(rnd, j, self.turns_before(rnd, j) + [m]), self.toks)[x]
                first = next(a for t, a, _ in built['spans'] if t == j)
                ids = built['ids'][:built['labels'][first][0] + 1]
        p = name(partner(x))
        link = (f"With the link on, {name(x)} also read {p}'s hidden states at every tick, from {p}'s prompt on."
                if rnd['link'] == 'on' else f"The link was cut: nothing from {p} reached {name(x)} but this text.")
        return dict(reply=reply_id, who=x, text=self.toks[x].decode(ids), tokens=len(ids), link=link,
                    seed=rnd['seed'], temperature=rnd['temperature'])

    def state(self):
        with self.lock:
            return dict(messages=[public(m) for m in self.messages], notes=self.notes, backend=self.backend,
                        about=self.about, framing=self.framing, max_tokens=self.max_tokens,
                        context_tokens=self.context_tokens, busy=self.busy.locked(),
                        names={x: name(x) for x in SIDES})


class Server(ThreadingHTTPServer):
    """Answers requests on their own threads and writes every round on the thread that runs serve(), the thread that
    loaded the models: an MLX array evaluates only on the thread whose stream made it, and a round written on another
    thread fails ("There is no Stream(gpu, 0) in current thread")."""
    daemon_threads = True

    def __init__(self, chat, port, idle_exit=0):
        self.chat, self.idle_exit, self.last = chat, idle_exit, time.time()
        self.jobs, self.closed = queue.Queue(), threading.Event()
        super().__init__(('127.0.0.1', port), Handler)
        if idle_exit:
            threading.Thread(target=self.watch, daemon=True).start()

    def serve(self):
        """Serves until shutdown(), running the rounds the handlers queue on this thread."""
        def requests():
            try:
                self.serve_forever()
            finally:
                self.closed.set()

        threading.Thread(target=requests, daemon=True).start()
        while not self.closed.is_set():
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            job()

    def watch(self):
        while True:
            time.sleep(min(15, self.idle_exit))
            if not self.chat.busy.locked() and time.time() - self.last > self.idle_exit:
                print(f'no request for {self.idle_exit:g} s: exiting', flush=True)
                self.shutdown()
                return


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass                                     # never log messages or private notes

    def allowed(self):
        host = urlparse('//' + (self.headers.get('Host') or '')).hostname
        origin = self.headers.get('Origin')
        if host not in ('127.0.0.1', 'localhost') or (origin and urlparse(origin).hostname != host):
            self.send_error(403, 'local same-origin requests only')
            return False
        return True

    def reply(self, status, data, kind='application/json; charset=utf-8'):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return
        u = urlparse(self.path)
        chat = self.server.chat
        if u.path != '/health':
            self.server.last = time.time()
        try:
            if u.path == '/':
                self.reply(200, (HERE / 'chat_page.html').read_bytes(), 'text/html; charset=utf-8')
            elif u.path[1:] in IMAGES:
                self.reply(200, (ASSETS / u.path[1:]).read_bytes(), 'image/png')
            elif u.path == '/health':
                self.reply(200, dict(ready=True, busy=chat.busy.locked(),
                                     idle=round(time.time() - self.server.last, 1)))
            elif u.path == '/state':
                self.reply(200, chat.state())
            elif u.path == '/view':
                self.reply(200, chat.view(int(parse_qs(u.query)['reply'][0])))
            else:
                self.send_error(404)
        except (ValueError, KeyError, IndexError) as e:
            self.reply(400, dict(error=str(e)))

    def do_POST(self):
        if not self.allowed():
            return
        self.server.last = time.time()
        chat = self.server.chat
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536 or self.headers.get_content_type() != 'application/json':
                raise ValueError('send a JSON body of at most 64 KB')
            data = json.loads(self.rfile.read(size))
            if self.path == '/stop':
                chat.stop.set()
                return self.reply(200, dict(stopping=True))
            if self.path == '/reset':
                chat.reset()
                return self.reply(200, dict(reset=True))
            if self.path == '/send':
                req = chat.prepare(data)
                run = lambda emit: chat.send(req, emit)
            elif self.path == '/compare':
                rid = data.get('reply') if isinstance(data, dict) else None
                if type(rid) is not int:
                    raise ValueError('reply must be a reply id')
                run = lambda emit: chat.compare(rid, emit)
            else:
                return self.send_error(404)
        except (ValueError, UnicodeError) as e:
            return self.reply(400, dict(error=str(e)))
        if not chat.busy.acquire(blocking=False):
            return self.reply(409, dict(error='The twins are still writing. Stop them first.'))
        try:
            self.stream(run)
        finally:
            chat.busy.release()
            self.server.last = time.time()

    def stream(self, run):
        """Queues run(emit) for the serving thread and sends what it emits as server-sent events."""
        chat, events = self.server.chat, queue.Queue()

        def job():
            try:
                run(lambda event, payload: events.put((event, payload)))
            except ValueError as e:
                events.put(('error', dict(error=str(e))))
            except Exception:
                traceback.print_exc()
                events.put(('error', dict(error='Generation failed, and the conversation is as it was before. The '
                                                'server log has the details.')))
            finally:
                events.put(None)

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.server.jobs.put(job)
        gone = False
        while (item := events.get()) is not None:
            if gone:
                continue
            try:
                self.wfile.write(f'event: {item[0]}\ndata: {json.dumps(item[1], ensure_ascii=False)}\n\n'.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                gone = True
                chat.stop.set()                  # the page went away: end the round, keep what is written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', help='step directory (quill/, ember/, bridge.*); default: Sauers/neural-pairing')
    ap.add_argument('--backend', choices=('torch', 'mlx'), default='torch')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--framing', choices=tuple(FRAMINGS), default='names')
    ap.add_argument('--max-tokens', type=int, default=200, help='the longest reply a request may ask for')
    ap.add_argument('--context-tokens', type=int, default=8192)
    ap.add_argument('--idle-exit', type=float, default=0, help='exit after this many seconds without a request (0: never)')
    args = ap.parse_args()
    run = args.run
    if run is None:
        from huggingface_hub import snapshot_download
        run = snapshot_download('Sauers/neural-pairing')
    if args.backend == 'mlx':
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(10 * 2**30)
        mx.set_cache_limit(256 * 2**20)
        about = '4-bit MLX copies of both base models (not the exact weights)'
    else:
        from evaluate import load_pair
        about = ('tiny random test models' if os.environ.get('NL_TINY') else
                 f'exact weights, bf16 on {torch.cuda.get_device_name()}' if str(S.DEVICE).startswith('cuda')
                 else 'exact weights on the CPU')
    pair, _ = load_pair(run)
    chat = Chat(pair, args.backend, about, args.framing, args.max_tokens, args.context_tokens)
    with Server(chat, args.port, args.idle_exit) as server:
        print(f'ready: http://localhost:{server.server_port} ({about})', flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve()


if __name__ == '__main__':
    main()
