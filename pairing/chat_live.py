"""Talk to Quill and Ember locally, using a trained 4-bit MLX pair.

Run from pairing/: python chat_live.py --run STEP_DIR --replies REPLIES
On the Mac use MPD_MEM_GIB=12 and /usr/bin/lockf -k
/Users/user/mpd-data/locks/macgpu.lock. Torch loading is only for NL_TINY=1 CPU tests.

Exactly what the twins see:
* Together: each sees games.system(x, 'duplex'), its own private note, every
  human message and its own earlier replies, with session's ChatML headers.
  Neither sees the other's reply text. Both write one token per session tick.
* Turns: each sees games.system(x, 'dialogue'), the short-message instruction
  and private-task layout used by coread.play_system, its own private note,
  every human message, its own replies as assistant messages and the partner's
  replies as user messages. Quill writes first; Ember listens live, then replies
  while Quill listens, exactly as session.generate schedules dialogue.
Private notes are inserted only into the owner's system message, never copied
to the partner. Their hidden states can cross the open link. In turns, a note
revealed in a reply can also travel as text, including with the link cut.

Each exchange rebuilds the visible text history at tick -1 in fresh KV caches;
hidden-state memory is not retained between human messages. Changing mode resets
history, preventing prior dialogue text from leaking into together mode. Reset
clears history and notes; changing a note alone does not erase earlier replies.
History is bounded by --context-tokens; overflow asks for reset, never silently
drops messages. There is one shared conversation per server, in memory only.

Generation and sampling are session.generate unchanged. A scoped prefix builder
adds prior ChatML messages; play_live.Observer's installation/restoration and
token_pieces are reused. The observer only watches the original Pair.step and
_exchange: it streams sampled IDs when consumed on the next tick, and annotates
them with the write at the position that predicted them. Cut uses mode='zero',
which bypasses exchanges entirely. Shading is the mean across exchanges of
RMS(h_after-h_before)/RMS(h_before), including dtype rounding, as in play_live;
it is descriptive write size, not causal attribution or proof of transfer.

HTTP: GET / serves the page; POST /chat accepts {message, mode: together|turns,
link: on|cut, notes: {quill, ember}, temperature, max_tokens?} and streams SSE
snapshot events followed by done (or error). POST /reset clears the conversation.
Requests are serialized on the server thread, including all MLX operations.
--replies is validated provenance only: no teacher replies are fed to the chat.
"""
import argparse
import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import threading
import traceback

os.environ.setdefault('NL_DEVICE', 'cpu')
import torch

import coread as C
import games as G
from link import SIDES, partner
import play_live as L
import session as S


# Prefix injection is scoped and restored even on disconnect. Also serialize
# separately constructed Chat instances in this process, since S.prefix is shared.
GENERATION_LOCK = threading.RLock()


def episode(message, mode, notes):
    systems = {}
    for x in SIDES:
        p = partner(x).title()
        text = G.system(x, 'duplex' if mode == 'together' else 'dialogue')
        text += ('\n\nA person is talking to both of you. Reply to their latest message. '
                 'Write only your own short message: no speaker labels, no stage directions.')
        if mode == 'turns':
            # Same identity, private-task placement and tone as play_system,
            # without inventing a game or a partner secret for free conversation.
            text += ' ' + C.PLAY_TONE.replace('This is a game between twins:', 'In this conversation,')
        if notes[x]:
            text += f'\n\nYour private task ({p} cannot see it): {notes[x]}'
        systems[x] = text
    ep = dict(game='live_chat', mode='duplex' if mode == 'together' else 'dialogue',
              system=systems, task={x: message for x in SIDES}, meta={})
    if mode == 'turns':
        ep['turns'] = [(x, '', 1.) for x in SIDES]
    return ep


@contextlib.contextmanager
def history_prefix(history, mode):
    """Use session's exact tokenized headers; no change to its tick scheduler."""
    original = S.prefix

    def prefix(tok, x, system, task=None):
        out = original(tok, x, system)
        for item in history:
            out += S.ids(tok, '<|im_start|>user\n' + item['message'] + '<|im_end|>\n')
            for y in SIDES:
                if y == x:
                    out += S.assistant_header(tok, x) + S.ids(tok, item['text'][x])
                    out += S.ids(tok, '<|im_end|>\n')
                elif mode == 'turns':
                    out += S.ids(tok, '<|im_start|>user\n' + item['text'][y] + '<|im_end|>\n')
        if task:
            out += S.ids(tok, '<|im_start|>user\n' + task + '<|im_end|>\n')
        return out

    S.prefix = prefix
    try:
        yield prefix
    finally:
        S.prefix = original


class ChatObserver(L.Observer):
    """Observe both duplex writers or two dialogue turns, without changing logits."""
    def __init__(self, pair, ep, emit):
        self.pair, self.ep, self.emit_event = pair, ep, emit
        self.tick, self.turn = -1, 0
        self.tokens = {x: [] for x in SIDES}
        self.metrics = {x: [] for x in SIDES}
        self.layers = {x: [] for x in SIDES}
        self.started = {x: False for x in SIDES}

    def snapshot(self, x, final=False, truncated=False):
        text = self.pair.toks[x].decode(self.tokens[x])
        pieces = L.token_pieces(self.pair.toks[x], self.tokens[x], text)
        if not final and pieces:
            pieces[-1] = pieces[-1].rstrip('\ufffd')
        return dict(speaker=x, text=''.join(pieces), final=final, truncated=truncated,
                    tokens=[dict(id=t, text=p, **m) for t, p, m in
                            zip(self.tokens[x], pieces, self.metrics[x])])

    def exchange(self, k, h, *args, **kwargs):
        out, gates = self.original_exchange(k, h, *args, **kwargs)
        if self.tick >= 0:
            for x in self.speakers:
                if self.valid.get(x):
                    q = self.valid[x][-1]
                    before, after = L.cpu(h[x][0, q]), L.cpu(out[x][0, q])
                    self.layers[x].append(float((after - before).norm() / before.norm().clamp_min(1e-12)))
        return out, gates

    def step(self, chunk, cache, qmask, tick, *args, **kwargs):
        self.tick = tick
        self.valid = {x: torch.where(v[1][0, -v[0].shape[1]:].bool())[0].tolist() for x, v in chunk.items()}
        dialogue = self.ep['mode'] == 'dialogue'
        if tick >= 0:
            speakers = [SIDES[self.turn]] if dialogue else SIDES
            for x in speakers:
                if self.started[x] and self.valid.get(x):
                    token = chunk[x][0][0, self.valid[x][0]].item()
                    if token == self.pair.toks[x].convert_tokens_to_ids('<|im_end|>'):
                        if dialogue:
                            self.emit_event('snapshot', self.snapshot(x, final=True))
                            self.turn += 1
                    else:
                        if len(self.valid[x]) != 1:
                            raise ValueError('chat observer requires unpaused free generation')
                        self.tokens[x].append(token)
                        self.emit_event('snapshot', self.snapshot(x))
        self.speakers = ([SIDES[self.turn]] if dialogue and self.turn < len(SIDES)
                         else [] if dialogue else SIDES)
        self.layers = {x: [] for x in SIDES}
        result = self.original_step(chunk, cache, qmask, tick, *args, **kwargs)
        if tick >= 0:
            for x in self.speakers:
                if self.valid.get(x):
                    self.started[x] = True
                    self.metrics[x].append(dict(tick=tick, effect=L.mean(self.layers[x])))
        return result

    def finish(self, result):
        turns = []
        for i, x in enumerate(SIDES):
            ids = result['tokens'][i] if self.ep['mode'] == 'dialogue' else result['tokens'][x]
            sampled = result['sampled'][i] if self.ep['mode'] == 'dialogue' else result['sampled'][x]
            if self.tokens[x] != ids[:-1] or len(self.metrics[x]) != len(ids):
                raise ValueError('chat observer disagrees with session.generate tokens or prediction positions')
            turn = self.snapshot(x, final=True, truncated=not sampled[-1])
            turn['end_token'] = dict(id=ids[-1], sampled=sampled[-1], **self.metrics[x][-1])
            self.emit_event('snapshot', turn)
            turns.append(turn)
        return turns


def checked_text(value, name, limit, nonempty=False):
    if not isinstance(value, str) or len(value) > limit or (nonempty and not value.strip()):
        raise ValueError(f'{name} must be {"nonempty " if nonempty else ""}text, at most {limit} characters')
    if '<|' in value:
        raise ValueError(f'{name} cannot contain ChatML control markers')
    return value


class Chat:
    def __init__(self, pair, max_tokens=128, context_tokens=4096):
        self.pair, self.max_tokens, self.context_tokens = pair, max_tokens, context_tokens
        self.reset()

    def reset(self):
        with GENERATION_LOCK:
            self.history, self.mode = [], None
            self.notes = {x: '' for x in SIDES}

    def prepare(self, data):
        if not isinstance(data, dict):
            raise ValueError('request must be a JSON object')
        message = checked_text(data.get('message'), 'message', 8000, nonempty=True)
        mode, link = data.get('mode', 'together'), data.get('link', 'on')
        if mode not in ('together', 'turns') or link not in ('on', 'cut'):
            raise ValueError('mode must be together/turns; link must be on/cut')
        notes = data.get('notes', self.notes)
        if not isinstance(notes, dict) or set(notes) - set(SIDES):
            raise ValueError('notes must contain only quill and ember')
        notes = {x: checked_text(notes.get(x, ''), x + ' note', 2000) for x in SIDES}
        temp = data.get('temperature', .7)
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not math.isfinite(temp) or not 0 <= temp <= 2:
            raise ValueError('temperature must be between 0 and 2')
        limit = data.get('max_tokens', self.max_tokens)
        if type(limit) is not int or not 1 <= limit <= self.max_tokens:
            raise ValueError(f'max_tokens must be between 1 and {self.max_tokens}')
        history = self.history if self.mode in (None, mode) else []
        ep = episode(message, mode, notes)
        with GENERATION_LOCK, history_prefix(history, mode):
            for x in SIDES:
                n = len(S.prefix(self.pair.toks[x], x, ep['system'][x], message))
                if n + 2 * limit + 64 > self.context_tokens:
                    raise ValueError('Conversation exceeds the context budget. Reset or shorten your message/notes.')
        return dict(message=message, mode=mode, link=link, notes=notes, temperature=temp,
                    limit=limit, history=history, episode=ep)

    def respond(self, request, emit):
        with GENERATION_LOCK:
            ep = request['episode']
            obs = ChatObserver(self.pair, ep, emit)
            seed = secrets.randbits(32)
            with history_prefix(request['history'], request['mode']), obs.installed():
                result = S.generate(self.pair, [ep], max_turn=request['limit'], temperature=request['temperature'],
                                    mode='zero' if request['link'] == 'cut' else None,
                                    gen=torch.Generator().manual_seed(seed), keep_tokens=True)[0]
            turns = obs.finish(result)
            # Only a fully generated exchange is committed. A broken connection
            # unwinds the observer and leaves the previous conversation intact.
            self.history = request['history'] + [dict(message=request['message'],
                                                     text={t['speaker']: t['text'] for t in turns})]
            self.mode, self.notes = request['mode'], request['notes']
            emit('done', dict(turns=turns, mode=self.mode, link=request['link'], seed=seed))


class Server(HTTPServer):
    def __init__(self, chat, port=8765):
        self.chat = chat
        super().__init__(('127.0.0.1', port), Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Do not log human messages or private notes.

    def allowed(self):
        host = self.headers.get('Host')
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        origin = self.headers.get('Origin')
        if host not in hosts or (origin and origin not in {'http://' + h for h in hosts}):
            self.send_error(403, 'Local same-origin requests only')
            return False
        return True

    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        body = data.encode() if isinstance(data, str) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return
        if self.path == '/':
            self.reply(200, PAGE, 'text/html; charset=utf-8')
        elif self.path == '/health':
            self.reply(200, dict(ready=True))
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.allowed():
            return
        if self.path not in ('/chat', '/reset'):
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536 or self.headers.get_content_type() != 'application/json':
                raise ValueError('send a JSON body of at most 65536 bytes')
            data = json.loads(self.rfile.read(size))
            if self.path == '/reset':
                self.server.chat.reset()
                self.reply(200, dict(reset=True))
                return
            request = self.server.chat.prepare(data)
        except (ValueError, UnicodeError) as e:
            self.reply(400, dict(error=str(e)))
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True

        def emit(event, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False)
            self.wfile.write(f'event: {event}\ndata: {body}\n\n'.encode())
            self.wfile.flush()

        try:
            emit('start', dict(mode=request['mode'], link=request['link']))
            self.server.chat.respond(request, emit)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            try:
                emit('error', dict(error='Generation failed; previous history retained. See the server terminal.'))
            except (BrokenPipeError, ConnectionResetError):
                pass


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Talk to Quill &amp; Ember</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:32px auto;max-width:960px;padding:0 20px;background:#10151e;color:#e7edf6;font:16px/1.6 system-ui}
h1{font-size:30px;margin-bottom:0}p{margin-top:6px}.muted,small{color:#acbbcd;font-size:13px}
fieldset{border:1px solid #354155;border-radius:14px;margin:20px 0;padding:16px}.controls,.notes{display:flex;gap:18px;flex-wrap:wrap}
label{display:block}select,input,textarea,button{font:inherit;border:1px solid #526079;border-radius:8px;background:#1b2636;color:inherit;padding:8px}
textarea{width:100%;resize:vertical}.notes label{flex:1;min-width:230px}.notes{margin-top:16px}button{cursor:pointer}button:disabled{opacity:.5;cursor:wait}
article{padding:18px 22px;margin:16px 0;border-radius:16px;background:#1c293b;max-width:88%}article.quill{border-left:3px solid #63d6ed}article.ember{border-left:3px solid #e69bd8;margin-left:auto}
article.human{background:#24352f;margin-left:auto}h2{font-size:14px;margin:0 0 10px}.message{white-space:pre-wrap;overflow-wrap:anywhere}.token{border-radius:2px}
.ramp{display:inline-block;width:100px;height:12px;background:linear-gradient(90deg,transparent,rgba(245,165,50,.85));vertical-align:middle}
#status{min-height:26px}form{padding-bottom:30px}#send{margin-top:10px}input[type=number]{width:85px}
@media(max-width:600px){article{max-width:100%}.notes label{min-width:100%}}
</style>
<h1>Quill &amp; Ember</h1><p class="muted">Two models, one hidden-state link. A local conversation on this Mac.</p>
<fieldset id="settings"><legend>Conversation</legend><div class="controls">
<label>Mode <select id="mode"><option value="together">Together · simultaneous</option><option value="turns">Turns · Quill, then Ember</option></select></label>
<label>Link <select id="link"><option value="on">On</option><option value="cut">Cut · zero write</option></select></label>
<label>Temperature <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7"></label>
<button id="reset" type="button">Reset conversation</button></div>
<p id="modehelp" class="muted"></p><div class="notes">
<label>Private note for Quill<textarea id="quillnote" rows="2" maxlength="2000" placeholder="You are in Lima today."></textarea></label>
<label>Private note for Ember<textarea id="embernote" rows="2" maxlength="2000" placeholder="Only Ember reads this note as text."></textarea></label></div>
<small>Notes go only in their owner's system prompt. An open link may carry them. Changing a note keeps previous replies; Reset clears replies and notes.</small>
</fieldset>
<p class="muted"><span class="ramp"></span> Clear = 0; full amber = 0.25 or more. Each token is shaded by link write RMS / residual RMS, averaged across exchanges at the position predicting it, including rounding. This is write size, not causal attribution, confidence, or proof that a note crossed. Hover for the ratio.</p>
<p class="muted">One conversation is shared by tabs. Text history is rebuilt for each message; hidden-state memory is not carried between messages. Changing mode resets the conversation.</p>
<main id="messages" aria-label="Conversation"></main><div id="status" role="status" aria-live="polite">Ready.</div>
<form id="chat"><label for="message">Your message to both twins</label><textarea id="message" rows="3" maxlength="8000" required placeholder="Hello, Quill and Ember. What would you like to talk about?"></textarea><button id="send">Send to both</button></form>
<script>
const $=id=>document.getElementById(id);let busy=false;
function help(){$('modehelp').textContent=$('mode').value==='together'?'Both answer at once. Each sees your messages and its own replies; only the link connects them.':'Quill answers first, then Ember. Each reads the other’s replies as text while the link stays open. With the link cut, text is still shared.'}help();
function setBusy(v){busy=v;$('settings').disabled=v;$('send').disabled=v;$('message').disabled=v}
function bubble(who,text){const a=document.createElement('article');a.className=who;const h=document.createElement('h2');h.textContent=who==='human'?'You':who[0].toUpperCase()+who.slice(1);a.append(h);const b=document.createElement('div');b.className='message';b.textContent=text;a.append(b);$('messages').append(a);return b}
async function reset(clearNotes=true){if(busy)return;setBusy(true);try{const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});if(!r.ok)throw Error('Reset failed');$('messages').replaceChildren();if(clearNotes){$('quillnote').value='';$('embernote').value=''}$('status').textContent='Conversation reset.'}catch(e){$('status').textContent=e.message}finally{setBusy(false)}}
$('reset').onclick=()=>reset();$('mode').onchange=()=>{help();reset(false)};
$('chat').onsubmit=async e=>{e.preventDefault();if(busy)return;const message=$('message').value;if(!message.trim())return;setBusy(true);bubble('human',message);const boxes={quill:bubble('quill','…'),ember:bubble('ember','…')};$('status').textContent='Reading…';let done=false;
function event(frame){let name='',data='';for(const l of frame.split('\n')){if(l.startsWith('event: '))name=l.slice(7);if(l.startsWith('data: '))data+=l.slice(6)}if(!data)return;const d=JSON.parse(data);if(name==='snapshot'){const box=boxes[d.speaker];box.replaceChildren();for(const t of d.tokens){const span=document.createElement('span');span.className='token';span.textContent=t.text;span.title='write / residual '+t.effect.toFixed(5);span.style.backgroundColor='rgba(245,165,50,'+Math.min(.85,t.effect/.25*.85)+')';box.append(span)}if(d.truncated){const small=document.createElement('small');small.textContent=' [token limit reached]';box.append(small)}$('status').textContent='Writing…'}if(name==='error')throw Error(d.error);if(name==='done'){done=true;$('message').value='';$('status').textContent='Ready.'}}
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,mode:$('mode').value,link:$('link').value,temperature:Number($('temperature').value),notes:{quill:$('quillnote').value,ember:$('embernote').value}})});if(!r.ok)throw Error((await r.json()).error||'Request failed');const reader=r.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const {value,done:end}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!end});let at;while((at=pending.indexOf('\n\n'))>=0){event(pending.slice(0,at));pending=pending.slice(at+2)}if(end)break}if(!done)throw Error('Connection ended before completion. Your message has been kept for retry.')}catch(err){$('status').textContent=err.message}finally{setBusy(false);$('message').focus()}};
</script></html>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--replies', required=True)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--context-tokens', type=int, default=4096)
    ap.add_argument('--backend', choices=('mlx', 'torch'), default='mlx')
    args = ap.parse_args()
    if args.backend == 'torch' and (os.environ.get('NL_TINY') != '1' or S.DEVICE != 'cpu'):
        ap.error('torch loading requires NL_TINY=1 NL_DEVICE=cpu')
    if not 1 <= args.port <= 65535 or not 1 <= args.max_tokens <= 512 or not 512 <= args.context_tokens <= 8192:
        ap.error('port: 1..65535; max-tokens: 1..512; context-tokens: 512..8192')
    json.loads(Path(args.replies).read_text())
    torch.set_num_threads(2)
    if args.backend == 'mlx':
        import mlx.core as mx
        from mlx_eval import load_pair
        mx.set_memory_limit(10 * 2**30)
        mx.set_cache_limit(128 * 2**20)
    else:
        from evaluate import load_pair
    pair, _ = load_pair(args.run)
    with Server(Chat(pair, args.max_tokens, args.context_tokens), args.port) as server:
        print(f'PID {os.getpid()} · {Path(args.run).resolve()} · http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
