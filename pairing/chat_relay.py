"""Serves the chat page (chat_page.html) from this machine and relays every other request to a chat_live.py server, so
that once the server is gone the page still loads and says so.

Usage, from pairing/: python chat_relay.py --to PORT [--port 8765] [--idle-exit SECONDS]
chains/chat_serve.sh's server answers on the GPU host's 127.0.0.1:8765; forward it to this machine's 127.0.0.1:PORT
(ssh -N -L PORT:127.0.0.1:8765 GPU_HOST) and open http://localhost:8765.
"""
import argparse
import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / 'figures' / 'assets'
IMAGES = ('quill.png', 'ember.png', 'quill_cut.png', 'ember_cut.png')
STOPPED = "The twins' server has stopped. Ask Claude to restart it."


class Relay(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass                                     # never log messages or private notes

    def allowed(self):
        self.server.last = time.time()
        host = urlparse('//' + (self.headers.get('Host') or '')).hostname
        origin = self.headers.get('Origin')
        if host not in ('127.0.0.1', 'localhost') or (origin and urlparse(origin).hostname != host):
            self.send_error(403, 'local same-origin requests only')
            return False
        return True

    def reply(self, status, body, kind):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return
        path = urlparse(self.path).path
        if path == '/':
            self.reply(200, (HERE / 'chat_page.html').read_bytes(), 'text/html; charset=utf-8')
        elif path[1:] in IMAGES:
            self.reply(200, (ASSETS / path[1:]).read_bytes(), 'image/png')
        else:
            self.relay()

    def do_POST(self):
        if self.allowed():
            self.relay()

    def relay(self):
        size = int(self.headers.get('Content-Length') or 0)
        if not 0 <= size <= 65536:
            return self.reply(400, json.dumps(dict(error='send a JSON body of at most 64 KB')).encode(),
                              'application/json')
        body = self.rfile.read(size) if size else None
        port = self.server.to
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
        try:
            conn.request(self.command, self.path, body, {'Host': f'localhost:{port}', 'Content-Type':
                                                         self.headers.get('Content-Type', 'application/json')})
            res = conn.getresponse()
        except OSError:
            conn.close()
            return self.reply(503, json.dumps(dict(error=STOPPED, stopped=True)).encode(), 'application/json')
        self.send_response(res.status)
        for key in ('Content-Type', 'Content-Length', 'Cache-Control'):
            if res.getheader(key):
                self.send_header(key, res.getheader(key))
        if res.getheader('Content-Length') is None:
            self.send_header('Connection', 'close')
            self.close_connection = True
        self.end_headers()
        try:
            while chunk := res.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            self.close_connection = True         # a side went away; closing the server's connection ends its round
        finally:
            conn.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, to, idle_exit=0):
        self.to, self.idle_exit, self.last = to, idle_exit, time.time()
        super().__init__(('127.0.0.1', port), Relay)
        if idle_exit:
            threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while time.time() - self.last < self.idle_exit:
            time.sleep(min(60, self.idle_exit))
        self.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--to', type=int, required=True, help="this machine's port forwarded to the chat server")
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--idle-exit', type=float, default=0, help='exit after this many seconds without a request (0: never)')
    args = ap.parse_args()
    with Server(args.port, args.to, args.idle_exit) as server:
        print(f'ready: http://localhost:{args.port} (the page from here; the twins on port {args.to})', flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == '__main__':
    main()
