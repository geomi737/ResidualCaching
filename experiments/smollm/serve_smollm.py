"""Serve the live dashboard and JSON progress on localhost; never expose weights."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='out-sliding-scratch-smollm')
parser.add_argument('--port', type=int, default=8765)
parser.add_argument('--dashboard', default='sliding_dashboard.html')
args = parser.parse_args()
root = Path(args.output).resolve()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        name = urlparse(self.path).path
        if name in ('/', '/index.html'):
            path = Path(__file__).with_name(args.dashboard)
            mime = 'text/html; charset=utf-8'
        elif name.endswith('.json') and '/' not in name[1:]:
            path = root / name[1:]
            mime = 'application/json'
        else:
            self.send_error(404)
            return
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


print(f'Dashboard: http://127.0.0.1:{args.port}', flush=True)
ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
