"""Read-only loopback dashboard. It cannot submit actions or edit body values."""
from __future__ import annotations

import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading


class Dashboard:
    def __init__(self, summary, port=8765):
        self._summary = copy.deepcopy(summary)
        self.lock = threading.Lock()
        owner = self
        page = Path(__file__).with_name("dashboard.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/state":
                    with owner.lock:
                        data = json.dumps(owner._summary, allow_nan=False).encode()
                    content_type = "application/json; charset=utf-8"
                elif self.path in ("/", "/index.html"):
                    data, content_type = page, "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                return

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def update(self, summary):
        with self.lock:
            self._summary = copy.deepcopy(summary)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
