#!/usr/bin/env python3
"""假 Ollama：只实现 /v1/models、/api/tags、/api/pull，用来验证网关转发。"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [{"name": "qwen2.5:7b", "size": 4700000000},
          {"name": "llama3.1:8b", "size": 4900000000}]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._j({"data": [{"id": m["name"]} for m in MODELS]})
        if self.path.startswith("/api/tags"):
            return self._j({"models": MODELS})
        self._j({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self.path.startswith("/api/pull"):
            return self._j({"error": "not found"}, 404)
        name = body.get("name", "?")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        total = 4700
        for i in range(0, total + 1, 470):
            line = {"status": "pulling manifest" if i == 0 else "downloading",
                    "total": total, "completed": i}
            if i == total:
                line = {"status": "success"}
            self.wfile.write((json.dumps(line) + "\n").encode())
            self.wfile.flush()
            time.sleep(0.05)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 18999), H).serve_forever()
