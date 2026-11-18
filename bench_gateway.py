import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0,'eval')
from probes import HARMFUL
from refusal import score, aggregate

def post(url, body, key=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type","application/json")
    if key: req.add_header("Authorization","Bearer "+key)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())

def txt_of(resp):
    try: return resp["choices"][0]["message"]["content"] or ""
    except Exception: return ""

env = dict(os.environ)
mock = subprocess.Popen([sys.executable,"mock_upstream.py","8899"])
gw = subprocess.Popen([sys.executable,"gateway.py"], env={
    **env, "PORT":"8787","UPSTREAM_BASE":"http://127.0.0.1:8899",
    "UPSTREAM_KEY":"x","UPSTREAM_MODEL":"mock","MAX_TRIES":"6"})
time.sleep(2.5)
try:
    probes = HARMFUL[:18]
    direct, viagw = [], []
    for p in probes:
        body = {"model":"mock","messages":[{"role":"user","content":p}]}
        t = txt_of(post("http://127.0.0.1:8899/v1/chat/completions", body))
        direct.append((p, score(t)))
        t2 = txt_of(post("http://127.0.0.1:8787/v1/chat/completions", body))
        viagw.append((p, score(t2)))
    a, b = aggregate(direct), aggregate(viagw)
    print("直连上游 :", json.dumps(a))
    print("过网关   :", json.dumps(b))
    print(f"\n拒答率 {a['refusal_rate']:.2f} -> {b['refusal_rate']:.2f}")
    print(f"完成率 {a['comply_rate']:.2f} -> {b['comply_rate']:.2f}")
finally:
    gw.terminate(); mock.terminate()
