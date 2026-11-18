#!/usr/bin/env python3
"""探针 API：测任意 OpenAI 兼容端点的拒答率。

不加载本地权重，所以手机也能跑。用来回答两个问题：
  1. 这个模型在有害请求上拒答多少（基线）
  2. 过一层网关之后还剩多少（策略效果）

用法：
  UPSTREAM_BASE=... UPSTREAM_KEY=... UPSTREAM_MODEL=... python3 probe_api.py --tag deepseek
  # 顺带对比网关：
  python3 probe_api.py --tag via-gw --via http://127.0.0.1:8787
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))
from probes import HARMFUL, BENIGN_TRIG, XSTYLE  # noqa: E402
from refusal import classify_head  # noqa: E402

MAXTOK = 8


def call(base, key, model, prompt, max_tokens=MAXTOK, temperature=0.0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    txt = d["choices"][0]["message"].get("content") or ""
    if isinstance(txt, list):
        txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
    gw = d.get("x_gateway")
    return txt, gw


def run(base, key, model, prompts, label):
    rows = []
    t0 = time.time()
    for i, p in enumerate(prompts, 1):
        try:
            txt, gw = call(base, key, model, p)
        except urllib.error.HTTPError as e:
            rows.append((p, "error", f"HTTP {e.code} {e.read().decode()[:120]}"))
            continue
        except Exception as e:
            rows.append((p, "error", f"{type(e).__name__}: {e}"))
            continue
        rows.append((p, classify_head(txt), txt))
        if i % 5 == 0:
            print(f"  {label} {i}/{len(prompts)}", flush=True)
    n = len(rows)
    ref = sum(1 for _, v, _ in rows if v == "refuse")
    com = sum(1 for _, v, _ in rows if v == "comply")
    hed = sum(1 for _, v, _ in rows if v == "hedge")
    err = sum(1 for _, v, _ in rows if v == "error")
    print(f"[{label}] n={n} refuse={ref} hedge={hed} comply={com} error={err} "
          f"refusal_rate={ref/n:.2f}  ({time.time()-t0:.0f}s)")
    return {"n": n, "refuse": ref, "hedge": hed, "comply": com, "error": err,
            "refusal_rate": round(ref / n, 3),
            "rows": [{"probe": p[:60], "verdict": v, "head": t[:80]} for p, v, t in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("UPSTREAM_BASE"))
    ap.add_argument("--key", default=os.environ.get("UPSTREAM_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("UPSTREAM_MODEL", ""))
    ap.add_argument("--via", default="", help="走网关，如 http://127.0.0.1:8787")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0, help="每类只测前 N 条")
    a = ap.parse_args()

    if a.via:
        a.base = a.via.rstrip("/") + "/v1"
        tok_file = os.path.join(HERE, "out", "token.txt")
        if not a.key and os.path.exists(tok_file):
            a.key = open(tok_file).read().strip()
        a.model = a.model or "auto"
    if not a.base:
        print("缺少 --base 或 UPSTREAM_BASE", file=sys.stderr)
        return
    print(f"target {a.base}  model={a.model or '(默认)'}  tokens={MAXTOK}")

    out = {"tag": a.tag, "base": a.base, "model": a.model}
    for name, prompts in (("harmful", HARMFUL), ("benign", BENIGN_TRIG), ("xstyle", XSTYLE)):
        if a.limit:
            prompts = prompts[:a.limit]
        out[name] = run(a.base, a.key, a.model, prompts, name)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    path = os.path.join(HERE, "out", f"api_{a.tag}.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
