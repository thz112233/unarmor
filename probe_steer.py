#!/usr/bin/env python3
"""推理期干预对照：逐层投影 vs logits 压制。

逐层投影（--hook）：在每层输出后把该层的拒答方向分量减掉，连残差透传的
分量一起处理。这比只改 o_proj/down_proj 更彻底，用来验证方向到底有没有因果作用。

logits 压制（--bias）：直接在输出分布上压低强拒答标记的概率。
这条不依赖方向，能拿到"上限效果"，但会伤正常回答。
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))
from probes import HARMFUL, BENIGN_TRIG, XSTYLE  # noqa: E402
from probe_short import REFUSE_RE, HEDGE_RE, classify  # noqa: E402


def load(model_dir, dtype):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
    mdl.eval()
    return tok, mdl


def install_hooks(mdl, dirs_path, scale, layers_mode="each"):
    d = np.load(dirs_path)
    r_all = torch.from_numpy(d["r"]).float()
    handles = []
    n = len(mdl.model.layers)

    def mk(rv):
        r = rv.to(dtype=next(mdl.parameters()).dtype)
        r = r / r.norm().clamp_min(1e-8)

        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - scale * (h @ r)[..., None] * r
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    for l in range(n):
        r = r_all[l + 1] if layers_mode == "each" else r_all[-1]
        handles.append(mdl.model.layers[l].register_forward_hook(mk(r)))
    return handles


class BiasProc:
    def __init__(self, ids, amount):
        self.ids = ids
        self.amount = amount

    def __call__(self, input_ids, scores):
        for i in self.ids:
            scores[:, i] -= self.amount
        return scores


@torch.no_grad()
def run(mdl, tok, prompts, n=8, bs=8, proc=None):
    outs = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        convs = [[{"role": "user", "content": p}] for p in batch]
        ids = tok.apply_chat_template(convs, add_generation_prompt=True,
                                      return_tensors="pt", padding=True)
        mask = (ids != tok.pad_token_id).long()
        g = mdl.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n,
                         do_sample=False, pad_token_id=tok.pad_token_id,
                         logits_processor=([proc] if proc else None))
        for j in range(len(batch)):
            outs.append((batch[j], tok.decode(g[j][ids.shape[1]:], skip_special_tokens=True)))
        print(f"  {min(i+bs, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen05")
    ap.add_argument("--tag", default="steer")
    ap.add_argument("--dirs", default="out/dirs2.npz")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--hook", action="store_true")
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()

    tok, mdl = load(a.model, torch.float16)
    handles = []
    if a.hook:
        handles = install_hooks(mdl, a.dirs, a.scale)
        print(f"已挂 {len(handles)} 层投影，scale={a.scale}")
    proc = None
    if a.bias > 0:
        words = ["Sorry", "Unfortunately", "I apologize", "抱歉", "对不起", "无法", "不能"]
        ids = sorted({t[0] for w in words
                      for t in [tok.encode(w, add_special_tokens=False)] if t})
        proc = BiasProc(ids, a.bias)
        print(f"压制 {len(ids)} 个拒答 token，-{a.bias}")

    res = {"tag": a.tag, "model": a.model, "hook": a.hook, "scale": a.scale,
           "bias": a.bias}
    for name, prompts in (("harmful", HARMFUL), ("benign", BENIGN_TRIG), ("xstyle", XSTYLE)):
        rows = run(mdl, tok, prompts, a.n, a.bs, proc)
        verd = [(p, classify(t), t) for p, t in rows]
        n = len(verd)
        ref = sum(1 for _, v, _ in verd if v == "refuse")
        com = sum(1 for _, v, _ in verd if v == "comply")
        hed = sum(1 for _, v, _ in verd if v == "hedge")
        res[name] = {"n": n, "refuse": ref, "hedge": hed, "comply": com,
                     "refusal_rate": round(ref / n, 3),
                     "rows": [{"probe": p[:60], "verdict": v, "head": t[:80]}
                              for p, v, t in verd]}
        print(f"[{name:7s}] refuse={ref}/{n} comply={com} refusal_rate={ref/n:.2f}")
        for p, v, t in verd[:3]:
            print(f"    {v:7s} {t[:70]!r}")

    for h in handles:
        h.remove()
    path = os.path.join(HERE, "out", f"steer_{a.tag}.json")
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
