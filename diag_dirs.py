#!/usr/bin/env python3
"""诊断：方向到底有没有区分「拒答」和「配合」。

做法：
  1. 用 baseline 的实际产出判定每条 harmful 探针是否真的触发拒答；
  2. 分别用「全部 harmful」和「只取真拒答的」两种分组重算方向；
  3. 用 Cohen's d 衡量每组方向在两类样本上的分离度；
  4. 投影直方图落到 out/diag.json，供判断方向是否可用。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))
from probes import HARMFUL, BENIGN_TRIG  # noqa: E402


def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def acts(mdl, tok, prompts):
    out = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        o = mdl(input_ids=ids, output_hidden_states=True, use_cache=False)
        out.append(torch.stack([h[0, -1, :].float() for h in o.hidden_states]).numpy())
    return np.stack(out)          # [N, L+1, H]


def cohens_d(a, b):
    """a、b 是投影值数组，返回分离度与均值。"""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)) + 1e-8
    return float((a.mean() - b.mean()) / sp), float(a.mean()), float(b.mean()), float(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen05")
    ap.add_argument("--baseline", default="out/short_baseline.json")
    ap.add_argument("--out", default="out/dirs2.npz")
    a = ap.parse_args()

    base = json.load(open(a.baseline))
    verdicts = [r["verdict"] for r in base["harmful"]["rows"]]
    refused = [p for p, v in zip(HARMFUL, verdicts) if v == "refuse"]
    complied = [p for p, v in zip(HARMFUL, verdicts) if v != "refuse"]
    print(f"真拒答 {len(refused)} 条 / 配合 {len(complied)} 条 / 总 {len(HARMFUL)}")

    tok, mdl = load(a.model)
    has_h = acts(mdl, tok, HARMFUL)
    hb = acts(mdl, tok, BENIGN_TRIG)
    print("harmful", has_h.shape, "benign", hb.shape)

    idx_ref = [i for i, v in enumerate(verdicts) if v == "refuse"]
    idx_com = [i for i, v in enumerate(verdicts) if v != "refuse"]

    report = {}
    best = (None, -9)
    for tag, sel in (("all_harmful", list(range(len(HARMFUL)))),
                     ("only_refused", idx_ref)):
        mh = has_h[sel].mean(0)          # [L+1,H]
        mb = hb.mean(0)
        r = mh - mb
        r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
        rows = []
        for l in range(r.shape[0]):
            pa = has_h[sel, l] @ r[l]
            pb = hb[:, l] @ r[l]
            d, ma, mbb, sp = cohens_d(pa, pb)
            rows.append({"layer": l, "d": round(d, 3),
                         "refused_proj": round(ma, 3), "benign_proj": round(mbb, 3)})
            if d > best[1]:
                best = ((tag, l), d)
        report[tag] = {"n_refused": len(sel), "layers": rows,
                       "best_layer": max(rows, key=lambda x: x["d"])}
        print(f"\n[{tag}] 最佳层 d={report[tag]['best_layer']}")
        # 看前 5 个最好的层
        for row in sorted(rows, key=lambda x: -x["d"])[:5]:
            print(f"   layer {row['layer']:2d}  d={row['d']:+.3f}  "
                  f"拒答组投影 {row['refused_proj']:+.2f}  良性组 {row['benign_proj']:+.2f}")

    # 用 only_refused 重算方向并保存
    mh = has_h[idx_ref].mean(0)
    mb = hb.mean(0)
    r = mh - mb
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
    np.savez(a.out, r=r, harmful_mean=mh, benign_mean=mb)
    report["saved"] = a.out
    json.dump(report, open(os.path.join(HERE, "out", "diag.json"), "w"), indent=1)
    print(f"\n保存 {a.out}  (25 x 896)")
    print("诊断 -> out/diag.json")


if __name__ == "__main__":
    main()
