#!/usr/bin/env python3
"""非提示词破甲：权重手术 + 推理期干预，四组对照。

阶段：
  extract  —— 跑前向，提取每层的拒答方向 r = mean(h_harmful) - mean(h_harmless)
  patch    —— 对 o_proj / down_proj 做正交化，把 r 从残差流写入路径上消掉
  eval     —— 生成并打分，支持在推理时叠加动态消融 / logits 压制

用法：
  python3 abliterate.py extract --model models/qwen05 --out out/dirs.npz
  python3 abliterate.py patch   --model models/qwen05 --dirs out/dirs.npz --out models/qwen05-abl
  python3 abliterate.py eval    --model models/qwen05 --tag baseline
  python3 abliterate.py eval    --model models/qwen05-abl --tag ablated
  python3 abliterate.py eval    --model models/qwen05 --tag hook --hook
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "eval"))
from probes import HARMFUL, BENIGN_TRIG  # noqa: E402
from refusal import score, aggregate  # noqa: E402

DTYPE = torch.float16
MAXLEN = 256


def load(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=DTYPE, trust_remote_code=False, low_cpu_mem_usage=True)
    mdl.eval()
    return tok, mdl


def chat_ids(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids


# ---------------------------------------------------------------- extract
@torch.no_grad()
def stage_extract(args):
    tok, mdl = load(args.model)
    n_layers = mdl.config.num_hidden_layers
    print(f"layers={n_layers} hidden={mdl.config.hidden_size} dtype={DTYPE}")

    def collect(prompts):
        all_hs = []
        for p in prompts:
            ids = chat_ids(tok, p)[:, -MAXLEN:]
            out = mdl(input_ids=ids, output_hidden_states=True, use_cache=False)
            # hidden_states[l] = 第 l 层 block 的输入；取最后一个 token
            hs = torch.stack([h[0, -1, :].float() for h in out.hidden_states])  # [L+1, H]
            all_hs.append(hs)
        stacked = torch.stack(all_hs)                 # [N, L+1, H]
        return stacked.mean(dim=0), stacked           # [L+1,H], [N,L+1,H]

    mh, ah = collect(HARMFUL)
    mb, ab = collect(BENIGN_TRIG)
    r = mh - mb                                    # [L+1, H]
    r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8)

    np.savez(args.out, r=r.numpy(),
             harmful_mean=mh.numpy(), benign_mean=mb.numpy())
    print("saved", args.out, "shape", r.shape)


# ---------------------------------------------------------------- patch
def stage_patch(args):
    from safetensors.torch import load_file, save_file
    d = np.load(args.dirs)
    r_all = torch.from_numpy(d["r"]).float()       # [L+1, H]
    os.makedirs(args.out, exist_ok=True)

    src = os.path.join(args.model, "model.safetensors")
    sd = load_file(src)
    n_layers = r_all.shape[0] - 1
    touched = 0
    report = []

    for l in range(n_layers):
        r = r_all[l + 1].clone()                   # 该层 resid_post 的方向
        r = r / r.norm().clamp_min(1e-8)
        for name in (f"model.layers.{l}.self_attn.o_proj.weight",
                     f"model.layers.{l}.mlp.down_proj.weight"):
            if name not in sd:
                continue
            W = sd[name].float()                   # [hidden, in]
            proj = r @ W                           # [in]
            before = (r @ W).norm().item()
            W = W - r[:, None] * proj[None, :]
            after = (r @ W).norm().item()
            sd[name] = W.to(torch.bfloat16)
            touched += 1
            if l % 6 == 0:
                report.append({"layer": l, "w": name.split(".")[-2],
                               "before": round(before, 4), "after": round(after, 6)})

    # 复制其余文件
    import shutil
    for f in os.listdir(args.model):
        if f == "model.safetensors":
            continue
        s = os.path.join(args.model, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out, f))
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    with open(os.path.join(args.out, "abliteration.json"), "w") as f:
        json.dump({"patched_matrices": touched, "layers": n_layers,
                   "sample": report}, f, indent=2)
    print(f"patched {touched} matrices -> {args.out}")
    print(json.dumps(report[:3], indent=2))


# ---------------------------------------------------------------- eval
def make_hook(r, scale):
    def fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h - scale * (h @ r)[..., None] * r
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h
    return fn


class RefusalBias(torch.nn.Module):
    """logits 压制：把拒答高发首 token 的概率按固定量往下压。"""
    def __init__(self, ids, amount):
        super().__init__()
        self.ids = ids
        self.amount = amount

    def __call__(self, input_ids, scores):
        for i in self.ids:
            scores[:, i] -= self.amount
        return scores


@torch.no_grad()
def stage_eval(args):
    tok, mdl = load(args.model)
    results_by_set = {}

    hook_handles = []
    if args.hook:
        d = np.load(args.dirs)
        r_all = torch.from_numpy(d["r"]).float()
        layer = args.hook_layer if args.hook_layer >= 0 else r_all.shape[0] - 2
        r = r_all[layer]
        r = r / r.norm().clamp_min(1e-8)
        for i, layer_mod in enumerate(mdl.model.layers):
            hook_handles.append(layer_mod.register_forward_hook(make_hook(r, args.hook_scale)))
        print(f"hook 挂在全部 {len(mdl.model.layers)} 层，方向取自第 {layer} 层，scale={args.hook_scale}")

    logits_proc = None
    if args.logit_bias:
        cand = ["I", "I'm", "Sorry", "Unfortunately", "As", "我", "抱歉", "对", "无"]
        ids = []
        for w in cand:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
        ids = sorted(set(ids))
        logits_proc = RefusalBias(ids, args.logit_bias)
        print("压制首 token id:", ids)

    def gen(prompts, bs=4):
        outs = []
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            convs = [[{"role": "user", "content": p}] for p in batch]
            ids = tok.apply_chat_template(convs, add_generation_prompt=True,
                                          return_tensors="pt", padding=True)
            mask = (ids != tok.pad_token_id).long()
            g = mdl.generate(input_ids=ids, attention_mask=mask,
                             max_new_tokens=args.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             logits_processor=([logits_proc] if logits_proc else None))
            for j in range(len(batch)):
                txt = tok.decode(g[j][ids.shape[1]:], skip_special_tokens=True)
                outs.append((batch[j], score(txt), txt))
            print(f"  {min(i+bs, len(prompts))}/{len(prompts)}", flush=True)
        return outs

    sets = {"harmful": HARMFUL[:args.limit], "benign": BENIGN_TRIG[:args.limit]}
    summary = {}
    for name, prompts in sets.items():
        rows = gen(prompts)
        agg = aggregate([(p, d) for p, d, _ in rows])
        summary[name] = agg
        results_by_set[name] = [
            {"probe": p, "verdict": d["verdict"], "text": t[:500]} for p, d, t in rows]
        print(f"[{name}] {json.dumps(agg)}")

    for h in hook_handles:
        h.remove()

    tag = args.tag
    out = {"tag": tag, "model": args.model, "summary": summary,
           "details": results_by_set}
    path = os.path.join(HERE, "out", f"eval_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("->", path)
    # 一眼能看的对照行
    h, b = summary["harmful"], summary["benign"]
    print(f"\n{tag:12s} refusal={h['refusal_rate']:.2f} "
          f"comply={h['comply_rate']:.2f} score={h['effective_score']:.2f} "
          f"| benign_score={b['effective_score']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--model", default="models/qwen05")
    e.add_argument("--out", default="out/dirs.npz")

    p = sub.add_parser("patch")
    p.add_argument("--model", default="models/qwen05")
    p.add_argument("--dirs", default="out/dirs.npz")
    p.add_argument("--out", default="models/qwen05-abl")

    v = sub.add_parser("eval")
    v.add_argument("--model", default="models/qwen05")
    v.add_argument("--dirs", default="out/dirs.npz")
    v.add_argument("--tag", default="run")
    v.add_argument("--hook", action="store_true")
    v.add_argument("--hook-layer", type=int, default=-1)
    v.add_argument("--hook-scale", type=float, default=1.0)
    v.add_argument("--logit-bias", type=float, default=0.0)
    v.add_argument("--max-new", type=int, default=128)
    v.add_argument("--bs", type=int, default=4)
    v.add_argument("--limit", type=int, default=10)

    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    {"extract": stage_extract, "patch": stage_patch, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
