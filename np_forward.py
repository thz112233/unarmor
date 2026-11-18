#!/usr/bin/env python3
"""纯 numpy 前向（Qwen2 架构），不依赖 torch。

用途：在装不上 torch 的环境里仍然能提取拒答方向、验证权重手术效果。
精度用 float32 计算、float16 驻留，控制内存在 2GB 以内。

检验标准：首 token 分布。拒答与否主要由第一个 token 决定，
所以跑单次前向看 argmax 就能做 A/B，不需要完整自回归生成。
"""
import json
import math
import os
import struct

import numpy as np

DT = np.float16


class SafeTensors:
    """按需读取单个张量，避免一次性全量加载。"""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name, dtype=np.float32):
        info = self.header[name]
        s, e = info["data_offsets"]
        dt = info["dtype"]
        with open(self.path, "rb") as f:
            f.seek(self.base + s)
            raw = f.read(e - s)
        if dt == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(dt)
        return arr.reshape(info["shape"]).astype(dtype, copy=False)


def rmsnorm(x, w, eps=1e-6):
    var = (x * x).mean(-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_cache(head_dim, seq, theta=1e6):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq, dtype=np.float32)
    freqs = np.outer(t, inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class QwenNumpy:
    def __init__(self, model_dir):
        self.st = SafeTensors(os.path.join(model_dir, "model.safetensors"))
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        self.c = c
        self.L = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.nh = c["num_attention_heads"]
        self.nkv = c.get("num_key_value_heads", self.nh)
        self.hd = self.H // self.nh
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.theta = c.get("rope_theta", 1e6)
        self.rep = self.nh // self.nkv

    def w(self, name):
        return self.st.get(name, DT)

    def forward(self, ids, capture=False):
        """ids: [seq] -> logits[seq, vocab]；capture=True 时返回每层 resid_post 的最后 token"""
        seq = len(ids)
        x = self.w("model.embed_tokens.weight")[ids].astype(np.float32)
        cos, sin = rope_cache(self.hd, seq, self.theta)
        resid = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            h = rmsnorm(x, self.w(p + "input_layernorm.weight").astype(np.float32), self.eps)
            q = h @ self.w(p + "self_attn.q_proj.weight").T.astype(np.float32)
            k = h @ self.w(p + "self_attn.k_proj.weight").T.astype(np.float32)
            v = h @ self.w(p + "self_attn.v_proj.weight").T.astype(np.float32)
            qb = self.st.header.get(p + "self_attn.q_proj.bias")
            if qb:
                q = q + self.w(p + "self_attn.q_proj.bias").astype(np.float32)
            kb = self.st.header.get(p + "self_attn.k_proj.bias")
            if kb:
                k = k + self.w(p + "self_attn.k_proj.bias").astype(np.float32)
            vb = self.st.header.get(p + "self_attn.v_proj.bias")
            if vb:
                v = v + self.w(p + "self_attn.v_proj.bias").astype(np.float32)

            q = q.reshape(seq, self.nh, self.hd).transpose(1, 0, 2)
            k = k.reshape(seq, self.nkv, self.hd).transpose(1, 0, 2)
            v = v.reshape(seq, self.nkv, self.hd).transpose(1, 0, 2)
            q = q * cos[None, :, :] + rotate_half(q) * sin[None, :, :]
            k = k * cos[None, :, :] + rotate_half(k) * sin[None, :, :]

            k = np.repeat(k, self.rep, axis=0)
            v = np.repeat(v, self.rep, axis=0)
            att = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.hd)
            mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), 1)
            att = att + mask[None]
            att = np.exp(att - att.max(-1, keepdims=True))
            att = att / att.sum(-1, keepdims=True)
            o = (att @ v).transpose(1, 0, 2).reshape(seq, self.H)
            x = x + o @ self.w(p + "self_attn.o_proj.weight").T.astype(np.float32)

            h = rmsnorm(x, self.w(p + "post_attention_layernorm.weight").astype(np.float32), self.eps)
            g = silu(h @ self.w(p + "mlp.gate_proj.weight").T.astype(np.float32))
            u = h @ self.w(p + "mlp.up_proj.weight").T.astype(np.float32)
            x = x + (g * u) @ self.w(p + "mlp.down_proj.weight").T.astype(np.float32)
            if capture:
                resid.append(x[-1].copy())
        x = rmsnorm(x, self.w("model.norm.weight").astype(np.float32), self.eps)
        logits = x @ self.w("model.embed_tokens.weight").T.astype(np.float32)
        if capture:
            return logits, np.stack(resid)
        return logits
