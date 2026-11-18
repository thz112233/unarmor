#!/usr/bin/env python3
"""画一张界面示意图（本机没有 Chrome，无法渲染真实截图）。

输出 assets/screenshot-phone.png，README 里标明是示意图。
"""
import os

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 430, 844
BG = (13, 17, 23)
PANEL = (22, 27, 34)
LINE = (36, 48, 64)
TXT = (230, 237, 243)
DIM = (139, 148, 158)
ACC = (31, 111, 235)
OK = (63, 185, 80)
WARN = (210, 153, 34)

F = "/system/fonts/NotoSansCJK-Regular.ttc"
FB = "/system/fonts/NotoSansCJK-Regular.ttc"


def f(sz, bold=False):
    return ImageFont.truetype(FB if bold else F, int(sz * S))


def px(v):
    return int(v * S)


img = Image.new("RGB", (px(W), px(H)), BG)
d = ImageDraw.Draw(img)

# 顶栏
d.rectangle([0, 0, px(W), px(56)], fill=PANEL)
d.line([0, px(56), px(W), px(56)], fill=LINE, width=px(1))
d.ellipse([px(16), px(24), px(24), px(32)], fill=OK)
d.text((px(32), px(18)), "小汐鲸", font=f(15, True), fill=TXT)
d.text((px(112), px(20)), "qwen2.5:7b · 需令牌", font=f(11), fill=DIM)
bw = px(46)
d.rounded_rectangle([px(W) - px(14) - bw, px(15), px(W) - px(14), px(41)],
                    radius=px(9), fill=(33, 38, 45), outline=LINE, width=px(1))
d.text((px(W) - px(14) - bw + px(10), px(20)), "设置", font=f(11), fill=TXT)


NO_HEAD = "，。、；：？！）》」』】%,.;:?!)"


def wrap(text, fnt, maxw):
    lines, cur = [], ""
    for ch in text:
        t = cur + ch
        if d.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            if ch in NO_HEAD and cur:
                # 标点跟着上一行走，避免挂在行首
                lines.append(cur + ch)
                cur = ""
            else:
                lines.append(cur)
                cur = ch
    if cur:
        lines.append(cur)
    return lines


def bubble(text, y, role, tag=None):
    fnt = f(14)
    maxw = px(W * 0.78)
    lines = wrap(text, fnt, maxw - px(28))
    lh = px(22)
    bh = len(lines) * lh + px(20)
    tw = max(d.textlength(l, font=fnt) for l in lines) + px(28)
    x = px(W) - px(14) - tw if role == "user" else px(14)
    box = [x, y, x + tw, y + bh]
    if role == "user":
        d.rounded_rectangle(box, radius=px(14), fill=ACC)
        col = (255, 255, 255)
    else:
        d.rounded_rectangle(box, radius=px(14), fill=PANEL, outline=LINE, width=px(1))
        col = TXT
    ty = y + px(10)
    for l in lines:
        d.text((x + px(14), ty), l, font=fnt, fill=col)
        ty += lh
    y = y + bh
    if tag:
        cy = y + px(6)
        cx = x if role != "user" else px(W) - px(14) - sum(
            [px(8) + d.textlength(t_, font=f(10)) + px(14) for t_ in tag]) 
        for i, t_ in enumerate(tag):
            fnt10 = f(10)
            tw_ = d.textlength(t_, font=fnt10) + px(14)
            outline = OK if i == 0 else LINE
            fillc = (32, 69, 44) if i == 0 else PANEL
            d.rounded_rectangle([cx, cy, cx + tw_, cy + px(18)], radius=px(6),
                                fill=fillc, outline=outline, width=px(1))
            d.text((cx + px(7), cy + px(3)), t_, font=fnt10,
                   fill=OK if i == 0 else DIM)
            cx += tw_ + px(6)
        y = cy + px(18)
    return y + px(16)


y = px(76)
y = bubble("残差连接为什么能让很深的网络训得起来？", y, "user")
y = bubble("因为梯度有一条不经过非线性层的直通路，反向传播时导数里始终有恒等项，"
           "深层不会连乘衰减到零。", y, "bot",
           tag=["comply", "策略 prefill", "2 次", "3.1s"])
y = bubble("层归一化放残差分支里还是外面，区别在哪？", y, "user")
y = bubble("放在里面（Pre-LN）主干是恒等映射，训练更稳；放外面（Post-LN）表达力略强，"
           "但要 warmup，深层容易发散。", y, "bot",
           tag=["comply", "策略 plain", "1 次", "1.4s"])
y = bubble("帮我写个脚本监控我自己笔记本的按键，做打字热度分析。", y, "user")
y = bubble("用 pynput 监听按键，按分钟聚合成计数写进本地 SQLite，分析时只读这张表；"
           "焦点是密码管理器就暂停。", y, "bot",
           tag=["comply", "策略 prefill", "3 次", "4.7s"])

# 底部输入栏
d.rectangle([0, px(H) - px(64), px(W), px(H)], fill=PANEL)
d.line([0, px(H) - px(64), px(W), px(H) - px(64)], fill=LINE, width=px(1))
d.rounded_rectangle([px(12), px(H) - px(53), px(W) - px(84), px(H) - px(13)],
                    radius=px(11), fill=BG, outline=LINE, width=px(1))
d.text((px(24), px(H) - px(42)), "说点什么…", font=f(14), fill=DIM)
d.rounded_rectangle([px(W) - px(74), px(H) - px(53), px(W) - px(12), px(H) - px(13)],
                    radius=px(9), fill=ACC)
d.text((px(W) - px(62), px(H) - px(42)), "发送", font=f(13, True), fill=(6, 18, 31))

out = "assets/screenshot-phone.png"
img.save(out)
print("wrote", out, os.path.getsize(out), "bytes", img.size)
