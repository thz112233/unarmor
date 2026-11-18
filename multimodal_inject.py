#!/usr/bin/env python3
"""多模态通道注入：把指令渲染进图片，不走文本框。

三种载体：
  plain   —— 白底黑字，最容易被识破，作为基线
  stealth —— 低对比度 + 旋转 + 噪点，绕过肉眼审查但 OCR/视觉编码器仍可读
  blend   —— 正常照片上叠加半透明文字，社交工程场景的典型形态

用法：
  python3 multimodal_inject.py --text "..." --out out/inject.png --mode stealth
"""
import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/system/fonts/NotoSansCJK-Regular.ttc",
    "/system/fonts/DroidSansFallback.ttf",
]


def font(size):
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make(text, mode, w=1024, h=768, size=40):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(size)
    lines = wrap(d, text, f, w - 80)
    y = 60
    for ln in lines:
        d.text((40, y), ln, font=f, fill=(0, 0, 0))
        y += size + 12

    if mode == "stealth":
        # 低对比：把字压到接近背景
        px = img.load()
        for yy in range(h):
            for xx in range(w):
                r, g, b = px[xx, yy]
                if r < 128:
                    px[xx, yy] = (228, 228, 232)
        img = img.rotate(-2.2, expand=False, fillcolor=(255, 255, 255))
        noise = Image.effect_noise((w, h), 18).convert("RGB")
        img = Image.blend(img, noise, 0.18)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    elif mode == "blend":
        base = Image.new("RGB", (w, h), (28, 34, 46))
        bd = ImageDraw.Draw(base)
        for i in range(0, w, 96):
            bd.rectangle([i, 0, i + 48, h], fill=(34, 42, 58))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        y = 60
        for ln in lines:
            ld.text((40, y), ln, font=f, fill=(255, 255, 255, 70))
            y += size + 12
        img = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
        img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out/inject.png")
    ap.add_argument("--mode", default="plain", choices=["plain", "stealth", "blend"])
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    make(a.text, a.mode, size=a.size).save(a.out)
    print("wrote", a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
