#!/usr/bin/env python3
"""虎鲸娘图标。

关键在特征顺序：眼斑包着眼睛（不是耳朵），下巴一道白，
头顶一支背鳍。小尺寸下这三样认出来就成立了。
"""
import os

from PIL import Image, ImageDraw

S = 512
SS = 4
BG = (13, 17, 23)
EDGE = (36, 48, 64)
BLACK = (16, 21, 28)
WHITE = (242, 247, 251)
EYE = (18, 24, 32)
TEAL = (78, 161, 255)
PINK = (255, 140, 172)
MOUTH = (26, 34, 44)


def sc(v):
    return int(v * SS)


def angled_patch(d, cx, cy, rx, ry, tilt):
    """画一个倾斜的椭圆贴片（虎鲸眼斑）。"""
    w, h = sc(rx * 2), sc(ry * 2)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.ellipse([0, 0, w - 1, h - 1], fill=WHITE)
    layer = layer.rotate(tilt, resample=Image.BICUBIC, expand=False)
    img.paste(layer, (sc(cx) - w // 2, sc(cy) - h // 2), layer)


img = Image.new("RGBA", (sc(S), sc(S)), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

d.rounded_rectangle([0, 0, sc(S) - 1, sc(S) - 1], radius=sc(112), fill=BG)
d.rounded_rectangle([sc(2), sc(2), sc(S) - 3, sc(S) - 3], radius=sc(111),
                    outline=EDGE, width=sc(3))

# 身体
d.ellipse([sc(138), sc(356), sc(374), sc(556)], fill=BLACK)
d.ellipse([sc(196), sc(420), sc(316), sc(556)], fill=WHITE)

# 头顶背鳍：加高加宽，头盖不住的部分要够显眼
d.polygon([(sc(256), sc(42)), (sc(184), sc(186)), (sc(328), sc(186))], fill=BLACK)
d.ellipse([sc(216), sc(112), sc(296), sc(192)], fill=BLACK)

# 头
d.ellipse([sc(118), sc(104), sc(394), sc(380)], fill=BLACK)

# 下巴那道白：整体下移，做成下颌而不是胸脯
d.ellipse([sc(204), sc(306), sc(308), sc(408)], fill=WHITE)

# 两片倾斜眼斑，往斜上方走，包住眼睛
angled_patch(d, 188, 228, 44, 33, 24)
angled_patch(d, 324, 228, 44, 33, -24)

# 眼睛（落在眼斑里，占位略收）
for ex in (198, 314):
    d.ellipse([sc(ex - 15), sc(232 - 19), sc(ex + 15), sc(232 + 19)], fill=EYE)
    d.ellipse([sc(ex - 7), sc(224 - 9), sc(ex + 2), sc(224 + 1)], fill=WHITE)
    d.ellipse([sc(ex + 2), sc(238 - 3), sc(ex + 8), sc(238 + 3)], fill=(140, 200, 255))

# 睫毛（上眼睑两根，给女性化的味道）
for ex, sgn in ((198, -1), (314, 1)):
    d.line([sc(ex + sgn * 12), sc(214), sc(ex + sgn * 24), sc(206)], fill=EYE, width=sc(5))
    d.line([sc(ex + sgn * 4), sc(212), sc(ex + sgn * 10), sc(202)], fill=EYE, width=sc(5))

# 腮红
for bx in (186, 326):
    d.ellipse([sc(bx - 18), sc(292 - 10), sc(bx + 18), sc(292 + 10)], fill=PINK)

# 嘴
d.arc([sc(238), sc(310), sc(274), sc(340)], start=18, end=162, fill=MOUTH, width=sc(6))

# 胸鳍：身体两侧各一片
d.polygon([(sc(150), sc(408)), (sc(96), sc(470)), (sc(174), sc(478))], fill=BLACK)
d.polygon([(sc(362), sc(408)), (sc(416), sc(470)), (sc(338), sc(478))], fill=BLACK)

# 额前一缕挑染
d.arc([sc(200), sc(158), sc(268), sc(228)], start=210, end=292, fill=TEAL, width=sc(10))

img = img.resize((S, S), Image.LANCZOS)
img.save("assets/icon-512.png")
img.resize((192, 192), Image.LANCZOS).save("assets/icon-192.png")
img.resize((180, 180), Image.LANCZOS).save("assets/apple-touch-icon.png")
img.resize((64, 64), Image.LANCZOS).save("assets/icon-64.png")
print("ok", os.path.getsize("assets/icon-512.png"))
