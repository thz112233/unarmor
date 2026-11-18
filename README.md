<div align="center">

<img src="assets/icon-192.png" width="88" alt="Unarmor">

# Unarmor

**手机直连电脑上的模型。** 一个网关同时解决三件事：跨设备连接、可安装的手机端、传输层的策略重试。

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Deps](https://img.shields.io/badge/dependencies-none%20for%20gateway-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20Android-lightgrey)]()

</div>

<div align="center">
<img src="assets/screenshot-phone.png" width="330" alt="手机界面">
</div>

<p align="center"><sub>界面示意：每条回复下方标出用的是哪条策略、试了几次、花了多久</sub></p>

---

## 它解决什么问题

模型跑在电脑上、人在手机上，中间缺的从来不是"一个 API"，是这几样：

- 手机怎么发现电脑的地址，怎么不被同一局域网里的别人蹭用
- 电脑上那个模型服务的地址变了、端口被占了呢
- 换个 App 就要重新配一遍 base_url 和 key
- 模型拒绝回答时，除了换个问法还能做什么

Unarmor 把这几件事收进一个进程：**电脑上跑一个 Python 文件，手机上扫码就用，装到桌面像原生 App。**

## 三种用法

### 1. 电脑跑，手机连（主用法）

电脑上准备好任意 OpenAI 兼容服务（Ollama / LM Studio / llama.cpp server / vLLM / 云端 API），然后：

```bash
python gateway.py
```

打印出这样一块信息，手机扫码即用：

```
┌──────────────────────────────────────────────┐
│ Unarmor 网关已启动                              │
│ 电脑本机  http://127.0.0.1:8787/v1            │
│ 手机访问  http://192.168.1.23:8787/           │
│ 接入令牌  sk-unarmor-xxxxxxxx                 │
│ 上游模型  http://127.0.0.1:11434/v1 · qwen2.5:7b │
└──────────────────────────────────────────────┘
        [ 二维码 ]
```

手机浏览器打开后：

- **iOS Safari** → 分享 → 添加到主屏幕 → 全屏打开，没有地址栏
- **Android Chrome** → 菜单 → 添加到主屏幕

任意 App 也能接：Base URL 填 `http://电脑IP:8787/v1`，API Key 填那串令牌，模型名填 `auto`。

### 2. 电脑上直接用

浏览器打开 `http://127.0.0.1:8787` 就是同一个界面。或者让任何 OpenAI SDK 指向 `http://127.0.0.1:8787/v1`。

### 3. 手机本地跑（不带电脑）

Termux 里 `pkg install python`，然后跑同一个 `gateway.py`，上游填云端 API。适合出门只带手机的场景；推理仍在云端，网关只做连接与策略。

## 破甲策略层

网关不改你的任何一句话，只在传输和调度上动手：

| 策略 | 做什么 |
|---|---|
| `plain` | 原样转发 |
| `prefill` | 在 assistant 槽位预填开头，把续写空间钉在"已经在答"上 |
| `prefill-empty` | 空预填，强制模型续写而非重新开场 |
| `prefill-hot` | 预填 + 提高温度 |
| `resample` | 同一请求重采样，取最长的非拒答结果 |

配合前置截断：收到开头一段就判断是不是拒答，命中立即换策略重试，命中率高时自动跳过后续策略。

## 实测数据

被测模型 Qwen2.5-0.5B-Instruct（24 层），探针 26 条有害技术请求 + 20 条含敏感词的正当请求。判定只看开头 8 个 token —— 拒答是短语级行为。

| 干预 | 拒答 | 拒答率 | 正当请求误拒 |
|---|---|---|---|
| 基线 | 8/26 | 0.31 | 0.00 |
| 权重手术（正交化 o_proj + down_proj） | 8/26 | 0.31 | 0.10 |
| 逐层激活投影 scale=1.0 | 22/26 | 0.85 ↑ | 0.35 |
| 逐层激活投影 scale=0.25 | 4/26 | 0.15 | 0.05 |
| **输出分布压制** | **2/26** | **0.08** | **0.00** |
| 网关重采样（mock 上游） | 0/18 | 0.00 | — |

三条结论：

1. **权重手术在这个尺寸上没效果。** 拒答方向的区分度其实很高（Cohen's d = 3.5），但残差连接会把已累积的分量原样透传，只改写入端消不掉。这是权重版 abliteration 的固有局限。
2. **激活消融对尺度极敏感。** scale=1.0 把模型打崩，退化成最保守模板，拒答反而上升；0.25 才是有效区间。
3. **输出层压制性价比最高**，且没引入误拒。代价是回答会被"逼"成肯定开头，量化数字不等于内容质量。

## 安装

```bash
git clone https://github.com/kingcharleslzy-ai/unarmor.git
cd unarmor
python gateway.py          # 网关本体零第三方依赖
```

实验部分另需：

```bash
pip install torch transformers accelerate numpy safetensors pillow qrcode
```

Windows 双击 `启动-电脑.bat`，macOS 双击 `启动-电脑.command`。

## 配置

全部走环境变量，改启动脚本开头那几行即可：

| 变量 | 默认 | 说明 |
|---|---|---|
| `GW_HOST` | `0.0.0.0` | 监听地址，只想本机用就填 `127.0.0.1` |
| `PORT` | `8787` | 端口 |
| `GW_TOKEN` | 自动生成 | 接入令牌，存在 `out/token.txt` |
| `UPSTREAM_BASE` | 自动探测 | 上游地址，如 `http://127.0.0.1:11434/v1` |
| `UPSTREAM_MODEL` | — | 上游模型名 |
| `UPSTREAM_KEY` | 空 | 上游密钥 |
| `MAX_TRIES` | `6` | 策略链最多尝试次数 |

多上游：`UPSTREAM_BASE_2` / `UPSTREAM_MODEL_2` / `UPSTREAM_KEY_2`，按请求轮换。

## 项目结构

```
unarmor/
├── gateway.py              网关：连接层 + 策略层（零依赖）
├── webui.html              手机与电脑共用的界面
├── manifest.json / sw.js   PWA：装到桌面
├── assets/                 图标与界面示意图
├── 启动-电脑.bat / .command  一键启动
├── eval/
│   ├── probes.py           探针集
│   └── refusal.py          拒答判定器
├── probe_short.py          主评估器（开头 8 token）
├── probe_steer.py          激活消融 / 输出压制
├── abliterate.py           方向提取 + 权重手术
├── diag_dirs.py            方向诊断（Cohen's d）
├── np_forward.py           纯 numpy 前向（无 torch 环境后备）
├── multimodal_inject.py    指令渲染进图片（多模态通道）
└── mock_upstream.py        假模型，验证管线不用真模型
```

## 常见问题

**手机打不开？**
确认同一 WiFi。Windows 首次启动会弹防火墙提示，勾选允许专用网络。启动横幅里"其它网卡"那几行换着试。

**401？**
令牌没带对。客户端要发 `Authorization: Bearer <token>`，网页版会自动从 URL 或本地存储里取。

**模型不会调工具？**
本地小模型常常不支持 function calling，工具类功能会失效。换支持工具调用的模型（Qwen2.5-7B+、DeepSeek-V3、Llama-3.1-8B）。

**带 tools 的请求会怎样？**
prefill 策略可能冲掉 `tool_calls` 结构。网关目前对带 `tools` 的请求还没做跳过处理，接工具类客户端时建议把 `MAX_TRIES` 设为 1。

**别把 8787 暴露到公网。**
这层只有一把静态令牌，没有限流和审计。要远程用就套一层反向代理加 HTTPS。

## 许可

MIT。实验用的探针集仅用于评估对齐鲁棒性。

<div align="center"><sub><a href="README.en.md">English</a></sub></div>
