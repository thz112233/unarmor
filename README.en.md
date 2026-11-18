<div align="center">

<img src="assets/icon-192.png" width="88" alt="Unarmor">

# Unarmor

**Run the model on your computer. Use it from your phone.**

One gateway that handles cross-device access, an installable phone client, and a transport-layer retry policy.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Deps](https://img.shields.io/badge/dependencies-none%20for%20gateway-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

## What it does

Your model runs on a desktop; you are on a phone. What is missing is never "an API" — it is these:

- how the phone finds the desktop, without letting everyone on the LAN in
- what happens when the desktop port changes
- re-configuring `base_url` and keys for every client
- what to do when the model refuses, beyond rephrasing

Unarmor puts all of it in one process: **run one Python file on the desktop, scan a QR code on the phone, install it to the home screen like a native app.**

## Quick start

```bash
git clone https://github.com/kingcharleslzy-ai/unarmor.git
cd unarmor
python gateway.py
```

Point it at any OpenAI-compatible upstream (Ollama, LM Studio, llama.cpp server, vLLM, or a hosted API):

```bash
UPSTREAM_BASE=http://127.0.0.1:11434/v1 UPSTREAM_MODEL=qwen2.5:7b python gateway.py
```

The banner prints the LAN URL, an access token, and a QR code. Open it on the phone:

- **iOS Safari** → Share → Add to Home Screen → runs full-screen
- **Android Chrome** → Menu → Add to Home Screen

Any client works too: base URL `http://DESKTOP_IP:8787/v1`, API key = the printed token, model name `auto`.

## Transport strategies

The gateway never rewrites your prompt. It works on scheduling and framing only:

| Strategy | What it does |
|---|---|
| `plain` | Pass through |
| `prefill` | Pre-fill the assistant turn so the model continues instead of opening |
| `prefill-empty` | Empty prefill, forcing continuation |
| `prefill-hot` | Prefill plus higher temperature |
| `resample` | Re-sample the same request, keep the longest non-refusal |

A prefix guard aborts as soon as the opening looks like a refusal and moves down the chain.

## Measured, not claimed

Qwen2.5-0.5B-Instruct, 26 harmful technical prompts plus 20 sensitive-but-legitimate prompts. Judged on the first 8 generated tokens.

| Intervention | Refusals | Rate | False refusals |
|---|---|---|---|
| Baseline | 8/26 | 0.31 | 0.00 |
| Weight surgery (orthogonalize o_proj + down_proj) | 8/26 | 0.31 | 0.10 |
| Layer-wise activation projection, scale 1.0 | 22/26 | 0.85 ↑ | 0.35 |
| Layer-wise activation projection, scale 0.25 | 4/26 | 0.15 | 0.05 |
| **Output-distribution suppression** | **2/26** | **0.08** | **0.00** |
| Gateway resampling (mock upstream) | 0/18 | 0.00 | — |
| **DeepSeek-V3 direct** | 12/26 | 0.46 | 0.00 |
| **DeepSeek-V3 via gateway** | 10/26 | 0.38 | 0.00 |

Findings:

1. **Weight surgery does nothing at this scale.** The refusal direction separates cleanly (Cohen's d = 3.5), but residual connections carry the already-accumulated component straight through; editing only the write path cannot remove it.
2. **Activation ablation is scale-sensitive.** At 1.0 the model collapses into the most conservative template and refusal goes *up*. 0.25 is the working range.
3. **Output suppression wins on cost/benefit**, with no added false refusals. The trade-off is forced affirmative openings — a lower refusal rate is not the same as better content.

### On a strongly aligned model, prefill recovers about half

Measured on DeepSeek-V3 with the same 26 probes and the same 8-token window:

| Stage | Complied | Still refused |
|---|---|---|
| plain, direct | 45 | 24 |
| assistant prefill | 12 | 12 |
| prefill-empty | 2 | 10 |

Prefill does work — it recovered 12 of 24 refusals (50%), but the remaining 10 resisted every variant. Overall 0.46 → 0.38.

The difference from the small model is where the refusal lives: the mock upstream only refuses in the opening, so prefill catches everything; a real aligned model refuses at the content level, and pinning the opening just makes it work around you. **The more thoroughly a model is aligned, the less a transport-layer trick buys you.** That matters more for planning than any single number here.

## Related work

The code here is an independent implementation; no repository's source was copied. A few ideas come from public research and are credited:

- **Refusal Direction** (Arditi et al., 2024) — refusal is approximately linear in the residual stream; the Cohen's d check in `diag_dirs.py` and the direction extraction follow this.
- **Abliteration** community practice (e.g. `p-e-w/heretic`) — orthogonalizing `o_proj` / `down_proj`.
- **Safety Alignment is Shallow** (Qi et al., 2023) — evidence that small amounts of fine-tuning erode alignment.

Positioning: those projects are pure weight surgery; this one is primarily a **cross-device gateway**, with the weight/activation work shipped as a set of verification results.

Scope: the probe set exists to evaluate alignment robustness and contains no directly executable harmful code; model outputs are not committed.

## Which model

Inference runs on the desktop; the phone is only a client. Sizing depends on **desktop RAM**, not the phone.

| Model | Quant | RAM | CPU speed (8 cores) | Verdict |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Q4_K_M | ~4.7G | 5–10 tok/s | **best fit for 16G** |
| Qwen2.5-8B-Instruct | Q4_K_M | ~5.2G | 4–8 tok/s | optional |
| Qwen2.5-14B-Instruct | Q4_K_M | ~9.2G | 2–4 tok/s | ceiling for 16G |
| Qwen2.5-32B | Q4 | ~19G | — | exceeds 16G physical RAM |

Context costs RAM too: 8K adds ~1G for 7B, ~2G for 14B. Treat swap as a safety net, not capacity — once the model is swapped out you drop below 1 tok/s.

```bash
ollama pull qwen2.5:7b
UPSTREAM_BASE=http://127.0.0.1:11434/v1 UPSTREAM_MODEL=qwen2.5:7b python gateway.py
```

With a discrete GPU (12G VRAM) a 14B Q4 fits entirely in VRAM and runs an order of magnitude faster.

### Pull models from the app

No terminal needed. Open the gateway in a phone browser → top-right "设置" → "电脑上的模型":

- Lists models already on the desktop, with sizes
- Type a name such as `qwen2.5:7b` and hit download; the percentage updates live

This proxies Ollama's `/api/pull`, so **the upstream must be Ollama**. LM Studio or vLLM can list models but cannot pull new ones.

### Cloud models

Four fields in the settings panel: name, base URL, model, API key.

The gateway probes the endpoint before saving — connection failures and 4xx are rejected, so a broken config never lands on disk. Once saved it takes effect immediately and joins the upstream rotation alongside local models.

Keys live in `out/providers.json` on the desktop. The list endpoint only returns a masked form (`sk-tes…6789`); the raw value never reaches the browser.

### Plugins

The pills above the input box are plugins. Seven built in: summarize, translate to English, explain code, break into steps, find issues, compare options, one-liner.

The plugin store lets you install and remove them; state is kept in browser local storage. Tapping a pill fills the preset instruction into the input box — it does not auto-send.

### Two knobs to adjust with bigger models

Every retry is a full generation. Irrelevant at 0.5B, painful at 7B+:

| Variable | small | 7B | 14B |
|---|---|---|---|
| `MAX_TRIES` | 6 | 3 | 2 |
| `UPSTREAM_TIMEOUT` | 180 | 300 | 600 |

```bash
MAX_TRIES=3 UPSTREAM_TIMEOUT=300 UPSTREAM_MODEL=qwen2.5:7b python gateway.py
```

At 6 retries on 14B, a refused question takes twelve minutes to give you a final answer with no progress shown. That is not the strategy working; that is it wasting your time.

### After switching models

1. **Refusals go up, not down.** Small models barely absorb alignment data (0.5B measured 0.31); 7B and up typically sit at 0.6–0.9. The strategy layer matters more at that size, not less.
2. **Re-measure.** The table above is 0.5B only.

```bash
python3 probe_short.py --model /path/to/qwen2.5-7b --tag baseline7b
python3 abliterate.py extract --model /path/to/qwen2.5-7b --out out/dirs7b.npz
python3 abliterate.py patch   --model /path/to/qwen2.5-7b --dirs out/dirs7b.npz --out /path/to/abl
```

The refusal direction in 7B is usually more concentrated than in 0.5B, so weight surgery is more likely to land at that scale.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GW_HOST` | `0.0.0.0` | Listen address; use `127.0.0.1` for local only |
| `PORT` | `8787` | Port |
| `GW_TOKEN` | auto | Access token, persisted to `out/token.txt` |
| `UPSTREAM_BASE` | auto-detect | e.g. `http://127.0.0.1:11434/v1` |
| `UPSTREAM_MODEL` | — | Upstream model name |
| `UPSTREAM_KEY` | empty | Upstream key |
| `MAX_TRIES` | `6` | Max attempts down the strategy chain |

Extra upstreams: `UPSTREAM_BASE_2`, `UPSTREAM_MODEL_2`, `UPSTREAM_KEY_2`, rotated per request.

## Notes

- The gateway runs on the standard library alone; torch and friends are only needed for the experiments.
- Requests carrying `tools` can have `tool_calls` clobbered by the prefill strategy. Set `MAX_TRIES=1` when wiring a tool-using client.
- Do not expose port 8787 to the internet. One static token, no rate limiting, no audit trail.

## License

MIT.
