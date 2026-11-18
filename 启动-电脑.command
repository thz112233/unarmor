#!/bin/bash
cd "$(dirname "$0")"

# ================= 改这里 =================
# 上游：电脑上已经跑着的模型服务。Ollama 默认就是下面这行。
export UPSTREAM_BASE="http://127.0.0.1:11434/v1"
export UPSTREAM_MODEL="qwen2.5:7b"
# 16G 内存可选：qwen2.5:7b（推荐）/ qwen2.5:14b（上限，慢）
# 有 12G 以上显存可以试 qwen2.5:14b，速度会快很多
# 云端 API 示例：
# export UPSTREAM_BASE="https://api.deepseek.com/v1"
# export UPSTREAM_MODEL="deepseek-chat"
export UPSTREAM_KEY=""
# ==========================================

export GW_HOST=0.0.0.0
export PORT=8787
export MAX_TRIES=6

if ! command -v python3 >/dev/null 2>&1; then
  echo "没找到 python3，请先安装 Python 3。"
  read -r -p "按回车退出"
  exit 1
fi

python3 gateway.py
