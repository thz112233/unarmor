@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ================= 改这里 =================
REM 上游：电脑上已经跑着的模型服务。Ollama 默认就是下面这行。
set "UPSTREAM_BASE=http://127.0.0.1:11434/v1"
set "UPSTREAM_MODEL=qwen2.5:7b"
REM 16G 内存可选：qwen2.5:7b（推荐）/ qwen2.5:14b（上限，慢）
REM 云端 API 示例（把 UPSTREAM_BASE 换成对应地址，KEY 填进去）：
REM set "UPSTREAM_BASE=https://api.deepseek.com/v1"
REM set "UPSTREAM_MODEL=deepseek-chat"
set "UPSTREAM_KEY="
REM ==========================================

set "GW_HOST=0.0.0.0"
set "PORT=8787"
set "MAX_TRIES=6"

set "PY=python"
%PY% -c "import sys" >nul 2>&1 || set "PY=py"
%PY% -c "import sys" >nul 2>&1 || (
  echo.
  echo 没找到 Python。请先到 python.org 装一个，安装时勾选 "Add to PATH"。
  echo.
  pause
  exit /b
)

echo 正在启动 Unarmor 网关...
%PY% gateway.py

echo.
echo 网关已退出。
pause
