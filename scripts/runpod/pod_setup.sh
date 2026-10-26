#!/usr/bin/env bash
# One-time setup of the RunPod pod for the PINE s4 experiment.
# Weights, dataset and virtualenvs live on the container disk; only agent_runs is on
# the network volume (/workspace), so the parser/index cache of every document survives.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"

mkdir -p /workspace/agent_runs /root/data /root/hf
export HF_HOME=/root/hf

apt-get update -qq >/dev/null
apt-get install -y -qq tmux rsync git >/dev/null

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# vLLM in its own environment (its torch pin must not touch PINE's)
if [ ! -x /root/vllm-venv/bin/vllm ]; then
  uv venv -q /root/vllm-venv --python 3.12
  # ninja: vLLM's kernel warm-up shells out to it and dies without it
  VIRTUAL_ENV=/root/vllm-venv uv pip install -q vllm ninja
fi
/root/vllm-venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# PINE on beta-index
if [ ! -d /root/pine ]; then
  git clone -q -b beta-index https://github.com/ecellolabs/pine.git /root/pine
fi
cd /root/pine && git pull -q && uv sync -q
uv run python -c "import pine, docling, rapidocr; print('pine ok')"

# serve the model
tmux kill-session -t vllm 2>/dev/null || true
tmux new -d -s vllm "export PATH=/root/vllm-venv/bin:/usr/local/cuda/bin:\$PATH CUDA_HOME=/usr/local/cuda HF_HOME=/root/hf; \
  vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  2>&1 | tee /workspace/vllm.log"
echo "setup done; vllm starting in tmux session 'vllm' (log /workspace/vllm.log)"
