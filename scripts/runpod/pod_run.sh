#!/usr/bin/env bash
# Run one sample set on the pod against the local vLLM server, thinking off everywhere,
# lean index, rule scoring (the gpt-oss judge is applied afterwards with usage/04_judge.py).
# usage: pod_run.sh <sample-set> <batch-name> <workers>
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SET="$1"; BATCH="$2"; WORKERS="${3:-8}"
# the parser's OCR and layout models stay on the CPU: sharing the GPU with vLLM
# ran the server out of memory (the document cache makes the parse a one-time cost)
export CUDA_VISIBLE_DEVICES=""
cd /root/pine
exec uv run usage/02_agent_qa.py mmlongbench_doc --max-samples "$SET" \
  --data-dir /root/data/mmlongbench_doc --output-dir /workspace/agent_runs \
  --api-url http://localhost:8000/v1 --api-key dummy \
  --vision-api-url http://localhost:8000/v1 --vision-api-key dummy \
  --text-model Qwen/Qwen3.5-9B --vision-model Qwen/Qwen3.5-9B \
  --no-reasoning --index lean \
  --workers "$WORKERS" --index-concurrency 4 --request-timeout 300 \
  --batch-name "$BATCH"
