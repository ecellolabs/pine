#!/bin/bash -l
set -uo pipefail

# One shard of a direct-mode MMLongBench-Doc run with a STaR adapter as the answer model:
#   1. starts a Qwen3.5-9B vLLM server (vLLM image) with the adapter loaded as $NAME,
#   2. waits until it answers,
#   3. runs usage/02_agent_qa.py on the shard's sample set inside the server's job
#      (planner/index on the base model, the answer call on the adapter),
#   4. cancels the server job when the shard is done or anything fails.
#
#   export OPENROUTER_API_KEY=sk-or-...          # the judge runs on OpenRouter
#   ADAPTER=/netscratch/$USER/outputs/star/k4_gold/v1-.../checkpoint-56 NAME=star56 \
#       BATCH=star56_direct_full bash train/mmlongbench_shard_cluster.sh mmlb_shard1
#
# Same flags as Experiment A (agent_runs/expA_direct_full) except --answer-model, so the two
# batches differ only in the answer model. All shards write into the same batch folder (disjoint
# documents); the batch manifest is rebuilt from the result.json files at the end.

SHARD="${1:?sample set name, e.g. mmlb_shard1}"
ADAPTER="$(realpath "${ADAPTER:?set ADAPTER to the checkpoint folder}")"
NAME="${NAME:?set NAME, the served adapter name (e.g. star56)}"
BATCH="${BATCH:?set BATCH, the batch folder name under agent_runs/}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (the judge runs on OpenRouter)}"
P=/netscratch/$USER/projects/pipeline-v2
PORT=${PORT:-$((10100 + RANDOM % 800))}
MOUNTS="/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER"
GPUS="${PARTITION:-A100-80GB,H100,H200,RTXA6000,L40S,A100-40GB}"
LOG="$P/agent_runs/$BATCH/.server_$SHARD.log"
[ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter_config.json in $ADAPTER" >&2; exit 1; }
[ -f "$P/sample_sets/$SHARD.json" ] || { echo "no sample set $SHARD" >&2; exit 1; }
mkdir -p "$P/agent_runs/$BATCH"

srun --container-image="/netscratch/$USER/envs/vllm.sqsh" --container-mounts="$MOUNTS" \
    -n 1 -c 4 --mem=60G --gpus-per-task=1 --partition="$GPUS" --time="${TIME:-12:00:00}" \
    --export=ALL,HF_HOME="/netscratch/$USER/cache/hf" \
    bash -c "/opt/env/.venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B --enable-lora --max-lora-rank 16 --lora-modules $NAME=$ADAPTER --port $PORT --host 0.0.0.0 --dtype auto" \
    > "$LOG" 2>&1 &
SERVER=$!
until grep -q "Running on node" "$LOG"; do
    kill -0 "$SERVER" 2>/dev/null || { echo "$SHARD: server exited before starting:"; tail -20 "$LOG"; exit 1; }
    sleep 10
done
JOB=$(grep -oE "Job [0-9]+: Running" "$LOG" | grep -oE "[0-9]+" | head -1)
NODE=$(grep -oE "Running on node\(s\) [^ ]+" "$LOG" | awk '{print $NF}')
trap 'scancel "$JOB" 2>/dev/null' EXIT    # never leave the GPU running
echo "$SHARD: job $JOB on $NODE:$PORT, loading"
until curl -sf "http://$NODE:$PORT/v1/models" | grep -q "\"$NAME\""; do
    kill -0 "$SERVER" 2>/dev/null || { echo "$SHARD: server exited while loading:"; tail -30 "$LOG"; exit 1; }
    sleep 15
done
echo "$SHARD: server ready, running the shard"

cd "$P"
URL="http://$NODE:$PORT/v1"
srun --jobid="$JOB" --overlap -n 1 -c 2 --mem=16G uv run usage/02_agent_qa.py mmlongbench_doc \
    --max-samples "$SHARD" --data-dir "/netscratch/$USER/data/mmlongbench_doc" \
    --api-url "$URL" --vision-api-url "$URL" \
    --text-model Qwen/Qwen3.5-9B --vision-model Qwen/Qwen3.5-9B \
    --navigator direct --answer-model "$NAME" \
    --thinking-sampling --max-tokens-scale 20 --request-timeout 1500 --index-no-reasoning \
    --workers 16 --batch-name "$BATCH" \
    --judge-model openai/gpt-oss-120b --judge-api-url https://openrouter.ai/api/v1 \
    --judge-api-key "$OPENROUTER_API_KEY" --judge-max-tokens 2000 \
    --judge-extra-body '{"reasoning_effort": "low"}'
echo "$SHARD: finished (exit $?); server job $JOB cancelled"
