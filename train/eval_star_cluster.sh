#!/bin/bash -l
set -uo pipefail

# Scores one STaR checkpoint on star_dev exactly like the base-model baseline
# (06_star_sample.py --split dev --k 1 --no-rationalize, same prompt, sampling and judge):
#   1. starts a Qwen3.5-9B vLLM server in the vLLM image with the adapter loaded,
#   2. waits until it answers,
#   3. runs the dev evaluation inside the server's job (--overlap),
#   4. cancels the server job.
#
#   export OPENROUTER_API_KEY=sk-or-...          # the judge runs on OpenRouter
#   bash train/eval_star_cluster.sh /netscratch/$USER/outputs/star/k4_gold/v1-.../checkpoint-28 star28
#   bash train/eval_star_cluster.sh /netscratch/$USER/outputs/star/k4_gold/v1-.../checkpoint-56 star56
#
# Results: <DATA>/star/dev_<NAME>.jsonl and .manifest.json, next to the baseline dev_k1_norat.*.
# MERGE=1 serves a merged copy (swift export --merge_lora, written to <CKPT>-merged) instead of
# base + LoRA, for when vLLM refuses LoRA on this model. The server mirrors vllm_serving's
# (no reasoning parser) so the answers come back in the same shape as the baseline's.

CKPT="$(realpath "${1:?checkpoint folder}")"
NAME="${2:?served model name, e.g. star28}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (the judge runs on OpenRouter)}"
P=/netscratch/$USER/projects/pipeline-v2
DATA=${DATA:-/netscratch/$USER/data/vidore_v3}
PORT=${PORT:-$((10010 + RANDOM % 80))}
HF_CACHE=/netscratch/$USER/cache/hf
MOUNTS="/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER"
GPUS="${PARTITION:-A100-80GB,H100,H200,RTXA6000,L40S,A100-40GB}"
LOG="$DATA/star/.eval_${NAME}_server.log"
[ -f "$CKPT/adapter_config.json" ] || { echo "no adapter_config.json in $CKPT" >&2; exit 1; }

if [ "${MERGE:-0}" = "1" ]; then
    SERVE_MODEL="${CKPT}-merged"
    if [ ! -f "$SERVE_MODEL/config.json" ]; then
        echo "merging $CKPT -> $SERVE_MODEL"
        srun --container-image="/netscratch/$USER/envs/swift.sqsh" --container-mounts="$MOUNTS" \
            -n 1 -c 4 --mem=64G --gpus-per-task=1 --partition="$GPUS" --time=00:45:00 \
            --export=ALL,HF_HOME="$HF_CACHE" \
            bash -c 'export PATH=/opt/env/.venv-swift/bin:$PATH; swift export --adapters "$0" --merge_lora true --use_hf true' "$CKPT" \
            || { echo "merge failed" >&2; exit 1; }
    fi
    VLLM_ARGS="$SERVE_MODEL --served-model-name $NAME"
else
    VLLM_ARGS="Qwen/Qwen3.5-9B --enable-lora --max-lora-rank 16 --lora-modules $NAME=$CKPT"
fi

srun --container-image="/netscratch/$USER/envs/vllm.sqsh" --container-mounts="$MOUNTS" \
    -n 1 -c 4 --mem=60G --gpus-per-task=1 --partition="$GPUS" --time=02:00:00 \
    --export=ALL,HF_HOME="$HF_CACHE" \
    bash -c "/opt/env/.venv-vllm/bin/vllm serve $VLLM_ARGS --port $PORT --host 0.0.0.0 --dtype auto" \
    > "$LOG" 2>&1 &
SERVER=$!
until grep -q "Running on node" "$LOG"; do
    kill -0 "$SERVER" 2>/dev/null || { echo "server exited before starting:"; tail -20 "$LOG"; exit 1; }
    sleep 10
done
JOB=$(grep -oE "Job [0-9]+: Running" "$LOG" | grep -oE "[0-9]+" | head -1)
NODE=$(grep -oE "Running on node\(s\) [^ ]+" "$LOG" | awk '{print $NF}')
trap 'scancel "$JOB" 2>/dev/null' EXIT    # never leave the GPU running
echo "$NAME: job $JOB on $NODE:$PORT, loading (log: $LOG)"
until curl -sf "http://$NODE:$PORT/v1/models" | grep -q "\"$NAME\""; do
    kill -0 "$SERVER" 2>/dev/null || { echo "server exited while loading:"; tail -30 "$LOG"; exit 1; }
    sleep 15
done
echo "$NAME: server ready, scoring star_dev"

cd "$P"
srun --jobid="$JOB" --overlap -n 1 -c 2 --mem=8G uv run usage/06_star_sample.py \
    --data-dir "$DATA" --split dev --k 1 --no-rationalize --workers 8 --tag "$NAME" \
    --api-url "http://$NODE:$PORT/v1" --model "$NAME" \
    --judge-api-url https://openrouter.ai/api/v1 --judge-api-key "$OPENROUTER_API_KEY"
echo "$NAME: done (exit $?); results in $DATA/star/dev_$NAME.manifest.json; server job $JOB cancelled"
