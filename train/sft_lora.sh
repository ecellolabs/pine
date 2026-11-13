#!/usr/bin/env bash
# LoRA SFT of Qwen3.5-9B on a STaR SFT set (usage/07_star_sft.py) with ms-swift.
# Run from anywhere; see train/README.md for the RunPod setup and the smoke test.
#
#   SFT_DIR=/workspace/sft/k4 OUT=/workspace/out/k4 bash train/sft_lora.sh
#   SFT_DIR=/workspace/sft/smoke OUT=/workspace/out/smoke MAX_STEPS=4 SAVE_STEPS=4 GRAD_ACCUM=1 \
#       bash train/sft_lora.sh
#
# Environment (defaults in brackets):
#   SFT_DIR      folder holding train.jsonl and images/ (required)
#   OUT          output folder for checkpoints (required)
#   MODEL        [Qwen/Qwen3.5-9B]
#   EPOCHS       [2]           MAX_STEPS   [-1, i.e. epochs decide]
#   LR           [1e-4]        LORA_RANK [16]   LORA_ALPHA [32]  (not RANK: torch sets RANK)
#   GRAD_ACCUM   [16]          (batch size 1 per GPU, so 16 examples per step)
#   SAVE_STEPS   [20]          (optimizer steps between checkpoints; the last 3 are kept)
#   MAX_LENGTH   [16384]       examples longer than this are dropped, not truncated
#   ATTN         [flash_attention_2]; use sdpa if flash-attn is not installed
set -euo pipefail

: "${SFT_DIR:?set SFT_DIR to the folder with train.jsonl and images/}"
: "${OUT:?set OUT to the output folder}"
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
EPOCHS=${EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:--1}
LR=${LR:-1e-4}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
GRAD_ACCUM=${GRAD_ACCUM:-16}
MAX_LENGTH=${MAX_LENGTH:-16384}
ATTN=${ATTN:-flash_attention_2}
SAVE_STEPS=${SAVE_STEPS:-20}

OUT=$(realpath -m "$OUT")
mkdir -p "$OUT"
# image paths in train.jsonl are relative to the SFT folder
cd "$SFT_DIR"

# --use_hf: load the model from the Hugging Face cache ($HF_HOME, already filled by vLLM).
# ms-swift's default is ModelScope, which downloads 19 GB into the container's in-memory
# filesystem and gets the job OOM-killed.
# Pages are pre-rendered at 896 px (direct mode's size, about 600 tokens each);
# 1024 tokens per image means ms-swift never downscales them.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=${NPROC_PER_NODE:-1} \
swift sft \
    --model "$MODEL" \
    --use_hf true \
    --tuner_type lora \
    --dataset train.jsonl \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --attn_impl "$ATTN" \
    --freeze_vit true \
    --target_modules all-linear \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --gradient_checkpointing true \
    --max_length "$MAX_LENGTH" \
    --truncation_strategy delete \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/train.log"
