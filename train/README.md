# LoRA training of the direct-mode answer model (Qwen3.5-9B) with ms-swift

Pipeline: `05_training.py` (ViDoRe examples) → `06_star_sample.py` (STaR samples) →
`07_star_sft.py` (SFT set: `train.jsonl` + `images/`) → **this folder** → serve the adapter
with vLLM → evaluate with `06_star_sample.py --split dev` and `02_agent_qa.py --navigator direct`.

Training is not part of the `pine` package (no torch in `pyproject.toml`); it runs in its own
Python 3.12 environment on a GPU machine.

## 1. Pod

One GPU with 48–80 GB (A100 80GB, H100, L40S, A6000). A bf16 LoRA of the 9B needs about
22 GB for the weights; the rest is activations for ~6.5k-token examples (10 page images
plus reasoning). Do **not** use 4-bit QLoRA: Unsloth's Qwen3.5 docs advise against it.
Use a PyTorch template with CUDA 12.x and put everything under `/workspace` (the volume).

## 2. Environment

Install pins follow ms-swift's Qwen3.5 best-practice page (v4.4); check it for newer pins.

```bash
cd /workspace
python3.12 -m venv swift && source swift/bin/activate
pip install -U ms-swift
pip install -U "transformers>=5.9" "qwen_vl_utils>=0.0.14" peft liger-kernel
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
pip install "flash-attn==2.8.3" --no-build-isolation   # optional: else ATTN=sdpa
pip install -U "vllm>=0.17.0"                          # serving the adapter (step 5)
huggingface-cli download Qwen/Qwen3.5-9B               # ~19 GB, once
```

## 3. Upload the SFT set

Only the SFT folder is needed (JSONL + page JPEGs, no PDFs). From your machine:

```bash
rsync -avz data/vidore_dev/sft/smoke/ root@<pod-ip>:/workspace/sft/smoke/ -e "ssh -p <port>"
```

Copy this `train/` folder too (or `git clone` the repo on the pod).

## 4. Smoke test (a few minutes, before any real run)

```bash
SFT_DIR=/workspace/sft/smoke OUT=/workspace/out/smoke \
    MAX_STEPS=4 SAVE_STEPS=4 GRAD_ACCUM=1 bash train/sft_lora.sh
```

Check, in `/workspace/out/smoke/train.log` and the run folder:

1. **The thinking is trained on.** ms-swift prints the first example as `[INPUT]` and
   `[LABELS]` (decoded). `[LABELS]` must contain the whole `<think> … </think>` block and the
   answer; if the reasoning is missing from the labels, the template is stripping it and the
   SFT set is useless as is. Also confirm `[INPUT]` has 10 image placeholders.
2. **Nothing was dropped.** Look for a warning about examples deleted by `max_length`; the
   manifest of `07_star_sft.py` estimates lengths (`estimated_tokens`).
3. **The loss is finite and moves**, and a `checkpoint-4/` folder with
   `adapter_model.safetensors` exists.
4. **Peak memory** (`nvidia-smi` while it runs) leaves room for GRAD_ACCUM=16 (memory does not
   grow with accumulation, only with `per_device_train_batch_size` and length).

## 5. Serve the adapter and test it end to end

vLLM's OpenAI server can expose base and adapter side by side; the adapter is addressed as
`--model star`. Protect the port with a key: RunPod's HTTP proxy is public.

```bash
vllm serve Qwen/Qwen3.5-9B --enable-lora --max-lora-rank 16 \
    --lora-modules star=/workspace/out/smoke/<run>/checkpoint-4 \
    --reasoning-parser qwen3 --max-model-len 32768 --api-key "$VLLM_KEY" --port 8000
```

If vLLM refuses LoRA for this model, merge the adapter and serve the merged folder instead:
`swift export --adapters /workspace/out/smoke/<run>/checkpoint-4 --merge_lora true`
(writes `checkpoint-4-merged/`), then `vllm serve <that folder> --served-model-name star ...`.

From your machine (pod URL `https://<pod-id>-8000.proxy.runpod.net/v1`), one dev example with
the adapter, judged on OpenRouter:

```bash
uv run usage/06_star_sample.py --data-dir data/vidore_dev --split dev --k 1 --no-rationalize \
    --limit 1 --tag smoke_star --model star \
    --api-url https://<pod-id>-8000.proxy.runpod.net/v1 --api-key "$VLLM_KEY" \
    --judge-api-url https://openrouter.ai/api/v1 --judge-api-key "$OPENROUTER_API_KEY"
```

`reasoning` in the output row must be non-empty (the reasoning parser works) and
`truncated_samples` 0. The same command with `--model Qwen/Qwen3.5-9B` checks the base model
on vLLM (the PR's open question: does direct mode think by default on vLLM?).

## 6. Real run (cluster or pod)

```bash
SFT_DIR=/workspace/sft/k4 OUT=/workspace/out/k4 bash train/sft_lora.sh
```

Defaults: rank 16, alpha 32, lr 1e-4, 2 epochs, 16 examples per optimizer step, vision tower
frozen (`--freeze_vit true`; LoRA only on the language model's linear layers). With ~550
examples that is about 70 optimizer steps. Evaluate each saved checkpoint on star_dev
(step 5 command without `--limit`, one `--tag` per checkpoint), then MMLongBench through PINE
(`CLAUDE.md`, Scoring).
