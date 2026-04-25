---
title: ODSE GRPO Trainer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---
# ODSE GRPO Training on Hugging Face Space

This folder is designed so you can push its contents to a **Docker HF Space** and have training start automatically with **TRL-only GRPO training**.

## Files in this folder

- `Dockerfile`: Builds the runtime and starts training.
- `requirements.txt`: Python dependencies (including TRL and ODSE from GitHub).
- `start_training.sh`: Validates env vars and runs `grpo.py`.
- `.env.example`: Suggested environment variables.

## Before you push

1. Keep your updated script at `odse_benchmark/grpo.py`.
2. Ensure your script contains the Hub push logic (`PUSH_TO_HUB`, `HF_REPO_ID`, `HF_TOKEN`).
3. Commit these files to your local repo.

## Step-by-step: run on HF Space

1. Create a new Space:
   - Type: **Docker**
   - Hardware: **GPU**
2. Open Space **Settings -> Variables and secrets** and add:
   - `HF_TOKEN` (required)
   - `HF_REPO_ID` (required if `PUSH_TO_HUB=1`)
   - `PUSH_TO_HUB=1`
   - `ENV_BASE_URL` (your ODSE env URL)
   - optional training vars from `.env.example`
3. Push this `odse_benchmark/` folder as the Space repository contents:
   - `Dockerfile`, `requirements.txt`, `start_training.sh`, `grpo.py`, etc. at repo root.
4. Wait for build to finish.
5. Training starts automatically (`CMD ["/app/start_training.sh"]`).
6. Track logs in the Space Logs tab.
7. Confirm model files in `https://huggingface.co/<HF_REPO_ID>`.

## Memory stability notes

- This setup is TRL-only and does not use vLLM.
- For CUDA OOM, reduce `MAX_COMPLETION_LENGTH` (for example `128`), keep `PER_DEVICE_BATCH_SIZE=1`, and lower `DATASET_SIZE` for pilot runs.
- Keep `GRADIENT_CHECKPOINTING=1` and `BF16=1` on supported GPUs.
- If OOM persists, set `MODEL_NAME` directly to a smaller model (for example `Qwen/Qwen2.5-1.5B-Instruct`).
- Keep `NUM_GENERATIONS>=2` for GRPO.
- If a very new model architecture is not recognized, set `FALLBACK_MODEL_NAME` (default is `Qwen/Qwen2.5-3B-Instruct`).
- If GPU memory is insufficient during initialization, the script falls back to `OOM_FALLBACK_MODEL_NAME` (default `Qwen/Qwen2.5-1.5B-Instruct`).
- If you see Triton/Inductor errors about missing C compiler, ensure Docker image includes `build-essential` and `CC/CXX` are set (already configured in this folder's `Dockerfile`).

## Push only benchmark to Space repo (example)

If your HF Space repo is cloned to `./hf-space`, sync files:

```bash
rsync -av --delete odse_benchmark/ hf-space/
cd hf-space
git add .
git commit -m "Setup ODSE GRPO auto-training Space"
git push
```

After push, Space rebuilds and starts training.
