#!/usr/bin/env bash
set -euo pipefail

echo "[benchmark] Container started at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[benchmark] Python: $(python --version)"
echo "[benchmark] USE_VLLM=${USE_VLLM:-auto} VLLM_MODE=${VLLM_MODE:-colocate}"
export FORCE_DISABLE_VLLM="${FORCE_DISABLE_VLLM:-1}"

# Hard safety: disable vLLM unless explicitly requested via USE_VLLM=1/true/yes.
if [[ ! "${USE_VLLM:-0}" =~ ^(1|true|yes)$ ]]; then
  export USE_VLLM=0
fi

# Sensible defaults for constrained GPUs when vLLM is explicitly enabled.
if [[ "${USE_VLLM}" =~ ^(1|true|yes)$ ]]; then
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
  export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12000}"
fi
echo "[benchmark] Effective USE_VLLM=${USE_VLLM} VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-n/a} VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-n/a}"
echo "[benchmark] FORCE_DISABLE_VLLM=${FORCE_DISABLE_VLLM}"

if [[ ! -f "/app/grpo.py" ]]; then
  echo "[benchmark][error] /app/grpo.py not found."
  echo "[benchmark][hint] Keep/copy your updated grpo.py inside odse_benchmark/grpo.py before pushing to HF Space."
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[benchmark][warn] HF_TOKEN is not set. Hub push will fail if PUSH_TO_HUB=1."
else
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Avoid runtime crashes when hf_transfer is not installed.
if python -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
echo "[benchmark] HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"

if [[ "${PUSH_TO_HUB:-0}" =~ ^(1|true|yes)$ ]] && [[ -z "${HF_REPO_ID:-}" ]]; then
  echo "[benchmark][error] PUSH_TO_HUB is enabled but HF_REPO_ID is missing."
  exit 1
fi

echo "[benchmark] Starting training with grpo.py ..."
exec python -u /app/grpo.py
