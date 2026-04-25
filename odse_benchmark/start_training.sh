#!/usr/bin/env bash
set -euo pipefail

echo "[benchmark] Container started at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[benchmark] Python: $(python --version)"
echo "[benchmark] TRL-only mode (vLLM removed)"

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
