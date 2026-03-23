#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_DIR="${ROOT_DIR}"

if [[ ! -x "${PG_DIR}/.venv/bin/python" ]]; then
  echo "Missing ${PG_DIR}/.venv. Create it with 'python3 -m venv .venv', install requirements.txt, then install mlx." >&2
  exit 1
fi

MODE="${RUN_MODE:-smoke}"
case "${MODE}" in
  smoke)
    DEFAULT_DATA_PATH="./data/datasets/fineweb10B_sp1024_smoke"
    DEFAULT_ITERATIONS="80"
    ;;
  promotion)
    DEFAULT_DATA_PATH="./data/datasets/fineweb10B_sp1024"
    DEFAULT_ITERATIONS="200"
    ;;
  *)
    echo "Unsupported RUN_MODE=${MODE}. Use smoke or promotion." >&2
    exit 1
    ;;
esac

if [[ "${MODE}" == "smoke" ]]; then
  if [[ ! -f "${PG_DIR}/data/datasets/fineweb10B_sp1024_smoke/fineweb_train_000000.bin" ]] || [[ ! -f "${PG_DIR}/data/datasets/fineweb10B_sp1024_smoke/fineweb_val_000000.bin" ]]; then
    echo "Missing Parameter Golf smoke data under ${PG_DIR}/data. See data/README.md." >&2
    exit 1
  fi
else
  if [[ ! -f "${PG_DIR}/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" ]] || [[ ! -f "${PG_DIR}/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]]; then
    echo "Missing Parameter Golf full data under ${PG_DIR}/data. See data/README.md." >&2
    exit 1
  fi
fi

if [[ ! -f "${PG_DIR}/data/tokenizers/fineweb_1024_bpe.model" ]]; then
  echo "Missing Parameter Golf tokenizer under ${PG_DIR}/data/tokenizers. See data/README.md." >&2
  exit 1
fi

cd "${PG_DIR}"

RUN_ID="${RUN_ID:-mlx_m4_${MODE}}" \
ITERATIONS="${ITERATIONS:-${DEFAULT_ITERATIONS}}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
VAL_MAX_BATCH_TOKENS="${VAL_MAX_BATCH_TOKENS:-524288}" \
MLX_MAX_MICROBATCH_TOKENS="${MLX_MAX_MICROBATCH_TOKENS:-8192}" \
WARMUP_STEPS="${WARMUP_STEPS:-0}" \
VERIFY_QUANTIZED_ROUNDTRIP="${VERIFY_QUANTIZED_ROUNDTRIP:-1}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
DATA_PATH="${DATA_PATH:-${DEFAULT_DATA_PATH}}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
"${PG_DIR}/.venv/bin/python" train_gpt_mlx.py
