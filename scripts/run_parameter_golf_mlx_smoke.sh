#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_DIR="${ROOT_DIR}"

if [[ ! -x "${PG_DIR}/.venv/bin/python" ]]; then
  echo "Missing ${PG_DIR}/.venv. Create it with 'python3 -m venv .venv' and install requirements.txt." >&2
  exit 1
fi

if [[ ! -f "${PG_DIR}/data/datasets/fineweb10B_sp1024_smoke/fineweb_train_000000.bin" ]] || [[ ! -f "${PG_DIR}/data/datasets/fineweb10B_sp1024_smoke/fineweb_val_000000.bin" ]] || [[ ! -f "${PG_DIR}/data/tokenizers/fineweb_1024_bpe.model" ]]; then
  echo "Missing Parameter Golf smoke data or tokenizer under ${PG_DIR}/data. See data/README.md." >&2
  exit 1
fi

cd "${PG_DIR}"

RUN_ID="${RUN_ID:-mlx_smoke}" \
ITERATIONS="${ITERATIONS:-200}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
VAL_MAX_BATCH_TOKENS="${VAL_MAX_BATCH_TOKENS:-524288}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024_smoke}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
"${PG_DIR}/.venv/bin/python" train_gpt_mlx.py
