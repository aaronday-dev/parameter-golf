#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $PYTHON_BIN" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

RUN_MODE="${RUN_MODE:-primary}"

INTX_PROFILE_DEFAULT="*.mlp.fc.weight:6,*.mlp.proj.weight:6,*.attn.c_q.weight:6,*.attn.c_k.weight:6,*.attn.c_v.weight:6,*.attn.proj.weight:6"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export INT8_COMPRESSOR="${INT8_COMPRESSOR:-lzma}"
export INTX_BITS_BY_NAME="${INTX_BITS_BY_NAME:-$INTX_PROFILE_DEFAULT}"
export VERIFY_QUANTIZED_ROUNDTRIP="${VERIFY_QUANTIZED_ROUNDTRIP:-1}"
export EVAL_MODE="${EVAL_MODE:-contiguous}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
export VAL_MAX_BATCH_TOKENS="${VAL_MAX_BATCH_TOKENS:-262144}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

case "$RUN_MODE" in
  primary)
    export RUN_ID="${RUN_ID:-cuda_5090_10l_mlp3x_mixedbits_v1}"
    export NUM_LAYERS="${NUM_LAYERS:-10}"
    export MLP_MULT="${MLP_MULT:-3}"
    ;;
  secondary)
    export RUN_ID="${RUN_ID:-cuda_5090_11l_mlp3x_mixedbits_v1}"
    export NUM_LAYERS="${NUM_LAYERS:-11}"
    export MLP_MULT="${MLP_MULT:-3}"
    ;;
  *)
    echo "Unsupported RUN_MODE=$RUN_MODE (expected primary or secondary)" >&2
    exit 1
    ;;
esac

exec "$PYTHON_BIN" train_gpt.py
