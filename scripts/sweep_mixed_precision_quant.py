#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_mlx as pg
from sweep_residual_sidecar import (
    DEFAULT_DATA_PATH,
    DEFAULT_FLOAT_ARTIFACT,
    DEFAULT_TOKENIZER_PATH,
    build_model,
    compress_bytes,
    evaluate_state,
    load_float_state,
    make_runtime_args,
    typed_state_from_npz,
)

DEFAULT_BITS_PROFILE = ",".join(
    [
        "*.mlp.fc.weight:5",
        "*.mlp.proj.weight:5",
        "*.attn.c_q.weight:6",
        "*.attn.c_k.weight:6",
        "*.attn.c_v.weight:6",
        "*.attn.proj.weight:6",
    ]
)
CURRENT_CAPPED_BPB = 2.35570158
CURRENT_CAPPED_ARTIFACT_BYTES = 15_109_864


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline exact-eval sweep for a mixed-bit quantization profile on the promoted float artifact "
            "used for the current capped-leader comparison. Compares the stock 8-bit export path against "
            "one lower-bit per-name-pattern profile."
        )
    )
    parser.add_argument("--float-artifact", type=Path, default=DEFAULT_FLOAT_ARTIFACT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--val-seqs", type=int, default=60568)
    parser.add_argument("--val-max-batch-tokens", type=int, default=131072)
    parser.add_argument("--bits-by-name", default=DEFAULT_BITS_PROFILE)
    parser.add_argument("--compare-capped-bpb", type=float, default=CURRENT_CAPPED_BPB)
    parser.add_argument("--compare-capped-artifact-bytes", type=int, default=CURRENT_CAPPED_ARTIFACT_BYTES)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--shared-core-blocks", type=int, default=9)
    parser.add_argument("--shared-core-schedule", type=str, default="cyclic")
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def parse_bits_profile(raw: str) -> tuple[tuple[str, int], ...]:
    pairs: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid bits profile item {item!r}; expected PATTERN:BITS")
        pattern, bits_text = item.split(":", 1)
        pairs.append((pattern.strip(), int(bits_text)))
    if not pairs:
        raise ValueError("need at least one PATTERN:BITS rule")
    return tuple(pairs)


def profile_text(profile: tuple[tuple[str, int], ...]) -> str:
    return ",".join(f"{pattern}:{bits}" for pattern, bits in profile)


def main() -> None:
    ns = parse_args()
    profile = parse_bits_profile(ns.bits_by_name)
    args = make_runtime_args(ns)
    float_state_np = load_float_state(ns.float_artifact)

    dataset_name, _, _ = pg.validate_dataset_tokenizer_pair(str(ns.data_path), str(ns.tokenizer_path))
    print(f"Dataset: {dataset_name}")
    sp = spm.SentencePieceProcessor(model_file=str(ns.tokenizer_path))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = pg.build_sentencepiece_luts(sp, args.vocab_size)
    full_val_tokens = pg.load_validation_tokens(args.val_files, args.train_seq_len)
    needed_tokens = ns.val_seqs * args.train_seq_len + 1
    if full_val_tokens.size < needed_tokens:
        raise ValueError(f"validation set too small: need {needed_tokens}, got {full_val_tokens.size}")
    val_tokens = full_val_tokens[:needed_tokens]
    print(f"Validation slice: {ns.val_seqs} seqs / {needed_tokens - 1} tokens")

    pg.mx.random.seed(args.seed)
    model = build_model(args)
    float_flat = typed_state_from_npz(model, float_state_np)
    compiled_loss = pg.mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_token_losses = pg.mx.compile(lambda x, y: model.token_losses(x, y), inputs=model.state, outputs=model.state)

    baseline_obj, baseline_stats = pg.quantize_state_dict_int8(float_flat)
    baseline_artifact_bytes = compress_bytes(baseline_obj)
    baseline_flat = pg.dequantize_state_dict_int8(baseline_obj)
    _, baseline_bpb = evaluate_state(
        model,
        compiled_loss,
        compiled_token_losses,
        baseline_flat,
        args,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_prefix="[8bit] ",
    )
    print(f"Baseline 8-bit artifact: {baseline_artifact_bytes:,} bytes")
    print(f"Baseline 8-bit bpb:      {baseline_bpb:.8f}")

    mixed_obj, mixed_stats = pg.quantize_state_dict_int8(float_flat, quant_bits_overrides=profile)
    mixed_artifact_bytes = compress_bytes(mixed_obj)
    mixed_flat = pg.dequantize_state_dict_int8(mixed_obj)
    _, mixed_bpb = evaluate_state(
        model,
        compiled_loss,
        compiled_token_losses,
        mixed_flat,
        args,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_prefix="[mixed] ",
    )
    print(f"Mixed profile:           {profile_text(profile)}")
    print(f"Mixed artifact:          {mixed_artifact_bytes:,} bytes")
    print(f"Mixed bpb:               {mixed_bpb:.8f}")
    print(f"Delta vs 8-bit:          bytes {mixed_artifact_bytes - baseline_artifact_bytes:+,} bpb {mixed_bpb - baseline_bpb:+.8f}")
    print(
        f"Delta vs capped leader:  bytes {mixed_artifact_bytes - ns.compare_capped_artifact_bytes:+,} "
        f"bpb {mixed_bpb - ns.compare_capped_bpb:+.8f}"
    )

    payload = {
        "float_artifact": str(ns.float_artifact),
        "bits_profile": profile_text(profile),
        "compare_capped_bpb": ns.compare_capped_bpb,
        "compare_capped_artifact_bytes": ns.compare_capped_artifact_bytes,
        "baseline_8bit": {
            "artifact_bytes": baseline_artifact_bytes,
            "val_bpb": baseline_bpb,
            "payload_bytes": baseline_stats["int8_payload_bytes"],
        },
        "mixed_profile": {
            "artifact_bytes": mixed_artifact_bytes,
            "val_bpb": mixed_bpb,
            "payload_bytes": mixed_stats["int8_payload_bytes"],
            "artifact_delta_vs_8bit": mixed_artifact_bytes - baseline_artifact_bytes,
            "bpb_delta_vs_8bit": mixed_bpb - baseline_bpb,
            "artifact_delta_vs_capped_leader": mixed_artifact_bytes - ns.compare_capped_artifact_bytes,
            "bpb_delta_vs_capped_leader": mixed_bpb - ns.compare_capped_bpb,
        },
    }
    if ns.output_json is not None:
        ns.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
