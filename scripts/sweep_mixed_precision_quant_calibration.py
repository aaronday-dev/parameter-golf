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
from sweep_mixed_precision_quant import (
    CURRENT_CAPPED_ARTIFACT_BYTES,
    CURRENT_CAPPED_BPB,
    DEFAULT_BITS_PROFILE,
    parse_bits_profile,
    profile_text,
)
from sweep_residual_sidecar import (
    ARTIFACT_BUDGET_BYTES,
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


DEFAULT_CALIBRATION_PROFILES = (
    ("mlp4_attn6", "*.mlp.fc.weight:4,*.mlp.proj.weight:4,*.attn.c_q.weight:6,*.attn.c_k.weight:6,*.attn.c_v.weight:6,*.attn.proj.weight:6"),
    ("mlp5_attn5", "*.mlp.fc.weight:5,*.mlp.proj.weight:5,*.attn.c_q.weight:5,*.attn.c_k.weight:5,*.attn.c_v.weight:5,*.attn.proj.weight:5"),
    ("mlp5_attn6", DEFAULT_BITS_PROFILE),
    ("mlp5_attn7", "*.mlp.fc.weight:5,*.mlp.proj.weight:5,*.attn.c_q.weight:7,*.attn.c_k.weight:7,*.attn.c_v.weight:7,*.attn.proj.weight:7"),
    ("mlp6_attn6", "*.mlp.fc.weight:6,*.mlp.proj.weight:6,*.attn.c_q.weight:6,*.attn.c_k.weight:6,*.attn.c_v.weight:6,*.attn.proj.weight:6"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline exact-eval calibration sweep around the current mixed-bit 5/6 profile on the "
            "promoted float artifact used for the capped-leader comparison. Evaluates a bounded "
            "neighborhood of nearby profiles against one shared 8-bit baseline."
        )
    )
    parser.add_argument("--float-artifact", type=Path, default=DEFAULT_FLOAT_ARTIFACT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--val-seqs", type=int, default=60568)
    parser.add_argument("--val-max-batch-tokens", type=int, default=131072)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "Repeatable LABEL=PATTERN:BITS,... override. "
            "If omitted, uses the bounded default neighborhood around the current 5/6 profile."
        ),
    )
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


def parse_profile_specs(raw_specs: list[str]) -> list[tuple[str, tuple[tuple[str, int], ...]]]:
    if not raw_specs:
        return [(label, parse_bits_profile(spec)) for label, spec in DEFAULT_CALIBRATION_PROFILES]
    specs: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    seen_labels: set[str] = set()
    for item in raw_specs:
        if "=" not in item:
            raise ValueError(f"invalid profile spec {item!r}; expected LABEL=PATTERN:BITS,...")
        label, raw_profile = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"invalid empty label in profile spec {item!r}")
        if label in seen_labels:
            raise ValueError(f"duplicate profile label {label!r}")
        seen_labels.add(label)
        specs.append((label, parse_bits_profile(raw_profile)))
    return specs


def detect_group_bits(profile: tuple[tuple[str, int], ...]) -> tuple[int | None, int | None]:
    mlp_bits: set[int] = set()
    attn_bits: set[int] = set()
    for pattern, bits in profile:
        if ".mlp." in pattern:
            mlp_bits.add(bits)
        if ".attn." in pattern:
            attn_bits.add(bits)
    return (
        next(iter(mlp_bits)) if len(mlp_bits) == 1 else None,
        next(iter(attn_bits)) if len(attn_bits) == 1 else None,
    )


def main() -> None:
    ns = parse_args()
    profile_specs = parse_profile_specs(ns.profile)
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

    rows: list[dict[str, object]] = []
    for label, profile in profile_specs:
        print(f"Testing profile {label}: {profile_text(profile)}")
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
            log_prefix=f"[{label}] ",
        )
        mlp_bits, attn_bits = detect_group_bits(profile)
        row = {
            "label": label,
            "bits_profile": profile_text(profile),
            "mlp_bits": mlp_bits,
            "attn_bits": attn_bits,
            "artifact_bytes": mixed_artifact_bytes,
            "artifact_delta_vs_8bit": mixed_artifact_bytes - baseline_artifact_bytes,
            "artifact_delta_vs_capped_leader": mixed_artifact_bytes - ns.compare_capped_artifact_bytes,
            "bytes_over_cap": mixed_artifact_bytes - ARTIFACT_BUDGET_BYTES,
            "val_bpb": mixed_bpb,
            "bpb_delta_vs_8bit": mixed_bpb - baseline_bpb,
            "bpb_delta_vs_capped_leader": mixed_bpb - ns.compare_capped_bpb,
            "payload_bytes": mixed_stats["int8_payload_bytes"],
        }
        rows.append(row)
        print(
            f"  artifact={mixed_artifact_bytes:,} over_cap={row['bytes_over_cap']:+,} "
            f"val_bpb={mixed_bpb:.8f} "
            f"d_8bit={row['bpb_delta_vs_8bit']:+.8f} "
            f"d_capped={row['bpb_delta_vs_capped_leader']:+.8f}"
        )

    best_bpb_row = min(rows, key=lambda row: row["val_bpb"])
    smallest_artifact_row = min(rows, key=lambda row: row["artifact_bytes"])

    payload = {
        "float_artifact": str(ns.float_artifact),
        "calibration_center_bits_profile": DEFAULT_BITS_PROFILE,
        "compare_capped_bpb": ns.compare_capped_bpb,
        "compare_capped_artifact_bytes": ns.compare_capped_artifact_bytes,
        "baseline_8bit": {
            "artifact_bytes": baseline_artifact_bytes,
            "val_bpb": baseline_bpb,
            "payload_bytes": baseline_stats["int8_payload_bytes"],
        },
        "summary": {
            "profile_count": len(rows),
            "best_bpb_label": best_bpb_row["label"],
            "best_bpb": best_bpb_row["val_bpb"],
            "smallest_artifact_label": smallest_artifact_row["label"],
            "smallest_artifact_bytes": smallest_artifact_row["artifact_bytes"],
        },
        "rows": rows,
    }
    if ns.output_json is not None:
        ns.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
