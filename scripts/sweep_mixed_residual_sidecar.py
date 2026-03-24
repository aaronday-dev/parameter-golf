#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_mlx as pg
from mlx.utils import tree_flatten
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
    trunc_svd_sidecar,
    typed_state_from_npz,
)
from sweep_tiled_residual_sidecar import build_tiled_sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline sweep for a mixed global+local residual sidecar on one sacred tensor. "
            "Starts from a global low-rank residual carrier, then adds a tiny tiled correction "
            "for whatever structure the global factors still miss."
        )
    )
    parser.add_argument("--float-artifact", type=Path, default=DEFAULT_FLOAT_ARTIFACT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--val-seqs", type=int, default=128)
    parser.add_argument(
        "--val-max-batch-tokens",
        type=int,
        default=131072,
        help="Maximum validation tokens per eval batch.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Skip keep-float/plain baselines and only exact-evaluate the requested carriers.",
    )
    parser.add_argument(
        "--target-tensor",
        default="blocks.0.mlp.proj.weight",
        help="Exact tensor name to carry with a mixed residual sidecar.",
    )
    parser.add_argument(
        "--global-ranks",
        default="64",
        help="Comma-separated global residual ranks to test.",
    )
    parser.add_argument(
        "--tile-cols-list",
        default="1024",
        help="Comma-separated local tile widths to test.",
    )
    parser.add_argument(
        "--local-ranks",
        default="4,8,12,16",
        help="Comma-separated local per-tile ranks to test.",
    )
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


def parse_positive_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"invalid integer value: {value}")
        values.append(value)
    if not values:
        raise ValueError("need at least one integer value")
    return values


def build_quant_obj(
    flat_state: dict[str, pg.mx.array],
    target_tensor: str,
    keep_float_target: bool,
    global_sidecar: tuple[np.ndarray, np.ndarray] | None,
    tiled_sidecar: list[dict[str, object]] | None,
) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    sidecars: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not pg.mx.issubdtype(arr.dtype, pg.mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if keep_float_target and name == target_tensor:
            kept = pg.keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue
        if int(arr.size) <= pg.INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = pg.keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue
        stats["num_float_tensors"] += 1
        q, s = pg.quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

    if global_sidecar is not None and tiled_sidecar:
        left, right = global_sidecar
        sidecars[target_tensor] = {
            "scheme": "residual_mixed_low_rank_v1",
            "global": {
                "scheme": "residual_low_rank_v1",
                "left": left,
                "right": right,
            },
            "tiles": tiled_sidecar,
        }
        stats["int8_payload_bytes"] += int(left.nbytes + right.nbytes)
        for tile in tiled_sidecar:
            stats["int8_payload_bytes"] += int(tile["left"].nbytes + tile["right"].nbytes)
    elif global_sidecar is not None:
        left, right = global_sidecar
        sidecars[target_tensor] = {
            "scheme": "residual_low_rank_v1",
            "left": left,
            "right": right,
        }
        stats["int8_payload_bytes"] += int(left.nbytes + right.nbytes)
    elif tiled_sidecar:
        sidecars[target_tensor] = {
            "scheme": "residual_tiled_low_rank_v1",
            "tiles": tiled_sidecar,
        }
        for tile in tiled_sidecar:
            stats["int8_payload_bytes"] += int(tile["left"].nbytes + tile["right"].nbytes)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if sidecars:
        obj["sidecars"] = sidecars
    return obj, stats


def sidecar_raw_bytes(global_sidecar: tuple[np.ndarray, np.ndarray] | None, tiled_sidecar: list[dict[str, object]] | None) -> int:
    total = 0
    if global_sidecar is not None:
        left, right = global_sidecar
        total += int(left.nbytes + right.nbytes)
    if tiled_sidecar:
        for tile in tiled_sidecar:
            total += int(tile["left"].nbytes + tile["right"].nbytes)
    return total


def main() -> None:
    ns = parse_args()
    global_ranks = parse_positive_int_list(ns.global_ranks)
    tile_cols_list = parse_positive_int_list(ns.tile_cols_list)
    local_ranks = parse_positive_int_list(ns.local_ranks)
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
    if ns.target_tensor not in float_flat:
        raise ValueError(f"missing target tensor {ns.target_tensor!r}")
    compiled_loss = pg.mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)

    keepf_artifact_bytes: int | None = None
    keepf_bpb: float | None = None
    keepf_stats: dict[str, int] | None = None
    if not ns.candidate_only:
        keepf_obj, keepf_stats = build_quant_obj(float_flat, ns.target_tensor, keep_float_target=True, global_sidecar=None, tiled_sidecar=None)
        keepf_artifact_bytes = compress_bytes(keepf_obj)
        keepf_flat = pg.dequantize_state_dict_int8(keepf_obj)
        _, keepf_bpb = evaluate_state(
            model,
            compiled_loss,
            keepf_flat,
            args,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_prefix="[keepf] ",
        )
        print(f"Keep-float artifact: {keepf_artifact_bytes:,} bytes")
        print(f"Keep-float bpb:      {keepf_bpb:.8f}")

    plain_obj, plain_stats = build_quant_obj(float_flat, ns.target_tensor, keep_float_target=False, global_sidecar=None, tiled_sidecar=None)
    plain_artifact_bytes = compress_bytes(plain_obj)
    plain_bpb: float | None = None
    if not ns.candidate_only:
        plain_flat = pg.dequantize_state_dict_int8(plain_obj)
        _, plain_bpb = evaluate_state(
            model,
            compiled_loss,
            plain_flat,
            args,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_prefix="[plain] ",
        )
        print(f"Plain-quant artifact: {plain_artifact_bytes:,} bytes")
        print(f"Plain-quant bpb:      {plain_bpb:.8f}")
    else:
        print(f"Plain-quant artifact: {plain_artifact_bytes:,} bytes")

    target_float = np.asarray(float_state_np[ns.target_tensor], dtype=np.float32)
    target_q = np.asarray(plain_obj["quantized"][ns.target_tensor], dtype=np.int8)
    target_scale = np.asarray(plain_obj["scales"][ns.target_tensor], dtype=np.float32)
    if target_scale.ndim > 0:
        target_dequant = target_q.astype(np.float32) * target_scale[:, None]
    else:
        target_dequant = target_q.astype(np.float32) * float(target_scale)
    residual = target_float - target_dequant
    residual_norm = float(np.linalg.norm(residual))
    weight_norm = float(np.linalg.norm(target_float))
    print(f"Residual relative norm: {residual_norm / max(weight_norm, 1e-12):.8f}")

    rows: list[dict[str, object]] = []
    for global_rank in global_ranks:
        print(f"Testing global rank {global_rank}")
        global_left, global_right = trunc_svd_sidecar(residual, global_rank)
        global_sidecar = (global_left, global_right)
        global_quant_obj, global_stats = build_quant_obj(
            float_flat,
            ns.target_tensor,
            keep_float_target=False,
            global_sidecar=global_sidecar,
            tiled_sidecar=None,
        )
        global_artifact_bytes = compress_bytes(global_quant_obj)
        global_flat = pg.dequantize_state_dict_int8(global_quant_obj)
        _, global_bpb = evaluate_state(
            model,
            compiled_loss,
            global_flat,
            args,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_prefix=f"[global {global_rank}] ",
        )
        global_row = {
            "mode": "global_only",
            "global_rank": global_rank,
            "tile_cols": None,
            "local_rank": None,
            "artifact_bytes": global_artifact_bytes,
            "artifact_delta_vs_keepf": None if keepf_artifact_bytes is None else global_artifact_bytes - keepf_artifact_bytes,
            "artifact_delta_vs_plain": global_artifact_bytes - plain_artifact_bytes,
            "bytes_over_cap": global_artifact_bytes - ARTIFACT_BUDGET_BYTES,
            "val_bpb": global_bpb,
            "bpb_delta_vs_keepf": None if keepf_bpb is None else global_bpb - keepf_bpb,
            "bpb_delta_vs_plain": None if plain_bpb is None else global_bpb - plain_bpb,
            "sidecar_raw_bytes": sidecar_raw_bytes(global_sidecar, None),
            "payload_bytes": global_stats["int8_payload_bytes"],
        }
        rows.append(global_row)
        print(
            f"  global-only artifact={global_artifact_bytes:,} "
            f"over_cap={global_row['bytes_over_cap']:+,} "
            f"val_bpb={global_bpb:.8f}"
        )

        global_approx = np.asarray(global_left, dtype=np.float32) @ np.asarray(global_right, dtype=np.float32)
        local_residual = residual - global_approx
        local_residual_rel = float(np.linalg.norm(local_residual) / max(weight_norm, 1e-12))
        print(f"  leftover residual after global rank {global_rank}: {local_residual_rel:.8f}")

        for tile_cols in tile_cols_list:
            if local_residual.shape[1] % tile_cols != 0:
                raise ValueError(f"tile_cols={tile_cols} does not divide tensor width {local_residual.shape[1]}")
            for local_rank in local_ranks:
                print(f"  testing mixed rank g={global_rank} tile_cols={tile_cols} l={local_rank}")
                tiles, _ = build_tiled_sidecar(local_residual, tile_cols, local_rank)
                quant_obj, stats = build_quant_obj(
                    float_flat,
                    ns.target_tensor,
                    keep_float_target=False,
                    global_sidecar=global_sidecar,
                    tiled_sidecar=tiles,
                )
                artifact_bytes = compress_bytes(quant_obj)
                quant_flat = pg.dequantize_state_dict_int8(quant_obj)
                _, val_bpb = evaluate_state(
                    model,
                    compiled_loss,
                    quant_flat,
                    args,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    log_prefix=f"[g{global_rank} tc{tile_cols} l{local_rank}] ",
                )
                row = {
                    "mode": "mixed",
                    "global_rank": global_rank,
                    "tile_cols": tile_cols,
                    "local_rank": local_rank,
                    "artifact_bytes": artifact_bytes,
                    "artifact_delta_vs_keepf": None if keepf_artifact_bytes is None else artifact_bytes - keepf_artifact_bytes,
                    "artifact_delta_vs_plain": artifact_bytes - plain_artifact_bytes,
                    "bytes_over_cap": artifact_bytes - ARTIFACT_BUDGET_BYTES,
                    "val_bpb": val_bpb,
                    "bpb_delta_vs_keepf": None if keepf_bpb is None else val_bpb - keepf_bpb,
                    "bpb_delta_vs_plain": None if plain_bpb is None else val_bpb - plain_bpb,
                    "bpb_delta_vs_global_same_rank": val_bpb - global_bpb,
                    "artifact_delta_vs_global_same_rank": artifact_bytes - global_artifact_bytes,
                    "sidecar_raw_bytes": sidecar_raw_bytes(global_sidecar, tiles),
                    "payload_bytes": stats["int8_payload_bytes"],
                    "leftover_residual_relative_norm": local_residual_rel,
                }
                rows.append(row)
                keepf_text = "n/a" if row["bpb_delta_vs_keepf"] is None else f"{row['bpb_delta_vs_keepf']:+.8f}"
                plain_text = "n/a" if row["bpb_delta_vs_plain"] is None else f"{row['bpb_delta_vs_plain']:+.8f}"
                global_text = f"{row['bpb_delta_vs_global_same_rank']:+.8f}"
                print(
                    f"    artifact={artifact_bytes:,} over_cap={row['bytes_over_cap']:+,} "
                    f"val_bpb={val_bpb:.8f} "
                    f"d_keepf={keepf_text} d_plain={plain_text} d_global={global_text}"
                )

    rows.sort(
        key=lambda row: (
            row["bytes_over_cap"] > 0,
            row["val_bpb"],
            row["artifact_bytes"],
        )
    )

    print()
    print("Top candidates")
    print("--------------")
    for row in rows[:10]:
        keepf_text = "n/a" if row["bpb_delta_vs_keepf"] is None else f"{row['bpb_delta_vs_keepf']:+.8f}"
        plain_text = "n/a" if row["bpb_delta_vs_plain"] is None else f"{row['bpb_delta_vs_plain']:+.8f}"
        global_text = "n/a" if row["mode"] == "global_only" else f"{row['bpb_delta_vs_global_same_rank']:+.8f}"
        print(
            f"mode={row['mode']} g={row['global_rank']} tile={row['tile_cols']} local={row['local_rank']} "
            f"artifact={row['artifact_bytes']:,} over_cap={row['bytes_over_cap']:+,} val_bpb={row['val_bpb']:.8f} "
            f"d_keepf={keepf_text} d_plain={plain_text} d_global={global_text}"
        )

    payload = {
        "keep_float": None if keepf_artifact_bytes is None else {
            "artifact_bytes": keepf_artifact_bytes,
            "val_bpb": keepf_bpb,
            "payload_bytes": keepf_stats["int8_payload_bytes"],
        },
        "plain_quant": {
            "artifact_bytes": plain_artifact_bytes,
            "val_bpb": plain_bpb,
            "payload_bytes": plain_stats["int8_payload_bytes"],
        },
        "target_tensor": ns.target_tensor,
        "residual_relative_norm": residual_norm / max(weight_norm, 1e-12),
        "rows": rows,
    }
    if ns.output_json is not None:
        ns.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
