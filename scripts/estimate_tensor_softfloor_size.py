#!/usr/bin/env python3
from __future__ import annotations

import argparse
import lzma
import pickle
from pathlib import Path
import zlib

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOAT_ARTIFACT = REPO_ROOT / "logs" / "mlx_full_seq_mlp4x_200_realval_vb524k_mlx_model.npz"
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_BASE_FLOOR = 1.0 / 127.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate compressed artifact size for the current int8 serializer, "
            "plus a block-specific soft-floor exception and optional row-subset "
            "keep-float comparison."
        )
    )
    parser.add_argument(
        "--float-artifact",
        type=Path,
        default=DEFAULT_FLOAT_ARTIFACT,
        help="Path to the saved float .npz artifact.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        metavar="SUBSTRING[,SUBSTRING...]",
        help="Tensor name substring to keep as passthrough; repeatable.",
    )
    parser.add_argument(
        "--soft-floor-block",
        action="append",
        default=["0"],
        metavar="BLOCK[,BLOCK...]",
        help="Block index to receive the soft-floor exception; repeatable.",
    )
    parser.add_argument(
        "--soft-floor-alpha",
        type=float,
        default=0.30,
        help=(
            "Interpolation factor between the current floor (1/127) and the "
            "tensor-specific floor estimate."
        ),
    )
    return parser.parse_args()


def load_float_state(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing float artifact: {path}")
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
            u16 = arr.view(np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        out[key] = np.asarray(arr)
    return out


def flatten_patterns(raw_patterns: list[str]) -> list[str]:
    patterns: list[str] = []
    for entry in raw_patterns:
        for pattern in entry.split(","):
            pattern = pattern.strip()
            if pattern:
                patterns.append(pattern)
    return patterns


def flatten_ints(raw_values: list[str]) -> list[int]:
    values: list[int] = []
    for entry in raw_values:
        for value in entry.split(","):
            value = value.strip()
            if value:
                values.append(int(value))
    return sorted(set(values))


def compressor_size(raw: bytes, compressor: str) -> int:
    if compressor == "zlib":
        return len(zlib.compress(raw, level=9))
    if compressor == "lzma":
        return len(lzma.compress(raw, preset=(9 | lzma.PRESET_EXTREME)))
    raise ValueError(f"unsupported compressor: {compressor}")


def keep_float_array(name: str, arr: np.ndarray, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if arr.dtype == np.float32:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(
            np.array(arr.astype(INT8_KEEP_FLOAT_STORE_DTYPE), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False)
        )
    return np.ascontiguousarray(np.array(arr, copy=True))


def tensor_block_index(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
        return int(parts[1])
    return None


def tensor_specific_floor(arr: np.ndarray) -> float:
    f32 = np.asarray(arr, dtype=np.float32)
    if f32.ndim == 2:
        row_clip_abs = (
            np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
            if f32.size
            else np.empty((f32.shape[0],), dtype=np.float32)
        )
        row_scales = row_clip_abs / 127.0
        if row_scales.size:
            return float(np.median(row_scales))
        return INT8_BASE_FLOOR

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    return clip_abs / 127.0 if clip_abs > 0.0 else INT8_BASE_FLOOR


def quantize_float_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.asarray(arr, dtype=np.float32)
    if f32.ndim == 2:
        clip_abs = (
            np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
            if f32.size
            else np.empty((f32.shape[0],), dtype=np.float32)
        )
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, INT8_BASE_FLOOR).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_float_array_soft_floor(arr: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.asarray(arr, dtype=np.float32)
    floor = (1.0 - alpha) * INT8_BASE_FLOOR + alpha * tensor_specific_floor(f32)
    if f32.ndim == 2:
        clip_abs = (
            np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
            if f32.size
            else np.empty((f32.shape[0],), dtype=np.float32)
        )
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, floor).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    raw_scale = clip_abs / 127.0 if clip_abs > 0.0 else 1.0
    scale = np.array(max(raw_scale, floor), dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / float(scale)), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def should_soft_floor(name: str, selected_blocks: set[int]) -> bool:
    block_idx = tensor_block_index(name)
    return block_idx is not None and block_idx in selected_blocks


def quantize_state_dict_int8(flat_state: dict[str, np.ndarray]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not np.issubdtype(arr.dtype, np.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

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
    return obj, stats


def build_selective_passthrough_variant(
    flat_state: dict[str, np.ndarray],
    base_quant_obj: dict[str, object],
    patterns: list[str],
) -> tuple[dict[str, object], list[str]]:
    matched_names = sorted(name for name in flat_state.keys() if any(pattern in name for pattern in patterns))
    moved_names = [name for name in matched_names if name in base_quant_obj["quantized"]]

    variant: dict[str, object] = {
        "__quant_format__": base_quant_obj["__quant_format__"],
        "quantized": dict(base_quant_obj["quantized"]),
        "scales": dict(base_quant_obj["scales"]),
        "dtypes": dict(base_quant_obj["dtypes"]),
        "passthrough": dict(base_quant_obj["passthrough"]),
    }
    if "qmeta" in base_quant_obj:
        variant["qmeta"] = dict(base_quant_obj["qmeta"])
    if "passthrough_orig_dtypes" in base_quant_obj:
        variant["passthrough_orig_dtypes"] = dict(base_quant_obj["passthrough_orig_dtypes"])

    passthrough_orig_dtypes = variant.setdefault("passthrough_orig_dtypes", {})
    for name in moved_names:
        variant["quantized"].pop(name, None)
        variant["scales"].pop(name, None)
        variant["dtypes"].pop(name, None)
        if "qmeta" in variant:
            variant["qmeta"].pop(name, None)
        variant["passthrough"][name] = keep_float_array(name, flat_state[name], passthrough_orig_dtypes)

    return variant, moved_names


def build_block_soft_floor_variant(
    flat_state: dict[str, np.ndarray],
    selected_blocks: set[int],
    alpha: float,
) -> tuple[dict[str, object], list[str]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    soft_floored_names: list[str] = []
    for name, arr in flat_state.items():
        if not np.issubdtype(arr.dtype, np.floating):
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = keep_float_array(name, arr, passthrough_orig_dtypes)
            continue

        if should_soft_floor(name, selected_blocks):
            q, s = quantize_float_array_soft_floor(arr, alpha)
            soft_floored_names.append(name)
        else:
            q, s = quantize_float_array(arr)

        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]

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
    return obj, soft_floored_names


def fmt_delta(value: int) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,}"


def fmt_names(names: list[str], limit: int = 5) -> str:
    if not names:
        return "<none>"
    head = ", ".join(names[:limit])
    if len(names) > limit:
        return f"{head}, ..."
    return head


def print_variant_sizes(
    title: str,
    raw_obj: dict[str, object],
    base_lzma: int,
    base_zlib: int,
    summary_lines: list[str],
) -> tuple[int, int]:
    raw = pickle.dumps(raw_obj, protocol=pickle.HIGHEST_PROTOCOL)
    lzma_size = compressor_size(raw, "lzma")
    zlib_size = compressor_size(raw, "zlib")
    print(title)
    for line in summary_lines:
        print(f"  {line}")
    print(f"  lzma: {lzma_size:,} bytes (delta {fmt_delta(lzma_size - base_lzma)})")
    print(f"  zlib: {zlib_size:,} bytes (delta {fmt_delta(zlib_size - base_zlib)})")
    return lzma_size, zlib_size


def main() -> None:
    ns = parse_args()
    patterns = flatten_patterns(ns.pattern)
    soft_floor_blocks = flatten_ints(ns.soft_floor_block)
    flat_state = load_float_state(ns.float_artifact)

    base_quant_obj, _ = quantize_state_dict_int8(flat_state)
    base_raw = pickle.dumps(base_quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    base_lzma = compressor_size(base_raw, "lzma")
    base_zlib = compressor_size(base_raw, "zlib")

    print(f"float_artifact: {ns.float_artifact}")
    print(f"tensors: {len(flat_state)}")
    print(f"baseline_lzma: {base_lzma:,} bytes")
    print(f"baseline_zlib:  {base_zlib:,} bytes")

    keep_float_lzma: int | None = None
    block_soft_floor_lzma: int | None = None

    if patterns:
        variant_obj, moved_names = build_selective_passthrough_variant(flat_state, base_quant_obj, patterns)
        keep_float_lzma, _ = print_variant_sizes(
            "row_keep_float:",
            variant_obj,
            base_lzma,
            base_zlib,
            [
                f"patterns: {', '.join(patterns)}",
                f"matched_quantized_tensors: {len(moved_names)}",
                f"moved_names: {fmt_names(moved_names)}",
            ],
        )
        print()
    else:
        print("row_keep_float: <skipped>")
        print()

    if soft_floor_blocks:
        variant_obj, soft_floored_names = build_block_soft_floor_variant(
            flat_state,
            set(soft_floor_blocks),
            ns.soft_floor_alpha,
        )
        block_soft_floor_lzma, _ = print_variant_sizes(
            "block_soft_floor:",
            variant_obj,
            base_lzma,
            base_zlib,
            [
                f"blocks: {', '.join(str(block) for block in soft_floor_blocks)}",
                f"alpha: {ns.soft_floor_alpha:.3f}",
                f"soft_floored_tensors: {len(soft_floored_names)}",
                f"soft_floored_names: {fmt_names(soft_floored_names)}",
            ],
        )
    else:
        print("block_soft_floor: <skipped>")

    if keep_float_lzma is not None and block_soft_floor_lzma is not None:
        print()
        print("comparison:")
        print(
            f"  row_keep_float - block_soft_floor: "
            f"{fmt_delta(keep_float_lzma - block_soft_floor_lzma)} bytes"
        )


if __name__ == "__main__":
    main()
