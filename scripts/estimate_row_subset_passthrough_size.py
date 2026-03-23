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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate compressed artifact size when selected rows of one 2D tensor "
            "are stored as fp16 passthrough while the remaining rows stay on the "
            "current per-row int8 quantizer path."
        )
    )
    parser.add_argument(
        "--float-artifact",
        type=Path,
        default=DEFAULT_FLOAT_ARTIFACT,
        help="Path to the saved float .npz artifact.",
    )
    parser.add_argument(
        "--tensor",
        required=True,
        help="Exact tensor name to split into quantized rows plus fp16 passthrough rows.",
    )
    parser.add_argument(
        "--rows",
        action="append",
        default=[],
        metavar="SPEC[,SPEC...]",
        help=(
            "Row selector for the target tensor. Accepts 0-based row indices and "
            "Python-style half-open slices such as 0,3,8:12,-1. Repeatable."
        ),
    )
    return parser.parse_args()


def load_float_state(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing float artifact: {path}")
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
                u16 = arr.view(np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
            out[key] = np.asarray(arr)
    return out


def flatten_specs(raw_specs: list[str]) -> list[str]:
    specs: list[str] = []
    for entry in raw_specs:
        for spec in entry.split(","):
            spec = spec.strip()
            if spec:
                specs.append(spec)
    return specs


def to_fp16_passthrough(arr: np.ndarray) -> np.ndarray:
    f_arr = np.asarray(arr)
    if np.issubdtype(f_arr.dtype, np.floating):
        return np.ascontiguousarray(np.asarray(f_arr, dtype=np.float16))
    return np.ascontiguousarray(np.array(f_arr, copy=True))


def keep_float_array(name: str, arr: np.ndarray, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating) and arr.dtype != np.float16:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.asarray(arr, dtype=INT8_KEEP_FLOAT_STORE_DTYPE))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.asarray(arr, dtype=np.float32)
    if f32.ndim == 2:
        clip_abs = (
            np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
            if f32.size
            else np.empty((f32.shape[0],), dtype=np.float32)
        )
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state: dict[str, np.ndarray]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
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


def compressor_size(raw: bytes, compressor: str) -> int:
    if compressor == "zlib":
        blob = zlib.compress(raw, level=9)
    elif compressor == "lzma":
        blob = lzma.compress(raw, preset=(9 | lzma.PRESET_EXTREME))
    else:
        raise ValueError(f"unsupported compressor: {compressor}")
    return len(blob)


def parse_row_indices(specs: list[str], num_rows: int) -> list[int]:
    row_set: set[int] = set()
    for spec in specs:
        if ":" in spec:
            start_text, end_text = spec.split(":", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else num_rows
            if start < 0:
                start += num_rows
            if end < 0:
                end += num_rows
            if start < 0 or end < 0 or start > num_rows or end > num_rows:
                raise ValueError(f"row slice out of bounds for {num_rows} rows: {spec}")
            if end < start:
                raise ValueError(f"invalid row slice (end before start): {spec}")
            row_set.update(range(start, end))
            continue

        row = int(spec)
        if row < 0:
            row += num_rows
        if row < 0 or row >= num_rows:
            raise ValueError(f"row index out of bounds for {num_rows} rows: {spec}")
        row_set.add(row)

    return sorted(row_set)


def format_row_indices(rows: list[int]) -> str:
    if not rows:
        return "<none>"

    chunks: list[str] = []
    start = prev = rows[0]
    for row in rows[1:]:
        if row == prev + 1:
            prev = row
            continue
        chunks.append(str(start) if start == prev else f"{start}:{prev + 1}")
        start = prev = row
    chunks.append(str(start) if start == prev else f"{start}:{prev + 1}")
    return ",".join(chunks)


def build_row_subset_variant(
    flat_state: dict[str, np.ndarray],
    base_quant_obj: dict[str, object],
    tensor_name: str,
    row_indices: list[int],
) -> dict[str, object]:
    variant: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_row_subset_fp16_v1",
        "quantized": dict(base_quant_obj["quantized"]),
        "scales": dict(base_quant_obj["scales"]),
        "dtypes": dict(base_quant_obj["dtypes"]),
        "passthrough": dict(base_quant_obj["passthrough"]),
    }
    if "qmeta" in base_quant_obj:
        variant["qmeta"] = dict(base_quant_obj["qmeta"])
    if "passthrough_orig_dtypes" in base_quant_obj:
        variant["passthrough_orig_dtypes"] = dict(base_quant_obj["passthrough_orig_dtypes"])

    target = np.asarray(flat_state[tensor_name])
    if target.ndim != 2:
        raise ValueError(f"tensor must be 2D: {tensor_name} has shape {target.shape}")
    if tensor_name not in base_quant_obj["quantized"]:
        raise ValueError(
            f"tensor is not on the current quantizer path (too small or non-float): {tensor_name}"
        )

    row_indices = sorted(set(row_indices))
    all_rows = set(range(int(target.shape[0])))
    row_set = set(row_indices)
    keep_rows = sorted(all_rows - row_set)

    target_q = np.asarray(base_quant_obj["quantized"][tensor_name])
    target_s = np.asarray(base_quant_obj["scales"][tensor_name])

    variant["quantized"][tensor_name] = np.ascontiguousarray(target_q[keep_rows])
    variant["scales"][tensor_name] = np.ascontiguousarray(target_s[keep_rows])
    variant["dtypes"][tensor_name] = str(target.dtype).split(".")[-1]
    if "qmeta" in variant:
        variant["qmeta"][tensor_name] = {"scheme": "per_row", "axis": 0, "row_subset_fp16": True}

    passthrough_rows = to_fp16_passthrough(target[row_indices])
    variant.setdefault("row_passthrough", {})[tensor_name] = passthrough_rows
    variant.setdefault("row_passthrough_rows", {})[tensor_name] = np.asarray(row_indices, dtype=np.int32)
    return variant


def fmt_delta(value: int) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,}"


def main() -> None:
    ns = parse_args()
    flat_state = load_float_state(ns.float_artifact)

    if ns.tensor not in flat_state:
        raise KeyError(f"tensor not found in artifact: {ns.tensor}")

    target = np.asarray(flat_state[ns.tensor])
    if target.ndim != 2:
        raise ValueError(f"tensor must be 2D: {ns.tensor} has shape {target.shape}")

    row_indices = parse_row_indices(flatten_specs(ns.rows), int(target.shape[0]))
    if not row_indices:
        raise ValueError("no rows selected; pass at least one row via --rows")

    base_quant_obj, _ = quantize_state_dict_int8(flat_state)
    base_raw = pickle.dumps(base_quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    base_lzma = compressor_size(base_raw, "lzma")
    base_zlib = compressor_size(base_raw, "zlib")

    variant_obj = build_row_subset_variant(flat_state, base_quant_obj, ns.tensor, row_indices)
    variant_raw = pickle.dumps(variant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    variant_lzma = compressor_size(variant_raw, "lzma")
    variant_zlib = compressor_size(variant_raw, "zlib")

    keep_rows = target.shape[0] - len(row_indices)
    print(f"float_artifact: {ns.float_artifact}")
    print(f"tensor: {ns.tensor}")
    print(f"tensor_shape: {tuple(target.shape)}")
    print(f"selected_rows: {format_row_indices(row_indices)}")
    print(f"selected_row_count: {len(row_indices)}")
    print(f"remaining_row_count: {keep_rows}")
    print(f"baseline_lzma: {base_lzma:,} bytes")
    print(f"baseline_zlib:  {base_zlib:,} bytes")
    print(
        f"lzma: {variant_lzma:,} bytes "
        f"(delta {fmt_delta(variant_lzma - base_lzma)})"
    )
    print(
        f"zlib: {variant_zlib:,} bytes "
        f"(delta {fmt_delta(variant_zlib - base_zlib)})"
    )


if __name__ == "__main__":
    main()
