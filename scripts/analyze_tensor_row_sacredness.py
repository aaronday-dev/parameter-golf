#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOAT_ARTIFACT = REPO_ROOT / "logs" / "mlx_full_seq_mlp4x_200_realval_vb524k_mlx_model.npz"
INT8_CLIP_Q = 99.99984 / 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one named 2D float tensor and rank its rows by simple proxies "
            "for preservation under the repo's per-row int8 quantizer."
        )
    )
    parser.add_argument(
        "--float-artifact",
        type=Path,
        default=DEFAULT_FLOAT_ARTIFACT,
        help="Path to the saved float .npz artifact.",
    )
    parser.add_argument(
        "--tensor-name",
        required=True,
        help="Exact tensor name to inspect.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="How many top rows to print, ranked by combined score.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Print all rows instead of only the top-k rows.",
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


def quantize_2d_rows(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f32 = np.asarray(arr, dtype=np.float32)
    if f32.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(f32.shape)}")

    if f32.size:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
    else:
        clip_abs = np.empty((f32.shape[0],), dtype=np.float32)
    scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
    clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
    q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
    deq = q.astype(np.float32) * scale[:, None]
    return q, scale, deq


def minmax01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32, copy=False)


def fmt_float(value: float) -> str:
    return f"{value:10.4e}"


def fmt_score(value: float) -> str:
    return f"{value:7.4f}"


def main() -> None:
    ns = parse_args()
    float_state = load_float_state(ns.float_artifact)
    if ns.tensor_name not in float_state:
        available = ", ".join(sorted(float_state.keys())[:20])
        suffix = "" if len(float_state) <= 20 else ", ..."
        raise KeyError(f"tensor not found: {ns.tensor_name!r}. Available: {available}{suffix}")

    tensor = np.asarray(float_state[ns.tensor_name])
    if tensor.ndim != 2:
        raise ValueError(f"tensor {ns.tensor_name!r} must be 2D, got shape {tuple(tensor.shape)}")

    q, _, deq = quantize_2d_rows(tensor)
    f32 = tensor.astype(np.float32, copy=False)
    if tensor.shape[1] == 0:
        rows = int(tensor.shape[0])
        row_norm = np.zeros((rows,), dtype=np.float32)
        row_rms_err = np.zeros((rows,), dtype=np.float32)
        row_rel_err = np.zeros((rows,), dtype=np.float32)
        row_max_code = np.zeros((rows,), dtype=np.int32)
        row_code_usage = np.zeros((rows,), dtype=np.float32)
    else:
        row_norm = np.sqrt(np.mean(f32 * f32, axis=1))
        row_rms_err = np.sqrt(np.mean((f32 - deq) ** 2, axis=1))
        row_rel_err = row_rms_err / np.maximum(row_norm, 1e-12)
        row_max_code = np.max(np.abs(q), axis=1).astype(np.int32, copy=False)
        row_code_usage = row_max_code.astype(np.float32) / 127.0

    norm_score = minmax01(row_norm)
    err_score = minmax01(row_rel_err)
    code_score = minmax01(row_code_usage)
    combined_score = 0.3 * norm_score + 0.5 * err_score + 0.2 * code_score

    order = np.argsort(-combined_score, kind="stable")
    if ns.all_rows or ns.top_k <= 0:
        limit = len(order)
    else:
        limit = min(ns.top_k, len(order))

    row_width = max(3, len(str(max(int(tensor.shape[0]) - 1, 0))))
    print(f"artifact: {ns.float_artifact}")
    print(f"tensor: {ns.tensor_name}")
    print(f"shape: {tuple(tensor.shape)}")
    print(
        "score: 0.3*norm01 + 0.5*rel_err01 + 0.2*code_usage01 "
        "(higher means more worth preserving)"
    )
    print("rank | row |        norm |     rel_rms | max_code | usage | score")

    for rank, row_idx in enumerate(order[:limit], start=1):
        print(
            f"{rank:>4} | {row_idx:>{row_width}d} | "
            f"{fmt_float(float(row_norm[row_idx]))} | "
            f"{fmt_float(float(row_rel_err[row_idx]))} | "
            f"{row_max_code[row_idx]:>8d} | "
            f"{fmt_score(float(row_code_usage[row_idx]))} | "
            f"{fmt_score(float(combined_score[row_idx]))}"
        )

    if limit < len(order):
        print(f"... {len(order) - limit} additional rows omitted; use --all-rows to print every row.")


if __name__ == "__main__":
    main()
