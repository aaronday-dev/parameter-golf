#!/usr/bin/env python3
"""
Tensor Salience Atlas for Parameter Golf.

Loads the float (.npz) and quantized (.ptx/.ptz) artifacts for a run,
computes per-tensor and per-row-family salience metrics, and ranks
"sacred" tensors — those too important to treat generically under int8+lzma.

Usage:
    python scripts/tensor_salience_atlas.py

Expects artifacts at the paths hardcoded below (best full run).
"""
from __future__ import annotations

import lzma
import math
import os
import pickle
import sys
import zlib

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOAT_PATH = os.path.join(BASE, "logs", "mlx_full_seq_mlp4x_200_realval_vb524k_mlx_model.npz")
# Try lzma first, fall back to zlib
QUANT_PTX = os.path.join(BASE, "logs", "mlx_full_seq_mlp4x_200_realval_vb524k_mlx_model.int8.ptx")
QUANT_PTZ = os.path.join(BASE, "logs", "mlx_full_seq_mlp4x_200_realval_vb524k_mlx_model.int8.ptz")

INT8_CLIP_Q = 99.99984 / 100.0


# ── load helpers ─────────────────────────────────────────────────────────────

def load_float_state(path: str) -> dict[str, np.ndarray]:
    """Load float .npz → flat dict of numpy f32 arrays.
    Handles bfloat16 stored as 2-byte void (|V2) by mlx's npz writer."""
    data = np.load(path)
    out = {}
    for k in data.files:
        arr = data[k]
        if arr.dtype.kind == 'V' and arr.dtype.itemsize == 2:
            # bfloat16 stored as void: reinterpret as uint16, then convert
            u16 = arr.view(np.uint16).astype(np.uint32)
            f32_bits = u16 << 16
            arr = f32_bits.view(np.float32)
        out[k] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    return out


def load_quant_obj(ptx_path: str, ptz_path: str) -> dict:
    """Load the pickled int8 quant object from .ptx (lzma) or .ptz (zlib)."""
    for path, decompress in [(ptx_path, lzma.decompress), (ptz_path, zlib.decompress)]:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                blob = f.read()
            raw = decompress(blob)
            return pickle.loads(raw)
    raise FileNotFoundError(f"No quant artifact found at {ptx_path} or {ptz_path}")


def dequantize_tensor(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Reproduce the dequantization path from train_gpt_mlx.py."""
    q_f32 = q.astype(np.float32)
    if scale.ndim > 0:
        return q_f32 * scale.astype(np.float32).reshape((q.shape[0],) + (1,) * (q.ndim - 1))
    return q_f32 * float(scale)


def simulate_quantize(f32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-quantize a float tensor using the same algorithm as train_gpt_mlx.py.
    Returns (q_int8, scale, dequantized_f32)."""
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8)
        deq = q.astype(np.float32) * scale[:, None]
        return q, scale, deq
    else:
        clip_abs = float(np.quantile(np.abs(f32).ravel(), INT8_CLIP_Q)) if f32.size else 0.0
        scale_val = clip_abs / 127.0 if clip_abs > 0.0 else 1.0
        scale = np.array(scale_val, dtype=np.float32)
        q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale_val), -127, 127).astype(np.int8)
        deq = q.astype(np.float32) * scale_val
        return q, scale, deq


def byte_entropy(data: np.ndarray) -> float:
    """Shannon entropy of the byte stream (bits per byte). Max = 8.0."""
    raw = data.tobytes()
    if len(raw) == 0:
        return 0.0
    counts = np.zeros(256, dtype=np.int64)
    for b in raw:
        counts[b] += 1
    probs = counts[counts > 0] / len(raw)
    return float(-np.sum(probs * np.log2(probs)))


# ── main atlas ───────────────────────────────────────────────────────────────

def build_atlas():
    print("Loading float artifact...")
    float_state = load_float_state(FLOAT_PATH)
    print(f"  {len(float_state)} tensors loaded")

    print("Loading quantized artifact...")
    quant_obj = load_quant_obj(QUANT_PTX, QUANT_PTZ)
    quantized = quant_obj.get("quantized", {})
    scales = quant_obj.get("scales", {})
    passthrough = quant_obj.get("passthrough", {})
    qmeta = quant_obj.get("qmeta", {})
    print(f"  {len(quantized)} quantized, {len(passthrough)} passthrough")

    # Total payload for byte-share computation
    total_payload_bytes = 0
    for name, q in quantized.items():
        total_payload_bytes += np.asarray(q).nbytes
        s = scales.get(name)
        if s is not None:
            total_payload_bytes += np.asarray(s).nbytes
    for name, arr in passthrough.items():
        total_payload_bytes += np.asarray(arr).nbytes

    print(f"  Total int8 payload: {total_payload_bytes:,} bytes")
    print()

    records = []

    # ── Process quantized tensors ────────────────────────────────────────
    for name in sorted(quantized.keys()):
        q_np = np.asarray(quantized[name], dtype=np.int8)
        s_np = np.asarray(scales[name], dtype=np.float32)
        is_per_row = s_np.ndim > 0

        # Get corresponding float tensor
        f32 = float_state.get(name)
        if f32 is None:
            continue
        f32 = f32.astype(np.float32)

        # Dequantize
        deq = dequantize_tensor(q_np, s_np)

        # Basic stats
        numel = int(q_np.size)
        q_bytes = q_np.nbytes + s_np.nbytes
        byte_share = q_bytes / total_payload_bytes

        # Quantized code stats
        zero_frac = float(np.mean(q_np == 0))
        small_frac = float(np.mean(np.abs(q_np) <= 2))

        # Quantization error
        abs_err = np.abs(f32 - deq)
        rel_err_numer = np.sqrt(np.mean(abs_err ** 2))
        rel_err_denom = np.sqrt(np.mean(f32 ** 2))
        rms_rel_err = rel_err_numer / max(rel_err_denom, 1e-12)

        # Max absolute error
        max_abs_err = float(np.max(abs_err))

        # Per-row scale stats (only for 2D per-row)
        scale_mean = scale_std = scale_cv = scale_max_med = 0.0
        if is_per_row and s_np.size > 1:
            s_f32 = s_np.astype(np.float32)
            scale_mean = float(np.mean(s_f32))
            scale_std = float(np.std(s_f32))
            scale_cv = scale_std / max(scale_mean, 1e-12)
            med = float(np.median(s_f32))
            scale_max_med = float(np.max(s_f32)) / max(med, 1e-12)

        # Per-row quantization error (for row-family analysis)
        row_rms_errors = None
        if f32.ndim == 2:
            row_abs_err = np.sqrt(np.mean((f32 - deq) ** 2, axis=1))
            row_rms_f32 = np.sqrt(np.mean(f32 ** 2, axis=1))
            row_rel_err = row_abs_err / np.maximum(row_rms_f32, 1e-12)
            row_rms_errors = row_rel_err

        # Byte entropy of quantized codes
        ent = byte_entropy(q_np)

        # --- Sacredness score ---
        # Combines:
        #   1. byte_share: how much of the artifact this tensor occupies
        #   2. rms_rel_err: how badly quantization damages it
        #   3. scale_cv: how uneven the per-row treatment is (row stress)
        #   4. entropy: high entropy = hard for lzma to compress = expensive bytes
        #
        # Sacredness = byte_share * (rms_rel_err * w1 + scale_cv * w2 + entropy/8 * w3)
        # where w1, w2, w3 are weights emphasizing error > stress > compressibility
        sacredness = byte_share * (
            3.0 * rms_rel_err +
            1.0 * scale_cv +
            0.5 * (ent / 8.0)
        )

        rec = {
            "name": name,
            "shape": tuple(q_np.shape),
            "numel": numel,
            "quantized": True,
            "per_row": is_per_row,
            "raw_bytes": int(f32.nbytes),
            "int8_bytes": q_bytes,
            "byte_share": byte_share,
            "zero_frac": zero_frac,
            "small_frac": small_frac,
            "rms_rel_err": rms_rel_err,
            "max_abs_err": max_abs_err,
            "scale_mean": scale_mean,
            "scale_std": scale_std,
            "scale_cv": scale_cv,
            "scale_max_med": scale_max_med,
            "entropy": ent,
            "sacredness": sacredness,
            "row_rms_errors": row_rms_errors,
        }
        records.append(rec)

    # ── Process passthrough tensors ──────────────────────────────────────
    for name in sorted(passthrough.keys()):
        arr = np.asarray(passthrough[name])
        f32 = float_state.get(name)
        numel = int(arr.size)
        pt_bytes = arr.nbytes
        byte_share = pt_bytes / total_payload_bytes

        # For passthrough fp16 tensors, compute the fp32→fp16 error
        rms_rel_err = 0.0
        if f32 is not None and arr.dtype == np.float16:
            f32_full = f32.astype(np.float32)
            deq = arr.astype(np.float32)
            abs_err = np.abs(f32_full - deq)
            rms_num = np.sqrt(np.mean(abs_err ** 2))
            rms_den = np.sqrt(np.mean(f32_full ** 2))
            rms_rel_err = rms_num / max(rms_den, 1e-12)

        ent = byte_entropy(arr)

        rec = {
            "name": name,
            "shape": tuple(arr.shape),
            "numel": numel,
            "quantized": False,
            "per_row": False,
            "raw_bytes": int(f32.nbytes) if f32 is not None else pt_bytes,
            "int8_bytes": pt_bytes,
            "byte_share": byte_share,
            "zero_frac": 0.0,
            "small_frac": 0.0,
            "rms_rel_err": rms_rel_err,
            "max_abs_err": 0.0,
            "scale_mean": 0.0,
            "scale_std": 0.0,
            "scale_cv": 0.0,
            "scale_max_med": 0.0,
            "entropy": ent,
            "sacredness": byte_share * (3.0 * rms_rel_err + 0.5 * (ent / 8.0)),
            "row_rms_errors": None,
        }
        records.append(rec)

    # ── Sort by sacredness ───────────────────────────────────────────────
    records.sort(key=lambda r: r["sacredness"], reverse=True)

    # ── Print full atlas ─────────────────────────────────────────────────
    print("=" * 120)
    print("TENSOR SALIENCE ATLAS — ranked by sacredness score")
    print("=" * 120)
    print()
    print(f"{'Rank':<5} {'Sacredness':>10} {'Name':<50} {'Shape':<20} "
          f"{'Numel':>10} {'Q?':>3} {'ByteShare':>9} {'RMSRelErr':>10} "
          f"{'ScaleCV':>8} {'Entropy':>8} {'ZeroFr':>7} {'SmallFr':>8}")
    print("-" * 120)

    for i, r in enumerate(records):
        q_flag = "Q" if r["quantized"] else "P"
        print(f"{i+1:<5} {r['sacredness']:>10.6f} {r['name']:<50} {str(r['shape']):<20} "
              f"{r['numel']:>10,} {q_flag:>3} {r['byte_share']:>9.4f} {r['rms_rel_err']:>10.6f} "
              f"{r['scale_cv']:>8.4f} {r['entropy']:>8.4f} {r['zero_frac']:>7.3f} {r['small_frac']:>8.3f}")

    # ── Top 10 detailed breakdown ────────────────────────────────────────
    print()
    print("=" * 120)
    print("TOP 10 — DETAILED BREAKDOWN")
    print("=" * 120)

    for i, r in enumerate(records[:10]):
        print()
        print(f"  #{i+1}  {r['name']}")
        print(f"       shape: {r['shape']}  numel: {r['numel']:,}  "
              f"{'QUANTIZED (per-row)' if r['per_row'] else 'QUANTIZED (per-tensor)' if r['quantized'] else 'PASSTHROUGH'}")
        print(f"       raw_bytes: {r['raw_bytes']:,}  int8_bytes: {r['int8_bytes']:,}  "
              f"byte_share: {r['byte_share']:.4f} ({r['byte_share']*100:.1f}%)")
        print(f"       rms_rel_err: {r['rms_rel_err']:.6f}  max_abs_err: {r['max_abs_err']:.6f}")
        print(f"       zero_frac: {r['zero_frac']:.3f}  small_frac (|q|≤2): {r['small_frac']:.3f}")
        print(f"       entropy: {r['entropy']:.4f} bits/byte")
        if r["per_row"]:
            print(f"       scale_mean: {r['scale_mean']:.6f}  scale_std: {r['scale_std']:.6f}  "
                  f"scale_cv: {r['scale_cv']:.4f}  scale_max/median: {r['scale_max_med']:.2f}")

        print(f"       SACREDNESS: {r['sacredness']:.6f}")

        # Row-family analysis for top quantized tensors
        if r["row_rms_errors"] is not None and r["quantized"]:
            errs = r["row_rms_errors"]
            n_rows = len(errs)
            # Find the top-stressed rows
            top_k = min(20, n_rows)
            top_idx = np.argsort(errs)[-top_k:][::-1]
            p90 = np.percentile(errs, 90)
            p99 = np.percentile(errs, 99)
            n_above_2x_median = int(np.sum(errs > 2 * np.median(errs)))

            print(f"       ROW ANALYSIS ({n_rows} rows):")
            print(f"         error p50: {np.median(errs):.6f}  p90: {p90:.6f}  p99: {p99:.6f}  "
                  f"max: {np.max(errs):.6f}")
            print(f"         rows > 2× median error: {n_above_2x_median} ({n_above_2x_median/n_rows*100:.1f}%)")
            print(f"         top 5 stressed rows: {list(top_idx[:5])} "
                  f"with rel_err: [{', '.join(f'{errs[j]:.4f}' for j in top_idx[:5])}]")

            # Check if a small row subset dominates
            sorted_errs = np.sort(errs)[::-1]
            top10_share = float(np.sum(sorted_errs[:max(1, n_rows//10)]) / np.sum(sorted_errs))
            print(f"         top 10% rows share of total error: {top10_share:.1%}")

            # Get the float scale for the top-stressed rows
            if r["per_row"] and r["name"] in scales:
                s_np = np.asarray(scales[r["name"]], dtype=np.float32)
                top5_scales = [float(s_np[j]) for j in top_idx[:5]]
                print(f"         top 5 row scales: [{', '.join(f'{s:.6f}' for s in top5_scales)}]")

    # ── Semantic grouping: aggregate by tensor family ────────────────────
    print()
    print("=" * 120)
    print("AGGREGATE BY TENSOR FAMILY")
    print("=" * 120)
    print()

    families = {}
    for r in records:
        name = r["name"]
        # Extract family: e.g. "blocks.3.mlp.proj.weight" → "mlp.proj.weight"
        parts = name.split(".")
        if parts[0] == "blocks" and len(parts) > 2:
            family = ".".join(parts[2:])
        else:
            family = name
        if family not in families:
            families[family] = {"count": 0, "total_bytes": 0, "total_sacredness": 0.0,
                                "total_byte_share": 0.0, "errors": [], "scale_cvs": [],
                                "entropies": [], "members": []}
        f = families[family]
        f["count"] += 1
        f["total_bytes"] += r["int8_bytes"]
        f["total_sacredness"] += r["sacredness"]
        f["total_byte_share"] += r["byte_share"]
        f["errors"].append(r["rms_rel_err"])
        f["scale_cvs"].append(r["scale_cv"])
        f["entropies"].append(r["entropy"])
        f["members"].append(r["name"])

    fam_list = sorted(families.items(), key=lambda x: x[1]["total_sacredness"], reverse=True)

    print(f"{'Rank':<5} {'Family':<35} {'Count':>5} {'TotalBytes':>12} {'ByteShare':>10} "
          f"{'MeanRMSErr':>10} {'MeanScaleCV':>11} {'MeanEntropy':>11} {'TotalSacred':>12}")
    print("-" * 120)

    for i, (fam_name, f) in enumerate(fam_list):
        mean_err = np.mean(f["errors"])
        mean_cv = np.mean(f["scale_cvs"])
        mean_ent = np.mean(f["entropies"])
        print(f"{i+1:<5} {fam_name:<35} {f['count']:>5} {f['total_bytes']:>12,} {f['total_byte_share']:>10.4f} "
              f"{mean_err:>10.6f} {mean_cv:>11.4f} {mean_ent:>11.4f} {f['total_sacredness']:>12.6f}")

    print()
    print("DONE.")
    return records, fam_list


if __name__ == "__main__":
    build_atlas()
