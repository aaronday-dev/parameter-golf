#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt_mlx as pg
from mlx.utils import tree_flatten


DEFAULT_FLOAT_ARTIFACT = REPO_ROOT / "logs" / "mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2_mlx_model.npz"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
ARTIFACT_BUDGET_BYTES = 16_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline budget-recovery sweep for the promoted sacred-tensor artifact. "
            "Preserves one sacred tensor in fp16 and makes one candidate tensor cheaper "
            "via harsher tensor-local clip percentiles."
        )
    )
    parser.add_argument("--float-artifact", type=Path, default=DEFAULT_FLOAT_ARTIFACT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--val-seqs", type=int, default=128)
    parser.add_argument(
        "--candidate-family",
        default="attn.c_q.weight",
        help="Tensor family suffix to sweep, e.g. attn.c_q.weight",
    )
    parser.add_argument(
        "--sacred-tensor",
        default="blocks.0.mlp.proj.weight",
        help="Exact tensor name to keep in fp16 passthrough.",
    )
    parser.add_argument(
        "--clip-percentiles",
        default="99.9,99.5,99.0,98.0,95.0",
        help="Comma-separated harsher clip percentiles to test on each candidate tensor.",
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


def parse_percentiles(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        pct = float(item)
        if not (0.0 < pct <= 100.0):
            raise ValueError(f"invalid clip percentile: {pct}")
        values.append(pct)
    if not values:
        raise ValueError("need at least one clip percentile")
    return values


def load_float_state(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
            u16 = arr.view(np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        out[key] = np.asarray(arr)
    return out


def make_runtime_args(ns: argparse.Namespace) -> pg.Hyperparameters:
    args = pg.Hyperparameters()
    args.data_path = str(ns.data_path)
    args.tokenizer_path = str(ns.tokenizer_path)
    args.vocab_size = ns.vocab_size
    args.num_layers = ns.num_layers
    args.shared_core_blocks = ns.shared_core_blocks
    args.shared_core_schedule = ns.shared_core_schedule
    args.model_dim = ns.model_dim
    args.num_heads = ns.num_heads
    args.num_kv_heads = ns.num_kv_heads
    args.mlp_mult = ns.mlp_mult
    args.train_seq_len = ns.train_seq_len
    args.val_max_batch_tokens = ns.val_seqs * ns.train_seq_len
    args.seed = ns.seed
    args.tie_embeddings = True
    return args


def build_model(args: pg.Hyperparameters) -> pg.GPT:
    return pg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        shared_core_blocks=args.shared_core_blocks,
        shared_core_schedule=args.shared_core_schedule,
        shared_core_custom_schedule=args.shared_core_custom_schedule,
        shared_core_role_gains=args.shared_core_role_gains,
        shared_core_pass_x0=args.shared_core_pass_x0,
        shared_core_revisit_gain=args.shared_core_revisit_gain,
        shared_core_revisit_count_gain=args.shared_core_revisit_count_gain,
        shared_core_phase_split_revisit_gain=args.shared_core_phase_split_revisit_gain,
        shared_core_revisit_damping=args.shared_core_revisit_damping,
        shared_core_correct_block=args.shared_core_correct_block,
        shared_core_correct_gain=args.shared_core_correct_gain,
        shared_core_adaptive_correct=args.shared_core_adaptive_correct,
        shared_core_adaptive_correct_block=args.shared_core_adaptive_correct_block,
        shared_core_adaptive_correct_sources=args.shared_core_adaptive_correct_sources,
        shared_core_adaptive_correct_max_gain=args.shared_core_adaptive_correct_max_gain,
        shared_core_adaptive_correct_target_amp=args.shared_core_adaptive_correct_target_amp,
        shared_core_adaptive_correct_log_band=args.shared_core_adaptive_correct_log_band,
        shared_core_adaptive_correct_revisit_only=args.shared_core_adaptive_correct_revisit_only,
        shared_core_directional_correct=args.shared_core_directional_correct,
        shared_core_directional_correct_block=args.shared_core_directional_correct_block,
        shared_core_directional_correct_sources=args.shared_core_directional_correct_sources,
        shared_core_directional_correct_max_gain=args.shared_core_directional_correct_max_gain,
        shared_core_directional_correct_target_amp=args.shared_core_directional_correct_target_amp,
        shared_core_directional_correct_log_band=args.shared_core_directional_correct_log_band,
        shared_core_directional_correct_revisit_only=args.shared_core_directional_correct_revisit_only,
        shared_core_directional_stress_guard=args.shared_core_directional_stress_guard,
        shared_core_directional_stress_band=args.shared_core_directional_stress_band,
        shared_core_directional_stress_min_factor=args.shared_core_directional_stress_min_factor,
        shared_core_orbit_gate=args.shared_core_orbit_gate,
        shared_core_orbit_gate_block=args.shared_core_orbit_gate_block,
        shared_core_orbit_gate_steps=args.shared_core_orbit_gate_steps,
        shared_core_orbit_gate_revisit_only=args.shared_core_orbit_gate_revisit_only,
        shared_core_attractor_pulse=args.shared_core_attractor_pulse,
        shared_core_attractor_pulse_block=args.shared_core_attractor_pulse_block,
        shared_core_attractor_pulse_steps=args.shared_core_attractor_pulse_steps,
        shared_core_attractor_pulse_gain=args.shared_core_attractor_pulse_gain,
        shared_core_attractor_pulse_trigger_amp=args.shared_core_attractor_pulse_trigger_amp,
        shared_core_attractor_pulse_margin=args.shared_core_attractor_pulse_margin,
        shared_core_attractor_pulse_revisit_only=args.shared_core_attractor_pulse_revisit_only,
        shared_core_mirror_cancel=args.shared_core_mirror_cancel,
        shared_core_mirror_cancel_block=args.shared_core_mirror_cancel_block,
        shared_core_mirror_cancel_threshold=args.shared_core_mirror_cancel_threshold,
        shared_core_mirror_cancel_revisit_only=args.shared_core_mirror_cancel_revisit_only,
        shared_core_stabilize_every=args.shared_core_stabilize_every,
        shared_core_stabilize_after=args.shared_core_stabilize_after,
        shared_core_stabilize_gain=args.shared_core_stabilize_gain,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        bigram_hash_on=args.bigram_hash_on,
        bigram_hash_bins=args.bigram_hash_bins,
        bigram_hash_init=args.bigram_hash_init,
        qk_gain_init=args.qk_gain_init,
    )


def typed_state_from_npz(model: pg.GPT, float_state_np: dict[str, np.ndarray]) -> dict[str, pg.mx.array]:
    reference = dict(tree_flatten(model.state))
    out: dict[str, pg.mx.array] = {}
    for name, ref in reference.items():
        out[name] = pg.mx.array(np.asarray(float_state_np[name]), dtype=ref.dtype)
    return out


def quantize_float_array_clip(arr: pg.mx.array, clip_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.array(arr.astype(pg.mx.float32), dtype=np.float32, copy=False)
    clip_q = clip_percentile / 100.0
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), clip_q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(pg.INT8_PER_ROW_SCALE_DTYPE, copy=False))
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), clip_q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def build_variant_quant_obj(
    flat_state: dict[str, pg.mx.array],
    sacred_tensor: str,
    candidate_tensor: str | None,
    clip_percentile: float | None,
) -> tuple[dict[str, object], dict[str, int]]:
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
        if not pg.mx.issubdtype(arr.dtype, pg.mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if name == sacred_tensor:
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
        if candidate_tensor is not None and name == candidate_tensor and clip_percentile is not None:
            q, s = quantize_float_array_clip(arr, clip_percentile)
        else:
            q, s = pg.quantize_float_array(arr)
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


def evaluate_state(
    model: pg.GPT,
    compiled_loss,
    state_flat: dict[str, pg.mx.array],
    args: pg.Hyperparameters,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> tuple[float, float]:
    model.update(pg.tree_unflatten(list(state_flat.items())))
    return pg.eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=None,
    )


def main() -> None:
    ns = parse_args()
    clip_percentiles = parse_percentiles(ns.clip_percentiles)
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

    baseline_quant_obj, baseline_stats = build_variant_quant_obj(
        float_flat,
        sacred_tensor=ns.sacred_tensor,
        candidate_tensor=None,
        clip_percentile=None,
    )
    baseline_blob = lzma.compress(pickle.dumps(baseline_quant_obj, protocol=pickle.HIGHEST_PROTOCOL), preset=(9 | lzma.PRESET_EXTREME))
    baseline_artifact_bytes = len(baseline_blob)
    baseline_quant_flat = pg.dequantize_state_dict_int8(baseline_quant_obj)
    baseline_loss, baseline_bpb = evaluate_state(
        model,
        compiled_loss,
        baseline_quant_flat,
        args,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    print(f"Baseline keep-float artifact: {baseline_artifact_bytes:,} bytes")
    print(f"Baseline keep-float bpb:      {baseline_bpb:.8f}")

    candidate_tensors = sorted(
        name for name in float_flat.keys() if name.endswith(ns.candidate_family)
    )
    if not candidate_tensors:
        raise ValueError(f"no tensors found for family suffix {ns.candidate_family!r}")

    rows: list[dict[str, object]] = []
    for tensor_name in candidate_tensors:
        for clip_percentile in clip_percentiles:
            print(f"Testing {tensor_name} @ clip {clip_percentile:.4f}")
            quant_obj, _ = build_variant_quant_obj(
                float_flat,
                sacred_tensor=ns.sacred_tensor,
                candidate_tensor=tensor_name,
                clip_percentile=clip_percentile,
            )
            blob = lzma.compress(pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL), preset=(9 | lzma.PRESET_EXTREME))
            artifact_bytes = len(blob)
            quant_flat = pg.dequantize_state_dict_int8(quant_obj)
            val_loss, val_bpb = evaluate_state(
                model,
                compiled_loss,
                quant_flat,
                args,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            row = {
                "tensor": tensor_name,
                "clip_percentile": clip_percentile,
                "artifact_bytes": artifact_bytes,
                "artifact_delta_vs_keepf": artifact_bytes - baseline_artifact_bytes,
                "bytes_over_cap": artifact_bytes - ARTIFACT_BUDGET_BYTES,
                "val_loss": val_loss,
                "val_bpb": val_bpb,
                "bpb_delta_vs_keepf": val_bpb - baseline_bpb,
            }
            rows.append(row)
            print(
                f"  artifact={artifact_bytes:,} "
                f"delta_vs_keepf={row['artifact_delta_vs_keepf']:+,} "
                f"bytes_over_cap={row['bytes_over_cap']:+,} "
                f"val_bpb={val_bpb:.8f} "
                f"delta_vs_keepf={row['bpb_delta_vs_keepf']:+.8f}"
            )

    rows.sort(key=lambda row: (row["bytes_over_cap"] > 0, row["bytes_over_cap"], row["bpb_delta_vs_keepf"]))

    print()
    print("Top candidates")
    print("--------------")
    for row in rows[:10]:
        print(
            f"{row['tensor']} clip={row['clip_percentile']:.4f} "
            f"artifact={row['artifact_bytes']:,} over_cap={row['bytes_over_cap']:+,} "
            f"val_bpb={row['val_bpb']:.8f} d_bpb={row['bpb_delta_vs_keepf']:+.8f}"
        )

    payload = {
        "baseline_keepf": {
            "artifact_bytes": baseline_artifact_bytes,
            "val_bpb": baseline_bpb,
            "payload_bytes": baseline_stats["int8_payload_bytes"],
        },
        "rows": rows,
    }
    if ns.output_json is not None:
        ns.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
