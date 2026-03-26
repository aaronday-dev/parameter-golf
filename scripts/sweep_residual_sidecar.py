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
            "Offline sweep for a low-rank residual sidecar on one sacred tensor. "
            "Builds a normal int8 artifact, then restores part of the tensor-specific "
            "quantization residual through a tiny fp16 low-rank carrier."
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
        help="Skip baseline evals and only exact-evaluate the requested residual ranks.",
    )
    parser.add_argument(
        "--target-tensor",
        default="blocks.0.mlp.proj.weight",
        help="Exact tensor name to carry with a low-rank residual sidecar.",
    )
    parser.add_argument(
        "--ranks",
        default="32,64,96,128,160,192,224",
        help="Comma-separated ranks to test.",
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


def parse_ranks(raw: str) -> list[int]:
    ranks: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        rank = int(item)
        if rank <= 0:
            raise ValueError(f"invalid rank: {rank}")
        ranks.append(rank)
    if not ranks:
        raise ValueError("need at least one rank")
    return ranks


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
    args.val_max_batch_tokens = ns.val_max_batch_tokens
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
        qk_gain_init=args.qk_gain_init,
    )


def typed_state_from_npz(model: pg.GPT, float_state_np: dict[str, np.ndarray]) -> dict[str, pg.mx.array]:
    reference = dict(tree_flatten(model.state))
    out: dict[str, pg.mx.array] = {}
    for name, ref in reference.items():
        out[name] = pg.mx.array(np.asarray(float_state_np[name]), dtype=ref.dtype)
    return out


def evaluate_state(
    model: pg.GPT,
    compiled_loss,
    compiled_token_losses,
    state_flat: dict[str, pg.mx.array],
    args: pg.Hyperparameters,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_prefix: str | None = None,
) -> tuple[float, float]:
    model.update(pg.tree_unflatten(list(state_flat.items())))
    log_fn = None
    if log_prefix is not None:
        def _log(msg: str) -> None:
            print(f"{log_prefix}{msg}", flush=True)
        log_fn = _log
    return pg.eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        compiled_token_losses=compiled_token_losses,
        log_fn=log_fn,
    )


def trunc_svd_sidecar(residual: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    rank = min(rank, s.shape[0])
    root_s = np.sqrt(s[:rank], dtype=np.float32)
    left = (u[:, :rank] * root_s[None, :]).astype(np.float16, copy=False)
    right = (root_s[:, None] * vt[:rank, :]).astype(np.float16, copy=False)
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def build_quant_obj(
    flat_state: dict[str, pg.mx.array],
    target_tensor: str,
    keep_float_target: bool,
    residual_sidecar: tuple[np.ndarray, np.ndarray] | None,
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
    if residual_sidecar is not None:
        left, right = residual_sidecar
        sidecars[target_tensor] = {
            "scheme": "residual_low_rank_v1",
            "left": left,
            "right": right,
        }
        stats["int8_payload_bytes"] += int(left.nbytes + right.nbytes)
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


def compress_bytes(quant_obj: dict[str, object]) -> int:
    return len(lzma.compress(pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL), preset=(9 | lzma.PRESET_EXTREME)))


def main() -> None:
    ns = parse_args()
    ranks = parse_ranks(ns.ranks)
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
    compiled_token_losses = pg.mx.compile(lambda x, y: model.token_losses(x, y), inputs=model.state, outputs=model.state)

    keepf_artifact_bytes: int | None = None
    keepf_bpb: float | None = None
    keepf_stats: dict[str, int] | None = None
    if not ns.candidate_only:
        keepf_obj, keepf_stats = build_quant_obj(float_flat, ns.target_tensor, keep_float_target=True, residual_sidecar=None)
        keepf_artifact_bytes = compress_bytes(keepf_obj)
        keepf_flat = pg.dequantize_state_dict_int8(keepf_obj)
        _, keepf_bpb = evaluate_state(
            model,
            compiled_loss,
            compiled_token_losses,
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

    plain_obj, plain_stats = build_quant_obj(float_flat, ns.target_tensor, keep_float_target=False, residual_sidecar=None)
    plain_artifact_bytes = compress_bytes(plain_obj)
    plain_bpb: float | None = None
    if not ns.candidate_only:
        plain_flat = pg.dequantize_state_dict_int8(plain_obj)
        _, plain_bpb = evaluate_state(
            model,
            compiled_loss,
            compiled_token_losses,
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
    for rank in ranks:
        print(f"Testing residual rank {rank}")
        left, right = trunc_svd_sidecar(residual, rank)
        quant_obj, stats = build_quant_obj(
            float_flat,
            ns.target_tensor,
            keep_float_target=False,
            residual_sidecar=(left, right),
        )
        artifact_bytes = compress_bytes(quant_obj)
        quant_flat = pg.dequantize_state_dict_int8(quant_obj)
        _, val_bpb = evaluate_state(
            model,
            compiled_loss,
            compiled_token_losses,
            quant_flat,
            args,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_prefix=f"[rank {rank}] ",
        )
        row = {
            "rank": rank,
            "artifact_bytes": artifact_bytes,
            "artifact_delta_vs_keepf": None if keepf_artifact_bytes is None else artifact_bytes - keepf_artifact_bytes,
            "artifact_delta_vs_plain": artifact_bytes - plain_artifact_bytes,
            "bytes_over_cap": artifact_bytes - ARTIFACT_BUDGET_BYTES,
            "val_bpb": val_bpb,
            "bpb_delta_vs_keepf": None if keepf_bpb is None else val_bpb - keepf_bpb,
            "bpb_delta_vs_plain": None if plain_bpb is None else val_bpb - plain_bpb,
            "sidecar_raw_bytes": int(left.nbytes + right.nbytes),
            "payload_bytes": stats["int8_payload_bytes"],
        }
        rows.append(row)
        keepf_text = "n/a" if row["bpb_delta_vs_keepf"] is None else f"{row['bpb_delta_vs_keepf']:+.8f}"
        plain_text = "n/a" if row["bpb_delta_vs_plain"] is None else f"{row['bpb_delta_vs_plain']:+.8f}"
        print(
            f"  artifact={artifact_bytes:,} over_cap={row['bytes_over_cap']:+,} "
            f"val_bpb={val_bpb:.8f} "
            f"d_keepf={keepf_text} "
            f"d_plain={plain_text}"
        )

    rows.sort(key=lambda row: (row["bytes_over_cap"] > 0, row["bytes_over_cap"], float("inf") if row["bpb_delta_vs_keepf"] is None else row["bpb_delta_vs_keepf"]))

    print()
    print("Top candidates")
    print("--------------")
    for row in rows[:10]:
        keepf_text = "n/a" if row["bpb_delta_vs_keepf"] is None else f"{row['bpb_delta_vs_keepf']:+.8f}"
        plain_text = "n/a" if row["bpb_delta_vs_plain"] is None else f"{row['bpb_delta_vs_plain']:+.8f}"
        print(
            f"rank={row['rank']} artifact={row['artifact_bytes']:,} "
            f"over_cap={row['bytes_over_cap']:+,} val_bpb={row['val_bpb']:.8f} "
            f"d_keepf={keepf_text} d_plain={plain_text}"
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
