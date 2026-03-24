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
from mlx.utils import tree_flatten, tree_unflatten


DEFAULT_FLOAT_ARTIFACT = REPO_ROOT / "logs" / "mlx_full_seq_mlp4x_200_realval_vb524k_mlx_model.npz"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024_smoke"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure tensor and tensor-family sacredness via float restoration against the int8 roundtrip baseline."
    )
    parser.add_argument(
        "--float-artifact",
        type=Path,
        default=DEFAULT_FLOAT_ARTIFACT,
        help="Path to the saved float .npz artifact.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Dataset root containing fineweb_val_*.bin shards.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="SentencePiece tokenizer .model path.",
    )
    parser.add_argument(
        "--val-seqs",
        type=int,
        default=128,
        help="Number of validation sequences to use for sacredness evaluation.",
    )
    parser.add_argument(
        "--top-families",
        type=int,
        default=3,
        help="How many top families to expand into single-tensor measurements.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the analysis as JSON.",
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
    missing = sorted(set(reference) - set(float_state_np))
    extra = sorted(set(float_state_np) - set(reference))
    if missing or extra:
        raise ValueError(
            "artifact/model state mismatch: "
            f"missing={missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}"
        )
    out: dict[str, pg.mx.array] = {}
    for name, ref in reference.items():
        out[name] = pg.mx.array(np.asarray(float_state_np[name]), dtype=ref.dtype)
    return out


def tensor_family(name: str) -> str:
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
        return ".".join(parts[2:])
    return name


def tensor_byte_cost(name: str, quant_obj: dict[str, object]) -> int:
    if name in quant_obj["quantized"]:
        q = np.asarray(quant_obj["quantized"][name])
        s = np.asarray(quant_obj["scales"][name])
        return int(q.nbytes + s.nbytes)
    if name in quant_obj["passthrough"]:
        return int(np.asarray(quant_obj["passthrough"][name]).nbytes)
    return 0


def tensor_rel_error(name: str, float_flat: dict[str, pg.mx.array], quant_flat: dict[str, pg.mx.array]) -> float:
    f32 = np.asarray(float_flat[name].astype(pg.mx.float32))
    q32 = np.asarray(quant_flat[name].astype(pg.mx.float32))
    diff = f32 - q32
    numer = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(f32 * f32)))
    return numer / max(denom, 1e-12)


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
    model.update(tree_unflatten(list(state_flat.items())))
    return pg.eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=None,
    )


def measure_variant(
    model: pg.GPT,
    compiled_loss,
    names: list[str],
    quant_flat: dict[str, pg.mx.array],
    float_flat: dict[str, pg.mx.array],
    args: pg.Hyperparameters,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    quant_loss: float,
    quant_bpb: float,
    quant_obj: dict[str, object],
) -> dict[str, object]:
    variant_flat = dict(quant_flat)
    for name in names:
        variant_flat[name] = float_flat[name]
    val_loss, val_bpb = evaluate_state(
        model,
        compiled_loss,
        variant_flat,
        args,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    byte_cost = sum(tensor_byte_cost(name, quant_obj) for name in names)
    restored_mb = byte_cost / 1_000_000.0
    rel_errors = [tensor_rel_error(name, float_flat, quant_flat) for name in names]
    return {
        "names": names,
        "member_count": len(names),
        "byte_cost": byte_cost,
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "loss_delta": quant_loss - val_loss,
        "bpb_delta": quant_bpb - val_bpb,
        "bpb_gain_per_mb_restored": (quant_bpb - val_bpb) / restored_mb if restored_mb > 0 else 0.0,
        "mean_rel_q_error": float(np.mean(rel_errors)) if rel_errors else 0.0,
        "max_rel_q_error": float(np.max(rel_errors)) if rel_errors else 0.0,
    }


def print_ranked_table(title: str, rows: list[dict[str, object]], key_name: str) -> None:
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'Rank':<5} {key_name:<36} {'Members':>7} {'Bytes':>10} "
        f"{'dLoss':>10} {'dBpb':>10} {'dBpb/MB':>12} {'MeanRelErr':>11}"
    )
    for idx, row in enumerate(rows, start=1):
        label = row[key_name]
        if len(label) > 36:
            label = label[:33] + "..."
        print(
            f"{idx:<5} {label:<36} {row['member_count']:>7} {row['byte_cost']:>10,} "
            f"{row['loss_delta']:>10.6f} {row['bpb_delta']:>10.6f} "
            f"{row['bpb_gain_per_mb_restored']:>12.6f} {row['mean_rel_q_error']:>11.6f}"
        )


def main() -> None:
    ns = parse_args()
    args = make_runtime_args(ns)
    if not ns.float_artifact.is_file():
        raise FileNotFoundError(f"missing float artifact: {ns.float_artifact}")
    if ns.val_seqs <= 0:
        raise ValueError("--val-seqs must be positive")

    print(f"Loading float artifact: {ns.float_artifact}")
    float_state_np = load_float_state(ns.float_artifact)

    dataset_name, _, _ = pg.validate_dataset_tokenizer_pair(str(ns.data_path), str(ns.tokenizer_path))
    print(f"Dataset: {dataset_name}")
    sp = spm.SentencePieceProcessor(model_file=str(ns.tokenizer_path))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = pg.build_sentencepiece_luts(sp, args.vocab_size)

    full_val_tokens = pg.load_validation_tokens(args.val_files, args.train_seq_len)
    needed_tokens = ns.val_seqs * args.train_seq_len + 1
    if full_val_tokens.size < needed_tokens:
        raise ValueError(
            f"validation set too small for {ns.val_seqs} sequences: need {needed_tokens} tokens, got {full_val_tokens.size}"
        )
    val_tokens = full_val_tokens[:needed_tokens]
    print(f"Validation slice: {ns.val_seqs} seqs / {needed_tokens - 1} tokens")

    pg.mx.random.seed(args.seed)
    model = build_model(args)
    float_flat = typed_state_from_npz(model, float_state_np)
    compiled_loss = pg.mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)

    print("Evaluating float baseline...")
    float_loss, float_bpb = evaluate_state(
        model,
        compiled_loss,
        float_flat,
        args,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    print("Building quantized roundtrip baseline...")
    quant_obj, quant_stats = pg.quantize_state_dict_int8(float_flat)
    quant_flat = pg.dequantize_state_dict_int8(quant_obj)
    quant_loss, quant_bpb = evaluate_state(
        model,
        compiled_loss,
        quant_flat,
        args,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )

    print()
    print("Baselines")
    print("---------")
    print(f"float_val_loss: {float_loss:.8f}")
    print(f"float_val_bpb:  {float_bpb:.8f}")
    print(f"quant_val_loss: {quant_loss:.8f}")
    print(f"quant_val_bpb:  {quant_bpb:.8f}")
    print(f"roundtrip_penalty_loss: {quant_loss - float_loss:.8f}")
    print(f"roundtrip_penalty_bpb:  {quant_bpb - float_bpb:.8f}")
    print(f"int8_payload_bytes: {quant_stats['int8_payload_bytes']:,}")

    families: dict[str, list[str]] = {}
    for name in float_flat:
        families.setdefault(tensor_family(name), []).append(name)

    family_rows: list[dict[str, object]] = []
    print()
    print(f"Evaluating {len(families)} tensor families...")
    for family_name in sorted(families):
        row = measure_variant(
            model,
            compiled_loss,
            names=sorted(families[family_name]),
            quant_flat=quant_flat,
            float_flat=float_flat,
            args=args,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            quant_loss=quant_loss,
            quant_bpb=quant_bpb,
            quant_obj=quant_obj,
        )
        row["family"] = family_name
        family_rows.append(row)
    family_rows.sort(key=lambda row: row["bpb_delta"], reverse=True)
    print_ranked_table("Family Sacredness", family_rows, "family")

    top_family_names = [row["family"] for row in family_rows[: max(ns.top_families, 0)]]
    tensor_rows: list[dict[str, object]] = []
    if top_family_names:
        print()
        print(f"Evaluating individual tensors inside top {len(top_family_names)} families...")
    for family_name in top_family_names:
        for tensor_name in sorted(families[family_name]):
            row = measure_variant(
                model,
                compiled_loss,
                names=[tensor_name],
                quant_flat=quant_flat,
                float_flat=float_flat,
                args=args,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                quant_loss=quant_loss,
                quant_bpb=quant_bpb,
                quant_obj=quant_obj,
            )
            row["tensor"] = tensor_name
            row["family"] = family_name
            tensor_rows.append(row)
    tensor_rows.sort(key=lambda row: row["bpb_delta"], reverse=True)
    if tensor_rows:
        print_ranked_table("Single-Tensor Sacredness", tensor_rows, "tensor")

    if ns.output_json is not None:
        ns.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "float_artifact": str(ns.float_artifact),
            "data_path": str(ns.data_path),
            "tokenizer_path": str(ns.tokenizer_path),
            "val_seqs": ns.val_seqs,
            "float_baseline": {"val_loss": float_loss, "val_bpb": float_bpb},
            "quant_baseline": {
                "val_loss": quant_loss,
                "val_bpb": quant_bpb,
                "payload_bytes": quant_stats["int8_payload_bytes"],
            },
            "family_rows": family_rows,
            "tensor_rows": tensor_rows,
        }
        with ns.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print()
        print(f"Wrote JSON report: {ns.output_json}")


if __name__ == "__main__":
    main()
