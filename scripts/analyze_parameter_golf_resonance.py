#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PG_DIR = ROOT
PG_MLX_PATH = PG_DIR / "train_gpt_mlx.py"
DEFAULT_DATA_PATH = PG_DIR / "data" / "datasets" / "fineweb10B_sp1024_smoke"
DEFAULT_RUN_IDS = (
    "mlx_shared3_mirror_cmp",
    "mlx_shared3_cyclic_cmp",
    "mlx_shared1_cyclic_cmp",
    "mlx_shared3_mirror_x0_cmp",
)
GAIN_ACTION_BAND = 0.05
DRIFT_ACTION_BAND = (0.8, 1.25)


def load_pg_module():
    spec = importlib.util.spec_from_file_location("parameter_golf_train_gpt_mlx", PG_MLX_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MLX trainer module from {PG_MLX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PG_DIR))
    spec.loader.exec_module(module)
    return module


def find_match(text: str, pattern: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {label} in log")
    return match.group(1)


def find_optional_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return None if match is None else match.group(1)


def parse_run_metadata(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8")
    pass_x0_match = find_optional_match(text, r"\bpass_x0_scales:(True|False)")
    revisit_match = find_optional_match(text, r"\brevisit_gains:(True|False)")
    revisit_count_match = find_optional_match(text, r"\brevisit_count_gains:(True|False)")
    damping_match = find_optional_match(text, r"\brevisit_damping:([0-9.]+)")
    phase_split_match = find_optional_match(text, r"\bphase_split_revisit_gain:(True|False)")
    custom_schedule_match = find_optional_match(text, r"\bcustom_schedule:([A-Z\-]+)")
    return {
        "vocab_size": int(find_match(text, r"\bvocab_size:(\d+)", "vocab_size")),
        "num_layers": int(find_match(text, r"\blayers:(\d+)", "layers")),
        "dim": int(find_match(text, r"\bdim:(\d+)", "dim")),
        "num_heads": int(find_match(text, r"\bheads:(\d+)", "heads")),
        "num_kv_heads": int(find_match(text, r"\bkv_heads:(\d+)", "kv_heads")),
        "seq_len": int(find_match(text, r"\bseq_len:(\d+)", "seq_len")),
        "shared_core_blocks": int(find_match(text, r"\bunique_blocks:(\d+)", "unique_blocks")),
        "shared_core_schedule": find_match(text, r"\bschedule:([a-z_]+)", "schedule"),
        "shared_core_custom_schedule": "" if custom_schedule_match in {None, "-"} else custom_schedule_match,
        "shared_core_pass_x0": pass_x0_match == "True",
        "shared_core_revisit_gain": revisit_match == "True",
        "shared_core_revisit_count_gain": revisit_count_match == "True",
        "shared_core_revisit_damping": float(damping_match) if damping_match is not None else 0.0,
        "shared_core_phase_split_revisit_gain": phase_split_match == "True",
        "val_bpb": float(find_match(text, r"final_int8_(?:zlib|lzma)_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", "val_bpb")),
    }


def infer_mlp_mult(flat_state: dict[str, object], dim: int) -> int:
    fc_weight = np.array(flat_state["blocks.0.mlp.fc.weight"])
    return int(fc_weight.shape[0] // dim)


def load_model(module, model_path: Path, metadata: dict[str, object]):
    flat_state = module.mx.load(str(model_path))
    args = module.Hyperparameters()
    phase_split_enabled = bool(
        metadata.get("shared_core_phase_split_revisit_gain", False)
        or "revisit_gains_encoder" in flat_state
        or "revisit_gains_decoder" in flat_state
    )
    count_gain_enabled = bool(
        metadata.get("shared_core_revisit_count_gain", False)
        or "revisit_count_gains" in flat_state
    )
    revisit_gain_enabled = bool(
        metadata.get("shared_core_revisit_gain", False)
        or "revisit_gains" in flat_state
        or phase_split_enabled
    )
    model = module.GPT(
        vocab_size=int(metadata["vocab_size"]),
        num_layers=int(metadata["num_layers"]),
        shared_core_blocks=int(metadata["shared_core_blocks"]),
        shared_core_schedule=str(metadata["shared_core_schedule"]),
        shared_core_custom_schedule=str(metadata.get("shared_core_custom_schedule", "")),
        shared_core_pass_x0=bool(metadata["shared_core_pass_x0"]),
        shared_core_revisit_gain=revisit_gain_enabled,
        shared_core_revisit_count_gain=count_gain_enabled,
        shared_core_revisit_damping=float(metadata.get("shared_core_revisit_damping", 0.0)),
        shared_core_phase_split_revisit_gain=phase_split_enabled,
        dim=int(metadata["dim"]),
        num_heads=int(metadata["num_heads"]),
        num_kv_heads=int(metadata["num_kv_heads"]),
        mlp_mult=infer_mlp_mult(flat_state, int(metadata["dim"])),
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    model.update(module.tree_unflatten(list(flat_state.items())))
    return model


def mx_to_np_f32(module, arr) -> np.ndarray:
    return np.array(arr.astype(module.mx.float32), dtype=np.float32, copy=False)


def rms(arr: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(arr, dtype=np.float64), dtype=np.float64)))


def tokenwise_cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    dots = np.sum(a64 * b64, axis=-1)
    a_norm = np.linalg.norm(a64, axis=-1)
    b_norm = np.linalg.norm(b64, axis=-1)
    return float(np.mean(dots / (a_norm * b_norm + 1e-8), dtype=np.float64))


def collect_batch_metrics(module, model, input_ids) -> dict[str, object]:
    x = module.rms_norm(model.tok_emb(input_ids).astype(module.COMPUTE_DTYPE))
    x0 = x
    skips = []
    x0_np = mx_to_np_f32(module, x0)
    prev_pass_np = x0_np
    prev_step_delta = None
    pass_rms = []
    gain = []
    rel_delta = []
    cos_prev = []
    update_cos = []
    block_indices = []
    phases = []

    def record(layer_idx: int, phase: str, x_in_np: np.ndarray, next_x) -> None:
        nonlocal prev_pass_np, prev_step_delta
        curr_np = mx_to_np_f32(module, next_x)
        curr_rms = rms(curr_np)
        prev_pass_rms = rms(prev_pass_np)
        x_in_rms = rms(x_in_np)
        step_delta = curr_np - x_in_np
        delta_rms = rms(step_delta)
        pass_rms.append(curr_rms)
        gain.append(curr_rms / max(prev_pass_rms, 1e-12))
        rel_delta.append(delta_rms / max(x_in_rms, 1e-12))
        cos_prev.append(tokenwise_cosine_mean(prev_pass_np, curr_np))
        if prev_step_delta is not None:
            update_cos.append(tokenwise_cosine_mean(prev_step_delta, step_delta))
        prev_step_delta = step_delta
        prev_pass_np = curr_np
        block_indices.append(model.block_index(layer_idx))
        phases.append(phase)

    for layer_idx in range(model.num_encoder_layers):
        x_in_np = prev_pass_np
        x = model.apply_block(layer_idx, x, x0)
        record(layer_idx, "encoder", x_in_np, x)
        skips.append(x)
    for decoder_idx in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[decoder_idx].astype(x.dtype)[None, None, :] * skips.pop()
        x_in_np = mx_to_np_f32(module, x)
        x = model.apply_block(model.num_encoder_layers + decoder_idx, x, x0)
        record(model.num_encoder_layers + decoder_idx, "decoder", x_in_np, x)

    return {
        "input_rms": rms(x0_np),
        "pass_rms": pass_rms,
        "gain": gain,
        "rel_delta": rel_delta,
        "cos_prev": cos_prev,
        "update_cos": update_cos,
        "encoder_boundary": model.num_encoder_layers,
        "block_indices": block_indices,
        "phases": phases,
    }


def mean_and_std(series: list[list[float]]) -> tuple[list[float], list[float]]:
    arr = np.asarray(series, dtype=np.float64)
    return arr.mean(axis=0).tolist(), arr.std(axis=0, ddof=0).tolist()


def actionability_summary(aggregate: dict[str, object]) -> dict[str, object]:
    gain_mean = np.asarray(aggregate["gain_mean"], dtype=np.float64)
    drift = float(aggregate["pass_rms_mean"][-1] / max(aggregate["pass_rms_mean"][0], 1e-12))
    action_passes = int(np.sum(np.abs(gain_mean - 1.0) >= GAIN_ACTION_BAND))
    return {
        "action_gain_band": [1.0 - GAIN_ACTION_BAND, 1.0 + GAIN_ACTION_BAND],
        "drift_band": list(DRIFT_ACTION_BAND),
        "passes_outside_gain_band": action_passes,
        "drift": drift,
        "gain_worthwhile": action_passes >= 3,
        "drift_worthwhile": drift < DRIFT_ACTION_BAND[0] or drift > DRIFT_ACTION_BAND[1],
    }


def analyze_run(module, run_id: str, data_path: Path, num_batches: int, batch_seqs: int) -> dict[str, object]:
    log_path = PG_DIR / "logs" / f"{run_id}.txt"
    model_path = PG_DIR / "logs" / f"{run_id}_mlx_model.npz"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log file: {log_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    metadata = parse_run_metadata(log_path)
    model = load_model(module, model_path, metadata)
    metadata["shared_core_revisit_gain"] = bool(metadata.get("shared_core_revisit_gain", False) or getattr(model, "revisit_gains", None) is not None)
    metadata["shared_core_revisit_count_gain"] = bool(
        metadata.get("shared_core_revisit_count_gain", False)
        or getattr(model, "revisit_count_gains", None) is not None
    )
    metadata["shared_core_phase_split_revisit_gain"] = bool(
        metadata.get("shared_core_phase_split_revisit_gain", False)
        or getattr(model, "revisit_gains_encoder", None) is not None
        or getattr(model, "revisit_gains_decoder", None) is not None
    )
    seq_len = int(metadata["seq_len"])
    val_tokens = module.load_validation_tokens(str(data_path / "fineweb_val_*.bin"), seq_len)
    total_seqs = (val_tokens.size - 1) // seq_len
    needed_seqs = num_batches * batch_seqs
    if total_seqs < needed_seqs:
        raise ValueError(
            f"Validation split only has {total_seqs} seqs for seq_len={seq_len}, need {needed_seqs}"
        )

    batches = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_seqs * seq_len
        end = start + batch_seqs * seq_len + 1
        chunk = val_tokens[start:end]
        input_ids = module.mx.array(chunk[:-1].reshape(batch_seqs, seq_len), dtype=module.mx.int32)
        batches.append(collect_batch_metrics(module, model, input_ids))

    pass_rms_mean, pass_rms_std = mean_and_std([batch["pass_rms"] for batch in batches])
    gain_mean, gain_std = mean_and_std([batch["gain"] for batch in batches])
    rel_delta_mean, rel_delta_std = mean_and_std([batch["rel_delta"] for batch in batches])
    cos_prev_mean, cos_prev_std = mean_and_std([batch["cos_prev"] for batch in batches])
    update_cos_mean, update_cos_std = mean_and_std([batch["update_cos"] for batch in batches])

    aggregate = {
        "input_rms_mean": float(np.mean([batch["input_rms"] for batch in batches], dtype=np.float64)),
        "pass_rms_mean": pass_rms_mean,
        "pass_rms_std": pass_rms_std,
        "gain_mean": gain_mean,
        "gain_std": gain_std,
        "rel_delta_mean": rel_delta_mean,
        "rel_delta_std": rel_delta_std,
        "cos_prev_mean": cos_prev_mean,
        "cos_prev_std": cos_prev_std,
        "update_cos_mean": update_cos_mean,
        "update_cos_std": update_cos_std,
        "encoder_boundary": int(batches[0]["encoder_boundary"]),
        "block_indices": list(batches[0]["block_indices"]),
        "phases": list(batches[0]["phases"]),
    }
    aggregate["actionability"] = actionability_summary(aggregate)

    return {
        "run_id": run_id,
        "metadata": metadata,
        "aggregate": aggregate,
        "batches": batches,
    }


def print_report(analyses: list[dict[str, object]]) -> None:
    print("run_id val_bpb gain_passes drift mean_rel_delta mean_cos_prev mean_update_cos")
    for analysis in analyses:
        agg = analysis["aggregate"]
        action = agg["actionability"]
        mean_rel_delta = float(np.mean(agg["rel_delta_mean"], dtype=np.float64))
        mean_cos_prev = float(np.mean(agg["cos_prev_mean"], dtype=np.float64))
        mean_update_cos = float(np.mean(agg["update_cos_mean"], dtype=np.float64))
        print(
            f"{analysis['run_id']} "
            f"{analysis['metadata']['val_bpb']:.8f} "
            f"{action['passes_outside_gain_band']} "
            f"{action['drift']:.4f} "
            f"{mean_rel_delta:.4f} "
            f"{mean_cos_prev:.4f} "
            f"{mean_update_cos:.4f}"
        )
    print()
    for analysis in analyses:
        agg = analysis["aggregate"]
        action = agg["actionability"]
        print(f"[{analysis['run_id']}]")
        print(
            f"  val_bpb={analysis['metadata']['val_bpb']:.8f} "
            f"schedule={analysis['metadata']['shared_core_schedule']} "
            f"custom_schedule={analysis['metadata'].get('shared_core_custom_schedule', '') or '-'} "
            f"unique_blocks={analysis['metadata']['shared_core_blocks']} "
            f"pass_x0={analysis['metadata']['shared_core_pass_x0']} "
            f"revisit_gain={analysis['metadata'].get('shared_core_revisit_gain', False)} "
            f"revisit_count_gain={analysis['metadata'].get('shared_core_revisit_count_gain', False)} "
            f"revisit_damping={analysis['metadata'].get('shared_core_revisit_damping', 0.0):.3f} "
            f"phase_split={analysis['metadata'].get('shared_core_phase_split_revisit_gain', False)}"
        )
        print(f"  pass_rms_mean={','.join(f'{x:.4f}' for x in agg['pass_rms_mean'])}")
        print(f"  gain_mean={','.join(f'{x:.4f}' for x in agg['gain_mean'])}")
        print(f"  rel_delta_mean={','.join(f'{x:.4f}' for x in agg['rel_delta_mean'])}")
        print(f"  cos_prev_mean={','.join(f'{x:.4f}' for x in agg['cos_prev_mean'])}")
        print(f"  update_cos_mean={','.join(f'{x:.4f}' for x in agg['update_cos_mean'])}")
        print(
            f"  actionability gain_band={tuple(action['action_gain_band'])} "
            f"passes_outside={action['passes_outside_gain_band']} "
            f"drift={action['drift']:.4f} drift_band={tuple(action['drift_band'])}"
        )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-pass resonance diagnostics for Parameter Golf MLX runs.")
    parser.add_argument("--run-id", action="append", dest="run_ids", help="Run ID to analyze. Repeat to compare runs.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Dataset directory containing fineweb_val_*.bin.")
    parser.add_argument("--num-batches", type=int, default=4, help="Number of fixed validation batches to analyze.")
    parser.add_argument("--batch-seqs", type=int, default=8, help="Number of sequences per fixed batch.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save full JSON analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_pg_module()
    run_ids = args.run_ids or list(DEFAULT_RUN_IDS)
    analyses = [analyze_run(module, run_id, args.data_path, args.num_batches, args.batch_seqs) for run_id in run_ids]
    print_report(analyses)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(analyses, indent=2), encoding="utf-8")
        print(f"wrote_json:{args.output_json}")


if __name__ == "__main__":
    main()
