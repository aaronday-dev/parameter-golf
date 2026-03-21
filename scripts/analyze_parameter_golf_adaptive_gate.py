#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PG_DIR = ROOT
PG_MLX_PATH = PG_DIR / "train_gpt_mlx.py"
DEFAULT_DATA_PATH = PG_DIR / "data" / "datasets" / "fineweb10B_sp1024_smoke"


def load_pg_module():
    spec = importlib.util.spec_from_file_location("parameter_golf_train_gpt_mlx", PG_MLX_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MLX trainer module from {PG_MLX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PG_DIR))
    spec.loader.exec_module(module)
    return module


def find_last_match(text: str, pattern: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"Missing {label} in log")
    match = matches[-1]
    return match[-1] if isinstance(match, tuple) else match


def find_last_optional(text: str, pattern: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    match = matches[-1]
    return match[-1] if isinstance(match, tuple) else match


def parse_bool(text: str | None, default: bool = False) -> bool:
    if text is None:
        return default
    return text == "True"


def parse_run_metadata(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8")
    role_gains_match = find_last_optional(text, r"\brole_gains:([0-9.,\-]+)")
    custom_schedule_match = find_last_optional(text, r"\bcustom_schedule:([A-Z\-]+)")
    correct_block_match = find_last_optional(text, r"\bcorrect_block:([A-Z\-]+)")
    adaptive_block_match = find_last_optional(text, r"\badaptive_correct_block:([A-Z\-]+)")
    adaptive_sources_match = find_last_optional(text, r"\badaptive_correct_sources:([A-Z\-]+)")
    directional_block_match = find_last_optional(text, r"\bdirectional_correct_block:([A-Z\-]+)")
    directional_sources_match = find_last_optional(text, r"\bdirectional_correct_sources:([A-Z\-]+)")
    orbit_block_match = find_last_optional(text, r"\borbit_gate_block:([A-Z\-]+)")
    pulse_block_match = find_last_optional(text, r"\battractor_pulse_block:([A-Z\-]+)")
    mirror_block_match = find_last_optional(text, r"\bmirror_cancel_block:([A-Z\-]+)")
    return {
        "vocab_size": int(find_last_match(text, r"\bvocab_size:(\d+)", "vocab_size")),
        "num_layers": int(find_last_match(text, r"\blayers:(\d+)", "layers")),
        "dim": int(find_last_match(text, r"\bdim:(\d+)", "dim")),
        "num_heads": int(find_last_match(text, r"\bheads:(\d+)", "heads")),
        "num_kv_heads": int(find_last_match(text, r"\bkv_heads:(\d+)", "kv_heads")),
        "seq_len": int(find_last_match(text, r"\bseq_len:(\d+)", "seq_len")),
        "shared_core_blocks": int(find_last_match(text, r"\bunique_blocks:(\d+)", "unique_blocks")),
        "shared_core_schedule": find_last_match(text, r"\bschedule:([a-z_]+)", "schedule"),
        "shared_core_custom_schedule": "" if custom_schedule_match in {None, "-"} else custom_schedule_match,
        "shared_core_role_gains": "" if role_gains_match in {None, "-"} else role_gains_match,
        "shared_core_pass_x0": parse_bool(find_last_optional(text, r"\bpass_x0_scales:(True|False)")),
        "shared_core_revisit_gain": parse_bool(find_last_optional(text, r"\brevisit_gains:(True|False)")),
        "shared_core_revisit_count_gain": parse_bool(find_last_optional(text, r"\brevisit_count_gains:(True|False)")),
        "shared_core_revisit_damping": float(find_last_optional(text, r"\brevisit_damping:([0-9.]+)") or 0.0),
        "shared_core_phase_split_revisit_gain": parse_bool(find_last_optional(text, r"\bphase_split_revisit_gain:(True|False)")),
        "shared_core_correct_block": "" if correct_block_match in {None, "-"} else correct_block_match,
        "shared_core_correct_gain": float(find_last_optional(text, r"\bcorrect_gain:([0-9.]+)") or 0.0),
        "shared_core_adaptive_correct": parse_bool(find_last_optional(text, r"\badaptive_correct:(True|False)")),
        "shared_core_adaptive_correct_block": "" if adaptive_block_match in {None, "-"} else adaptive_block_match,
        "shared_core_adaptive_correct_sources": "" if adaptive_sources_match in {None, "-"} else adaptive_sources_match,
        "shared_core_adaptive_correct_revisit_only": parse_bool(
            find_last_optional(text, r"\badaptive_correct_revisit_only:(True|False)"),
            default=True,
        ),
        "shared_core_adaptive_correct_max_gain": float(find_last_optional(text, r"\badaptive_correct_max_gain:([0-9.]+)") or 0.0),
        "shared_core_adaptive_correct_target_amp": float(find_last_optional(text, r"\badaptive_correct_target_amp:([0-9.]+)") or 1.1),
        "shared_core_adaptive_correct_log_band": float(find_last_optional(text, r"\badaptive_correct_log_band:([0-9.]+)") or 0.35),
        "shared_core_directional_correct": parse_bool(find_last_optional(text, r"\bdirectional_correct:(True|False)")),
        "shared_core_directional_correct_block": "" if directional_block_match in {None, "-"} else directional_block_match,
        "shared_core_directional_correct_sources": "" if directional_sources_match in {None, "-"} else directional_sources_match,
        "shared_core_directional_correct_revisit_only": parse_bool(
            find_last_optional(text, r"\bdirectional_correct_revisit_only:(True|False)"),
            default=True,
        ),
        "shared_core_directional_correct_max_gain": float(find_last_optional(text, r"\bdirectional_correct_max_gain:([0-9.]+)") or 0.0),
        "shared_core_directional_correct_target_amp": float(find_last_optional(text, r"\bdirectional_correct_target_amp:([0-9.]+)") or 1.1),
        "shared_core_directional_correct_log_band": float(find_last_optional(text, r"\bdirectional_correct_log_band:([0-9.]+)") or 0.35),
        "shared_core_directional_stress_guard": parse_bool(find_last_optional(text, r"\bdirectional_stress_guard:(True|False)")),
        "shared_core_directional_stress_band": float(find_last_optional(text, r"\bdirectional_stress_band:([0-9.]+)") or 0.02),
        "shared_core_directional_stress_min_factor": float(find_last_optional(text, r"\bdirectional_stress_min_factor:([0-9.]+)") or 0.5),
        "shared_core_orbit_gate": parse_bool(find_last_optional(text, r"\borbit_gate:(True|False)")),
        "shared_core_orbit_gate_block": "" if orbit_block_match in {None, "-"} else orbit_block_match,
        "shared_core_orbit_gate_steps": int(find_last_optional(text, r"\borbit_gate_steps:(\d+)") or 1),
        "shared_core_orbit_gate_revisit_only": parse_bool(
            find_last_optional(text, r"\borbit_gate_revisit_only:(True|False)"),
            default=True,
        ),
        "shared_core_attractor_pulse": parse_bool(find_last_optional(text, r"\battractor_pulse:(True|False)")),
        "shared_core_attractor_pulse_block": "" if pulse_block_match in {None, "-"} else pulse_block_match,
        "shared_core_attractor_pulse_steps": int(find_last_optional(text, r"\battractor_pulse_steps:(\d+)") or 0),
        "shared_core_attractor_pulse_gain": float(find_last_optional(text, r"\battractor_pulse_gain:([0-9.]+)") or 0.0),
        "shared_core_attractor_pulse_trigger_amp": float(find_last_optional(text, r"\battractor_pulse_trigger_amp:([0-9.]+)") or 0.0),
        "shared_core_attractor_pulse_margin": float(find_last_optional(text, r"\battractor_pulse_margin:([0-9.\-]+)") or 0.0),
        "shared_core_attractor_pulse_revisit_only": parse_bool(
            find_last_optional(text, r"\battractor_pulse_revisit_only:(True|False)"),
            default=True,
        ),
        "shared_core_mirror_cancel": parse_bool(find_last_optional(text, r"\bmirror_cancel:(True|False)")),
        "shared_core_mirror_cancel_block": "" if mirror_block_match in {None, "-"} else mirror_block_match,
        "shared_core_mirror_cancel_threshold": float(find_last_optional(text, r"\bmirror_cancel_threshold:([0-9.]+)") or 0.0),
        "shared_core_mirror_cancel_revisit_only": parse_bool(
            find_last_optional(text, r"\bmirror_cancel_revisit_only:(True|False)"),
            default=True,
        ),
        "shared_core_stabilize_every": int(find_last_optional(text, r"\bstabilize_every:(\d+)") or 0),
        "shared_core_stabilize_after": int(find_last_optional(text, r"\bstabilize_after:(\d+)") or 0),
        "shared_core_stabilize_gain": float(find_last_optional(text, r"\bstabilize_gain:([0-9.]+)") or 0.0),
        "val_bpb": float(find_last_match(text, r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", "val_bpb")),
    }


def infer_mlp_mult(flat_state: dict[str, object], dim: int) -> int:
    fc_weight = np.array(flat_state["blocks.0.mlp.fc.weight"])
    return int(fc_weight.shape[0] // dim)


def load_model(module, run_id: str):
    log_path = PG_DIR / "logs" / f"{run_id}.txt"
    model_path = PG_DIR / "logs" / f"{run_id}_mlx_model.npz"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log file: {log_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    metadata = parse_run_metadata(log_path)
    flat_state = module.mx.load(str(model_path))
    args = module.Hyperparameters()
    model = module.GPT(
        vocab_size=int(metadata["vocab_size"]),
        num_layers=int(metadata["num_layers"]),
        shared_core_blocks=int(metadata["shared_core_blocks"]),
        shared_core_schedule=str(metadata["shared_core_schedule"]),
        shared_core_custom_schedule=str(metadata["shared_core_custom_schedule"]),
        shared_core_role_gains=str(metadata["shared_core_role_gains"]),
        shared_core_pass_x0=bool(metadata["shared_core_pass_x0"]),
        shared_core_revisit_gain=bool(metadata["shared_core_revisit_gain"]),
        shared_core_revisit_count_gain=bool(metadata["shared_core_revisit_count_gain"]),
        shared_core_phase_split_revisit_gain=bool(metadata["shared_core_phase_split_revisit_gain"]),
        shared_core_revisit_damping=float(metadata["shared_core_revisit_damping"]),
        shared_core_correct_block=str(metadata["shared_core_correct_block"]),
        shared_core_correct_gain=float(metadata["shared_core_correct_gain"]),
        shared_core_adaptive_correct=bool(metadata["shared_core_adaptive_correct"]),
        shared_core_adaptive_correct_block=str(metadata["shared_core_adaptive_correct_block"]),
        shared_core_adaptive_correct_sources=str(metadata["shared_core_adaptive_correct_sources"]),
        shared_core_adaptive_correct_max_gain=float(metadata["shared_core_adaptive_correct_max_gain"]),
        shared_core_adaptive_correct_target_amp=float(metadata["shared_core_adaptive_correct_target_amp"]),
        shared_core_adaptive_correct_log_band=float(metadata["shared_core_adaptive_correct_log_band"]),
        shared_core_adaptive_correct_revisit_only=bool(metadata["shared_core_adaptive_correct_revisit_only"]),
        shared_core_directional_correct=bool(metadata["shared_core_directional_correct"]),
        shared_core_directional_correct_block=str(metadata["shared_core_directional_correct_block"]),
        shared_core_directional_correct_sources=str(metadata["shared_core_directional_correct_sources"]),
        shared_core_directional_correct_max_gain=float(metadata["shared_core_directional_correct_max_gain"]),
        shared_core_directional_correct_target_amp=float(metadata["shared_core_directional_correct_target_amp"]),
        shared_core_directional_correct_log_band=float(metadata["shared_core_directional_correct_log_band"]),
        shared_core_directional_correct_revisit_only=bool(metadata["shared_core_directional_correct_revisit_only"]),
        shared_core_directional_stress_guard=bool(metadata["shared_core_directional_stress_guard"]),
        shared_core_directional_stress_band=float(metadata["shared_core_directional_stress_band"]),
        shared_core_directional_stress_min_factor=float(metadata["shared_core_directional_stress_min_factor"]),
        shared_core_orbit_gate=bool(metadata["shared_core_orbit_gate"]),
        shared_core_orbit_gate_block=str(metadata["shared_core_orbit_gate_block"]),
        shared_core_orbit_gate_steps=int(metadata["shared_core_orbit_gate_steps"]),
        shared_core_orbit_gate_revisit_only=bool(metadata["shared_core_orbit_gate_revisit_only"]),
        shared_core_attractor_pulse=bool(metadata["shared_core_attractor_pulse"]),
        shared_core_attractor_pulse_block=str(metadata["shared_core_attractor_pulse_block"]),
        shared_core_attractor_pulse_steps=int(metadata["shared_core_attractor_pulse_steps"]),
        shared_core_attractor_pulse_gain=float(metadata["shared_core_attractor_pulse_gain"]),
        shared_core_attractor_pulse_trigger_amp=float(metadata["shared_core_attractor_pulse_trigger_amp"]),
        shared_core_attractor_pulse_margin=float(metadata["shared_core_attractor_pulse_margin"]),
        shared_core_attractor_pulse_revisit_only=bool(metadata["shared_core_attractor_pulse_revisit_only"]),
        shared_core_mirror_cancel=bool(metadata["shared_core_mirror_cancel"]),
        shared_core_mirror_cancel_block=str(metadata["shared_core_mirror_cancel_block"]),
        shared_core_mirror_cancel_threshold=float(metadata["shared_core_mirror_cancel_threshold"]),
        shared_core_mirror_cancel_revisit_only=bool(metadata["shared_core_mirror_cancel_revisit_only"]),
        shared_core_stabilize_every=int(metadata["shared_core_stabilize_every"]),
        shared_core_stabilize_after=int(metadata["shared_core_stabilize_after"]),
        shared_core_stabilize_gain=float(metadata["shared_core_stabilize_gain"]),
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
    return metadata, model


def mean_and_std_nan(series: list[list[float]]) -> tuple[list[float], list[float]]:
    arr = np.asarray(series, dtype=np.float64)
    return np.nanmean(arr, axis=0).tolist(), np.nanstd(arr, axis=0, ddof=0).tolist()


def collect_batch_metrics(module, model, input_ids) -> dict[str, object]:
    x = module.rms_norm(model.tok_emb(input_ids).astype(module.COMPUTE_DTYPE))
    x0 = x
    anchor = x
    anchor_count = 1
    block_deltas = [module.mx.zeros_like(x) for _ in range(model.num_unique_blocks)]
    block_delta_ready = [False] * model.num_unique_blocks
    layer_deltas = [module.mx.zeros_like(x) for _ in range(model.num_layers)]
    layer_delta_ready = [False] * model.num_layers
    skips = []
    adaptive_lambda = []
    adaptive_cos_ab = []
    adaptive_amp_ratio = []
    orbit_match = []
    aux_metric_2 = []
    aux_metric_3 = []
    block_indices = []

    def record(layer_idx: int, adaptive_metrics) -> None:
        block_indices.append(model.block_index(layer_idx))
        if adaptive_metrics is None:
            adaptive_lambda.append(float("nan"))
            adaptive_cos_ab.append(float("nan"))
            adaptive_amp_ratio.append(float("nan"))
            orbit_match.append(float("nan"))
            aux_metric_2.append(float("nan"))
            aux_metric_3.append(float("nan"))
            return
        adaptive_lambda.append(float(np.array(adaptive_metrics[0].astype(module.mx.float32), dtype=np.float32)))
        adaptive_cos_ab.append(float(np.array(adaptive_metrics[1].astype(module.mx.float32), dtype=np.float32)))
        adaptive_amp_ratio.append(float(np.array(adaptive_metrics[2].astype(module.mx.float32), dtype=np.float32)))
        if len(adaptive_metrics) >= 4:
            orbit_match.append(float(np.array(adaptive_metrics[3].astype(module.mx.float32), dtype=np.float32)))
        else:
            orbit_match.append(float("nan"))
        if len(adaptive_metrics) >= 5:
            aux_metric_2.append(float(np.array(adaptive_metrics[4].astype(module.mx.float32), dtype=np.float32)))
        else:
            aux_metric_2.append(float("nan"))
        if len(adaptive_metrics) >= 6:
            aux_metric_3.append(float(np.array(adaptive_metrics[5].astype(module.mx.float32), dtype=np.float32)))
        else:
            aux_metric_3.append(float("nan"))

    for layer_idx in range(model.num_encoder_layers):
        x_in = x
        x = model.apply_block(layer_idx, x, x0)
        x, anchor, anchor_count, adaptive_metrics = model.apply_post_block_controls(
            layer_idx,
            x0,
            x_in,
            x,
            anchor,
            anchor_count,
            block_deltas,
            block_delta_ready,
            layer_deltas,
            layer_delta_ready,
        )
        record(layer_idx, adaptive_metrics)
        skips.append(x)
    for decoder_idx in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[decoder_idx].astype(x.dtype)[None, None, :] * skips.pop()
        layer_idx = model.num_encoder_layers + decoder_idx
        x_in = x
        x = model.apply_block(layer_idx, x, x0)
        x, anchor, anchor_count, adaptive_metrics = model.apply_post_block_controls(
            layer_idx,
            x0,
            x_in,
            x,
            anchor,
            anchor_count,
            block_deltas,
            block_delta_ready,
            layer_deltas,
            layer_delta_ready,
        )
        record(layer_idx, adaptive_metrics)

    return {
        "adaptive_lambda": adaptive_lambda,
        "adaptive_cos_ab": adaptive_cos_ab,
        "adaptive_amp_ratio": adaptive_amp_ratio,
        "orbit_match": orbit_match,
        "aux_metric_2": aux_metric_2,
        "aux_metric_3": aux_metric_3,
        "block_indices": block_indices,
    }


def analyze_run(module, run_id: str, data_path: Path, num_batches: int, batch_seqs: int) -> dict[str, object]:
    metadata, model = load_model(module, run_id)
    seq_len = int(metadata["seq_len"])
    val_tokens = module.load_validation_tokens(str(data_path / "fineweb_val_*.bin"), seq_len)
    total_seqs = (val_tokens.size - 1) // seq_len
    needed_seqs = num_batches * batch_seqs
    if total_seqs < needed_seqs:
        raise ValueError(f"Validation split has {total_seqs} seqs, need {needed_seqs}")

    batches = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_seqs * seq_len
        end = start + batch_seqs * seq_len + 1
        chunk = val_tokens[start:end]
        input_ids = module.mx.array(chunk[:-1].reshape(batch_seqs, seq_len), dtype=module.mx.int32)
        batches.append(collect_batch_metrics(module, model, input_ids))

    adaptive_lambda_mean, adaptive_lambda_std = mean_and_std_nan([b["adaptive_lambda"] for b in batches])
    adaptive_cos_ab_mean, adaptive_cos_ab_std = mean_and_std_nan([b["adaptive_cos_ab"] for b in batches])
    adaptive_amp_ratio_mean, adaptive_amp_ratio_std = mean_and_std_nan([b["adaptive_amp_ratio"] for b in batches])
    orbit_match_mean, orbit_match_std = mean_and_std_nan([b["orbit_match"] for b in batches])
    aux_metric_2_mean, aux_metric_2_std = mean_and_std_nan([b["aux_metric_2"] for b in batches])
    aux_metric_3_mean, aux_metric_3_std = mean_and_std_nan([b["aux_metric_3"] for b in batches])
    active_mask = [not np.isnan(x) and x > 0.0 for x in adaptive_lambda_mean]
    active_passes = [i + 1 for i, is_active in enumerate(active_mask) if is_active]
    metric_semantics = {
        "metric_4": "orbit_match",
        "metric_5": "aux_metric_2",
        "metric_6": "aux_metric_3",
    }
    if bool(metadata["shared_core_directional_stress_guard"]) and not bool(metadata["shared_core_orbit_gate"]) and not bool(metadata["shared_core_attractor_pulse"]):
        metric_semantics = {
            "metric_4": "stress_guard_factor",
            "metric_5": "stress_guard_delta",
            "metric_6": "aux_metric_3",
        }

    return {
        "run_id": run_id,
        "metadata": metadata,
        "metric_semantics": metric_semantics,
        "adaptive_lambda_mean": adaptive_lambda_mean,
        "adaptive_lambda_std": adaptive_lambda_std,
        "adaptive_cos_ab_mean": adaptive_cos_ab_mean,
        "adaptive_cos_ab_std": adaptive_cos_ab_std,
        "adaptive_amp_ratio_mean": adaptive_amp_ratio_mean,
        "adaptive_amp_ratio_std": adaptive_amp_ratio_std,
        "orbit_match_mean": orbit_match_mean,
        "orbit_match_std": orbit_match_std,
        "aux_metric_2_mean": aux_metric_2_mean,
        "aux_metric_2_std": aux_metric_2_std,
        "aux_metric_3_mean": aux_metric_3_mean,
        "aux_metric_3_std": aux_metric_3_std,
        "active_passes": active_passes,
        "mean_active_lambda": float(np.nanmean(np.asarray(adaptive_lambda_mean, dtype=np.float64))),
        "mean_active_cos_ab": float(np.nanmean(np.asarray(adaptive_cos_ab_mean, dtype=np.float64))),
        "mean_active_amp_ratio": float(np.nanmean(np.asarray(adaptive_amp_ratio_mean, dtype=np.float64))),
        "mean_orbit_match": float(np.nanmean(np.asarray(orbit_match_mean, dtype=np.float64))),
        "mean_aux_metric_2": float(np.nanmean(np.asarray(aux_metric_2_mean, dtype=np.float64))),
        "mean_aux_metric_3": float(np.nanmean(np.asarray(aux_metric_3_mean, dtype=np.float64))),
        "block_indices": batches[0]["block_indices"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze adaptive contractive C-gate metrics for parameter-golf MLX runs.")
    parser.add_argument("--run-id", action="append", required=True, help="Run id to inspect, may be passed multiple times.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--batch-seqs", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    module = load_pg_module()
    results = {
        run_id: analyze_run(module, run_id, args.data_path, args.num_batches, args.batch_seqs)
        for run_id in args.run_id
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
