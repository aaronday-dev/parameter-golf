#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ModuleNotFoundError as exc:
    raise SystemExit(
        "missing dependency: huggingface_hub. Install repo requirements first, "
        "for example '.venv/bin/pip install -r requirements.txt'."
    ) from exc

DEFAULT_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DEFAULT_REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
DATASET_VARIANTS = {
    "sp1024": {
        "dataset_name": "fineweb10B_sp1024",
        "smoke_name": "fineweb10B_sp1024_smoke",
        "tokenizer_name": "fineweb_1024_bpe.model",
    }
}
SMOKE_VAL_TOKENS = 1_047_553
HEADER_WORDS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")
MAGIC = 20240520
VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the published Parameter Golf dataset prefix.")
    parser.add_argument("--variant", default="sp1024", choices=sorted(DATASET_VARIANTS))
    parser.add_argument("--train-shards", type=int, default=80, help="Number of train shards to download.")
    parser.add_argument(
        "--build-smoke",
        action="store_true",
        help="Also create data/datasets/<dataset>_smoke from train shard 0 plus a truncated validation shard.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--remote-root-prefix", default=DEFAULT_REMOTE_ROOT_PREFIX)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def dataset_remote_dir(remote_root_prefix: str, dataset_name: str) -> str:
    return f"{remote_root_prefix}/datasets/{dataset_name}"


def tokenizer_remote_path(remote_root_prefix: str, tokenizer_name: str) -> str:
    return f"{remote_root_prefix}/tokenizers/{tokenizer_name}"


def copy_remote_file(repo_id: str, filename: str, dest: Path) -> Path:
    src = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def write_smoke_val_shard(src: Path, dest: Path) -> None:
    header = np.fromfile(src, dtype=HEADER_DTYPE, count=HEADER_WORDS)
    if header.size != HEADER_WORDS or int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise ValueError(f"unexpected shard header in {src}")
    src_tokens = int(header[2])
    if src_tokens < SMOKE_VAL_TOKENS:
        raise ValueError(f"{src} only has {src_tokens} tokens, need at least {SMOKE_VAL_TOKENS}")

    tokens = np.fromfile(
        src,
        dtype=TOKEN_DTYPE,
        count=SMOKE_VAL_TOKENS,
        offset=HEADER_WORDS * HEADER_DTYPE.itemsize,
    )
    if tokens.size != SMOKE_VAL_TOKENS:
        raise ValueError(f"short read from {src}: got {tokens.size} tokens")

    header_out = header.copy()
    header_out[2] = SMOKE_VAL_TOKENS
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        header_out.astype(HEADER_DTYPE).tofile(f)
        tokens.astype(TOKEN_DTYPE, copy=False).tofile(f)


def main() -> None:
    args = parse_args()
    if args.train_shards <= 0:
        raise ValueError("--train-shards must be positive")

    variant = DATASET_VARIANTS[args.variant]
    dataset_name = variant["dataset_name"]
    smoke_name = variant["smoke_name"]
    tokenizer_name = variant["tokenizer_name"]

    output_root = args.output_root.resolve()
    full_dir = output_root / "datasets" / dataset_name
    smoke_dir = output_root / "datasets" / smoke_name
    tokenizer_dir = output_root / "tokenizers"

    remote_dataset_dir = dataset_remote_dir(args.remote_root_prefix, dataset_name)
    remote_tokenizer = tokenizer_remote_path(args.remote_root_prefix, tokenizer_name)

    files = list_repo_files(args.repo_id, repo_type="dataset")
    train_files = sorted(
        f for f in files if f.startswith(f"{remote_dataset_dir}/fineweb_train_") and f.endswith(".bin")
    )
    val_files = sorted(
        f for f in files if f.startswith(f"{remote_dataset_dir}/fineweb_val_") and f.endswith(".bin")
    )
    if not train_files:
        raise FileNotFoundError(f"no train shards found under {remote_dataset_dir}")
    if not val_files:
        raise FileNotFoundError(f"no validation shards found under {remote_dataset_dir}")
    if args.train_shards > len(train_files):
        raise ValueError(f"requested {args.train_shards} train shards, only {len(train_files)} available")

    tokenizer_path = copy_remote_file(args.repo_id, remote_tokenizer, tokenizer_dir / tokenizer_name)
    print(f"tokenizer:{tokenizer_path}")

    downloaded_train: list[Path] = []
    for remote_train in train_files[: args.train_shards]:
        local_train = copy_remote_file(args.repo_id, remote_train, full_dir / Path(remote_train).name)
        downloaded_train.append(local_train)
        print(f"train:{local_train}")

    downloaded_val: list[Path] = []
    for remote_val in val_files:
        local_val = copy_remote_file(args.repo_id, remote_val, full_dir / Path(remote_val).name)
        downloaded_val.append(local_val)
        print(f"val:{local_val}")

    if args.build_smoke:
        smoke_train = smoke_dir / "fineweb_train_000000.bin"
        smoke_val = smoke_dir / "fineweb_val_000000.bin"
        shutil.copy2(downloaded_train[0], smoke_train)
        write_smoke_val_shard(downloaded_val[0], smoke_val)
        print(f"smoke_train:{smoke_train}")
        print(f"smoke_val:{smoke_val} tokens:{SMOKE_VAL_TOKENS}")


if __name__ == "__main__":
    main()
