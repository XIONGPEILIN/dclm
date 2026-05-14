"""
JumpLLaDA Data Preparation
============================
Download and tokenize a dataset for JumpLLaDA training.

Uses the same GPT-NeoX tokenizer as DCLM (EleutherAI/gpt-neox-20b, vocab=50432).
Tokenized data is saved as memory-mapped .bin files (uint16).

Data is downloaded to /export/ssd2/xiong-p/dclm/data and symlinked to ./data.
If dataset size > 2TB, data is stored at /export/space0/data/HF instead.

Usage:
    python prepare_data.py --dataset fineweb-edu --num_tokens 30B
    python prepare_data.py --dataset c4 --num_tokens 30B
"""

import os
import sys
import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


# Size threshold: 2TB
SIZE_THRESHOLD_BYTES = 2 * 1024 ** 4  # 2 TiB
LARGE_DATA_DIR = "/export/space0/data/HF"
DEFAULT_DATA_DIR = "/export/ssd2/xiong-p/dclm/data"
LOCAL_LINK_DIR = "./data"


def parse_token_count(s: str) -> int:
    """Parse human-readable token count like '30B', '1T', '100M'."""
    s = s.strip().upper()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def estimate_dataset_size(num_tokens: int, dtype_bytes: int = 2) -> int:
    """Estimate on-disk size of tokenized data in bytes."""
    return num_tokens * dtype_bytes


def get_data_dir(estimated_size: int) -> str:
    """Choose data directory based on estimated size."""
    if estimated_size > SIZE_THRESHOLD_BYTES:
        print(f"Estimated data size: {estimated_size / 1e12:.2f} TB > 2 TB")
        print(f"Using large storage directory: {LARGE_DATA_DIR}")
        data_dir = os.path.join(LARGE_DATA_DIR, "jump_llada_data")
    else:
        print(f"Estimated data size: {estimated_size / 1e9:.2f} GB (< 2 TB)")
        print(f"Using SSD storage: {DEFAULT_DATA_DIR}")
        data_dir = DEFAULT_DATA_DIR
    return data_dir


def create_symlink(real_dir: str, link_dir: str):
    """Create a symlink from link_dir -> real_dir if not already linked."""
    link_path = Path(link_dir)
    real_path = Path(real_dir)

    if link_path.is_symlink():
        current_target = link_path.resolve()
        if current_target == real_path.resolve():
            print(f"Symlink already exists: {link_dir} -> {real_dir}")
            return
        else:
            print(f"Updating symlink: {link_dir} -> {real_dir} (was {current_target})")
            link_path.unlink()
    elif link_path.exists():
        print(f"Warning: {link_dir} exists and is not a symlink. Skipping symlink creation.")
        return

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(real_path)
    print(f"Created symlink: {link_dir} -> {real_dir}")


def tokenize_and_save(
    dataset_name: str,
    output_dir: str,
    tokenizer_name: str = "EleutherAI/gpt-neox-20b",
    num_tokens: int = 30_000_000_000,
    seq_len: int = 2048,
    shard_size: int = 100_000_000,  # 100M tokens per shard
    num_proc: int = 8,
    dataset_subset: str = None,
    dataset_split: str = "train",
    text_column: str = "text",
):
    """Download, tokenize, and save dataset as .bin shards.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'c4', 'HuggingFaceFW/fineweb-edu')
        output_dir: Where to save tokenized .bin files
        tokenizer_name: Tokenizer to use
        num_tokens: Target number of tokens
        seq_len: Sequence length for training
        shard_size: Tokens per shard file
        num_proc: Number of processes for tokenization
        dataset_subset: Dataset subset/config name
        dataset_split: Dataset split
        text_column: Name of the text column
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    # Check if already tokenized
    existing_bins = list(Path(output_dir).glob("*.bin"))
    if existing_bins:
        total_existing = sum(f.stat().st_size for f in existing_bins)
        existing_tokens = total_existing // 2  # uint16
        print(f"Found existing tokenized data: {existing_tokens:,} tokens in {len(existing_bins)} shards")
        if existing_tokens >= num_tokens * 0.95:  # 95% threshold
            print("Dataset already prepared. Skipping.")
            return output_dir
        else:
            print("Incomplete dataset. Re-tokenizing...")

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Save mask_id info
    mask_id = len(tokenizer) - 1  # Use last token as mask
    print(f"Vocab size: {len(tokenizer)}, Mask ID: {mask_id}")

    # Load dataset (streaming to handle large datasets)
    print(f"Loading dataset: {dataset_name} (subset={dataset_subset}, split={dataset_split})")
    ds_kwargs = {"path": dataset_name, "split": dataset_split, "streaming": True}
    if dataset_subset:
        ds_kwargs["name"] = dataset_subset
    dataset = load_dataset(**ds_kwargs)

    # Tokenize and write shards
    print(f"Tokenizing and saving to {output_dir}...")
    print(f"Target: {num_tokens:,} tokens, Shard size: {shard_size:,} tokens")

    token_buffer = []
    shard_idx = 0
    total_tokens = 0
    pbar = tqdm(total=num_tokens, unit="tok", unit_scale=True, desc="Tokenizing")

    for example in dataset:
        text = example.get(text_column, "")
        if not text:
            continue

        tokens = tokenizer.encode(text)
        token_buffer.extend(tokens)

        # Write shard when buffer is full
        while len(token_buffer) >= shard_size:
            shard_tokens = token_buffer[:shard_size]
            token_buffer = token_buffer[shard_size:]

            # Save as uint16
            arr = np.array(shard_tokens, dtype=np.uint16)
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
            arr.tofile(shard_path)

            total_tokens += len(shard_tokens)
            shard_idx += 1
            pbar.update(len(shard_tokens))

        if total_tokens >= num_tokens:
            break

    # Write remaining tokens
    if token_buffer and total_tokens < num_tokens:
        # Pad to multiple of seq_len
        remainder = len(token_buffer) % seq_len
        if remainder > 0:
            token_buffer = token_buffer[:len(token_buffer) - remainder]

        if token_buffer:
            arr = np.array(token_buffer, dtype=np.uint16)
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
            arr.tofile(shard_path)
            total_tokens += len(token_buffer)
            pbar.update(len(token_buffer))

    pbar.close()

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "dataset_subset": dataset_subset,
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "mask_id": mask_id,
        "total_tokens": total_tokens,
        "num_shards": shard_idx + 1,
        "shard_size": shard_size,
        "seq_len": seq_len,
        "dtype": "uint16",
    }
    import json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_size = sum(f.stat().st_size for f in Path(output_dir).glob("*.bin"))
    print(f"\nDone! Tokenized {total_tokens:,} tokens in {shard_idx + 1} shards")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Saved to: {output_dir}")

    return output_dir


# ============================================================================
# Preset Datasets
# ============================================================================

DATASET_PRESETS = {
    "c4": {
        "dataset_name": "allenai/c4",
        "dataset_subset": "en",
        "text_column": "text",
    },
    "fineweb-edu": {
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_subset": "sample-10BT",
        "text_column": "text",
    },
    "fineweb": {
        "dataset_name": "HuggingFaceFW/fineweb",
        "dataset_subset": "sample-10BT",
        "text_column": "text",
    },
    "openwebtext": {
        "dataset_name": "Skylion007/openwebtext",
        "dataset_subset": None,
        "text_column": "text",
    },
    "redpajama": {
        "dataset_name": "togethercomputer/RedPajama-Data-1T-Sample",
        "dataset_subset": None,
        "text_column": "text",
    },
    "slimpajama": {
        "dataset_name": "cerebras/SlimPajama-627B",
        "dataset_subset": None,
        "text_column": "text",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Prepare data for JumpLLaDA training")
    parser.add_argument("--dataset", type=str, default="c4",
                        help=f"Dataset preset or HuggingFace name. Presets: {list(DATASET_PRESETS.keys())}")
    parser.add_argument("--num_tokens", type=str, default="30B",
                        help="Number of tokens to tokenize (e.g., '30B', '1T', '100M')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-selected based on size if not specified)")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b",
                        help="Tokenizer name")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes")
    parser.add_argument("--shard_size", type=str, default="100M",
                        help="Tokens per shard (e.g., '100M')")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Text column name (auto-detected from preset)")
    args = parser.parse_args()

    num_tokens = parse_token_count(args.num_tokens)
    shard_size = parse_token_count(args.shard_size)

    # Resolve dataset preset
    if args.dataset in DATASET_PRESETS:
        preset = DATASET_PRESETS[args.dataset]
        dataset_name = preset["dataset_name"]
        dataset_subset = preset.get("dataset_subset")
        text_column = args.text_column or preset.get("text_column", "text")
        print(f"Using preset: {args.dataset} -> {dataset_name}")
    else:
        dataset_name = args.dataset
        dataset_subset = None
        text_column = args.text_column or "text"
        print(f"Using custom dataset: {dataset_name}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        estimated_size = estimate_dataset_size(num_tokens)
        base_dir = get_data_dir(estimated_size)
        safe_name = dataset_name.replace("/", "_").replace("-", "_")
        output_dir = os.path.join(base_dir, "tokenized", safe_name)

        # Create symlink: ./data -> real data dir
        create_symlink(base_dir, LOCAL_LINK_DIR)

    print(f"Output directory: {output_dir}")

    # Tokenize
    tokenize_and_save(
        dataset_name=dataset_name,
        output_dir=output_dir,
        tokenizer_name=args.tokenizer,
        num_tokens=num_tokens,
        seq_len=args.seq_len,
        shard_size=shard_size,
        num_proc=args.num_proc,
        dataset_subset=dataset_subset,
        text_column=text_column,
    )

    print(f"\n{'='*60}")
    print(f"Data ready! To train, run:")
    print(f"  python train.py --dataset_path {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
