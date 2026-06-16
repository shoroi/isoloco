import os
import argparse
import multiprocessing as mp
import random
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from tqdm import tqdm
import math

# Global tokenizer for multiprocessing
_tokenizer = None


def init_worker(tokenizer_name):
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Set a very large max length to suppress warning by effectively disabling truncation
    _tokenizer.model_max_length = int(1e9)
    if _tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} must have an EOS token.")


def tokenize_doc(doc):
    """Tokenize a single document and append EOS token."""
    global _tokenizer
    text = doc.get("text")

    if not text or not isinstance(text, str):
        return np.array([], dtype=np.uint16)

    tokens = _tokenizer.encode(text, add_special_tokens=False)
    tokens.append(_tokenizer.eos_token_id)

    tokens_array = np.array(tokens, dtype=np.uint16)

    if not ((0 <= tokens_array) & (tokens_array < 2**16)).all():
        raise ValueError(
            f"Token IDs exceed uint16 range. Vocab size: {_tokenizer.vocab_size}"
        )

    return tokens_array


def main(args):
    target_tokens = args.total_tokens
    shard_size = args.shard_size
    expected_shards = math.ceil(target_tokens / shard_size)

    # Check if output directory exists and is not empty
    val_path = os.path.join(args.output_dir, "valid_000000.npy")
    test_path = os.path.join(args.output_dir, "test_000000.npy")
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise RuntimeError(
            f"Output directory {args.output_dir} exists and is not empty. "
            "Please remove it or specify a different output directory."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate split fractions
    if args.val_split < 0 or args.test_split < 0 or (args.val_split + args.test_split) >= 1.0:
        raise ValueError("Require val_split >= 0, test_split >= 0, and val_split + test_split < 1.0")
    train_frac = 1.0 - args.val_split - args.test_split
    val_frac = args.val_split
    test_frac = args.test_split

    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Training target: {target_tokens / 1e9:.1f}B tokens (from {train_frac:.0%} of files)")
    print(f"Validation target: {args.val_tokens / 1e9:.1f}B tokens (from {val_frac:.0%} of files)")
    print(f"Test target: {args.test_tokens / 1e9:.1f}B tokens (from {test_frac:.0%} of files)")
    print(f"Shard size: {shard_size / 1e6:.0f}M tokens")
    print(f"Expected shards: {expected_shards}")

    # Query dataset files and split them
    print("\nQuerying dataset files...")
    fs = HfFileSystem()
    all_files = sorted(fs.glob(f"datasets/{args.dataset}/**/*.parquet"))

    if not all_files:
        raise ValueError(f"No parquet files found for dataset {args.dataset}")

    # Shuffle files before splitting to avoid bias from file ordering
    rng = random.Random(args.seed)
    rng.shuffle(all_files)

    # Split files between train, validation, and test
    train_end_idx = int(len(all_files) * train_frac)
    val_end_idx = int(len(all_files) * (train_frac + val_frac))
    train_files = [f"hf://{f}" for f in all_files[:train_end_idx]]
    val_files = [f"hf://{f}" for f in all_files[train_end_idx:val_end_idx]]
    test_files = [f"hf://{f}" for f in all_files[val_end_idx:]]

    # Validate split sizes to fail fast on tiny datasets
    if len(train_files) == 0:
        raise ValueError(f"No training files after split. Dataset has {len(all_files)} files, train_frac={train_frac:.2f}")
    if args.val_tokens > 0 and len(val_files) == 0:
        raise ValueError(f"No validation files after split. Dataset has {len(all_files)} files, val_frac={val_frac:.2f}")
    if args.test_tokens > 0 and len(test_files) == 0:
        raise ValueError(f"No test files after split. Dataset has {len(all_files)} files, test_frac={test_frac:.2f}")

    print(f"Found {len(all_files)} parquet files")
    print(f"Training: {len(train_files)} files, Validation: {len(val_files)} files, Test: {len(test_files)} files")

    # Save file lists
    with open(os.path.join(args.output_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(args.output_dir, "valid_files.txt"), "w") as f:
        f.write("\n".join(val_files))
    with open(os.path.join(args.output_dir, "test_files.txt"), "w") as f:
        f.write("\n".join(test_files))

    num_proc = args.num_proc if args.num_proc > 0 else max(1, os.cpu_count() * 3 // 4)
    print(f"Using {num_proc} processes")

    # Load and shuffle training dataset
    print(f"\nLoading training data from {len(train_files)} files...")
    train_dataset = load_dataset("parquet", data_files=train_files, split="train", streaming=True)
    train_dataset = train_dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)
    train_iter = iter(train_dataset)

    # Initialize processing variables
    shard_idx = 0
    shard_buffer = np.empty(shard_size, dtype=np.uint16)
    tokens_in_shard = 0
    total_tokens = 0

    with mp.Pool(num_proc, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        with tqdm(total=target_tokens, unit="tokens", desc="Training") as pbar:
            for doc_tokens in pool.imap(
                tokenize_doc, train_iter, chunksize=args.chunk_size
            ):
                if total_tokens >= target_tokens:
                    break

                if len(doc_tokens) == 0:
                    continue

                # Process tokens from this document
                doc_idx = 0
                while doc_idx < len(doc_tokens) and total_tokens < target_tokens:
                    space_left = shard_size - tokens_in_shard
                    doc_left = len(doc_tokens) - doc_idx
                    global_left = target_tokens - total_tokens

                    take = min(space_left, doc_left, global_left)
                    if take == 0:
                        break

                    # Copy tokens to shard buffer
                    shard_buffer[tokens_in_shard : tokens_in_shard + take] = doc_tokens[
                        doc_idx : doc_idx + take
                    ]
                    tokens_in_shard += take
                    total_tokens += take
                    pbar.update(take)
                    doc_idx += take

                    # Save shard if full
                    if tokens_in_shard == shard_size:
                        shard_path = os.path.join(
                            args.output_dir, f"train_{shard_idx:06d}.npy"
                        )
                        np.save(shard_path, shard_buffer)
                        tqdm.write(
                            f"Saved shard {shard_idx} ({shard_size / 1e6:.0f}M tokens)"
                        )
                        shard_idx += 1
                        tokens_in_shard = 0

    # Save final partial shard
    if tokens_in_shard > 0:
        shard_path = os.path.join(args.output_dir, f"train_{shard_idx:06d}.npy")
        np.save(shard_path, shard_buffer[:tokens_in_shard])
        tqdm.write(
            f"Saved final shard {shard_idx} ({tokens_in_shard / 1e6:.1f}M tokens)"
        )

    print(f"Training completed. Total tokens: {total_tokens:,}")

    # Process validation set from separate files (guaranteed no file overlap)
    if args.val_tokens > 0:
        print(f"\nLoading validation data from {len(val_files)} files...")
        val_dataset = load_dataset("parquet", data_files=val_files, split="train", streaming=True)
        val_iter = iter(val_dataset)

        val_buffer = np.empty(args.val_tokens, dtype=np.uint16)
        val_tokens_collected = 0

        with mp.Pool(num_proc, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
            with tqdm(total=args.val_tokens, unit="tokens", desc="Validation") as pbar:
                for doc_tokens in pool.imap(
                    tokenize_doc, val_iter, chunksize=args.chunk_size
                ):
                    if val_tokens_collected >= args.val_tokens:
                        break

                    if len(doc_tokens) == 0:
                        continue

                    space_left = args.val_tokens - val_tokens_collected
                    take = min(space_left, len(doc_tokens))

                    val_buffer[val_tokens_collected : val_tokens_collected + take] = doc_tokens[:take]
                    val_tokens_collected += take
                    pbar.update(take)

        # Save validation shard
        np.save(val_path, val_buffer[:val_tokens_collected])
        print(f"Saved validation shard ({val_tokens_collected / 1e9:.1f}B tokens)")

    # Process test set from separate files (guaranteed no file overlap)
    if args.test_tokens > 0:
        print(f"\nLoading test data from {len(test_files)} files...")
        test_dataset = load_dataset("parquet", data_files=test_files, split="train", streaming=True)
        test_iter = iter(test_dataset)

        test_buffer = np.empty(args.test_tokens, dtype=np.uint16)
        test_tokens_collected = 0

        with mp.Pool(num_proc, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
            with tqdm(total=args.test_tokens, unit="tokens", desc="Test") as pbar:
                for doc_tokens in pool.imap(
                    tokenize_doc, test_iter, chunksize=args.chunk_size
                ):
                    if test_tokens_collected >= args.test_tokens:
                        break

                    if len(doc_tokens) == 0:
                        continue

                    space_left = args.test_tokens - test_tokens_collected
                    take = min(space_left, len(doc_tokens))

                    test_buffer[test_tokens_collected : test_tokens_collected + take] = doc_tokens[:take]
                    test_tokens_collected += take
                    pbar.update(take)

        # Save test shard
        np.save(test_path, test_buffer[:test_tokens_collected])
        print(f"Saved test shard ({test_tokens_collected / 1e9:.1f}B tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset and save as shards"
    )

    parser.add_argument(
        "--dataset",
        default="mlfoundations/dclm-baseline-1.0-parquet",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--tokenizer",
        default="togethercomputer/LLaMA-2-7B-32K",
        help="Transformers tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        default="~/datasets/dclm_tokenized",
        help="Output root directory for shards",
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=1024**3,
        help="Tokens per shard (default: 1B tokens)",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=10 * 1024**3,
        help="Total tokens to process (default: 10B tokens)",
    )
    parser.add_argument(
        "--val_tokens",
        type=int,
        default=1024**3,
        help="Validation tokens to process (default: 1B tokens)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of dataset to reserve for validation (default: 0.05)",
    )
    parser.add_argument(
        "--test_tokens",
        type=int,
        default=1024**3,
        help="Test tokens to process (default: 1B tokens)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.05,
        help="Fraction of dataset to reserve for test (default: 0.05)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for dataset shuffling"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=10000, help="Shuffle buffer size"
    )

    parser.add_argument(
        "--num_proc", type=int, default=-1, help="Number of processes (-1 for auto)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=256, help="Processing chunk size"
    )

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    main(args)
