#!/usr/bin/env python3
"""Download the hendrycks-MATH benchmark dataset from HuggingFace."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download the hendrycks-MATH benchmark dataset from HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="nlile/hendrycks-MATH-benchmark",
        help="HuggingFace dataset ID to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/math",
        help="Directory to save the dataset files",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required. Install with: pip install datasets")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {args.dataset_id}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Splits: {args.splits}")
    print()

    try:
        dataset = load_dataset(args.dataset_id)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    for split in args.splits:
        if split not in dataset:
            print(f"Warning: Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
            continue

        output_file = output_dir / f"{split}.jsonl"

        if output_file.exists() and not args.force:
            print(f"Skipping {split}: {output_file} already exists (use --force to overwrite)")
            continue

        print(f"Saving {split} split ({len(dataset[split])} examples) to {output_file}")

        with open(output_file, "w") as f:
            for example in dataset[split]:
                f.write(json.dumps(example) + "\n")

        print(f"  Saved {len(dataset[split])} examples")

    print()
    print("Download complete!")


if __name__ == "__main__":
    main()
