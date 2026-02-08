#!/usr/bin/env python3
"""Download models from HuggingFace Hub."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model ID to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save the model. If not specified, uses HuggingFace cache.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision (branch, tag, or commit hash)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for private/gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Only download files matching these patterns (e.g., '*.safetensors')",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Exclude files matching these patterns",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Download to HuggingFace cache only (default if --output-dir not specified)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: 'huggingface_hub' package is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading model: {args.model_id}")
    print(f"Revision: {args.revision}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    else:
        print("Output: HuggingFace cache directory")
    print()

    try:
        local_dir = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=args.output_dir,
            token=args.token,
            allow_patterns=args.include,
            ignore_patterns=args.exclude,
        )
        print(f"Model downloaded successfully!")
        print(f"Location: {local_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
