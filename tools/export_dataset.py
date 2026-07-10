"""
Export TokiLM training data to HuggingFace.

TokiLM's training data is prepared locally by `python -m tokilm prepare`,
which downloads the monolingual and Tatoeba translation corpora and writes:
    data/train.jsonl
    data/eval.jsonl
each line: {"text": "<|im_start|>assistant\\n...<|im_end|>"}

This tool pushes those prepared files to a HuggingFace dataset repo.

Usage:
    python tools/export_dataset.py --repo your-username/tokilm-data --token hf_xxx
    python tools/export_dataset.py --local-only          # just save locally
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def push_to_hub(train_data, eval_data, repo_id, token):
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "eval": Dataset.from_list(eval_data),
    })

    print(f"\nDataset:")
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Eval:  {len(eval_data):,} samples")
    print(f"  Columns: {list(ds['train'].column_names)}")

    print(f"\nPushing to {repo_id}...")
    ds.push_to_hub(repo_id, token=token)

    card_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_card.md")
    if os.path.exists(card_path):
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Pushed README.md")

    print(f"Done! https://huggingface.co/datasets/{repo_id}")


def save_local(train_data, eval_data, output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train.jsonl", train_data), ("eval.jsonl", eval_data)]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {path} ({len(data):,} samples)")


def main():
    parser = argparse.ArgumentParser(description="Export TokiLM dataset to HuggingFace")
    parser.add_argument("--repo", default=None, help="HuggingFace dataset repo (e.g. your-username/tokilm-data)")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument("--data-dir", default="data", help="Directory with train.jsonl/eval.jsonl")
    parser.add_argument("--local-only", action="store_true", help="Save locally without pushing to HF")
    parser.add_argument("--output-dir", default="dataset", help="Local output directory")
    args = parser.parse_args()

    load_env()

    train_path = os.path.join(args.data_dir, "train.jsonl")
    eval_path = os.path.join(args.data_dir, "eval.jsonl")
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        print(f"Error: missing {train_path} / {eval_path}. Run `python -m tokilm prepare` first.")
        sys.exit(1)

    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)

    save_local(train_data, eval_data, args.output_dir)

    if not args.local_only:
        token = args.token or os.environ.get("HF_TOKEN")
        repo = args.repo or os.environ.get("HF_DATASET")
        if not token:
            print("Error: No HF token. Set HF_TOKEN in .env or pass --token")
            sys.exit(1)
        if not repo:
            print("Error: No HF repo. Set HF_DATASET in .env or pass --repo")
            sys.exit(1)
        push_to_hub(train_data, eval_data, repo, token)


if __name__ == "__main__":
    main()
