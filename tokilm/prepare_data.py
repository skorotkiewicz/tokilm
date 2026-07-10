"""Prepare Toki Pona training data for TokiLM."""

import json
import os
import random

DATA_DIR = "data"
DATASET_NAME = "finnnnnnnnnnnn/toki-pona-sentences"
VOCAB_SIZE = 4096

SPECIAL_TOKENS = [
    "<pad>",         # 0
    "<|im_start|>",  # 1
    "<|im_end|>",    # 2
]


def clean_text(text):
    return " ".join((text or "").split())


def format_sample(text):
    return f"<|im_start|>assistant\n{clean_text(text)}<|im_end|>"


def split_texts(texts, n_samples=None, eval_ratio=0.05, seed=42):
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio must be >= 0 and < 1")
    if n_samples is not None and n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    items = [clean_text(text) for text in texts]
    items = [text for text in items if text]
    random.Random(seed).shuffle(items)
    if n_samples is not None:
        items = items[:n_samples]

    n_eval = int(len(items) * eval_ratio)
    return items[n_eval:], items[:n_eval]


def write_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": format_sample(text)}, ensure_ascii=False) + "\n")


def train_tokenizer(texts, save_path, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(texts)} texts...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def prepare(data_dir=DATA_DIR, n_samples=None, eval_ratio=0.05, seed=42):
    os.makedirs(data_dir, exist_ok=True)

    print(f"Loading {DATASET_NAME}...")
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="train")
    train_texts, eval_texts = split_texts(
        (row["text"] for row in ds),
        n_samples=n_samples,
        eval_ratio=eval_ratio,
        seed=seed,
    )

    train_path = os.path.join(data_dir, "train.jsonl")
    eval_path = os.path.join(data_dir, "eval.jsonl")
    write_jsonl(train_path, train_texts)
    write_jsonl(eval_path, eval_texts)
    print(f"Prepared {len(train_texts) + len(eval_texts):,} samples:")
    print(f"  Train: {len(train_texts):,}")
    print(f"  Eval:  {len(eval_texts):,}")

    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = train_tokenizer(
        [format_sample(text) for text in train_texts + eval_texts],
        tokenizer_path,
    )

    test = "<|im_start|>assistant\nsina pona<|im_end|>"
    ids = tokenizer.encode(test).ids
    decoded = tokenizer.decode(ids)
    print(f"\nTokenizer test:")
    print(f"  Input:   {test}")
    print(f"  Tokens:  {len(ids)} ids")
    print(f"  Decoded: {decoded}")


if __name__ == "__main__":
    prepare()
