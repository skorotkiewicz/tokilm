"""Prepare Toki Pona training data for TokiLM."""

import json
import os
import random

DATA_DIR = "data"
DATASET_NAME = "finnnnnnnnnnnn/toki-pona-sentences"
TRANSLATION_DATASET_NAME = "NetherQuartz/tatoeba-tokipona"
VOCAB_SIZE = 4096

SPECIAL_TOKENS = [
    "<pad>",         # 0
    "<|im_start|>",  # 1
    "<|im_end|>",    # 2
]


def clean_text(text):
    return " ".join((text or "").split())


def format_sample(*lines):
    return f"<|im_start|>assistant\n{'\n'.join(clean_text(line) for line in lines)}<|im_end|>"


def translation_samples(rows, n_samples=None):
    samples = []
    for row in rows:
        source = clean_text(row.get("source"))
        source_lang = clean_text(row.get("source_lang"))
        toki_pona = clean_text(row.get("tok"))
        if not source or not source_lang or not toki_pona:
            continue

        # ponytail: source/tok already spans the source languages; fan-out columns can wait for benchmarks.
        samples.extend([
            format_sample(f"{source_lang}: {source}", f"Toki Pona: {toki_pona}"),
            format_sample(f"Toki Pona: {toki_pona}", f"{source_lang}: {source}"),
        ])
        if n_samples is not None and len(samples) >= n_samples * 2:
            break
    return samples


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


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for text in samples:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


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

    sentences = load_dataset(DATASET_NAME, split="train")
    train_texts, eval_texts = split_texts(
        (row["text"] for row in sentences),
        n_samples=n_samples,
        eval_ratio=eval_ratio,
        seed=seed,
    )
    train_samples = [format_sample(text) for text in train_texts]
    eval_samples = [format_sample(text) for text in eval_texts]

    print(f"Loading {TRANSLATION_DATASET_NAME}...")
    translations = load_dataset(TRANSLATION_DATASET_NAME)
    train_samples.extend(translation_samples(translations["train"], n_samples))
    eval_samples.extend(translation_samples(translations["validation"], n_samples))
    random.Random(seed).shuffle(train_samples)
    random.Random(seed).shuffle(eval_samples)

    train_path = os.path.join(data_dir, "train.jsonl")
    eval_path = os.path.join(data_dir, "eval.jsonl")
    write_jsonl(train_path, train_samples)
    write_jsonl(eval_path, eval_samples)
    print(f"Prepared {len(train_samples) + len(eval_samples):,} samples:")
    print(f"  Train: {len(train_samples):,}")
    print(f"  Eval:  {len(eval_samples):,}")

    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = train_tokenizer(
        train_samples + eval_samples,
        tokenizer_path,
    )

    test = format_sample("English: I am happy.", "Toki Pona: mi pilin pona.")
    ids = tokenizer.encode(test).ids
    decoded = tokenizer.decode(ids)
    print(f"\nTokenizer test:")
    print(f"  Input:   {test}")
    print(f"  Tokens:  {len(ids)} ids")
    print(f"  Decoded: {decoded}")


if __name__ == "__main__":
    prepare()
