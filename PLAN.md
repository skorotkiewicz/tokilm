# TokiLM Plan

## Goal

Create a small Toki Pona-only language model based on GuppyLM.

The lazy path is to keep GuppyLM's existing tiny transformer, tokenizer flow,
training loop, and inference code, then replace only the dataset preparation.
Do not fine-tune the published fish model: train from scratch so the model does
not inherit GuppyLM's English/fish behavior.

## Dataset

Use `finnnnnnnnnnnn/toki-pona-sentences` from Hugging Face:

- 19,515 rows
- 597 kB total file size
- one `train` split
- one text column: `text`
- license: CC BY-SA 4.0
- source material from `liputenpo.org`, `toki-ramble`, and `lipukule.org`

## Implementation

1. Reuse the existing GuppyLM runtime code in the `tokilm` package.
2. Replace synthetic data generation in `prepare_data.py` with Hugging Face
   dataset loading:
   ```python
   from datasets import load_dataset

   ds = load_dataset("finnnnnnnnnnnn/toki-pona-sentences", split="train")
   ```
3. Shuffle with seed `42`, split 95/5, and write the existing JSONL format:
   ```json
   {"text": "<|im_start|>assistant\nsina pona\n<|im_end|>"}
   ```
4. Train a fresh tokenizer from the Toki Pona text only.
5. Keep the default GuppyLM model config unless training proves too slow:
   - vocab size: `4096`
   - max sequence length: `128`
   - layers: `6`
   - hidden size: `384`
   - heads: `6`
6. Keep the existing train and chat entrypoints:
   ```bash
   python -m tokilm prepare
   python -m tokilm train
   python -m tokilm chat --prompt "toki"
   ```

## Checks

- `prepare` creates `data/train.jsonl`, `data/eval.jsonl`, and
  `data/tokenizer.json`.
- tokenizer encode/decode round-trips a few Toki Pona examples.
- one short smoke training run reaches a finite loss and writes a checkpoint.
- chat loads the checkpoint and produces non-empty Toki Pona-looking text.

## Later Only If Needed

- Reduce `vocab_size` to `1024` or `2048` if the tokenizer is wasteful on this
  small corpus.
- Add stricter output filtering if the model emits non-Toki Pona text.
- Export to ONNX only after the PyTorch checkpoint is usable.
