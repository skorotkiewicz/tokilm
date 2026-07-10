# TokiLM

<p align="center">
  <img src="assets/tokilm.png" alt="TokiLM" width="400"/>
</p>

<p align="center"><em>A ~9M parameter LLM that speaks and translates Toki Pona.</em></p>

<p align="center">
  <a href="https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences"><img src="https://img.shields.io/badge/🤗_Dataset-toki--pona--sentences-blue" alt="Dataset"/></a>&nbsp;
  <a href="https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona"><img src="https://img.shields.io/badge/🤗_Dataset-tatoeba--tokipona-blue" alt="Translation dataset"/></a>&nbsp;
  <a href="https://huggingface.co/Grizzlykw/tokilm-9m-chat"><img src="https://img.shields.io/badge/🤗_Model-tokilm--9m-orange" alt="Model"/></a>&nbsp;
  <a href="https://github.com/skorotkiewicz/tokilm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
</p>

---

> **This project exists to show that training your own language model is not magic.**
> No PhD required. No massive GPU cluster. A small repo, a few minutes on a T4 GPU, and you have a working LLM that you built from scratch — data prep, tokenizer, model architecture, training loop, and inference. If you can run a script, you can train a language model.
>
> It won't write essays. But it will show you exactly how every piece works — from raw text to trained weights to generated output — so the big models stop feeling like black boxes.

---

```
You> toki
TokiLM> toki a. sina pona. mi wile toki pona.

You> moku li pona
TokiLM> moku li pona tawa mi. mi moku e kili.

You> sina pilin seme
TokiLM> mi pilin pona. mi lukin e suno.
```

---

## What is TokiLM?

TokiLM is a tiny language model that speaks [Toki Pona](https://en.wikipedia.org/wiki/Toki_Pona) — a minimalist constructed language with ~120 root words. It generates short sentences and learns translation between Toki Pona and the languages in its aligned corpus.

It's trained from scratch on the [Toki Pona sentence corpus](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences) and [Tatoeba translation pairs](https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona), runs on a single GPU, and produces a model small enough to run in a browser.

---

## Architecture

| | |
|---|---|
| **Parameters** | 8.7M |
| **Layers** | 6 |
| **Hidden dim** | 384 |
| **Heads** | 6 |
| **FFN** | 768 (ReLU) |
| **Vocab** | 4,096 (BPE) |
| **Max sequence** | 128 tokens |
| **Norm** | LayerNorm |
| **Position** | Learned embeddings |
| **LM head** | Weight-tied with embeddings |

Vanilla transformer. No GQA, no RoPE, no SwiGLU, no early exit. As simple as it gets.

---

## Quick Start

### Chat locally

```bash
pip install torch tokenizers
python -m tokilm chat --prompt "toki"
```

Or interactively:

```bash
python -m tokilm chat
```

The CLI runs a second inference for every reply and appends its English translation in parentheses.

```
You> sina pona
TokiLM> pona. sina pilin seme. (Good. How do you feel?)
```

### Train your own

```bash
python -m tokilm prepare   # download corpus + train BPE tokenizer -> data/
python -m tokilm train     # train for 10k steps -> checkpoints/
```

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync --extra cpu # or --extra gpu

uv run python -m tokilm prepare
uv run python -m tokilm train --device cuda --max-steps 50000
uv run python -m tokilm chat --checkpoint checkpoints/final_model.pt --tokenizer data/tokenizer.json --prompt "toki"

# Translation prompts use `Source language: text` followed by `Target language:`
uv run python -m tokilm chat --checkpoint checkpoints/final_model.pt --tokenizer data/tokenizer.json --prompt $'English: I am happy.\nToki Pona:'

uv run python -m tokilm chat --checkpoint hf_export/pytorch_model.bin --tokenizer hf_export/tokenizer.json
```

Quick smoke run:

```bash
uv run python -m tokilm prepare --n-samples 64 --data-dir data
uv run python -m tokilm train --max-steps 1 --data-dir data --output-dir .smoke-checkpoints --device cpu # or --device cuda
```

---

## Export

Export a trained checkpoint for sharing or browser inference:

```bash
python tools/export_model.py --repo Grizzlykw/tokilm-9m-chat --token hf_xxx   # HuggingFace: pytorch_model.bin + config.json + tokenizer.json
python tools/export_onnx.py --push                                          # ONNX (quantized uint8) for onnxruntime-web
python tools/export_dataset.py --repo your-username/tokilm-data --token hf_xxx  # push prepared data/*.jsonl to HF
```

Set `HF_TOKEN` and `HF_REPO` (and `HF_DATASET`) in `.env` to avoid passing flags.

---

## Dataset

`python -m tokilm prepare` combines:

- [finnnnnnnnnnnn/toki-pona-sentences](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences) — monolingual Toki Pona sentences.
- [NetherQuartz/tatoeba-tokipona](https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona) — aligned multilingual ↔ Toki Pona pairs. Each `source`/`tok` row is emitted in both directions; its official validation split stays in eval data.

| | |
|---|---|
| Format | `{"text": "<|im_start|>assistant\n...<|im_end|>"}` (one JSON object per line) |
| Translation prompt | `English: I am happy.\nToki Pona:` |
| Preparation | `python -m tokilm prepare` formats both corpora and trains a 4096-token BPE tokenizer |

```python
from datasets import load_dataset
ds = load_dataset("NetherQuartz/tatoeba-tokipona")
print(ds["train"][0]["source"], ds["train"][0]["tok"])
```

---

## Project Structure

```
tokilm/
├── config.py               Hyperparameters (model + training)
├── model.py                Vanilla transformer
├── dataset.py              Data loading + batching
├── train.py                Training loop (cosine LR, AMP)
├── eval_cases.py           Held-out test cases
├── prepare_data.py         Data prep + tokenizer training
└── inference.py            Chat interface

tools/
├── make_colab.py           Generates Colab notebooks (train_tokilm.ipynb, use_tokilm.ipynb)
├── export_onnx.py          Export model to ONNX (quantized uint8)
├── export_model.py         Export model to HuggingFace format
└── export_dataset.py       Push prepared data to HuggingFace
```

---

## Design Decisions

**Why no system prompt?** Generation remains completion-style. Translation uses short `Source language:` / `Target language:` labels, which fit the 128-token context without adding an instruction template.

**Why single-turn only?** Multi-turn degrades at turn 3-4 due to the 128-token context window. Single-turn is reliable.

**Why vanilla transformer?** GQA, SwiGLU, RoPE, and early exit add complexity that doesn't help at 9M params. Standard attention + ReLU FFN + LayerNorm produces the same quality with simpler code.

**Why Toki Pona?** A tiny vocabulary and simple grammar are a good fit for a tiny model — it can actually learn the language within 10k steps on a single GPU.

---

## License

MIT

---

## Credits

TokiLM is based on [GuppyLM](https://github.com/arman-bd/guppylm) by Arman Hossain — a from-scratch tiny LLM tutorial. The model architecture, training loop, and tooling follow GuppyLM, retargeted to speak Toki Pona.
