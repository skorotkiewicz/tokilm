---
license: mit
language:
  - tok
tags:
  - tokipona
  - tiny-llm
  - text-generation
  - from-scratch
pipeline_tag: text-generation
---

<p align="center">
  <a href="https://github.com/skorotkiewicz/tokilm"><img src="https://img.shields.io/badge/GitHub-tokilm-181717?logo=github" alt="GitHub"/></a>&nbsp;
  <a href="https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences"><img src="https://img.shields.io/badge/🤗_Dataset-toki--pona--sentences-blue" alt="Dataset"/></a>&nbsp;
  <a href="https://colab.research.google.com/github/skorotkiewicz/tokilm/blob/main/train_tokilm.ipynb"><img src="https://img.shields.io/badge/Open_in-Colab-F9AB00?logo=googlecolab" alt="Colab"/></a>
</p>

# TokiLM — 9M Parameter Toki Pona Language Model

A ~9M parameter LLM trained from scratch that speaks [Toki Pona](https://en.wikipedia.org/wiki/Toki_Pona).

This project exists to show that training your own language model is not magic. One small repo,
a few minutes on a T4 GPU, and you have a working LLM built from scratch.

## Example

```
You> toki
TokiLM> toki a. sina pona. mi wile toki pona.

You> moku li pona
TokiLM> moku li pona tawa mi. mi moku e kili.

You> sina pilin seme
TokiLM> mi pilin pona. mi lukin e suno.
```

## Architecture

| | |
|---|---|
| **Parameters** | 8.7M |
| **Type** | Vanilla transformer (from scratch) |
| **Layers** | 6 |
| **Hidden dim** | 384 |
| **Heads** | 6 |
| **FFN** | 768 (ReLU) |
| **Vocab** | 4,096 (BPE) |
| **Max sequence** | 128 tokens |
| **Norm** | LayerNorm |
| **Position** | Learned embeddings |
| **LM head** | Weight-tied with embeddings |

No GQA, no RoPE, no SwiGLU, no early exit. As simple as it gets.

## Training

- **Data:** [finnnnnnnnnnnn/toki-pona-sentences](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences) — a corpus of Toki Pona sentences
- **Steps:** 10,000
- **Optimizer:** AdamW (cosine LR schedule)
- **Hardware:** T4 GPU (~5 min)
- **No system prompt** — the language is baked into the weights

## Usage

```bash
python -m tokilm chat --checkpoint pytorch_model.bin --tokenizer tokenizer.json
```

Or programmatically:

```python
from tokilm.inference import TokiInference

engine = TokiInference("pytorch_model.bin", "tokenizer.json")
r = engine.chat_completion([{"role": "user", "content": "toki"}])
print(r["choices"][0]["message"]["content"])
```

## Links

- **Repo:** [github.com/Grizzlykw/tokilm](https://github.com/Grizzlykw/tokilm)
- **Dataset:** [huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences)

## License

MIT
