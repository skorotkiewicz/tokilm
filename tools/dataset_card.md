---
license: mit
task_categories:
  - text-generation
language:
  - tok
tags:
  - tokipona
  - tiny-llm
  - from-scratch
pretty_name: TokiLM Training Data
---

# TokiLM Training Data

The data used to train [TokiLM](https://huggingface.co/Grizzlykw/tokilm-9m-chat), a ~9M parameter
language model that speaks [Toki Pona](https://en.wikipedia.org/wiki/Toki_Pona).

## Source

The raw corpus is [finnnnnnnnnnnn/toki-pona-sentences](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences)
— a collection of Toki Pona sentences.

## Preparation

Each sentence is wrapped as a single assistant turn in ChatML and written to `data/train.jsonl`
(and a small eval split to `data/eval.jsonl`):

```
<|im_start|>assistant
toki a. sina pona.<|im_end|>
```

A 4096-token BPE tokenizer (special tokens `<pad>`, `<|im_start|>`, `<|im_end|>`) is trained on the
formatted text. This is done by:

```bash
python -m tokilm prepare
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("finnnnnnnnnnnn/toki-pona-sentences")
print(ds["train"][0])
```

## Links

- **Model:** [huggingface.co/Grizzlykw/tokilm-9m-chat](https://huggingface.co/Grizzlykw/tokilm-9m-chat)
- **Source dataset:** [huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences)

## License

MIT
