---
license: cc-by-2.0
task_categories:
  - text-generation
  - translation
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

The source corpora are [finnnnnnnnnnnn/toki-pona-sentences](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences)
and [NetherQuartz/tatoeba-tokipona](https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona).

## Preparation

Monolingual sentences and both directions of every multilingual `source`/`tok` pair are written to
`data/train.jsonl`; held-out data goes to `data/eval.jsonl`:

```
<|im_start|>assistant
toki a. sina pona.<|im_end|>

<|im_start|>assistant
English: I am happy.
Toki Pona: mi pilin pona.<|im_end|>
```

A 4096-token BPE tokenizer (special tokens `<pad>`, `<|im_start|>`, `<|im_end|>`) is trained on the
formatted text. This is done by:

```bash
python -m tokilm prepare
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("NetherQuartz/tatoeba-tokipona")
print(ds["train"][0])
```

## Links

- **Model:** [huggingface.co/Grizzlykw/tokilm-9m-chat](https://huggingface.co/Grizzlykw/tokilm-9m-chat)
- **Source dataset:** [huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences](https://huggingface.co/datasets/finnnnnnnnnnnn/toki-pona-sentences)
- **Translation dataset:** [huggingface.co/datasets/NetherQuartz/tatoeba-tokipona](https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona)

## License

Prepared translation data retains the source dataset's CC BY 2.0 license.
