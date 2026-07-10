# TokiLM

Tiny Toki Pona language model.

## Usage

```bash
python -m tokilm prepare
python -m tokilm train
python -m tokilm chat --prompt "toki"
```

For a quick training smoke run:

```bash
python -m tokilm prepare --n-samples 64 --data-dir /tmp/tokilm-data
python -m tokilm train --max-steps 1 --data-dir /tmp/tokilm-data --output-dir /tmp/tokilm-checkpoints --device cpu
```
