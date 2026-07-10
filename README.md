# TokiLM

Tiny Toki Pona language model.

## Usage

```bash
python -m tokilm prepare
python -m tokilm train
python -m tokilm chat --prompt "toki"
```

```bash
uv run python -m tokilm prepare
uv run python -m tokilm train
uv run python -m tokilm chat

uv run python -m tokilm train --data-dir data --output-dir checkpoints --device cpu --max-steps 5000
uv run python -m tokilm train --data-dir data --output-dir checkpoints --device cuda --max-steps 10000
uv run python -m tokilm chat --checkpoint checkpoints/final_model.pt --tokenizer data/tokenizer.json --prompt "toki"
```

For a quick training smoke run:

```bash
uv run python -m tokilm prepare --n-samples 64 --data-dir data
uv run python -m tokilm train --max-steps 1 --data-dir data --output-dir .smoke-checkpoints --device cpu
```
