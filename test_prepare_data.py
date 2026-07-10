import json
import tempfile
from pathlib import Path

from tokilm.prepare_data import format_sample, split_texts, write_jsonl


def main():
    train, eval_ = split_texts(["sina pona", "", " mi   moku ", "toki"], eval_ratio=0.34, seed=1)

    assert len(train) == 2
    assert len(eval_) == 1
    assert all(text.strip() == text for text in train + eval_)
    assert format_sample("sina pona") == "<|im_start|>assistant\nsina pona<|im_end|>"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        write_jsonl(path, ["sina pona"])
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row == {"text": "<|im_start|>assistant\nsina pona<|im_end|>"}


if __name__ == "__main__":
    main()
