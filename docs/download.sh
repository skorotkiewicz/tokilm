#!/usr/bin/env bash
# Download model.onnx and tokenizer.json from HuggingFace into this directory.

set -e

REPO="Grizzlykw/tokilm-9m-chat"
BASE="https://huggingface.co/${REPO}/resolve/main"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Downloading from ${REPO}..."
curl -fSL "${BASE}/model.onnx" -o "${DIR}/model.onnx"
echo "  model.onnx     $(du -h "${DIR}/model.onnx" | cut -f1)"

curl -fSL "${BASE}/tokenizer.json" -o "${DIR}/tokenizer.json"
echo "  tokenizer.json $(du -h "${DIR}/tokenizer.json" | cut -f1)"

echo "Done. Run: cd docs && python -m http.server 8080"

To resume this session: pi --session 019f4b75-47b2-72c0-ad61-a3eaaaa2e0c1
