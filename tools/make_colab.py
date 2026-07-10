"""Generate the training and chat Colab notebooks.

Writes `train_tokilm.ipynb` and `use_tokilm.ipynb` into the repo root.
Open them in Colab (https://colab.research.google.com/github/skorotkiewicz/tokilm/blob/main/train_tokilm.ipynb)
to train TokiLM or chat with a pretrained checkpoint in the browser.

Usage:
    python tools/make_colab.py
"""

import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(path):
    with open(path) as f:
        return f.read()


def read_for_colab(path):
    """Read a Python file and flatten relative imports for Colab."""
    content = read_file(path)
    content = re.sub(r'from \.(\w+) import', r'from \1 import', content)
    return content


def cell(source, cell_type="code"):
    lines = source.split("\n")
    formatted = [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    base = {"cell_type": cell_type, "metadata": {}, "source": formatted}
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


def md(text):
    return cell(text, "markdown")


def code(text):
    return cell(text, "code")


# Source files to embed in the notebook
FILES = [
    ("config.py",       "tokilm/config.py"),
    ("model.py",        "tokilm/model.py"),
    ("dataset.py",      "tokilm/dataset.py"),
    ("prepare_data.py", "tokilm/prepare_data.py"),
    ("train.py",        "tokilm/train.py"),
    ("inference.py",    "tokilm/inference.py"),
]


def build():
    cells = []

    # ══════════════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "# TokiLM — Tiny Toki Pona Language Model\n"
        "\n"
        "Train a ~9M parameter LLM that speaks and translates Toki Pona.\n"
        "\n"
        "**What this notebook does:**\n"
        "1. Downloads monolingual and aligned translation data from HuggingFace\n"
        "2. Trains a BPE tokenizer on the data\n"
        "3. Trains a 6-layer vanilla transformer (8.7M params)\n"
        "4. Tests generation and translation prompts\n"
        "\n"
        "**Architecture:** 6 layers, 384 dim, 6 heads, ReLU FFN, LayerNorm, 4096 vocab\n"
        "\n"
        "**Runtime:** ~5 min on T4 GPU\n"
        "\n"
        "**Result:** A model that generates and translates short Toki Pona sentences."
    ))

    # ══════════════════════════════════════════════════════════════════
    #  1. SETUP
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 1. Setup\n"
        "\n"
        "Install dependencies and create a clean working directory."
    ))

    cells.append(code(
        "!pip install -q torch tokenizers tqdm numpy datasets huggingface_hub\n"
        "\n"
        "import torch\n"
        "print(f'PyTorch {torch.__version__}')\n"
        "print(f'CUDA: {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
    ))

    cells.append(code(
        "import os, shutil\n"
        "\n"
        "# Start fresh — removes stale files from previous runs\n"
        "if os.path.exists('/content/tokilm'):\n"
        "    shutil.rmtree('/content/tokilm')\n"
        "os.makedirs('/content/tokilm')\n"
        "os.chdir('/content/tokilm')\n"
        "print(f'Working dir: {os.getcwd()}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  2. SOURCE FILES
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 2. Source Files\n"
        "\n"
        "Write the model code to disk. These are the only files needed:\n"
        "- `config.py` — model and training hyperparameters\n"
        "- `model.py` — transformer architecture\n"
        "- `dataset.py` — data loading and batching\n"
        "- `prepare_data.py` — download corpus + train tokenizer\n"
        "- `train.py` — training loop\n"
        "- `inference.py` — chat interface"
    ))

    for display_name, src_path in FILES:
        full_path = os.path.join(PROJECT_ROOT, src_path)
        content = read_for_colab(full_path)
        cells.append(code(f"%%writefile {display_name}\n{content}"))

    # ══════════════════════════════════════════════════════════════════
    #  3. PREPARE DATA
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 3. Prepare Data\n"
        "\n"
        "Download Toki Pona sentences plus multilingual Tatoeba pairs and train a BPE tokenizer.\n"
        "\n"
        "Each sentence is wrapped as a single assistant turn in ChatML:\n"
        "```\n"
        "<|im_start|>assistant\n"
        "toki a. sina pona.<|im_end|>\n"
        "\n"
        "<|im_start|>assistant\n"
        "English: I am happy.\n"
        "Toki Pona: mi pilin pona.<|im_end|>\n"
        "```"
    ))

    cells.append(code(
        "import os\n"
        "from prepare_data import prepare\n"
        "\n"
        "# prepare() downloads both corpora, formats generation and bidirectional translation\n"
        "# samples, and trains a 4096-token BPE tokenizer.\n"
        "os.makedirs('data', exist_ok=True)\n"
        "prepare('data')\n"
        "\n"
        "import json\n"
        "with open('data/train.jsonl') as f:\n"
        "    sample = json.loads(f.readline())\n"
        "print(f'\\nSample:\\n{sample[\"text\"]}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  4. VERIFY ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 4. Verify Architecture\n"
        "\n"
        "Quick sanity check — build the model, print param count, run a dummy forward pass."
    ))

    cells.append(code(
        "from config import TokiConfig\n"
        "from model import TokiLM\n"
        "import torch\n"
        "\n"
        "config = TokiConfig()\n"
        "model = TokiLM(config)\n"
        "print(model.param_summary())\n"
        "print(f'  Layers: {config.n_layers}, Heads: {config.n_heads}, FFN: {config.ffn_hidden}')\n"
        "print(f'  Vocab: {config.vocab_size}, Max seq: {config.max_seq_len}')\n"
        "\n"
        "# Dummy forward pass\n"
        "x = torch.randint(0, config.vocab_size, (2, 32))\n"
        "logits, _ = model(x)\n"
        "print(f'  Forward pass: {x.shape} -> {logits.shape} OK')\n"
        "del model"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  5. TRAIN
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 5. Train\n"
        "\n"
        "10,000 steps with cosine LR schedule. Takes ~2 min on T4.\n"
        "\n"
        "The model learns to:\n"
        "- Generate valid Toki Pona words\n"
        "- Stay in the Toki Pona language\n"
        "- Translate short aligned sentences to and from Toki Pona\n"
        "- Stop generating at the right time (learn the `<|im_end|>` token)"
    ))

    cells.append(code("from train import train\ntrain()"))

    # ══════════════════════════════════════════════════════════════════
    #  6. TEST
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 6. Test\n"
        "\n"
        "Chat with the trained model. Each message is independent (single-turn)."
    ))

    cells.append(code(
        "from inference import TokiInference\n"
        "import torch\n"
        "\n"
        "engine = TokiInference(\n"
        "    'checkpoints/best_model.pt', 'data/tokenizer.json',\n"
        "    device='cuda' if torch.cuda.is_available() else 'cpu'\n"
        ")\n"
        "\n"
        "def chat(prompt):\n"
        "    r = engine.chat_completion([{'role': 'user', 'content': prompt}], max_tokens=64)\n"
        "    return r['choices'][0]['message'].get('content', '').strip()\n"
        "\n"
        "# Test across different Toki Pona seeds\n"
        "tests = [\n"
        "    ('toki',           'greeting'),\n"
        "    ('sina pona',      'praise'),\n"
        "    ('moku li pona',   'food'),\n"
        "    ('telo li lili',   'water'),\n"
        "    ('mi wile musi',   'fun'),\n"
        "    ('sina pilin seme','feelings'),\n"
        "    ('mi sona ala',    'confused'),\n"
        "    ('pipi li lukin',  'bug'),\n"
        "    ('o toki e jan',   'people'),\n"
        "    ('mi olin e sina','love'),\n"
        "    ('sewi li seme',   'sky'),\n"
        "    ('mi lape',        'sleep'),\n"
        "    ('English: I am happy.\\nToki Pona:', 'en → tok'),\n"
        "    ('Toki Pona: sina pona.\\nEnglish:', 'tok → en'),\n"
        "]\n"
        "\n"
        "print(f'{\"Topic\":<12s}  {\"You\":<35s}  TokiLM')\n"
        "print('=' * 100)\n"
        "for prompt, topic in tests:\n"
        "    reply = chat(prompt)\n"
        "    print(f'{topic:<12s}  {prompt:<35s}  {reply[:128]}')\n"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  7. EXPORT & UPLOAD TO HUGGINGFACE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 7. Export & Upload to HuggingFace\n"
        "\n"
        "Export the model in both PyTorch and ONNX (quantized uint8, ~9 MB) formats,\n"
        "then upload everything to HuggingFace in one go.\n"
        "\n"
        "Set your token and repo below."
    ))

    cells.append(code(
        "!pip install -q onnx onnxruntime onnxscript\n"
        "\n"
        "from huggingface_hub import HfApi, login\n"
        "import torch, json, os, shutil\n"
        "from config import TokiConfig\n"
        "from model import TokiLM\n"
        "\n"
        "HF_TOKEN = os.environ.get('HF_TOKEN', '')  # Or paste your token here\n"
        "HF_REPO = os.environ.get('HF_REPO', 'Grizzlykw/tokilm-9m-chat')  # Or change this\n"
        "\n"
        "# Load checkpoint\n"
        "ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)\n"
        "cfg = ckpt['config']\n"
        "os.makedirs('hf_export', exist_ok=True)\n"
        "\n"
        "# ── PyTorch format ──\n"
        "torch.save(ckpt['model_state_dict'], 'hf_export/pytorch_model.bin')\n"
        "\n"
        "with open('hf_export/config.json', 'w') as f:\n"
        "    json.dump({\n"
        "        'model_type': 'tokilm',\n"
        "        'architectures': ['TokiLM'],\n"
        "        'vocab_size': cfg['vocab_size'],\n"
        "        'max_position_embeddings': cfg['max_seq_len'],\n"
        "        'hidden_size': cfg['d_model'],\n"
        "        'num_hidden_layers': cfg['n_layers'],\n"
        "        'num_attention_heads': cfg['n_heads'],\n"
        "        'intermediate_size': cfg['ffn_hidden'],\n"
        "        'hidden_dropout_prob': cfg.get('dropout', 0.1),\n"
        "        'pad_token_id': cfg['pad_id'],\n"
        "        'bos_token_id': cfg['bos_id'],\n"
        "        'eos_token_id': cfg['eos_id'],\n"
        "    }, f, indent=2)\n"
        "\n"
        "shutil.copy('data/tokenizer.json', 'hf_export/tokenizer.json')\n"
        "print(f'pytorch_model.bin: {os.path.getsize(\"hf_export/pytorch_model.bin\")/1e6:.1f} MB')\n"
        "\n"
        "# ── ONNX format (quantized uint8) ──\n"
        "valid_fields = {f.name for f in TokiConfig.__dataclass_fields__.values()}\n"
        "config = TokiConfig(**{k: v for k, v in cfg.items() if k in valid_fields})\n"
        "model = TokiLM(config)\n"
        "model.load_state_dict(ckpt['model_state_dict'])\n"
        "model.eval()\n"
        "\n"
        "dummy = torch.randint(0, config.vocab_size, (1, 32))\n"
        "fp32_path = 'hf_export/model_fp32.onnx'\n"
        "torch.onnx.export(\n"
        "    model, (dummy,), fp32_path,\n"
        "    input_names=['input_ids'], output_names=['logits'],\n"
        "    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'},\n"
        "                  'logits': {0: 'batch', 1: 'seq_len'}},\n"
        "    opset_version=17,\n"
        ")\n"
        "\n"
        "from onnxruntime.quantization import quantize_dynamic, QuantType\n"
        "quantize_dynamic(fp32_path, 'hf_export/model.onnx', weight_type=QuantType.QUInt8)\n"
        "os.remove(fp32_path)\n"
        "print(f'model.onnx:       {os.path.getsize(\"hf_export/model.onnx\")/1e6:.1f} MB (uint8)')\n"
        "\n"
        "# ── Upload ──\n"
        "if HF_TOKEN:\n"
        "    login(token=HF_TOKEN)\n"
        "    api = HfApi()\n"
        "    api.create_repo(HF_REPO, exist_ok=True)\n"
        "    api.upload_folder(folder_path='hf_export', repo_id=HF_REPO, repo_type='model')\n"
        "    print(f'Done! https://huggingface.co/{HF_REPO}')\n"
        "else:\n"
        "    print('No HF_TOKEN — exported locally to hf_export/')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  8. DOWNLOAD
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 8. Download\n"
        "\n"
        "Or download the model locally as a tar.gz."
    ))

    cells.append(code(
        "import os\n"
        "\n"
        "!cd /content && tar czf tokilm.tar.gz \\\n"
        "    tokilm/checkpoints/best_model.pt \\\n"
        "    tokilm/checkpoints/config.json \\\n"
        "    tokilm/data/tokenizer.json \\\n"
        "    tokilm/model.py \\\n"
        "    tokilm/config.py \\\n"
        "    tokilm/inference.py \\\n"
        "    tokilm/hf_export/model.onnx\n"
        "\n"
        "sz = os.path.getsize('/content/tokilm.tar.gz') / 1e6\n"
        "print(f'Package: /content/tokilm.tar.gz ({sz:.1f} MB)')\n"
        "\n"
        "try:\n"
        "    from google.colab import files\n"
        "    files.download('/content/tokilm.tar.gz')\n"
        "except ImportError:\n"
        "    print('Not in Colab — download manually from the file browser.')"
    ))

    # ══════════════════════════════════════════════════════════════════

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": "TokiLM — Train"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def build_use():
    """Build the use_tokilm notebook — download model from HF and chat."""
    cells = []

    cells.append(md(
        "# TokiLM — Chat with Toki Pona\n"
        "\n"
        "Download a pre-trained 9M parameter Toki Pona LLM and chat with it. Just run all cells.\n"
        "\n"
        "**Model:** [Grizzlykw/tokilm-9m-chat](https://huggingface.co/Grizzlykw/tokilm-9m-chat)"
    ))

    cells.append(code(
        "# Setup + Download\n"
        "!pip install -q torch tokenizers huggingface_hub\n"
        "import os, shutil\n"
        "if os.path.exists('/content/tokilm'): shutil.rmtree('/content/tokilm')\n"
        "os.makedirs('/content/tokilm'); os.chdir('/content/tokilm')\n"
        "\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(repo_id='Grizzlykw/tokilm-9m-chat', local_dir='.')\n"
        "print('Model downloaded.')"
    ))

    cells.append(md(
        "## Source files\n"
        "\n"
        "The HF repo ships weights but not the model code, so write it here.\n"
        "Relative imports are flattened to plain imports for Colab (no package)."
    ))
    for display_name, src_path in [
        ("config.py", "tokilm/config.py"),
        ("model.py", "tokilm/model.py"),
        ("inference.py", "tokilm/inference.py"),
    ]:
        full_path = os.path.join(PROJECT_ROOT, src_path)
        content = read_for_colab(full_path)
        cells.append(code(f"%%writefile {display_name}\n{content}"))

    cells.append(code(
        "# Load model\n"
        "from inference import TokiInference\n"
        "import torch\n"
        "\n"
        "engine = TokiInference('pytorch_model.bin', 'tokenizer.json',\n"
        "                        device='cuda' if torch.cuda.is_available() else 'cpu')\n"
        "\n"
        "def chat(prompt):\n"
        "    return engine.chat_completion(\n"
        "        [{'role': 'user', 'content': prompt}], max_tokens=64\n"
        "    )['choices'][0]['message'].get('content', '').strip()\n"
        "\n"
        "# Quick test\n"
        "for p in ['toki', 'sina pona', 'moku li pona', 'sina sona ala sona e toki pona', 'mi lape']:\n"
        "    print(f'You> {p}\\nTokiLM> {chat(p)}\\n')"
    ))

    cells.append(code(
        "# Interactive chat — type your messages\n"
        "while True:\n"
        "    try:\n"
        "        p = input('You> ').strip()\n"
        "    except (KeyboardInterrupt, EOFError):\n"
        "        break\n"
        "    if not p or p.lower() in ('quit', 'exit', 'q'):\n"
        "        print('TokiLM> bye.'); break\n"
        "    print(f'TokiLM> {chat(p)}\\n')"
    ))

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "TokiLM — Chat"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def write_notebook(nb, filename):
    out = os.path.join(PROJECT_ROOT, filename)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    n = len(nb["cells"])
    sz = os.path.getsize(out) / 1024
    print(f"Generated {out}: {n} cells, {sz:.1f} KB")


if __name__ == "__main__":
    write_notebook(build(), "train_tokilm.ipynb")
    write_notebook(build_use(), "use_tokilm.ipynb")
