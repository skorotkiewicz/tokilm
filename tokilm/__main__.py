"""Entry point for: python -m tokilm"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="tokilm")
    sub = parser.add_subparsers(dest="cmd")

    prep = sub.add_parser("prepare", help="Download Toki Pona and translation data, then train tokenizer")
    prep.add_argument("--data-dir", default="data")
    prep.add_argument("--n-samples", type=int)
    prep.add_argument("--eval-ratio", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=42)

    train_p = sub.add_parser("train", help="Train TokiLM")
    train_p.add_argument("--data-dir", default="data")
    train_p.add_argument("--output-dir", default="checkpoints")
    train_p.add_argument("--max-steps", type=int)
    train_p.add_argument("--device")

    chat = sub.add_parser("chat", help="Chat with a trained checkpoint")
    chat.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    chat.add_argument("--tokenizer", default="data/tokenizer.json")
    chat.add_argument("--device", default="cpu")
    chat.add_argument("--prompt", "-p")

    args = parser.parse_args()
    if args.cmd == "prepare":
        from .prepare_data import prepare

        prepare(args.data_dir, args.n_samples, args.eval_ratio, args.seed)
    elif args.cmd == "train":
        from .train import train

        train(args.max_steps, args.data_dir, args.output_dir, args.device)
    elif args.cmd == "chat":
        from .inference import main as chat_main

        sys.argv = [sys.argv[0]]
        for name in ("checkpoint", "tokenizer", "device", "prompt"):
            value = getattr(args, name)
            if value is not None:
                sys.argv.extend([f"--{name.replace('_', '-')}", str(value)])
        chat_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
