"""
Load a trained ndigit=3 adder checkpoint and run a few predictions.

加载已经训练好的 ndigit=3 加法模型，并运行若干预测样例。
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mingpt.model import GPT
from projects.adder.adder import AdditionDataset, get_config


def build_model(checkpoint_path, device):
    config = get_config()
    config.data.ndigit = 3

    dataset = AdditionDataset(config.data, split="test")
    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()

    model = GPT(config.model)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict_add(model, a, b, ndigit, device):
    if not (0 <= a < 10**ndigit and 0 <= b < 10**ndigit):
        raise ValueError(f"inputs must be in [0, {10**ndigit - 1}] for ndigit={ndigit}")

    prompt = f"{a:0{ndigit}d}{b:0{ndigit}d}"
    x = torch.tensor([[int(ch) for ch in prompt]], dtype=torch.long, device=device)
    y = model.generate(x, ndigit + 1, do_sample=False)

    # The model emits the answer digits reversed, matching the training encoding.
    # 模型输出的是反向结果数字，这和训练时的编码方式一致。
    reversed_digits = y[0, -(ndigit + 1):]
    normal_digits = reversed_digits.flip(0)
    return int("".join(str(int(d)) for d in normal_digits))


def parse_pair(text):
    if "+" in text:
        left, right = text.split("+", 1)
    elif "," in text:
        left, right = text.split(",", 1)
    else:
        raise argparse.ArgumentTypeError("pair must look like 123+456 or 123,456")
    return int(left), int(right)


def main():
    parser = argparse.ArgumentParser(description="Test a trained ndigit=3 adder model.")
    parser.add_argument("--checkpoint", default="out/adder/model.pt", help="path to model checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device to run on")
    parser.add_argument("pairs", nargs="*", type=parse_pair, help="addition pairs, e.g. 123+456")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}. "
            "Train first with: bash examples/adder/train_ndigit3.sh"
        )

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model, config = build_model(args.checkpoint, device)
    pairs = args.pairs or [(123, 456), (7, 39), (999, 999), (314, 159), (500, 500)]

    for a, b in pairs:
        pred = predict_add(model, a, b, config.data.ndigit, device)
        truth = a + b
        mark = "OK" if pred == truth else "WRONG"
        print(f"{a:03d} + {b:03d} = {pred:04d}  gt={truth:04d}  {mark}")


if __name__ == "__main__":
    main()
