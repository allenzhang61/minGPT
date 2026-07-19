"""
Load a trained character-level GPT checkpoint and generate text.

加载训练好的字符级 GPT checkpoint，并生成文本。
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mingpt.model import GPT
from projects.chargpt.chargpt import CharDataset, get_config


def build_dataset(input_path, block_size):
    with open(input_path, "r") as f:
        text = f.read()
    config = CharDataset.get_default_config()
    config.block_size = block_size
    return CharDataset(config, text)


def build_model(checkpoint_path, dataset, device):
    config = get_config()
    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()

    model = GPT(config.model)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def encode_prompt(dataset, prompt, device):
    missing = sorted(set(prompt) - set(dataset.stoi))
    if missing:
        chars = "".join(repr(ch) for ch in missing)
        raise ValueError(f"prompt contains characters not seen during training: {chars}")
    return torch.tensor([dataset.stoi[ch] for ch in prompt], dtype=torch.long, device=device)[None, ...]


@torch.no_grad()
def sample(model, dataset, prompt, max_new_tokens, temperature, top_k, device):
    x = encode_prompt(dataset, prompt, device)
    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
    )[0]
    return "".join(dataset.itos[int(i)] for i in y)


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained chargpt model.")
    parser.add_argument("--checkpoint", default="out/chargpt/model.pt", help="path to model checkpoint")
    parser.add_argument("--input", default="projects/chargpt/input.txt", help="training text used to build the vocabulary")
    parser.add_argument("--prompt", default="O God, O God!", help="text prompt to continue")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="number of characters to generate")
    parser.add_argument("--block-size", type=int, default=128, help="same block size used during training")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=10, help="sample only from the top-k logits")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device to run on")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}. "
            "Train first with: bash examples/chargpt/train.sh"
        )
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input text not found: {args.input}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    dataset = build_dataset(args.input, args.block_size)
    model = build_model(args.checkpoint, dataset, device)
    print(sample(model, dataset, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device))


if __name__ == "__main__":
    main()
