#!/usr/bin/env bash
set -euo pipefail

# Train the character-level GPT demo.
# 训练字符级 GPT 示例。
#
# Usage:
#   1. Put a training corpus at projects/chargpt/input.txt
#      先把训练文本放到 projects/chargpt/input.txt
#
#   2. Run:
#      bash examples/chargpt/train.sh
#
# Optional overrides:
#   MAX_ITERS=50000 bash examples/chargpt/train.sh
#   BLOCK_SIZE=128 BATCH_SIZE=64 bash examples/chargpt/train.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_FILE="$ROOT/projects/chargpt/input.txt"
PYTHON_BIN="${PYTHON:-python}"

if [[ -x "$ROOT/.venv/bin/python" && -z "${PYTHON:-}" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Missing $INPUT_FILE"
  echo "Create it first, for example: cp /path/to/your/text.txt $INPUT_FILE"
  exit 1
fi

cd "$ROOT/projects/chargpt"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" chargpt.py \
  --system.work_dir="'$ROOT/out/chargpt'" \
  --system.sample_interval="${SAMPLE_INTERVAL:-2000}" \
  --system.sample_tokens="${SAMPLE_TOKENS:-200}" \
  --model.model_type="${MODEL_TYPE:-gpt-mini}" \
  --trainer.max_iters="${MAX_ITERS:-50000}" \
  --trainer.batch_size="${BATCH_SIZE:-64}" \
  --trainer.num_workers="${NUM_WORKERS:-0}" \
  --trainer.device="${DEVICE:-auto}" \
  --data.block_size="${BLOCK_SIZE:-128}"
