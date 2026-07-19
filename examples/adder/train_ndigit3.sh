#!/usr/bin/env bash
set -euo pipefail

# Train the adder demo on 3-digit addition.
# 训练 3 位数加法模型。
#
# Usage:
#   bash examples/adder/train_ndigit3.sh
#
# Optional overrides:
#   MAX_ITERS=50000 bash examples/adder/train_ndigit3.sh
#   BATCH_SIZE=128 MAX_ITERS=50000 bash examples/adder/train_ndigit3.sh

python projects/adder/adder.py \
  --data.ndigit=3 \
  --trainer.max_iters="${MAX_ITERS:-20000}" \
  --trainer.batch_size="${BATCH_SIZE:-64}"
