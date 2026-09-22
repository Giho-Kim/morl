#!/usr/bin/env bash
set -euo pipefail

seeds="${SEEDS:-0 1 2 3}"
output="${OUTPUT:-runs_ant_d_prior}"
device="${DEVICE:-auto}"

for seed in $seeds; do
  python train.py \
    --env mo_ant \
    --method d \
    --seed "$seed" \
    --device "$device" \
    --no-tilted-behavior \
    --output "$output"
done
