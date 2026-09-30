#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/hard_iit}"
device="${2:-cuda}"
for arm in gold deep r3 r3_state r3_iit; do
  for split in id_test ood_test; do
    python -m kev.latent_iit_trials evaluate --out "$out" --device "$device" --arm "$arm" --split "$split" --allow-locked-test
  done
done
