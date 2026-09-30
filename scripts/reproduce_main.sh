#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/hard_iit}"
device="${2:-cuda}"
if [[ -e "$out" ]]; then echo "Choose a new output directory: $out already exists" >&2; exit 1; fi
python -m kev.latent_iit_run prepare --out "$out"
python -m kev.latent_iit_run gold_smoke --out "$out" --device "$device"
python -m kev.latent_iit_run gold_train --out "$out" --device "$device"
python -m kev.latent_iit_run gold_stress --out "$out" --device "$device"
python -m kev.latent_iit_run onepass_smoke --out "$out" --device "$device"
python -m kev.latent_iit_run onepass_train --out "$out" --device "$device" --epochs 4
python -m kev.latent_iit_run onepass_stress --out "$out" --device "$device"
python -m kev.latent_iit_trials init --out "$out" --device "$device"
for arm in deep r3 r3_state r3_iit; do
  python -m kev.latent_iit_trials smoke --out "$out" --device "$device" --arm "$arm"
  python -m kev.latent_iit_trials train --out "$out" --device "$device" --arm "$arm" --epochs 4
done
echo "Validation checkpoints ready. Inspect selections, then run scripts/evaluate_main.sh $out $device"
