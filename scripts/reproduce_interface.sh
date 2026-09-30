#!/usr/bin/env bash
set -euo pipefail
parser="${1:-runs/hard_iit/onepass_rule_best.pt}"
out="${2:-runs/state_interface}"
device="${3:-cuda}"
python -m kev.latent_interface_run prepare --parser-checkpoint "$parser" --out "$out"
python -m kev.latent_interface_run cache --out "$out" --device "$device"
python -m kev.latent_interface_run train --out "$out"
python -m kev.latent_interface_run freeze --out "$out"
python -m kev.latent_interface_run cache_locked --out "$out" --device "$device"
python -m kev.latent_interface_run evaluate --out "$out"
