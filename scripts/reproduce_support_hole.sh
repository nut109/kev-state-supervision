#!/usr/bin/env bash
set -euo pipefail
out="${1:-runs/support_hole/results.json}"
python -m kev.latent_support_hole --seed 20260929 --out "$out"
