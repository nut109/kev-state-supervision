#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q tests/test_latent_data.py tests/test_latent_iit.py tests/test_latent_iit_encoder.py tests/test_latent_iit_trials.py tests/test_release.py
