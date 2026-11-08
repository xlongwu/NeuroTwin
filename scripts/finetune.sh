#!/usr/bin/env bash
set -euo pipefail
# MODEL_CONFIG=configs/neurotwin_multiscale_phase.json HC_WEIGHT=<新HC best> bash scripts/finetune.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" finetune "$@"
