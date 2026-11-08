#!/usr/bin/env bash
set -euo pipefail
# 新实验：MODEL_CONFIG=configs/neurotwin_multiscale_phase.json bash scripts/pretrain.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" pretrain "$@"
