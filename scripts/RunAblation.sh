#!/bin/bash
# ============================================================================
# NeuroTwin 消融实验薄入口：透传参数给 experiments/run_experiments.py
#
# 用法：
#   bash scripts/RunAblation.sh --group G14_MOE --dry-run   # 核对命令
#   bash scripts/RunAblation.sh --only baseline --smoke     # 端到端冒烟
#   bash scripts/RunAblation.sh --priority P0 --seeds 2024  # 正式 P0 组
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[CMD] python experiments/run_experiments.py $*"
python experiments/run_experiments.py "$@"
