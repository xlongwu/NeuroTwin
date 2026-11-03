#!/usr/bin/env bash
# B0–B3 共用原训练入口与被试划分；其余参数原样传给 main.py。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
experiment="${1:?用法: bash scripts/train_individualized.sh B0|B1|B2|B3 [main.py 参数]}"
shift
case "$experiment" in
  B0|B1|B2|B3) ;;
  *) echo "未知实验: $experiment" >&2; exit 1 ;;
esac
args=(--dynamics_experiment "$experiment")
[[ -z "${DATA_ROOT:-}" ]] || args+=(--data_root "$DATA_ROOT")
[[ -z "${CHECKPOINT_DIR:-}" ]] || args+=(--checkpoint_dir "$CHECKPOINT_DIR")
[[ -z "${DEVICE:-}" ]] || args+=(--device "$DEVICE")
exec "${PYTHON:-python}" -u main.py "${args[@]}" "$@"
