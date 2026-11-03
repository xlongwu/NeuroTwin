#!/usr/bin/env bash
# 从零启动修改后的 B2，再用本次 B2 最佳权重启动 B3；不读取历史模型。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
run_dir="${1:?用法: bash scripts/train_individualized_full.sh 新输出目录}"
PYTHON="${PYTHON:-python}"
data_root="${DATA_ROOT:-$ROOT/data}"
mkdir "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
source_dir="$run_dir/运行源码"
mkdir -p "$ROOT/.runtime_tmp"
# torch_shm_manager 的 Unix socket 有路径长度限制，临时目录保持短名。
export TMPDIR="$(mktemp -d "$ROOT/.runtime_tmp/n.XXXXXX")"
export TORCH_SHARING_STRATEGY=file_system
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
"$PYTHON" - "$ROOT" "$source_dir" "$data_root" <<'PY'
import shutil
import sys
from pathlib import Path
root, dest, data = map(Path, sys.argv[1:])
dest.mkdir()
for name in ('main.py', 'model.py'):
    shutil.copy2(root / name, dest / name)
for name in ('models', 'train', 'utils', 'analysis', 'configs'):
    for src in (root / name).rglob('*'):
        if src.is_file() and src.suffix in ('.py', '.json'):
            out = dest / src.relative_to(root)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
(dest / 'data').symlink_to(data.resolve(), target_is_directory=True)
PY
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
cp "${BASH_SOURCE[0]}" "$run_dir/启动完整训练.sh"
echo "$$" > "$run_dir/进程.pid"
stage=准备
trap 'code=$?; printf "FAILED stage=%s exit=%s time=%s\n" "$stage" "$code" "$(date -Is)" > "$run_dir/运行状态.txt"; exit "$code"' ERR
common=(--data_root "$data_root" --checkpoint_dir "$run_dir"
        --device "${DEVICE:-cuda:0}" --train_epochs 150
        --train_samples_per_subject 8 --eval_anchors_per_subject 8
        --eval_context_length 64 --eval_batch_size 256 --num_workers 6
        --num_threads 4 --pin_memory False --subject_cache_size 4096
        --save_training_state)
stage=健康预训练
printf 'RUNNING stage=%s time=%s\n' "$stage" "$(date -Is)" > "$run_dir/运行状态.txt"
"$PYTHON" -u "$source_dir/main.py" --dynamics_experiment B2 "${common[@]}" \
    --split_file "$data_root/subject_split_pretrain.json" \
    > "$run_dir/健康预训练.log" 2>&1
stage=病理微调
printf 'RUNNING stage=%s time=%s\n' "$stage" "$(date -Is)" > "$run_dir/运行状态.txt"
"$PYTHON" -u "$source_dir/main.py" --dynamics_experiment B3 "${common[@]}" \
    --split_file "$data_root/subject_split_finetune.json" \
    --hc_reference_split "$data_root/subject_split_pretrain.json" \
    --pretrained_weight "$run_dir/neurotwin_B2_pretrain/base_best.pt" \
    > "$run_dir/病理微调.log" 2>&1
printf 'COMPLETE time=%s\n' "$(date -Is)" > "$run_dir/运行状态.txt"
