#!/usr/bin/env bash
# ============================================================================
# NeuroTwin-TFM 预训练（HC 队列，graph_ode 动态路线，方案 docs/Version4_docs_1012）
#
# 任务口径（graph_ode twin 家族 hc_demo_twin）：Raw BOLD（前 32 TR 固定 mu0/s0 校准）
#   → 可见历史 FC → SoftAnatomicalPrior → TFM 四层主干
#   → 受控条件图 ODE（Heun RK2 同步推进 z/q，持续状态）
#   + CPM 头（无控制辅助预测，cpm_intervention=False）
#   + VarianceHead（逐 ROI/horizon 对角方差）
#   + ProfileEncoder（personalized=True，基线前 32 TR → 32 维固定 profile）
# 损失：原 one-step/CPM 项 + 0.5·动态多步(Huber+PCC) + 0.05·方差 NLL
#
# 用法
#   PYTHON="$HOME/anaconda3/envs/mamba/bin/python" bash scripts/Pretrain_HC_tfm.sh
#   # 可选环境变量：
#   #   CHECKPOINT_DIR=<输出目录>  默认 $ROOT/checkpoints（想隔离本轮产物就指到本轮目录）
#   #   DEVICE=cuda:1              不给则沿用 main.py 的设备默认
#   #   DATA_ROOT=<数据根目录>      默认 $ROOT/data
#   #   PRINT_CMD=1                打印完整命令行（默认只打印关键参数摘要）
#
# 条件注入（2026-10-10 选定方案，见 TASK_HANDOFF §0.26）：HC 侧固定
#   ``--condition_group hc3 --condition_arm demo``。hc3 只有共享三字段
#   （Sex/Age/education），结构上不可能用融合臂（``build_condition_encoder`` 会直接拒），
#   所以「cat」落在下游的 MDD 微调脚本里；两者共用同一个锚点宽度 C=32。
#
# ⚠️ 2026-10-10 起本轮不再有 study/fresh 身份闸门（`experiments/condition_study.py` 已删除，
#   §0.27）：主干是否从随机初始化开始，完全由「本脚本不传 --pretrained_weight」这一事实保证，
#   而不是由代码校验。产物是否落在本轮目录由 CHECKPOINT_DIR 决定，脚本不再强制。
#
# ⚠️ 与旧 discrete 路线（neurotwin_tfm_pretrain 等）产物互不兼容：graph_ode 结构
#   由 ARCH_VERSION=6 的 checkpoint 快照重建，旧 discrete 权重不会迁移到 ODE 头。
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# DataLoader 共享内存：固定到项目内目录，避免 Qoder 会话切换清理 TMPDIR 后
# multiprocessing resource_sharer socket 消失（长训练中断的根因）。
export TMPDIR="$ROOT/.runtime_tmp"
mkdir -p "$TMPDIR"
export TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"

PYTHON_BIN="${PYTHON:-python}"
data_root="${DATA_ROOT:-$ROOT/data}"
checkpoint_dir="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
DEVICE="${DEVICE:-}"
# 预训练产物的固定目录名：下游 scripts/Finetune_MDD_tfm.sh 按
# ${name}*/base_best.pt 找起点，两处改名必须同步。
# twin 家族命名（方案 §2.1）：hc_demo_twin / mdd_cat_twin，不与旧
# neurotwin_tfm_pretrain/finetune 产物混用。
name="neurotwin_tfm_pretrain"

# ---- 实验规模（方案 §17 首选起点：D=256, N=4, patch=4, heads=4）----
# 随机来源（模型 seed / 采样 seed / Condition Dropout seed）只在
# utils/condition_schema.py 登记一次，这里读出来用；脚本里抄字面量必然漂移。
# 历史缺陷：本脚本曾把 --sampling_seed 直接写成 --seed 的同一个值（都是 2024），
# 于是模型初始化流与采样流并成一条，方案 §四 的「分离」名存实亡。
read -r seed sampling_seed cond_dropout_seed < <(
  PYTHONPATH=. "$PYTHON_BIN" -c 'from utils.condition_schema import (DROPOUT_SEED, MODEL_SEED, SAMPLING_SEED); print(MODEL_SEED, SAMPLING_SEED, DROPOUT_SEED)') \
  || { echo "[错误] $PYTHON_BIN 读取 utils.condition_schema 的随机来源常量失败（需 PYTHON=<≥3.10 且带 torch 的解释器>）" >&2; exit 1; }
if [[ "$seed" == "$sampling_seed" || "$seed" == "$cond_dropout_seed" \
      || "$sampling_seed" == "$cond_dropout_seed" ]]; then
  echo "[错误] 登记处自身的随机来源就不同值（seed=$seed sampling=$sampling_seed dropout=$cond_dropout_seed），违反方案 §四" >&2
  exit 1
fi
num_rois=116
context_min=16
context_max=64
eval_context_length=64
eval_anchors_per_subject=16
eval_rollout_tasks_per_subject=2
eval_rollout_horizons="1,2,4,8,16"
tfm_dim=256
tfm_layers=4
tfm_heads=4
tfm_patch_len=8                    # 方案 §4.3：{1,4,8} 消融，不用 TimesFM 的 32
cpm_horizon=8                      # 方案 §8：首选 8，随后测 {4,8,16}

# ---- P0 损失扩展（默认=E2 正式基线：Temporal Diff 0.05 + Cross-Head 0.02）----
# 消融可经环境变量覆盖：
#   E0: LAMBDA_TEMPORAL_DIFF=0 LAMBDA_CROSS_HEAD=0 bash scripts/Pretrain_HC_tfm.sh
#   E1: LAMBDA_TEMPORAL_DIFF=0.05 bash scripts/Pretrain_HC_tfm.sh
# 说明：temporal_diff / cross_head 为固定权重项，不进入 uncertainty 聚合；
#       cross_head_start_epoch 仅在 LAMBDA_CROSS_HEAD>0 时生效。
lambda_temporal_diff="${LAMBDA_TEMPORAL_DIFF:-0.05}"
lambda_cross_head="${LAMBDA_CROSS_HEAD:-0.02}"
cross_head_start_epoch="${CROSS_HEAD_START_EPOCH:-5}"
cpm_horizon_weight="${CPM_HORIZON_WEIGHT:-gamma}"
cpm_horizon_alpha="${CPM_HORIZON_ALPHA:-0.5}"
# ---- S2 历史记忆（正式基线默认开启）----
dynamics_history_memory="${DYNAMICS_HISTORY_MEMORY:-True}"

train_epochs="${TRAIN_EPOCHS:-150}"
warmup_epochs=15
# 数据加载并发（只影响吞吐，不影响训练结果）：默认关闭 pin_memory 以避免长时
# 运行中 resource_sharer/pin-memory 线程因临时 socket 消失而中断训练。
num_workers="${NUM_WORKERS:-6}"
pin_memory="${PIN_MEMORY:-False}"
batch_size=256
patience=20
lr_init=5e-5
lr_peak=1e-4
lr_final=5e-5

# ---- 预检查 ----
# graph_ode 路线用 raw + history_fc：只要求 Raw 目录，不要求 Mask（图来自
# 当前可见历史），也不回退 Normlize/窗口缓存（方案 §4.1）。
[ -d "$data_root/Raw/HC" ] || {
  echo "[错误] 缺少 HC Raw 目录（$data_root/Raw/HC/ROISignals_<id>.mat，[T,F] 布局）"; exit 1; }

ARGS=(
  # ---- 基础与路径 ----
  --mode pretrain
  --seed "$seed"
  --data_root "$data_root"
  --checkpoint_dir "$checkpoint_dir"
  --name "$name"
  --num_rois "$num_rois"
  # ---- Next-Timepoint 任务口径 ----
  --task_mode next_timepoint
  --context_min "$context_min"
  --context_max "$context_max"
  --forecast_offsets 1               # TFM 的 one-step 头只建模 +1 偏移（方案 §10）
  --eval_context_length "$eval_context_length"
  --eval_anchors_per_subject "$eval_anchors_per_subject"
  --eval_rollout_tasks_per_subject "$eval_rollout_tasks_per_subject"
  --eval_rollout_horizons "$eval_rollout_horizons"
  --fc_min_length 32
  --eval_fc True
  --eval_spectral False
  # ---- TFM 模型规模 ----
  --tfm_dim "$tfm_dim"
  --tfm_layers "$tfm_layers"
  --tfm_heads "$tfm_heads"
  --tfm_patch_len "$tfm_patch_len"
  --tfm_ff_ratio 4
  --tfm_one_step_hidden_dim 256
  --cpm_horizon "$cpm_horizon"
  --cpm_layers 2
  --cpm_intervention False           # V1 仅预留接口，不做干预预测（方案 §13.1/§30）
  # ---- SC 软先验（A_eff 一次计算全层共享，方案 §7）----
  --sc_prior_mode soft_prior
  --sc_lambda_mode global
  --sc_lambda_init 0.7
  --sc_prior_rank 12
  --sc_delta_a True
  --sc_sinkhorn_iters 64
  # ---- TFM 损失权重（方案 §20；λ_cpm=0 可退化 Stage 0 基线）----
  --lambda_one 1.0
  --lambda_cpm 1.0
  --lambda_pcc 0.1
  --cpm_gamma 1.0                    # 1=逐 horizon 均匀权重
  --cpm_huber_delta 1.0
  # ---- P0 损失扩展（默认关；见文件顶部环境变量说明）----
  --lambda_temporal_diff "$lambda_temporal_diff"
  --lambda_cross_head "$lambda_cross_head"
  --cross_head_start_epoch "$cross_head_start_epoch"
  --cpm_horizon_weight "$cpm_horizon_weight"
  --cpm_horizon_alpha "$cpm_horizon_alpha"
  --norm True                        # Context-only RevIN（方案 §15）
  --dropout 0.1
  # ---- 优化与训练规模 ----
  --train_epochs "$train_epochs"
  --warmup_epochs "$warmup_epochs"
  --batch_size "$batch_size"
  --patience "$patience"
  --lr_init "$lr_init"
  --lr_peak "$lr_peak"
  --lr_final "$lr_final"
  --weight_decay 0.01
  --grad_clip 1.0
  --use_ema True
  --amp True
  --num_workers "$num_workers"       # 见文件顶部 PIN_MEMORY/NUM_WORKERS 说明
  --pin_memory "$pin_memory"
  --eval_batch_size "${EVAL_BATCH_SIZE:-512}"  # 仅影响验证吞吐；降低 Val 峰值内存
  --subject_cache_size 4096         # 覆盖全部被试，消除每 epoch 对全宽 Raw mat 的重复 loadmat
  --train_samples_per_subject 0    # 0=auto（每被试每 epoch min(合法组合数, 12)）
  --sampling_seed "$sampling_seed"
  # --refresh_split_manifest 是 store_true 开关（不接受取值），默认 False 不刷新
)

# ---- 本轮条件注入（选定方案：hc3 + demo）----
ARGS+=(
  --condition_group hc3 --condition_arm demo     # hc3 无新增字段，结构上只能用 demo
  --cond_dropout_seed "$cond_dropout_seed"
)
# ---- graph_ode 动态路线（方案 §2 默认参数；twin 家族固定口径）----
ARGS+=(
  --dynamics_mode graph_ode        # 受控条件图 ODE（Heun RK2 同步 z/q）
  --graph_source history_fc        # 功能耦合图来自当前可见历史，不要求 Mask
  --bold_source raw                # Raw/HC/ROISignals_<id>.mat，前 32 TR 校准
  --personalized True              # 固定基线个体 profile
  --profile_dim 32
  --calibration_length 32          # Raw 标准化与 profile 用同一前缀
  --ode_steps 2
  --dynamics_dt 1.0
  --lambda_dynamics 0.5
  --lambda_variance 0.05
  --dynamics_history_memory "$dynamics_history_memory"
)
# 宽度不在这里重复字面量：main.py 的 argparse 默认就是锚点
# (d_demo=16, d_new=16, C=32, H_delta=64, delta_style=mlp, dropout_p=0, inject=heads)。
# HC 侧**不带** --pretrained_weight：主干从随机初始化训练（方案 §一.1）。
if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi

# 生效参数已随 checkpoint 的 config.args 快照保存，完整命令行只在 PRINT_CMD=1 时打印。
arg_value() {   # 从 ARGS 取开关取值，避免把同一个字面量抄进第二处
  local key="--$1" i
  for ((i = 0; i < ${#ARGS[@]}; i++)); do
    [[ ${ARGS[i]} == "$key" ]] && { echo "${ARGS[i + 1]}"; return 0; }
  done
}

if [[ "${PRINT_CMD:-0}" == "1" ]]; then
  echo "[CMD] $PYTHON_BIN -u main.py ${ARGS[*]}"
else
  echo "[CMD] main.py --mode $(arg_value mode) --name $(arg_value name)" \
       "| condition=$(arg_value condition_group)/$(arg_value condition_arm)" \
       "| epochs=$(arg_value train_epochs) bs=$(arg_value batch_size)" \
       "| ckpt=$(arg_value checkpoint_dir)" \
       "| ${#ARGS[@]} 个参数（PRINT_CMD=1 看完整命令行）"
fi
PYTHONPATH=. "$PYTHON_BIN" -u main.py "${ARGS[@]}"
