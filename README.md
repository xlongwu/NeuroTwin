# NeuroTwin

当前唯一模型为 **多尺度相位协同预测方案（原 V2）**：历史功能图引导的时空 Transformer、多尺度历史状态、局部—全局频率状态和带未来时距交互的条件 CPM。输入为116 ROI、64 TR可见历史，一次输出未来32 TR。

固定流程：扫描前32 TR校准 → 历史RevIN → 历史功能图与四层TFM编码，同时提取多尺度历史及16/32/64 TR复数状态 → 动态查询和额外历史记忆 → 条件CPM → 未来时距因果卷积 → 振荡残差校正 → 一次反RevIN。

唯一配置为 [configs/neurotwin.json](configs/neurotwin.json)。代码不再提供B/V0/V1结构、旧权重初始化或方案选择开关。现有V2参数键及权重文件保持不变。

- [模型结构](docs/复现说明/多尺度相位模型结构.md)
- [训练、权重与运行说明](docs/复现说明/多尺度相位预测实施说明.md)
- [训练结果与验证边界](results/模型优化/长程相位预测V1与V2_20261109/说明/训练结果总结.md)
- [模块索引](models/README.md)

## 使用已有权重

从仓库根目录运行：

```bash
python model.py --mode pretrain --device cpu
python model.py --mode finetune --device cpu
```

权重目录为 `checkpoints/多尺度相位协同预测方案_1108/`，沿用训练开始日期1108。默认严格加载该方案的HC/MDD EMA best，并核对SHA256、配置及全部参数键。两阶段best均为第149轮；last及完整续训状态也保留。当前只完成训练与验证，尚未评价测试集。

```python
import torch
from model import build_model

model, config = build_model(device='cpu', mode='finetune')
# history: [B,64,116]，按扫描前32 TR固定校准。
# condition：使用该权重训练统计量编码的字段张量。
with torch.no_grad():
    prediction = model.forecast_horizon(history, condition=condition)  # [B,32,116]
```

原始扫描和字段表推理使用 `inference_twin.py` 的 `TwinSession`；H1由同一次H32前向截取，不存在独立单步预测头。

## 训练与评价

```bash
# HC：从现有V2 HC best热启动，新建输出目录，不覆盖已有训练。
DEVICE=cuda:0 bash scripts/pretrain.sh

# MDD：从当前V2 HC best开始；若要使用新HC，显式传入其实际路径。
DEVICE=cuda:0 bash scripts/finetune.sh

# 显式评价验证集；不会默认自动评价测试集。
python -m analysis.evaluate --mode finetune --splits val --device cuda:0 \
  --out_dir results/模型优化/当前V2验证评价 --save_arrays
```

算法配置只读取 `configs/neurotwin.json`，没有 `MODEL_CONFIG`、`--model_config` 或结构开关。两阶段默认使用 `data/subject_split_pretrain.json`、`data/subject_split_finetune.json` 固定划分。训练默认EMA=0.995并记录原始权重/EMA两遍验证，按EMA验证分数选择best，训练结束不自动运行测试集。

## 基础检查

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_model_config.py tests/test_tfm.py tests/test_multiscale_phase.py \
  tests/test_long_horizon_phase.py tests/test_forecast_fidelity_loss.py \
  tests/test_train_optim_scheduler.py tests/test_training_scripts_contract.py \
  tests/test_evaluation_report.py
bash -n scripts/train.sh scripts/pretrain.sh scripts/finetune.sh
```

历史实验报告保留其原始数字和来源，用于追溯，不再作为可执行的模型配置或旧模型加载入口。无关分析、数据与用户已有改动不参与本次收敛。
