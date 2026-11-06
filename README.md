# NeuroTwin

当前唯一正式模型为已训练验证的 **方案 B**：历史功能图引导的时空 Transformer 编码器，加字段条件 CPM，一次预测未来 32 TR。输入为 116 ROI、64 TR 可见历史；输出为校准尺度的 BOLD 预测。

实际流程：扫描最早 32 TR 固定校准 → 历史 RevIN → 历史 FC 与可学习图融合 → 时间 patch 与四层时空编码 → 人口学／临床条件 CPM → 反 RevIN → H1–32。

正式配置只有 [configs/neurotwin.json](configs/neurotwin.json)。模型、训练、严格权重加载和下游接口说明见 [方案 B 工程整理说明](docs/复现说明/方案B工程整理说明.md)，模块索引见 [models/README.md](models/README.md)。

当前代码删除了 ODE、联合预测、频域记忆、个体 profile、方差预测及其训练分支。B 原本未执行这些模块；历史权重、训练记录和结果没有删除。

## 直接运行

从仓库根目录运行，使用现有 PyTorch 环境及 `data/` 中的原始 BOLD、merged 条件表和固定被试划分。

```bash
# 核对正式最佳权重的 SHA256、配置及全部参数键
python model.py --mode pretrain --device cpu
python model.py --mode finetune --device cpu

# HC 训练：复现已验证 B 的 H8 初始化 → H32 预训练配方
DEVICE=cuda:0 bash scripts/pretrain.sh

# MDD 微调：默认读取当前正式 HC best；使用新训练的 HC 时显式传入
DEVICE=cuda:0 bash scripts/finetune.sh
HC_WEIGHT=checkpoints/方案B/训练/neurotwin_pretrain/base_best.pt \
  DEVICE=cuda:0 bash scripts/finetune.sh

# 完整 H32 固定锚点评价，附逐时距／阶段／FC／频域指标
python -m analysis.evaluate --mode finetune --device cuda:0 \
  --out_dir results/方案B/抑郁症评估 --save_arrays
```

训练输出目录存在时自动另建带序号的目录，避免覆盖历史权重。微调指定新 HC 时，以实际训练打印的 best 路径为准。

```python
import torch
from model import build_model

model, config = build_model(device='cpu', mode='finetune')
# history: [B,64,116]，已按扫描前32TR校准；condition为同一权重训练统计量编码的字段张量
with torch.no_grad():
    prediction = model.forecast_horizon(history, condition=condition)  # [B,32,116]
```

原始序列和条件表推理可使用 `inference_twin.py` 的 `TwinSession`，详见工程说明。

## 检查

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
python -m compileall -q main.py model.py inference_twin.py models train analysis
bash -n scripts/train.sh scripts/pretrain.sh scripts/finetune.sh
```

本次整理没有启动正式重训。相同原始 B 权重、输入及随机种子下，HC／MDD 预测、六项损失，以及冻结／解冻阶段一次 AdamW 更新的差异均为零；训练入口与下游入口另用临时小数据验证。原研究文档属于历史记录，不再作为运行配方。
