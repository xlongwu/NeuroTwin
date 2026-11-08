# 方案 B 与多尺度相位模型模块

共用模型类为 `tfm.NeuroTwinTFM`。构建与严格加载统一通过仓库根目录 `model.py`；默认正式B读取 `configs/neurotwin.json`，已训练实验模型显式选择 `configs/neurotwin_multiscale_phase.json`。

| 文件 | 职责 |
|---|---|
| `tfm.py` | RevIN、历史图、时空编码器、字段条件及 CPM 的单一路径；公开预测接口 |
| `tfm_backbone.py` | 时间 patch、因果时间注意力、图引导 ROI 注意力、RMSNorm、前馈层 |
| `temporal_multiscale.py` | 多尺度因果历史状态；等价窗口线性计算，保留Conv1d参数 |
| `phase_forecast.py` | 复数频率/相位编码、有界未来动态与振荡校正门控 |
| `graph_prior.py` | 历史 FC 与可学习功能图融合、低秩残差及对称 Sinkhorn |
| `conditioning.py` | HC 三字段编码；MDD 在共享编码上增加五个临床字段 |
| `common.py` | 可逆归一化、图归一化和共享张量运算 |
| `__init__.py` | 精简结构版本 `ARCH_VERSION=7` |

输入 `x` 为 `[B,116,1,64]`，前向返回 `(prediction, aux)`：预测为 `[B,116,32]`；辅助信息包含有效图、RevIN 统计量、条件表示。传入 `return_tokens=True` 可取得 `[B,116,8,256]` 历史 token，供个体表征分析使用。

`forecast_horizon(history, condition)` 接收 `[B,64,116]`，返回 `[B,32,116]`。`predict_next_state` 截取同一 CPM 前向的 H1，用于观测刷新滚动评价。

CPM 使用 ROI 与时距嵌入作为查询，通过两层交叉注意力读取历史 token，并接受条件 FiLM。历史图只由可见历史计算，不读取未来。没有 ODE、状态积分、独立单步头、profile 或方差读出。

启用多尺度相位配置后，新增历史状态和未来相位特征参与CPM查询，额外历史记忆以共享投影的独立注意力读取后残差融合；振荡先验与CPM预测在同一RevIN空间相加，再统一反归一化。默认B不创建这些参数。详见[当前结构框架](../docs/复现说明/多尺度相位模型结构.md)。

```bash
python model.py --model_config configs/neurotwin_multiscale_phase.json --mode finetune --device cpu
```

原 v6 B 权重已经显式迁移；本地现存精简权重位于 `checkpoints/CPM直接长程预测方案_1106/`，原配置中的历史目录未覆盖。需要加载已移动的B权重时可显式传 `--ckpt`，仍核对原配置哈希与结构。迁移记录保存在相应目录，新旧模型都使用严格参数加载。
