# 方案 B 模型模块

唯一模型类为 `tfm.NeuroTwinTFM`。正式构建与严格加载统一通过仓库根目录 `model.py`；正式参数读取 `configs/neurotwin.json`。

| 文件 | 职责 |
|---|---|
| `tfm.py` | RevIN、历史图、时空编码器、字段条件及 CPM 的单一路径；公开预测接口 |
| `tfm_backbone.py` | 时间 patch、因果时间注意力、图引导 ROI 注意力、RMSNorm、前馈层 |
| `graph_prior.py` | 历史 FC 与可学习功能图融合、低秩残差及对称 Sinkhorn |
| `conditioning.py` | HC 三字段编码；MDD 在共享编码上增加五个临床字段 |
| `common.py` | 可逆归一化、图归一化和共享张量运算 |
| `__init__.py` | 精简结构版本 `ARCH_VERSION=7` |

输入 `x` 为 `[B,116,1,64]`，前向返回 `(prediction, aux)`：预测为 `[B,116,32]`；辅助信息包含有效图、RevIN 统计量、条件表示。传入 `return_tokens=True` 可取得 `[B,116,8,256]` 历史 token，供个体表征分析使用。

`forecast_horizon(history, condition)` 接收 `[B,64,116]`，返回 `[B,32,116]`。`predict_next_state` 截取同一 CPM 前向的 H1，用于观测刷新滚动评价。

CPM 使用 ROI 与时距嵌入作为查询，通过两层交叉注意力读取历史 token，并接受条件 FiLM。历史图只由可见历史计算，不读取未来。没有 ODE、状态积分、独立单步头、profile 或方差读出。

原 v6 B 权重已经显式迁移到 `checkpoints/方案B/`；保留模块的参数键不变，删除键及 H8 初始化时距嵌入扩展均记录在各目录的 `权重迁移记录.json`。新结构只接受严格参数加载，原始权重文件保留。
