# 当前模型模块

本目录保留固定预测结构与仍有入口引用的结局读出，不保留未采用的候选网络。
现有预训练/微调权重的参数键、形状和计算结果保持不变。

| 文件 | 用途 |
|---|---|
| `tfm.py` | 组装主模型、预测头 FiLM、CPM、状态预测与更新接口 |
| `tfm_backbone.py` | 时间 patch、因果时间注意力、图引导 ROI 注意力与编码块 |
| `conditioning.py` | HC 共享字段编码、MDD 拼接增量 MLP 与条件遮蔽 |
| `graph_prior.py` | 历史功能图与学习图的全局混合、邻接残差和 Sinkhorn 归一化 |
| `tfm_dynamics.py` | Graph-ODE 向量场、Heun 积分和逐 ROI 方差读出 |
| `tfm_personal.py` | 扫描前缀 profile 编码与个体 FiLM |
| `common.py` | RevIN、图归一化和变长输入的前缀投影 |
| `twin_outcome.py` | 独立临床结局读出；由 `train/twin_outcome.py` 与 `inference_twin.py` 使用 |
| `__init__.py` | 权重结构版本 |

条件只使用字段张量，经预测头 FiLM 注入。已删除旧数值病理归一化文件
`pathology.py`，以及主干条件 RMSNorm、条件门控残差、条件低秩、线性增量生成器、
CPM 干预嵌入和未采用的图先验分支。控制输入仍通过现有 Graph-ODE 接口处理。

权重构造从 `config.args` 读取；旧快照中未使用的候选配置字段不再创建模块。
`python model.py --mode pretrain` / `--mode finetune` 可检查当前两份权重的严格加载。
