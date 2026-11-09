# 多尺度相位协同预测模型（原V2）

唯一模型类为 `tfm.NeuroTwinTFM`，唯一运行配置为 `configs/neurotwin.json`。构建、SHA256核对和严格加载统一通过根目录 `model.py`；不支持旧方案权重或模块关闭开关。

| 文件 | 职责 |
|---|---|
| `tfm.py` | RevIN、历史图、TFM、动态CPM查询、额外历史记忆、未来时距交互及公开预测接口 |
| `tfm_backbone.py` | 时间patch、时间注意力、图引导ROI注意力、RMSNorm和前馈层 |
| `temporal_multiscale.py` | 多尺度因果历史编码及确定性窗口卷积实现 |
| `phase_forecast.py` | 唯一的LocalGlobalPhaseForecast：16/32/64 TR复数状态融合与逐时距振荡校正 |
| `graph_prior.py` | 历史FC、可学习图、低秩残差及对称Sinkhorn |
| `conditioning.py` | HC三字段；MDD共享编码与新增五个临床字段 |
| `common.py` | 可逆归一化、图归一化和共享张量运算 |
| `__init__.py` | 权重容器格式ARCH_VERSION=7；严格参数键进一步限定为完整V2结构 |

输入 `[B,116,1,64]`，前向返回 `(prediction, aux)`，预测为 `[B,116,32]`。`return_tokens=True` 可取得 `[B,116,8,256]` 历史token、相位特征、动态状态和振荡校正信息。

`forecast_horizon(history, condition)` 接收 `[B,64,116]`，返回 `[B,32,116]`。`predict_next_state` 截取同一H32前向的H1，用于观测刷新预测。

V2始终创建多尺度编码、局部—全局相位模块、动态查询/记忆及未来时距卷积；没有可选的旧预测路径。振荡残差与CPM预测在同一RevIN空间相加，之后统一反归一化。所有模块只读取可见历史、字段条件和已知时距，不输入真实未来。

```bash
python model.py --mode finetune --device cpu
```

[完整结构](../docs/复现说明/多尺度相位模型结构.md) · [训练与权重](../docs/复现说明/多尺度相位预测实施说明.md)
