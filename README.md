# NeuroTwin

以可见 BOLD 历史预测未来 116 ROI 状态。当前保留的最优长程模型为
**S2 History Memory + Graph-ODE + CPM，完整 H32 监督、原分组 E2 目标**。

## 保留的模型与结果

| 用途 | 路径 |
|---|---|
| 最优 H32 推理权重 | `checkpoints/final/h32_best.pt` |
| H32 的 HC 预训练起点 | `checkpoints/phase2_S2/neurotwin_tfm_pretrain/base_best.pt` |
| 现有 H8 基线契约所需权重 | `checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt` |
| 正式 val/test 结果、逐被试表与图 | [results/final](results/final/README.md) |
| 权重 SHA256 与结果来源 | [manifest.json](results/final/manifest.json) |

H8 权重仅用于保持 `baseline.py` 及已有基线契约可用，不作为本轮最优长程模型。
所有未采用候选、旧版本权重、长任务目录、重复日志和缓存已按清理要求移除。
原始数据及 `data/` 中的划分清单保持不变。

## 最优模型结果

下表为同协议的 225 名被试、每人 16 个完整 K64/H32 锚点，被试等权 ODE MAE。

| Split | H1–8 | H9–16 | H17–32 | H1–32 |
|---|---:|---:|---:|---:|
| val | 0.519091 | 0.857499 | 0.898595 | 0.793445 |
| test | 0.518049 | 0.856258 | 0.896604 | 0.791879 |

同 H32 网络、同预算监督对照确认长程监督的收益；但远期时间相关仍约 0.18，
相对历史均值只改善约 3%。当前结果不证明临床病理因果、稳定跨访视个体机制或真实刺激响应。
实验结论和负结果摘要见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 运行

使用带 PyTorch、NumPy、pandas、SciPy 等项目依赖的 Python 环境。
数据与权重不入 Git；远程运行需另行准备，权重按 manifest 校验。

```bash
# 单被试自然轨迹推理：替换 SUBJECT 为实际文件/临床表中的 ID
python inference_twin.py \
  --checkpoint checkpoints/final/h32_best.pt \
  --raw data/Raw/MDD/ROISignals_SUBJECT.mat --subject_id SUBJECT \
  --condition_file Rest-meta-MDD-V1V2-Merged-MDD.xlsx \
  --horizon 32 --device cpu

# 统一评价入口：这是重新评价命令，已有正式结果直接读取 results/final 即可
python -m experiments.evaluate_variant \
  --ckpt checkpoints/final/h32_best.pt --out_dir results/h32_eval --splits val

# 旧 H8 基线的配置/strict 加载自检
python baseline.py
```

训练入口保留，不在清理或收尾时自动执行：

```bash
# 最优 H32 配方，保留原 run 的 BS64/eval128、种子、划分和其余科学参数
# --execute 才会正式训练；默认仅打印命令
python -m experiments.long_horizon.train_supervision --arm h32 --device cuda:0

# 预训练与微调的通用入口
bash scripts/Pretrain_HC_tfm.sh
HC_WEIGHT=checkpoints/phase2_S2/neurotwin_tfm_pretrain/base_best.pt \
  bash scripts/Finetune_MDD_tfm.sh
```

`configs/long_horizon_h32.json` 来自所保留最优 run，未来输出写入
`checkpoints/h32_runs/`、`results/h32_runs/`，不会覆盖 `final/`。
完整 epoch 状态保存与恢复仍由 `SAVE_TRAINING_STATE` / `RESUME_TRAINING_STATE` 控制。
旧最佳推理权重不能恢复历史 optimizer/RNG；worker 增强 RNG 与 CUDA/compile 逐位续接未验证。

## 目录

```text
main.py, models/, train/, utils/   训练与模型实现
inference_twin.py                 Graph-ODE 持续状态推理
baseline.py                      保留的 H8 契约入口
analysis/                        公共评价与分析工具
experiments/evaluate_variant.py  checkpoint 严格重建与统一评价
experiments/long_horizon/         H32 训练/配对评价与状态刷新工具
configs/, scripts/, tests/       配方、启动脚本和对应测试
data/                            原始数据与划分，清理不修改
checkpoints/final/                最优 H32 权重
checkpoints/phase2_S2/            必需的 HC 起点与 H8 契约权重
results/final/                   最优结果和压缩后的总实验数值摘要
docs/                            实验结论、未解问题和清理说明
```

[未解决问题](docs/RESEARCH_PLAN.md)仅作后续另行立项记录，不自动追加训练。
[清理范围与检查](docs/CLEANUP.md)说明本次保留和删除内容。
