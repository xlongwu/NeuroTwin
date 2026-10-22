# 正式基线评估（S2 + E2）

本目录是正式基线的评估入口与参照。

- `S2_metrics_test.json`：S2 基线（History Memory + Graph-ODE + CPM，E2 损失）
  在 225 个 test 被试上的最终评估指标（复制自 `results/phase2/eval_S2/`，2026-10-21）。
- 基线权重：`checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt`
  （SHA256 契约见 `configs/baseline_s2_e2.json`）。
- 复评命令：

```bash
python -m experiments.evaluate_variant \
    --ckpt checkpoints/phase2_S2/neurotwin_tfm_finetune/finetuned_best.pt \
    --out_dir results/baseline --splits test
```

后续损失函数实验（results/loss/）与长时域实验（results/long_horizon/）均以本目录
指标为对照基线；完整实验记录见 `docs/EXPERIMENTS.md`。
