# TASK_HANDOFF — NeuroTwin RECC 执行计划会话交接

> 最后更新：2026-10-07（第三次会话：收尾归档）。研究线状态：**已结题**。
> 结题文档：[RECC结题与负结果归档_20261007.md](docs/Version3_docs_1003/RECC结题与负结果归档_20261007.md)。
> 新会话从本记录继续，不重复已锁定的工作。

## 停止决定（已确认，2026-10-07）

接受执行计划 §8.4：**保留现有简单预测模型，停止当前专家库的预测型 RECC 扩展，负结果归档**。维持暂停：正确 RECC 重训、λ/τ 搜索与 Pair-RECC 第二批、CPM 专家与 ROI 分区、S5 多 seed/条件重训/独立临床验证。

适用范围限定：结论仅覆盖当前 one-step 同构专家库、当前多步冻结库与当前 Router 特征/路由方法、patch=4/seed=2024 协议；**不**构成"MoE/RECC 原理上无效"。统一声明：未重训主预测模型、专家或 RECC；仅进行了冻结权重推理和小型 Router 拟合。

## 验收门终态

| 门 | 判定 |
|---|---|
| G1 工程正确性 | 通过（77 测试，含 18 项契约测试；正式输出历史复现 ≤20 ppm） |
| G2 one-step | 未通过（oracle 空间 0.12%–0.31% < 1%，val+test 一致） |
| G3 one-step | 不投入（+0.056%） |
| G2-multi 多步库 | 通过（H8 4.26% / MAE 口径 2.58%；H16 8.18% / 5.18%） |
| G3-multi 多步库 | 未通过（4 设置 Router 一致劣于固定最优：−0.25% ~ −4.93%） |

部署最优解（只读推理）：近程固定 cpm_film/cpm_k64；长程固定 cpm_rollout。

## 会话历史

1. 第一次会话：S1 契约修复 + 77 测试；S0 provenance；S2 全量 val 复评（G2 判定）；S3 脚本。
2. 第二次会话：test 口径修订（正式输出复现验证）；E1/E2 epoch 恢复（36/41）；多步冻结库 H8 检验（G2-multi 过 / G3-multi 败）。
3. 第三次会话（收尾）：H16 + network 特征补充检验（四设置一致）；本地证据独立核查（`analysis_results/recc_repair_20261007/closure_audit.json`：10/10 strict+smoke、10/10 sha 一致、77 测试、历史 test 复现 ≤20 ppm、MAE 口径 oracle 复核）；协议核查（被试分离代码证据、oracle 覆盖边界、参照=最强可部署）；结题文档与归档提交。

## 证据索引

全部产物：`analysis_results/recc_repair_20261007/`（git 忽略目录）——
provenance 台账、逐权重 reeval（val/test）、router 与 multistep 检验、
`closure_audit.json`（归档核查汇总）、`epoch_recovery.json`。
复现命令与逐项证据路径：见结题文档 §7 与各小节。

## 未解决项 / 重新开启条件

1. 真实模型 diagnose/RNG 单步更新对照未完成（仅合成单测证据）；
   B/C/D stdout/TensorBoard 缺失，best epoch 不可恢复；
   FC/频谱未重算；多 seed 与独立临床验证未完成（结论不含临床主张）。
2. 重新开启路由/专家研究的条件：新结构假设 + 预设主检验与门槛 + 冻结库可学习性
   预验证（复用本仓协议），见结题文档 §9。
3. 未提交内容及原因：`RECC问题核查与分阶段解决方案_20261007.md`（含历史机器路径
   记录，仅工作区更新状态）；`results/tfm_experts_*`（被试级 CSV）；
   `analysis_results/`（忽略目录，npz 含被试级数组）；用户既有 legacy 清理改动与
   各类删除（非本任务范围，保持原状）。
