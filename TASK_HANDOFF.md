# TASK_HANDOFF — NeuroTwin 临床条件注入（condition injection）实施交接

> 最后更新：2026-10-10。任务范围（2026-10-10 起）：把 8 个字段级临床条件真实接入
> NeuroTwin-TFM 双头训练，并按**选定的单一注入方式**（MDD=`cat`、HC=`demo`）交付可直接启动的
> fresh 训练入口。原任务里「交付**受预算约束的搜索工具链**，用来判断非线性编码 / 融合方式 /
> Condition Dropout / 注入位置 / 参数配比是否值得保留」这一半已在 2026-10-10 按用户决定移除
> （见 §0.26），不再作为交付目标。
> 权威规格：[NeuroTwin_condition_injection_v1_revised_20261008.md](docs/NeuroTwin_condition_injection_v1_revised_20261008.md)
> （该文件优先于更早的 v1/v2 方案里关于 HC 两字段、MDD 七字段、5F 限制、`-9999` 视为缺失、
> HAMA 与 HAMATotal 合并等冲突要求）。
> 上一研究线（RECC/MoE）已结题，其交接记录见 `git show HEAD:TASK_HANDOFF.md`；本轮**未**复活
> 该线的任何模块。

## 0. 2026-10-09 本轮（fresh 从头实验）工程复核

> 本节由本轮复核追加，**不覆盖**下面 §1 起的历史记录。历史里的「全绿」「已完成」是上一轮
> （允许复用历史权重的搜索工具链建设）的状态，本轮按 §一 的硬约束重新核实过，两者不能混用。
> 本轮的关键差别：**所有 HC 必须随机初始化，MDD 只能引用本轮新训出的 HC 权重，且必须可追溯**。

> **§0.1–§0.25 里凡是涉及 `condition_runner / condition_space / condition_budget /
> condition_analysis / condition_benchmark / condition_cohort / migrate_condition_checkpoint`
> 与 `Run_condition_study.sh / Run_condition_full.sh / Condition_search_full.sh` 的命令行，
> 2026-10-10 起都不可再执行**（那些文件已物理删除，§0.26）。这些小节按交接要求原样保留，
> 作为「当时实测到的缺陷与证据」的记录；其中被 §0.26 明确列为「替代/保留」的三项
> ——随机来源登记（`utils/condition_schema.py`）、study 身份建立
> （`python -m experiments.condition_study`）、正式训练入口（两个 `*_tfm.sh` 脚本）——
> 才是当前有效的落点。**其中「study 身份建立」已在同日第三次口径变更（§0.27）里连模块一起删除**，
> 现在有效落点只剩随机来源登记与两个 `*_tfm.sh` 脚本。当前执行方式统一见 §7.1 与 README 快速开始。

> **行号口径**：§0.1–§0.26 里引用的 `main.py` 行号都是 **§0.27 之前**的位置（那次删了 141 行，
> 之后行号整体上移约 140）。§0.27、§1 状态表、§2.5、§7、§8 里的行号已按当前文件重新核对。

### 0.1 本轮修好的真实缺陷（都带复现证据）

| # | 缺陷 | 修法 | 证据命令 |
|---|---|---|---|
| 1 | `train/training_state.py` 的 `_RNG_KEYS` 与 `capture_rng_state()` 实际输出的键名不一致，`load_training_state` 会把**正常保存**的状态判成残缺（Python 状态是 625 个整数，没有单一 `python` 键） | 键名对齐为 `python_version/python_state/python_has_gauss/numpy_*/torch_cpu/torch_cuda` | `pytest tests/test_condition_training_workflow.py -q`（10 通过） |
| 2 | `build_exec_config` 直接读 `ns.study_id`，用旧 `Namespace` 构造时 `AttributeError` 打断排程 | 改 `getattr(ns, 'study_id', '')` | `pytest tests/test_condition_fresh_gate.py::test_10... -q` |
| 3 | 子进程退出码 0 就被登记 `completed`：checkpoint 缺失、主指标 NaN/inf、逐被试文件为空都能进排名 | `run_trials` 在标 completed 前调用 `completed_row_is_valid`，不合格直接 `TrialError` 走 failed 分支 | `tests/test_condition_fresh_gate.py::test_2` |
| 4 | `best_condition_config.json` 只校验「completed+跑满预算」，单 seed / 缺对照 / 缺 bootstrap / 缺护栏也能冻结 | 新增 `_freeze_gate`：fresh 模式下强制过 `condition_study.check_freeze_eligibility`（≥3 seed、5 项预注册对照实跑、护栏齐全、被试级配对 bootstrap ≥5000 且 CI 支持、同族 demo 基线存在），并把 Pareto/身份写进 payload | `tests/test_condition_fresh_gate.py::test_6/7/8` |
| 5 | 配对 bootstrap 只在 `diagnostics_summary` 里被**打印**，不参与任何决定 | `_paired_comparisons` 读取 `<study>/analysis/paired__<config_hash>.json`，缺文件＝未测量＝不能冻结；跨 study 的产物直接拒绝 | `test_8/9` |
| 6 | 阶段 3+ 的晋级臂核对读的是**无 study 约束**的 registry，旧 registry 的排名能替本轮决定晋级 | `validate_promotion(Registry(cfg.registry_dir, study_id=cfg.study_id))` | `test_3`（外来行与无身份行都报错） |

同时确认：`main.py` 已有 fresh 身份入口（`--init_source/--study_id/--study_root/--trial_id/--source_trial_id`，
`--init_source` 缺省时旧入口行为不变），checkpoint 内嵌 `config.study` 块，`read_provenance`
只用 `weights_only=True` 解码（历史/需反序列化任意对象的权重被判不可信）。

### 0.2 本轮三档清单（已实现且已验证 / 已实现但未验证 / 尚未实现）

**已实现且已验证**（本轮真实执行过，命令见 §0.4）

- 字段 schema、编码替换、有效性 mask、merged 唯一来源、用药列冲突、train-only 统计量、
  HC→MDD 逐位继承、`layout_hash` 拒绝字段顺序变化 —— 对**真实** merged 文件跑过只读审计
  （`medication` 两列 0 冲突、`scale_source` 全为 `iqr`、无未登记类别码）；
- 条件编码器/双头 FiLM/融合臂的可训练性、优化器分组与解冻保 Adam 状态、零增量恒等、
  恢复接口与 RNG 流等价 —— `tests/test_condition_training_workflow.py`（10 通过，真 torch）；
- fresh 身份/去重/产物校验/冻结门槛/来源校验 —— `tests/test_condition_fresh_gate.py`（10 通过，
  → §0.6 已扩到 13 通过，含 `main.fresh_study_gate` 的真实代码路径与「同起点家族」有牙校验）；
- 性能画像工具链本身 —— `experiments.condition_benchmark` 的 CLI 在 CPU 上实跑成功
  （tiny 合成预设；MDD 案例实测 `train_step.mean_ms≈23.3`、condition 参数 5976、
  `rho_conditioning≈0.1237`、CPU 侧 RSS 增量口径）。

**已实现但未验证 / 验证不完整**（不当作验收通过）

- `train/condition_activation.py`（约 46 KB，含 200 步硬上限、逐桶首梯度/首更新/output-witness）：
  模块能导入、`__all__` 与常量齐备，但**尚无测试文件执行过真实前反向**，激活步数未实测；
  （→ §0.6 已闭合：`tests/test_condition_activation.py` 24 通过，且两条真实 smoke 都落盘了
  `condition_activation.json`，首梯度/首更新步数见 §0.6 表）
- `experiments/condition_benchmark.py`：CLI 与画像实跑通过，但其测试里
  `test_non_finite_timing_and_empty_step_count_raise` 在**与其他进程并发**时出现过 1 次失败、
  单独复跑通过 → 该测试对时序/资源敏感，需要加固成确定性断言后才算稳定；
  （→ §0.22 已加固并压测通过：根因是 ①case 用 `seed=None` 建、数值轨迹落在全局 torch RNG
  的当前位置上，②一段与墙钟无关的 NaN 断言被绑在真实计时上。改为固定 seed + 常量假时钟，
  并新增「分位数必须等于注入常量」的精确断言；原 12 个异常断言一个未删。）
- `experiments/condition_runner.py` 的 fresh 分支：闸门与 provenance 写入是单测验证的，
  **尚未**用真实 `main.py` 端到端跑过一次 fresh 训练（只跑过合成 subprocess 模拟）。
  （→ 部分闭合，见 §0.15：真实 `main.py` 的 fresh 两条腿已在真实 merged 数据上跑通
  ——HC H0 随机初始化预训练、以及从该 H0 起点新建条件模块的 MDD 微调，
  `[fresh] MDD 起点校验通过` 与 `[migrate] … missing=0` 都有实测读数，并接上了独立 strict
  重建评估与置换/全缺失诊断。**仍未跑的是 runner 自己编排的那一层**：阶段循环、registry 行写入、
  依赖解析与晋级判定还没有在真实训练上走过一遍（§0.15 是直接调 `main.py`／`evaluate_variant`／
  `condition_diagnostics`，绕过了 runner）。这一层**没法廉价补**：runner 没有 tiny 训练档
  （`--cost_preset` 只管成本核算，`--ladder` 要求至少两档），而手工把计划里的 `train_epochs` 改小会被
  `trial_from_dict` 的哈希校验拒绝（§0.16 第 1 条）。所以它只能在用户正式执行阶段 1 时一并验证；
  在此之前，runner 侧已实测的是 dry-run 层的三道闸门（计划防篡改 / 预算 fail-closed / 起点记录），
  见 §0.16。
  （→ §0.21 已闭合这一条：runner 编排层在真实 `--execute` fresh 运行上跑完了 8 试次排程
  —— 6 completed / 1 failed / 1 blocked，registry 行、逐被试评估与 GPU·h 成本都落盘，
  冻结门槛也按 `formal=false` 正确拒绝生成 `best_condition_config.json`；过程中修掉
  `source_checkpoint_sha256` 与 `split_hash` 两处**静默空值**（§一.12/§八.7）。
  但那是 2-epoch 资源档的工程 smoke，不构成阶段 2 的验收通过。））

**尚未实现（本轮明确没做，不要当成已完成）**

- 真正的 fresh 正式实验（阶段 1–5）：**0 个完成的学习试次**，因此没有任何关于最优注入方式、
  融合优劣或参数收益的结论；
- test 留出评估：按方案只能在冻结后一次性执行，未运行；
- `experiments/condition_cohort.py`、`experiments/condition_analysis.py`、
  `scripts/Run_condition_study.sh`、`experiments/migrate_condition_checkpoint.py`：
  本轮复核开始时**文件不存在**，现正在并行补齐；补齐后必须先跑 §0.4 的命令再更新本表。
  注意迁移工具即使建好，也**不得**被本轮 fresh 路径调用（§一.1）。
  （→ 四个文件均已落地并有独立测试：`test_condition_cohort.py` 31、`test_condition_analysis.py` 35、
  `test_condition_migration.py` 21、`test_condition_migrate_checkpoint.py` 24 通过；
  `condition_cohort` 与 `condition_analysis` 的 CLI 在真实数据/真实 study 目录上跑过，
  结果见 §0.5/§0.6。迁移工具仍是**独立入口**，fresh 路径没有调用它。）

### 0.3 协议与数据事实的偏差登记（不改数据，只如实记）

- 方案 §四 写的是 `On medication?` 与 `OnMedication`；当前 merged MDD 表里实际存在的是
  `medication` 与 `OnMedication`（没有字面 `On medication?` 列）。`utils/condition_schema.py`
  把三个名字都登记为来源，实测 `alias_overlap['medication'] = 0`（两列从不同行同时出现，
  因此没有触发冲突报错）。这里**没有**引入「来自哪一列」的身份特征，也**没有**静默改列序。
- HC 表里 `Sex=0` 的唯一一行（`S23-2-0005`）不在这轮三个 split 的任何一侧，所以 HC 词表
  不会因它越界；如果以后把它纳入队列，必须显式登记类别码再重新拟合（`--condition_extra_codes`）。

### 0.4 复核命令（本轮实际跑过的）

```bash
cd /data3/Digital_Brain/NeuroTwin
export PYTHONPATH=.
PY=$HOME/anaconda3/envs/mamba/bin/python      # 本轮全部实测数字都出自这个解释器（torch 2.6）

$PY -m pytest tests/ -q                                            # 全量回归
$PY -m pytest tests/test_condition_fresh_gate.py -q                 # fresh 闸门（控制流，非训练证据）
$PY -m pytest tests/test_condition_training_workflow.py -q          # 真 torch 前反向/优化器/恢复等价
$PY -m experiments.condition_benchmark --out_dir <空目录> --device cpu \
    --warmup 3 --n_steps 10 --eval_passes 1 --memory_steps 2 --print_json
```

### 0.5 同日追加：接入缺失项的修复与实测（2026-10-09 下午）

| 项 | 事实 | 证据 |
|---|---|---|
| 激活记录未接入训练 | `train/condition_activation.py` 此前**只被测试引用**，训练循环里没有钩子。现已在 `main.py` 训练循环接入：新增 `--cond_activation_steps N`（1..200，0=关闭），每步 `optimizer.zero_grad` 后取步前快照、`scaler.step` 后记录，epoch 末把摘要写成 `<save_dir>/condition_activation.json` 并打印。零条件臂（`arm=none`）明确不建探针，避免把「不存在」写成「未激活」。 | 真实 fresh smoke 日志出现 `[activation] 激活记录已启用，预算 8 个 train step`，且 step 1–8 记录未抛错 |
| 继承校验只覆盖结构、不覆盖数值 | `layout_hash`/`subset_layout_hash` 只哈希字段顺序/类型/来源列/类别词表，**不含连续字段的 mu/scale/n_valid/scale_source**；因此「MDD 侧把 Age/education 重新拟合一遍」在数值漂移、词表不变时能通过共享前缀检查。迁移工具当时的报错文案自己写着「mu/scale 不参与哈希，需另行核对」——即当时并没有那道核对。修复方向：新增精确数值指纹（float 用 `hex()`）并在迁移与继承路径上同时比对结构哈希与数值指纹。 | 复现命令：`$PY -m pytest tests/test_condition_migration.py -q` → 修复前 `test_hc3_to_mdd8_inheritance_requires_bitwise_shared_prefix` 失败（反例是空断言） |
| 一键脚本无法表达合法的 HC→MDD 依赖 | `scripts/Run_condition_study.sh` 原先把 `--pretrained_weight`/`--condition_inherit_stats` 无条件列入禁用清单，而 fresh MDD 的起点必须通过 `--pretrained_weight` 传入并由 `verify_current_study_hc` 从权重内嵌身份校验。现改为**目录限定放行**：这两个开关只接受落在 `$STUDY_ROOT_ABS/` 之内的路径，其余一律 `die`。真正的历史权重加载、`--resume/--retry_failed/--adopt_orphan/--hc_weight/--load_backbone_only/--save_training_state` 仍然全禁。 | `bash -n` 通过；守卫逻辑见脚本 `FORBIDDEN`/`STUDY_SCOPED`/`assert_no_forbidden` |
| legacy HAMD FiLM 是否被误注入 | 读码确认：预训练态 `tfm_patho_mode` 只在存在活动非 none 臂时被置为 `film`，其输入是新条件编码器；`forward` 里同时收到 `pathology_score` 与 `condition` 直接报错（两通路互斥），`arm=none` 时 `tfm_patho_mode` 强制回 `none`。日志那行「模型生效 tfm_patho_mode=film（args 值 none）」指的是 FiLM 槽位被条件向量复用，不是残留 HAMD FiLM。 | `models/tfm.py:539-553`、`models/tfm.py:700-703` |
| 分析模块与冻结门闸的契约 | `experiments/condition_analysis.py` 写 `<study>/analysis/paired__<config_hash>.json`，未真跑时把 `n_resamples` 如实写成 0 而不填造 CI；门闸读的正是这段结构，因此「CI 未测量 ≠ CI 支持」成立。空 study 上跑 CLI 只报缺登记表、不落任何产物。 | `$PY -m experiments.condition_analysis --study-root .tmp_cond/smoke_study --candidate cat --baseline-arm demo` → 打印「登记表不存在」，`find .tmp_cond/smoke_study/analysis -type f` 为空 |
| 阶段 0 仍缺 MDD 腿的 smoke | 脚本阶段 0.3 只有 HC smoke（`phase0_smoke_hc`）。fresh MDD 起点校验目前由编排层测试与 `verify_current_study_hc` 覆盖，**没有**在真实 `main.py` 上跑过一次 HC→MDD 串接。这是本轮已知的未验证项，未补完之前不要把「HC→MDD 真实串接已验证」写进结论。 | 见脚本 0.3 段；补齐需要确定 stats 继承载荷文件的产出方（`condition_cohort` CLI 目前只写审计报告，不写可继承 stats json） |

本轮实测（fresh、真实 merged 表 + 真实影像、CPU、1 epoch、`--train_samples_per_subject 1`、`--is_probe`，
study_id=`smoke20261009`，输出目录 `.tmp_cond/smoke_study/`）：条件参占 6 064 880 总参的
`rho_cond=0.0061`，编码器 `arm=demo hc3` 3408 参数、含双头 FiLM 后条件模块合计 37200，
`[params] 条件参数组校验通过（全部入优化器、无重复、numel 一致）`；train-only 统计量
Age `mu=29.000 scale=16.308 (iqr, n=1759)`、education `mu=14.000 scale=3.706 (iqr, n=1754)`、
Sex 词表 `[1.0, 2.0] + Unknown`，`layout_hash=0b4ae75e58b7fe2f`。
这些是 **探针/smoke 口径**（tiny、1 epoch、无 AMP），不能当作正式 150 epoch HC 预训练的性能画像。

### 0.6 同日第二次追加：真实 HC→MDD 串接 + 四处完整性修复（2026-10-09 晚）

§0.2 与 §0.5 里三条「未验证」现已闭合，本节是本轮最新事实；旧条目保留原文，只在下面标明状态。

| 项 | 事实 | 证据（本轮真实执行） |
|---|---|---|
| 阶段 0 缺 MDD 腿 smoke（§0.5 最后一条）→ **已闭合** | 脚本新增阶段 0.3b：用 0.3 刚训出的探针 HC 权重做真实 `main.py` 微调腿，`--init_source current_study_hc_checkpoint --source_trial_id … --condition_inherit_stats <同一本轮 HC 权重> --is_probe`；`EXECUTE` 下若 0.3 权重不存在直接 `die`（不会静默跳过，也不会去别处找权重） | 日志 `.tmp_cond/smoke3_mdd_cat.log` 首两行：`[fresh] MDD 起点校验通过：study=smoke20261009 source_trial=s1_hc_demo#smoke2 sha256=eb64a0ac3528 seed=2024` + `[fresh][warn] 起点是阶段 0 探针/smoke 权重…` |
| 探针权重不能冒充正式起点，但 0.3b 必须能接它 | `verify_current_study_hc` 新增 `allow_probe_source`，且**只有消费方自己带 `--is_probe` 时才为 True**。它只放宽两条与探针身份直接冲突的检查（路径在 `probe_dir` 下、来源自标 `is_probe`）加「必须在 registry 有行」；study 内路径、内嵌 `study_id/kind/init_source/trial_id/seed/split_hash` 一致性与文件 SHA256 一条不减 | `pytest tests/test_condition_fresh_gate.py::test_5 -q`；`test_13` 断言正式 MDD 接同一探针权重必须抛 `ProvenanceError`（"探针"） |
| 身份块里 `condition_schema_version` 恒为空（真实缺陷） | `main.py` 读的是不存在的 `args.condition_schema_version`，于是每条正式 trial 的 `config.study` 都记录 `''`：事后无法判断两份权重是否同一条件语义。改为在训练前由 `utils.condition_schema.SCHEMA_VERSION` 推导 `run_schema`（`arm=none` 如实留空），写入 provenance，并与来源 HC 的记录值显式比对；来源缺该字段时打印 warn 而不是静默通过 | `pytest tests/test_condition_fresh_gate.py::test_13 -q`（记录、留空、继承 SHA、版本冲突拒绝四项） |
| 「同一起点家族」校验是空转（严重） | `_same_init_family` 比较的是 `dedup_signature` 的 payload 键 `split_sha256`/`init_checkpoint_sha256`，而 registry **行里没有这两列** → 两边都取空串、空==空恒真，任何 `arm=demo` 的行都会被判同族基线，预注册的相对改善门槛因此形同不设。改为比较真实列：`cohort`、`split_hash`、`init_source`、`source_checkpoint_sha256`（MDD 行缺来源哈希直接判非同族） | `pytest tests/test_condition_fresh_gate.py::test_11 test_12 -q`：把 demo 基线的来源哈希改成别的值、或清空 `split_hash`、或换队列，冻结门必须重新拒绝 |
| 退化 bootstrap 报出非法 p 值 | 当配对差恒为 0（$b-a\equiv 0$）时 $p=2\min(P(D\le 0),P(D\ge 0))=2.0$。现按 $p\leftarrow\min\!\big(1,\max(p,\frac{1}{n_{\text{boot}}+1})\big)$ 夹紧 | `paired_bootstrap(x, x)` 修复前 `p_two_sided=2.0`、修复后 `1.0`；正常用例 `p=0.001996`、CI `[-0.1456,-0.096]` |
| 数值继承闸（§0.5 第二条）→ **已落地** | `utils/condition_schema.py` 新增 `field_fingerprint`（`mu/scale/codes` 走 `float.hex()`，不做容差比较）与 `shared_stats_hash`（顺序敏感的规范 JSON sha256 前 16 位）；`experiments/migrate_condition_checkpoint.py::_check_shared_inheritance` 在迁移与继承两条路上同时比结构哈希与数值指纹，并点名漂移量 | `pytest tests/test_condition_migration.py tests/test_condition_migrate_checkpoint.py -q`（21 + 24 通过） |
| 入口拒绝用户写 `--fresh`，脚本文档与实现不一致 | `Run_condition_study.sh` 的 fresh 是恒开的（`FRESH_BASE` 里就带 `--fresh`），但 CLI 没有这个键，用户照 `main.py` 的写法输入会得到「未知参数」硬错；头部注释还写着「编排命令里永不出现 `--pretrained_weight`」，与 0.3b 的目录限定放行矛盾。现改为：接受 `--fresh`（无副作用，因为无法关闭），并把注释改成事实 —— `--pretrained_weight/--condition_inherit_stats` 只接受 `$STUDY_ROOT_ABS/` 之内路径，其余历史权重加载开关仍全禁 | `bash -n` 通过；`--fresh --dry-run --study-id doret1009 --study-root ./.tmp_cond/study/doret1009 --stages 0..5` 退出码 0 且**未创建任何目录**（`ls` 确认不存在）；`--execute` 不给 `--device` → 退出码 1；`--study-root ./checkpoints/evil` → 退出码 1（受保护目录守卫）；dry-run 文本含阶段 0.3b |

本轮 MDD 腿的真实数字（探针口径：dim 64 / 2 层 / `cpm_horizon 4` / 1 epoch / CPU，
`arm=cat mdd8`，`--cond_activation_steps 8`）：

```
[migrate] loaded/copied=136 | new_initialized=20 | zero_expanded=0 | skipped=0 | missing=0
[params]  total=6 075 600 | conditioning=47 920 | trainable=47 920 | rho_cond=0.0079
          distinct_tensors=35     # 阶段 1 冻结主干，只有条件参数入优化器
        编码器内部：feature_continuous=1600 feature_categorical=176
                    demo_encoder=2720 delta_encoder=9632 total=14128 encoded_fields=8
        layout_hash=2db26880c6eb8cd0（mdd8）；来源 HC 的 0b4ae75e58b7fe2f 是其 demo 子集
[NextPoint] checkpoint criterion = val next-state MAE=0.5129 | ΔMAE(模型-持续性)=+0.0268
            val rollout MAE (free, R=16) | H1 0.5136 | H2 0.9129 | H4 1.1963 | H8 1.2177 | H16 1.0558
[activation] 8/8 步全部记录，all_activated=True
            cpm  firstGrad=1 firstUpdate=1 witness=witnessed（cpm_head.film，2 参数）
            os   firstGrad=1 firstUpdate=1 witness=witnessed（one_step_head.film，2 参数）
            encoder firstGrad=1 firstUpdate=1（31 参数；delta 末层零初始化 → 部分子参数首非零梯度在第 2 步）
            backbone status=not_built（本配置 `cond_inject=heads`，主干未建 AdaRMSNorm，属如实登记）
```

HC 腿同一口径下：`total=6 064 880`、`conditioning=37 200`、$\rho_{\text{cond}}=0.0061$、
`val next-state MAE=0.5146`、$\Delta\text{MAE}=+0.0118$、
rollout $\text{H1}=0.5102,\text{H2}=0.9164,\text{H4}=1.2028,\text{H8}=1.2149,\text{H16}=1.0417$。
**两条腿都是探针**：1 epoch、tiny 结构、无 AMP、`--is_probe`，其权重只用于验证流程，
不能当正式 150 epoch HC 预训练的性能画像，也不能进排名或晋级（§一.1、§七 阶段 0 的要求）。


本轮全量回归：`$PY -m pytest tests/ -q` → **336 passed, 11 skipped in 102 s**
（本节之前的记录是 333 passed / 11 skipped；本轮净增 `test_11/12/13` 三项——
同族校验有牙、MDD 缺来源哈希不算同族、`main.fresh_study_gate` 记录并比对条件协议版本；
原有断言一条未删。单跑 `$PY -m pytest tests/test_condition_fresh_gate.py -q` → 13 passed）。

`.tmp_cond/smoke_study/` 的实际产物状态（用来核对「没有提前生成结论」）：
`registry/` 只有 `initial_condition_config.json`，`analysis/` 为空，
**没有** `best_condition_config.json`，`phase0_probe/` 为空 —— 即 §八.8「缺证据不冻结」
在真实目录上成立，而不是只在测试里成立。

#### 0.6.1 仍未闭合 / 已知限制（不要读成已验证）

- **正式实验仍未开始**：阶段 1–5 完成的学习试次为 **0**，因此没有任何关于融合方式、
  注入位置、宽度或 Dropout 优劣的结论；上面两条 smoke 数字不构成效果证据。
- ~~`--seed / --sampling_seed / --split_seed` 在本轮探针里同为 2024（只登记，未改动默认），
  正式跑法必须让三者取值不同（§四 要求三类 seed 分离）。~~
  → **§0.7 已修**：live 随机流（model / sampling / dropout / permutation）四条已两两分离并
  有三层闸门；仅剩「冻结划分 seed 与首个模型 seed 数值相同」这一条作为偏差登记在 §0.7 末尾。
- 直接用 `main.py` 起的手工试次 `config_hash=''`（该字段由编排器计算），
  因此这类权重只能靠 `trial_id + SHA256` 追溯，不能参与配置去重。
- 本轮 MDD 探针权重的 `condition_schema_version` 仍是 `''`：它在修复落地（15:08）之前
  的 15:03 就已启动，Python 不会在运行中重读源码。**新起的试次**才会写版本号
  （由 `test_13` 证明代码路径正确）。
- `weights_only=False` 的 `torch.load` 在既有入口仍有残留（`main.py:192`、
  `experiments/evaluate_variant.py:56,74`、`experiments/condition_runner.py:449`、
  `train/optim.py`、`analysis/*`、`inference.py`）。新的身份读取路径
  `condition_study.read_provenance` 与新迁移工具一律走 `weights_only=True`；
  旧入口的完整改造超出本轮范围，属安全债务而非功能缺陷。
- `experiments/condition_space.py` 未纳入版本控制（`git status` 显示 untracked），
  其「与上一轮相同」的说法无法用 diff 核实；本轮按当前文件内容对待。
- 编排器的 fresh 分支现在既有合成测试、也有真实 `main.py` 双腿串接，但
  **runner 自身的 execute 路径**仍未拿真实数据跑过一个完整预算试次
  （§0.2 第三条对这一点仍然成立）。

### 0.7 同日第三次追加：随机来源 seed 分离的真实缺陷与修复（2026-10-09 晚，§四）

§四 要求「split seed、model seed、sampling seed 和 Condition Dropout seed 分离」。逐处核对
后确认：**这条要求在修复前从未真的成立**，而且不是"没接线"，是"接成了同一条流"。

| # | 实测缺陷 | 修法 | 证据 |
|---|---|---|---|
| 1 | `scripts/Run_condition_study.sh` 阶段 0.3/0.3b 写 `--seed "$first_seed" --sampling_seed "$first_seed"`，而 `main.py` 的 `--seed` 与 `--sampling_seed` 默认值又都是 2024 ⇒ 模型初始化流与采样流是**同一条随机流**；Condition Dropout seed 从未显式给出 | 两条 smoke 改为显式 `--sampling_seed "$SAMPLING_SEED"` + `--cond_dropout_seed "$DROPOUT_SEED"`；值在 `preflight()` 里从登记处读，不在脚本抄字面量 | dry-run 打印 `--seed 2024 --sampling_seed 20261009 --cond_dropout_seed 20261008`；`test_15` 断言脚本里 `--sampling_seed "$first_seed"` 这一旧模式已消失且两处常量各出现 2 次 |
| 2 | `condition_space.DIAG_DEFAULTS['perm_seed']` 登记的 20261008 与 `DROPOUT_SEED` **同值**，而它上方注释自己写着「置换种子与…Dropout seed 全部分离」 | 新增 `PERM_SEED = 20261010`，`DIAG_DEFAULTS['perm_seed']` 引用它；注释保留原文并说明改值原因 | `test_15` 的三常量两两不同断言 |
| 3 | `condition_budget.run_probe` 的回退 `recipe.get('sampling_seed', 2024)`：配方缺键时**静默**造出一条与 `--seed` 合并的探针流 | 回退值改为 `SAMPLING_SEED` 常量 | 修好后全量回归仍全绿（见下面的 `pytest tests -q`，cohort/预算/排程用例都在其中） |

闸门分三层（任何一层单独就足够拦下回归）：

1. **登记处**：`experiments/condition_space.validate_trials` 对每条试次核对 model / sampling /
   dropout / permutation 四个值两两不同，同值直接 `ValueError`（阶段 1–5 的排程在生成命令前就停）。
2. **入口**：`main.fresh_study_gate` 对 fresh 运行核对 `--seed / --sampling_seed / --cond_dropout_seed`；
   **参数缺失也拒绝**（不能"看不到就跳过"）。`main.py` 的历史默认值不动 ⇒ 非 fresh 旧入口行为不变。
3. **脚本**：`preflight()` 读取登记处常量，并与 `--seeds`（或预注册三种子）做碰撞检查后打印一行
   `随机来源分离：…`，供人工核对。

实测（本轮真实跑过）：

```bash
$PY -m pytest tests/test_condition_fresh_gate.py -q     # 15 passed（新增 test_14/15）
$PY -m pytest tests -q                                   # 338 passed, 11 skipped in 107.64s
bash -n scripts/Run_condition_study.sh                   # 语法 OK
bash scripts/Run_condition_study.sh --dry-run --stages 0..5 --max-trials 200 --gpu-budget-hours 300
#   => exit=0；两条 smoke 的 seed 三值分离；study 目录**未创建**（dry-run 无副作用）
bash ... --stages 0 --seeds 20261009                     # => exit=1「随机来源未分离（方案 §四）」
bash ... --execute（无 --device）                        # => exit=1
bash ... --study-root ./checkpoints/brain                # => exit=1（受保护目录）
```

顺带核清了两条此前只"看着实现"的项（都找到真实证据，不再算未验证）：

- §三「`condition_extra_codes` 是否真的传入拟合」：链路是
  `condition_space.cli_flags`（`--condition_extra_codes`）→ `main.py:657` 的
  `parse_extra_codes` → `utils/dataloader.py:1137/1300`（显式预注册，禁止运行时静默扩表）
  → `utils/condition_schema.py:482` 拒绝未知字段名（`--condition_extra_codes 含未知字段`）。
  四段都在；训练集之外出现未登记的有效类别码时，`utils/dataloader.py:1368` 是
  **`raise ValueError`**（报错里附带建议的登记串），而不是静默扩表或伪装成缺失。
- §六.3「增量为零时 OS/CPM 输出最大误差 <1e-6，优先逐位一致」：
  `tests/test_condition_migrate_checkpoint.py::test_model_level_zero_increment_equivalence_hc3_to_mdd8`
  在**真模型输出级**断言 bitwise 相等（`:620` 的 `max_abs_err < 1e-6` 只是下限口径），
  并由 `test_model_level_equivalence_breaks_when_delta_is_not_zero`（`:626`）做反向对照，
  证明这道断言不是空转。

#### 0.7.1 登记但未改：划分 seed 与首个模型 seed 数值相同

`data/subject_split_pretrain.json` 与 `..._finetune.json` 的 `seed` 字段都是 **2024**，
与预注册模型 seed 的第一个值 2024 数值相同。机制上两者不共用随机流：划分 seed 只在 manifest
生成期消费，训练期只读名单 + `split_hash` 指纹，`condition_runner.ensure_seed_split` 甚至专门
用「同名单副本 + `review_seed_of_run`」把模型 seed 与划分 seed 解耦。但
`condition_cohort.seed_distinctness` 是**按值**比较四个槽位的，因此 seed=2024 的试次会被它报成
`collisions` 非空（该函数自述"只报告不修正"，不阻断训练）。

要彻底消除需要把 `PREREGISTERED_SEEDS` 与 `condition_study` 预注册里的 `seeds` 一起换成不含 2024
的三个值，同时改 `condition_space` 里若干 `seed: int = 2024` 默认值和 `if seed != 2024` 的副本规则。
本轮完成的学习试次为 **0**，改与不改都不影响任何已有结果；这属于**预注册内容的重排**，
按 AGENTS.md「遇到不确定的项目约定先查证、无法确认就说明假设」，这里只登记事实并交给你定夺，
不擅自重排预注册种子。

### 0.8 同日第四次追加：两条真的能走通的「历史权重/统计量」入口（2026-10-09 晚，§一.2 / §三 / §四）

继续核 §三 清单里的两条自问时，找到的不是"没接线"，而是**两条实际可被走通的入口**。

| # | 实测缺陷 | 修法 | 证据 |
|---|---|---|---|
| 1 | `main.py` 的 `--pretrained_weight` 默认值是历史权重 `./checkpoints/neurotwin_pretrain/base_best.pt`。`args.pretrained_weight` 的消费点确实全在 `if args.mode == 'finetune':` 内（`main.py:745-795`），所以 fresh HC 今天**读不到**它 —— 但 §一.2 禁的是"隐藏的默认路径"本身，`fresh_study_gate` 的 hc 分支此前只拦 `--load_backbone_only` / `--save_training_state`，不拦这个默认值；同理 `--condition_inherit_stats` 在 `main.py:638` 是**跨模式消费**的，一条 fresh HC 命令带着历史 .pt 就会把旧拟合统计量带进本轮（§一.8 / §四） | 默认值改为空串，历史路径抽成 `main.LEGACY_PRETRAINED_WEIGHT`，只在**非 fresh 的 finetune** 下于 `parse_args` 末尾回填；`fresh_study_gate` 的 hc 分支现在对非空 `--pretrained_weight` / `--condition_inherit_stats` 直接抛 `ForeignArtifactError`，mdd 分支要求继承快照落在本轮 `study_root` 之内 | `test_16`（正向：HC 默认值为空、MDD 本轮内快照放行；反向：历史路径、本轮外快照均报错）+ `parse_args` 三条默认值断言（`--mode finetune` 无参仍得到旧路径 ⇒ 旧脚本行为逐字不变） |
| 2 | `utils/dataloader._fit_condition_stats` 的**继承分支**只传 `only_fields=MDD_NEW_GROUP`，**漏传 `extra_codes`**。于是新增五字段的预注册码被静默丢弃，而同一函数下方的 audit 报错又建议「传 `--condition_extra_codes "…"`」⇒ 用户照提示改完仍然报同一个错。反向问题：给继承字段（`Sex`/`education`）传码会**空转**，看起来像"扩大 HC 词表"却毫无提示 | 继承分支补传 `extra_codes`，并对 `DEMO_GROUP ∩ extra_codes` **显式拒绝**（要求回 HC 试次登记后重训 HC 再继承）；audit 报错按字段分流：继承字段给"回 HC 侧登记"的文案，新增字段保留原文案（该串的效力现在是真的） | `test_E_extra_codes_reach_the_inherit_path`（哨兵码 7.0 在合成表里一处都没有 ⇒ 只有真的传进拟合才会出现在词表；同时断言共享三字段逐位不变、`Sex` 不含 7.0、传 `Sex:7.0` 报错）；schema 级对照实测：`episode codes (old, no extra_codes)=(0.0, 1.0)`，`(fixed)=(0.0, 1.0, 7.0)`，`Sex codes` 两边都是 `(1.0, 2.0)` |

§三「MDD 是否会扩大继承的 HC Sex 字典」的结论（本轮核到底，非推断）：**不会**，三重机制各自独立成立 ——
① 继承分支的 `only_fields` 不含 `DEMO_GROUP`，`ConditionStats.fit` 只对 target 字段整体替换 `FieldStats`；
② `utils/dataloader.py:1337-1343` 逐字段比对 `(fitted, mu, scale, codes)`，被覆盖即 `AssertionError`；
③ `ConditionStats.encode` 对每个有效类别码调 `check_codes`（`utils/condition_schema.py:421-429`），
未登记码 `raise ValueError`，既不落 Unknown 也不扩表 —— 评估路径同理（`condition_stats_mode='inherit'`
在 audit 之前 `return`，编码期照样被 `check_codes` 拦住）。

实测（本轮真实跑过）：

```bash
$PY -m pytest tests/test_condition_fresh_gate.py -q   # 16 passed（新增 test_16）
$PY -m pytest tests/test_condition_injection.py -q -k extra_codes   # 2 passed
$PY -m pytest tests -q                                 # 340 passed, 11 skipped in 99.37s
bash -n scripts/Run_condition_study.sh                 # 语法 OK
bash scripts/Run_condition_study.sh --study-id ... --study-root /tmp/... --stages 0
#   => exit=0；dry-run 未创建任何目录；HC 腿命令行里**没有** --pretrained_weight，
#      MDD 腿的 --pretrained_weight / --condition_inherit_stats 都指向本轮 phase0_probe 之内
```

> 修正 §0.7 的一处表述：那里写「`condition_extra_codes` 链路四段都在」，对**非继承**路径成立，
> 但继承路径当时确实漏传（即上表第 2 条）。以本节为准；旧文字保留以留痕。

#### 0.8.1 仍未闭合（不要把上面读成实验已完成）

- 阶段 1–5 的正式学习试次依然是 **0**，因此没有任何关于融合方式、注入位置、宽度、Dropout 的
  效果结论；§0.7 的两条 smoke 与本轮的合成测试都只证明**链路与闸门**，不是训练证据。
- 本轮改动只碰了 fresh 入口的拒绝条件、`parse_args` 的默认值回填时机、以及 dataloader 的继承拟合
  调用；**没有**改任何非条件配方（宽度、层数、目标、损失、增强、采样）与预注册阈值。
- `--condition_inherit_stats` 与 `--pretrained_weight` 的**目录限定**在 `main.py` 层只覆盖到
  "落在本轮 study 内"；更细的"快照必须来自那个已通过 SHA256 校验的同一个 .pt"目前由
  `condition_study.assert_command_fresh` + 脚本的 `STUDY_SCOPED` 保证，`main.py` 未重复实现。
- 旧入口的 `weights_only=False` 安全债（`main.py:192` 等）与 §0.7.1 的划分 seed 数值巧合，
  两条都仍在原处，未在本轮处理。

### 0.9 同日第五次追加：冻结链路的五处真实断裂（2026-10-09 晚，§五 / §七 / §八）

这一轮不再问「模块在不在」，只问一件事：**照交付命令跑完阶段 5，能不能生成
`best_condition_config.json`**。答案是当时**永远不能**，而且失败点全在控制流里，不在阈值松紧上。
下面五条都是读代码 + 跑测试实测出来的（每条先有反例，再修）。

| # | 实测断裂 | 修法 | 证据 |
|---|---|---|---|
| 1 | 预注册护栏里的 `variance_ratio_distance_from_1` 与 `all_missing_degradation` **没有任何生产者**：registry 无此列，runner 也不写。而 `check_freeze_eligibility` 把「缺测」判为不通过（`condition_study.py:702-706`）⇒ 冻结被**构造性**拒绝 | 两列都改成**由实测产物派生**：前者 $=\lvert \text{VR} - 1 \rvert$（`condition_runner.read_metrics`，VR 缺测则仍为 `None`），后者 $= \Delta_{\text{all\_missing}} / \text{MAE}_{\text{identity}}$（`read_diagnostics` 读 `scenario_deltas.all_missing_repr`）；`CSV_COLUMNS` 补两列，`execute_diagnostic` 落盘 | `test_19`（VR 0.92 ⇒ 距离 0.08；`rollout.enabled=false` ⇒ 两列 `None`；identity 2.0 / all-missing 2.2 ⇒ 0.1；场景缺席 ⇒ `None`） |
| 2 | 交付脚本从不传 `--full_guardrails`（runner 默认 `False`），于是 `rollout_h16_mae` / `fc_edge_pcc_mean` / `variance_ratio` 在正式试次上恒为空 ⇒ 第 1 条修完仍然缺测 | `runner_stage` 在阶段 5 追加 `--full_guardrails` 并打印原因；阶段 1–4 不追加（只对最后一档生效：runner 内条件为 `full_guardrails and final_rung`，代价如实计入预算） | dry-run 第 39 行 `[CMD] ... --stages 5 --auto_promote --full_guardrails` |
| 3 | **决定性一条**：`Trial.config()` 含 `seed` ⇒ 三个预注册 seed 天然是三个 `config_hash`。`write_best_config` 的「同配置分组」与 `condition_analysis` 的「一次只分析一个配置」都按 `config_hash` 取组 ⇒ 候选组恒为 1 个 seed，`min_seeds_for_freeze=3` 在构造上不可满足；docstring 承诺的「`config_hash`（该配置的全部 seed）」也**从未**成立 | 新增 **`Trial.family_config_hash`**：去掉 `seed` 的同一配置身份，并归一化 `--review-scope full` 造成的 `#seedN` 家族别名。**只用于汇总**（冻结候选组、配对产物命名、best 的 `completed_seeds`）；去重与复用仍按含 seed 的 `config_hash`，每个 trial 保留独立身份（§八.3） | `test_17`（换 seed ⇒ `config_hash` 变、`family_config_hash` 不变；`fam` 与 `fam#seed2025` 同族；换臂不同族；`deduplicate([a,b])==2`）+ `test_21`（真实逐 seed 行形：bootstrap 只按 `config_hash` 命名时门槛拒绝，按 family 键命名后放行；单 seed 组仍被拒；best 里 `stability=multi_seed`、`config_hash` 仍指向具体那条 trial） |
| 4 | `all_missing_degradation` 的真实读数落在**诊断行**（`kind=diagnostic`，无选优指标）上，而门槛只读候选行自己的列 ⇒ 即使诊断跑过也判缺测。另外阶段 5 的诊断只生成 seed 2024 一份，2025/2026 永远无护栏 | 门槛新增 `_guardrail_evidence`：只把**同 seed、同权重**（`compared_against` 引用或 `config_hash` 相同）的已有读数搬到护栏列，找不到就保持 `None`，不生成任何新数值；`condition_space.review_diagnostics` 逐 seed 展开诊断并接进 `generate_stages` | `test_18`（3 个 seed 各一份诊断，`path_slots['ckpt'] == f'trial:{config_hash}'`，`arm=none` 不生成）+ `test_20`（补诊断行前门槛报 `all_missing_degradation` 缺失，补后放行） |
| 5 | 两处「看起来在工作、实际空转」的选优口径：① `_pareto_front` / `deterministic_tiebreak` 读 `condition_params`/`probe_step_ms`/`missing_robustness`，registry 写的是 `n_params_conditioning`/`wall_seconds`/`all_missing_degradation` ⇒ 目标恒 `None` ⇒「任一目标缺测就跳过支配判定」退化成**所有行都在前沿**；② `deterministic_tiebreak` 生产链路**没有调用者**，只有 1 个测试在调；③ 脚本从不调用 `experiments.condition_analysis`，即门槛要求的 `analysis/paired__*.json` **在交付的一键流程里根本没有生产者** | ①② `condition_study` 加 `_OBJECTIVE_SOURCES` + `_objective()` 别名解析（缺失鲁棒性取**负的** `all_missing_degradation`，同一实测值换方向），未测量目标按「最差」参与裁决而非占便宜；best 产物新增 `freeze_gate.deterministic_tiebreak_order`，把预注册排序写进冻结依据（冻结仍是人工 `--frozen_trial`，脚本不自动挑赢家）。③ 脚本阶段 5 之后自动跑一条**只读统计腿** `python -m experiments.condition_analysis --study-root <本轮 study>`（bootstrap 次数用模块默认 5000＝预注册下限；失败只 warn，因为缺 CI 时门槛本就拒绝冻结） | `test_22`（被全面支配的行必须被剔除；只测主指标的行不被当成支配对象；`_objective(partial,'condition_params') is None`）+ `test_23`（排序覆盖该配置全部 seed 且无重复）+ CLI 空 study 实测 `exit=2`「登记表不存在」⇒ warn 分支可达 |

实测（本轮真实跑过，命令与数字逐字记录）：

```bash
$PY -m pytest tests/test_condition_fresh_gate.py -q   # 23 passed（新增 test_17..test_23）
$PY -m pytest tests/test_condition_workflow.py -q      # 20 passed（原有断言未删未放宽）
$PY -m pytest tests -q                                  # 347 passed, 11 skipped in 103.87s
bash -n scripts/Run_condition_study.sh                  # 语法 OK
# 脚本自带「Python≥3.10」环境闸门：不带 PYTHON= 直接跑会在探测处 die（exit=1，见下）
PYTHON="$PY" bash scripts/Run_condition_study.sh --dry-run --stages 0..5 \
  --study-id dr09d --study-root /tmp/.../study --max-trials 200 --gpu-budget-hours 300
#   => exit=0；未创建任何目录；第 37 行说明阶段 5 追加护栏，第 39 行
#      [CMD] ... --stages 5 --auto_promote --full_guardrails；
#      第 42 行 [CMD] ... -m experiments.condition_analysis --study-root <本轮 study>
PYTHON="$PY" bash scripts/Run_condition_study.sh --dry-run --stages 5..1   # exit=1「区间必须从小到大」
PYTHON="$PY" bash scripts/Run_condition_study.sh --dry-run --stages 0..9   # exit=1「只接受 0..5」
bash scripts/Run_condition_study.sh --dry-run --stages 0..5 --study-id dr09d ...   # exit=1：python3 非≥3.10
$PY -m experiments.condition_analysis --study-root <空 study>  # exit=2，脚本据此走 warn 分支
```

> **更正 §0.7 表末与 §6 的一条旧记录**：那里写「`--stages 0..5` 退出码 0 且未创建目录」，
> 但当时的解析正则 `^[0-9][0-9,[:space:]]*$` 不收点号，`0..5` 会 die 在
> 「`--stages 需要 0..5 的阶段号`」。脚本头部 `--stages <0..5，空格或逗号分隔>` 里的
> `0..5` 本是**取值范围**注记，被文档当成语法抄进了命令。现按区间语法实现（`0..5`、`3..5`
> 均可，与逐个数字等价、自动去重、越界与非升序一律 die），头部参数一览同步改成
> `--stages <0 1 2 或 0..5 / 3..5>`；旧文字保留以留痕。

#### 0.9.1 仍未闭合（读到这里请不要把 §0.9 当成实验完成）

- **completed 试次仍为 0**，`best_condition_config.json` 不存在，也没有任何冻结。§0.9 全部
  是「闸门能不能被合法满足」的工程修复，不是训练结果；**最优条件注入方式仍未确定**。
- `--frozen_trial` 在**非 fresh** 的旧入口仍绕过 `_freeze_gate`（fresh 分支 `condition_runner.py:1905`，
  旧入口走 `:1917-1918` 的 `else: decision = None`），这是刻意保留的向后兼容（§六.8）；
  本轮交付的所有命令都走 `--fresh`，该路径不在本轮链路上。
- ~~`check_freeze_eligibility` 里 `if required_controls_present and missing_ctrl` 在传空列表时
  不报缺失（旧行为未动）；fresh 路径由 `_freeze_gate` 自己的 `预注册对照缺实跑结果` 兜住，
  所以本轮口径仍是 fail-closed~~ **已闭合，见 §0.11**（旧文字保留以留痕；实测旧写法在空列表与
  漏传参数两种情况下都返回 `can_freeze=True`，fresh 路径确实被 `_freeze_gate` 兜住，但库层契约
  与 §八.8 不符，已改为 fail-closed）。
- ~~`same_budget_ranking` 仍只按 `(stage, train_epochs, cohort)` 分组，未校验初始化家族与
  非条件配方（§八.7 的更严口径只落在 `_same_init_family`/`dedup_signature` 这两处）~~
  **已闭合，见 §0.10 第 2 条**（旧文字保留以留痕）；
  ~~`_completed_hc` 取第一条 `provides_family` 行的风险同样仍在~~
  **已闭合，见 §0.10 第 3 条**（旧文字保留以留痕）。
- Pareto 的「时延」目标目前回退到 `wall_seconds`（整试次训练时长），不是纯推理 `probe_step_ms`；
  两者口径不同这一点已写进 `_OBJECTIVE_SOURCES` 注释，但**没有**为它单独造一个测量产物。
- **旧一键脚本与本轮硬约束冲突（本轮新发现，已在 §1 顶部改指向）**：
  `scripts/Condition_search_full.sh:38` 的 `REG_DIR` 默认值是共享目录
  `results/condition_search`（实测该目录已存在旧 `budget_profile.json`、`plan_stage12.json`、
  `initial_condition_config.json`），`:39` 的画像默认从同目录读，且 `:130`、`:158`、`:188`
  三处调用**都不带 `--fresh`／`--study_id`** ⇒ 用它 `--execute` 会把旧登记目录与旧拟合画像
  卷进选择链路，直接违反 §一.7 / §一.8。本轮只把它**从推荐入口降级为历史兼容入口**（保留文件、
  不删不改），合法入口是 `scripts/Run_condition_study.sh`（恒 `--fresh`，registry/画像/产物
  全部钉在 `$STUDY_ROOT` 内）。**未**给该脚本加拒绝分支，因为这会改动旧正常入口的行为，需要用户先确认范围。
  §7.1 的启动命令原先也抄了这个旧入口，已改写成 fresh 三段式（dry-run / execute / 从某阶段重来），
  实测三条分别 exit=0（不落目录）/ exit=1（`--execute` 缺 `--device`）/ exit=0。同处顺带更正一处
  **文档与实现不符**：旧文字说「脚本用宽松的显式闸门（500 次 / 5000 GPU·h）」，那两个默认值只存在于
  旧脚本（`Condition_search_full.sh:45-46`）；fresh 脚本没有预算默认值（`Run_condition_study.sh:123`），
  `--execute` 跑阶段 1..5 必须显式给 `--max-trials` 与 `--gpu-budget-hours`（`:246-253`），缺则 die。
  §0.4 那行「`$HOME/anaconda3/envs/mamba/bin/python` 是唯一能导入 `models.graph_prior` 的环境」也已按实测
  改写：base 环境（Python 3.11.5 + torch 2.5.1）同样能导入，保留 mamba 只因为本轮全部数字都出自它。
- 模型侧仍未闭合的四条与 §0.5/§0.6 的记录一致：~~`ConditionNormGenerator(mode='shared')` 有
  未使用的 `ident` 词表行且缺 `shared`+`last1/last2` 用例（`models/conditioning.py:734-735`、
  `:758-762`）~~ **已闭合，见 §0.10 第 5 条**（旧文字保留以留痕）；
  ~~`test_3_no_dead_condition_parameters_in_any_arm` 只查 `conditioner.*` 且默认
  `inject='heads'`~~ **已闭合，见 §0.10 第 4 条**（旧文字保留以留痕）；OS-only / CPM-only
  注入范围未实现（`models/tfm.py:642-647`）；
  ~~`main.py --condition_arm none` 配默认 `--tfm_patho_mode film` 仍会建 legacy HAMD FiLM
  （`main.py` 从不传 `condition_arm=`：`:711-719`、`:736`、`:1538`），且没有钉住该行为的测试~~
  **已闭合，见 §0.10 第 1 条**（旧文字保留以留痕）。

### 0.10 同日第六次追加：五处真实断裂 + 一处审计加固（臂声明往返 / §八.7 同预算可比性 / HC 起点选择 / 死条件参数扫描口径 / shared 生成器死行 / 惰性宽度轴被当搜索维度）（2026-10-09 晚）

第 1、2、3、5、6 条都不是设计缺陷，而是**用探针跑出来的实测断裂**：先复现反例，再修，再回归。
第 4 条不是断裂——那次审计结果本身是干净的（15 个组合里 `grad is None` 的条件参数为 0），
但那条验收测试的**扫描口径过窄**，会漏掉主干注入档建立的条件参数，因此按「验收必须真的扫到东西」
的标准把它补齐，并加了防自欺断言。第 5 条紧接着说明第 4 条的口径**仍然不够**：
逐张量的 `grad is None` 扫不出「一张表里只有部分行是死的」，所以第 5 条的判据下沉到**逐元素**。
第 6 条换了一个层面：模型侧没有死参数，但**搜索空间把结构惰性的轴当成真维度**，于是同一个模型
被登记成 5 个不同 `trial_id` 的试次、要跑 5 次完整训练。写第 6 条时又查出两处**同源缺口**：
`condition_runner` 的 `--plan` 导入路径一条语义校验都没有（闸门只长在生成侧），以及起点产物
`initial_condition_config.json` 用一个假臂名算「每个晋级臂多少条试次」。两处都记在第 6 条的
修法 ④⑤ 里，不单列条目。

| # | 实测断裂 | 修法 | 证据 |
|---|---|---|---|
| 1 | **独立 strict 重建与训练快照互相矛盾**（§六.7）。`main.py` 的 `--condition_arm` 默认值是字符串 `'none'`，但构造模型时**从不把该参数传给 `NeuroTwinTFM`** ⇒ 用「不给 `--condition_arm` + `--tfm_patho_mode film`」训练，得到的是**带 legacy HAMD FiLM 的模型**，而写进快照的 `condition_arm` 是 `'none'`。独立评估按快照重建时，`models/tfm.py:549-553` 会因「显式无条件」把 FiLM 关掉 ⇒ 探针实测 `strict load: FAILED`，10 个 unexpected keys（`pathology_normalizer.*` / `one_step_head.film*`）。即：**训练出来的权重无法被自己的快照重建** | 把 CLI 默认值改成「未声明」哨兵 `None`（`main.py:1385-1394`，`default=None`，帮助文本区分「未声明＝保留 legacy 行为」与「显式 `none`＝本轮 H0/D_none，legacy FiLM 也不建」），并在构造处声明 `condition_arm=args.condition_arm`（`main.py:746`）。两个方向都进快照，因此快照值必须与训练时真正建出的结构逐一对应 | `test_M_undeclared_vs_declared_none_arm_both_rebuild_strict`：未声明 ⇒ FiLM 建出、重建后键集**完全一致**、strict 加载 `missing`/`unexpected` 均空；显式 `none` ⇒ 两侧都不建 FiLM、strict 同样干净。另有 `test_M_main_forwards_condition_arm_to_model_constructor` 作**源码级接线守卫**（检查 `main()` 里构造调用是否含 `condition_arm=args.condition_arm`）——它明确标注**不是模型验收**，只防止再次漏接线 |
| 2 | **同预算排名只按 `(stage, train_epochs, cohort)` 分组**（§八.7）。于是同一 stage/epoch/队列但 **split 不同、初始化家族不同、来源权重不同、batch/lr/架构/采样/冻结档不同**的试次会被并排比较，进而驱动晋级与冻结——而 `--auto_promote` 正是从这个排名反推阶段 3+ 的臂名与冻结宽度 | `condition_runner` 新增可比性签名：`COMPARABILITY_FIELDS`（`mode/kind/split_hash/condition_schema_version/init_source/init_seed/requires_family/source_checkpoint_sha256/recipe_hash`，`:1134-1137`）、`COMPARABILITY_REQUIRED`（`:1139`）、`comparability_signature`（`:1142`）、`comparability_missing`（`:1152`）、fail-closed 的 `assert_comparability_evidenced`（`:1157`）；`same_budget_ranking` 的分组键改为 `(stage, train_epochs, cohort, signature)`（`:1177`）；`_row_base` 落 `recipe_hash` 列＝非条件配方指纹（排除 `train_epochs`，因为它还受 `--ladder` 覆盖，`:690`）；`PROVENANCE_COLUMNS` 补 `recipe_hash`（`:46`）；两个推导入口都接上校验（`stage2_top_arms :1216`、`_stage_completed :1287`）；排程报告逐组打印 split/init/来源权重/recipe 证据，缺证据的组显式标注「不能驱动晋级」（`:1482-1492`） | `test_Q_same_budget_group_checks_split_init_family_and_recipe`：证据全同 ⇒ 1 组；分别改动 `split_hash`／`source_checkpoint_sha256`／`recipe_hash`／`init_seed` ⇒ 各变 2 组；带 `study_id` 但缺必需证据 ⇒ `SystemExit`（消息含「可比性证据」）；旧入口行（无 `study_id`）仍按原行为晋级（§六.8 兼容） |
| 3 | **同一家族的 HC 起点由登记表写入顺序决定**（§一.4/§一.12）。`_completed_hc` 旧实现是「遍历 `registry.latest()`，返回第一条 `provides_family` 匹配的 completed 行」。探针实测：家族 `hc_demo_d16_C32` 同时存在 seed 2024 与 seed 2025 两条完成行时，**seed 2025 的 MDD 复核试次静默拿到 seed 2024 的权重**，`blocked` 为空、日志无任何提示 ⇒ 同一融合比较家族可能有两个不同的共同起点，而 §一.4 要求共享同一个 H1 checkpoint、§一.12 要求来源可追溯 | `_completed_hc` 改为返回 `(checkpoint_path, 冲突说明)`（`condition_runner.py:607`）：唯一权重直接用它；多个不同权重按**最大训练预算**（`train_epochs`，并列看 `rung`）确定性选取，并把全部候选打印一次（按家族去重，`_hc_family_warned`，`:529`/`:655`）；最大预算处仍有两条不同权重、或预算列全缺无从排序 ⇒ 返回冲突说明，调用方把该试次置为 `blocked`（`:578`），不给任何起点 | `test_24_hc_family_start_is_deterministic_and_ambiguity_is_refused`：唯一提供者正常解析且权重与继承统计量同源；低档+高档**两种写入顺序**都选中跑满预算的那条；最大预算处平手 ⇒ `blocked` 含家族名与两个候选路径且 `pretrained_weight` 缺席；预算列全缺 ⇒ 同样拒绝；同一权重登记两次不算歧义 |
| 4 | **死条件参数扫描只覆盖一个注入档、一类参数名**（审计加固，不是断裂）。`tests/test_condition_training_workflow.py:107` 的 `test_3_no_dead_condition_parameters_in_any_arm` 旧口径是「只过滤 `conditioner.*`，且只建默认 `inject='heads'` 的模型」。而主干注入档建立的条件参数叫 `cond_norm.gen.L1.weight/bias`（AdaRMSNorm 生成器），**既不以 `conditioner.` 开头，也只在 `backbone` / `heads+backbone` 档存在**。探针实测：`arm=cat, inject=backbone` 下旧口径扫到 31 个条件参数、新口径扫到 33 个，差的正是那两个 `cond_norm.gen.*` ⇒ 这条测试对「主干档建了却从不参与计算」的条件参数是盲的，真出现死参数也不会红（方案 §三「backbone-only 是否建立了永远不使用的条件参数」） | 先用探针实测 5 臂 × 3 档共 **15 个组合**的 `p.grad is None` 计数（全为 0，模型侧本身干净），再把测试口径补齐（`:119-132`）：过滤条件放宽到 `conditioner.*` ∪ 名字含 `film` ∪ 名字含 `cond_norm`；注入档从只测 `heads` 扩到三档全扫；判据仍是 `grad is None`（没进计算图），不改成「首步梯度为零」——增量末层零初始化会让首步梯度为 0，那是设计。另加一条**防自欺断言**：`arm != 'none'` 且 `inject != 'heads'` 时若一个条件参数都没扫到就直接判失败（说明过滤名与真实参数名不符、断言会空转）。原有断言（`conditioner.*` 无死参数）一条未删、未放宽 | `$PY -m pytest tests/test_condition_training_workflow.py -q -k test_3` ⇒ `1 passed, 9 deselected in 21.78s`；探针实测条件参数个数/元素数（`d_new=16, C=32, H=64, dim=16, layers=2`）：`arm=cat` heads `n=35 numel=16176`、backbone `n=33 numel=17232`、heads+backbone `n=37 numel=19344`；`arm=sum/mean` 分别 `12080 / 13136 / 15248`；15 个组合 `dead=0`。注意这是**合成前反向验收**（CPU、随机张量、单次 backward），不是训练证据；OS-only / CPM-only 注入档未实现，所以 15 个组合只覆盖已实现的三档 |
| 5 | **`ConditionNormGenerator(mode='shared')` 的层身份表为未选中的 block 建行**（§三「永不使用的条件参数」）。旧实现建 `nn.Embedding(num_layers, ident_dim)`，但 `forward` 只读 `self.blocks` 里的行 ⇒ 未选中的行**永远拿不到梯度**，却仍被 `n_params()` 计入条件参数预算（`models/tfm.py:681` 打印的「额外参数」正是这个数）。探针实测（`ident_dim=8`；先把零初始化的输出层扰动成非零，否则首步全表梯度为 0、看不出结构死行）：`num_layers=2, scope=last1` 死行 `[0]`＝8/16 个元素；`num_layers=4, scope=last1` 死行 `[0,1,2]`＝**24/32**；`last2`/`all` 无死行。第 4 条那种 `p.grad is None` 判据**扫不出来**——整张表都有 `grad`，死的只是行；而旧测试 `test_P_backbone_conditioning_*` 恰好只测了 `shared`+`all`，即唯一没有死行的档位 | 身份表改为只为被选中的 block 建行：`nn.Embedding(len(self.blocks), ident_dim)`，并显式登记「绝对 block 下标 → 行号」映射 `ident_row`（`models/conditioning.py:726-746`），`forward` 按行号查表（`:763-766`）；类文档同步更正（原「按绝对层下标建模，切换注入范围时已有行仍可用」与本修复冲突，而阶段 4 的注入位置比较恒为**新建恒等条件模块**、不跨范围继承权重，见 §六.4）。`per_layer` 分支一行未改 | 新增 `test_P_shared_norm_generator_has_no_structurally_dead_identity_rows`：2 种生成器 × 3 个范围，判据下沉到**逐元素**（`(p.grad == 0)`），并要求「登记预算 == 实际拿到梯度的元素数」，另验 4 个 block 的调制两两不同（防身份表退化成同一行复制）⇒ `1 passed, 66 deselected in 3.47s`。修后探针复测：`last1/last2/all` 死行均为空，`n_params()` 由 3968→3944（4 层 last1）。模型级 6 个组合（`inject=backbone` × 3 范围 × 2 生成器，`dim=16, layers=4, C=32`）实测恒等起点 `True`、strict 加载 `missing/unexpected` 均空、`cond_norm.*` 元素数 == `gen.n_params()`；顺带得到阶段 4 要登记的实测参数量：`per_layer` 随范围增长 **3168 / 6336 / 12672**，`shared` 近似恒定 **3944 / 3952 / 3968** |
| 6 | **搜索空间把结构惰性的宽度轴当成真维度**（阶段 3 明令禁止；§八 的「登记搜索规模」也因此报大）。`stage3_promoted()` 对**每个**晋级臂都铺开 `STAGE3_WIDTH_SUBSET` 的 5 个 `(d_new, H_delta)` 组合，但只有 `cat`/`sum`/`mean` 真的消费这两个轴：`demo` 不建增量编码器（`delta is None`）也不消费新增五字段，`direct` 的新增组走**规范向量**（宽度由 schema 与词表决定，与 `d_new` 无关）+ **线性**增量（没有隐藏层，`h_delta` 无处可用）。探针实测（真实 merged 表拟合的统计量，`d_demo=16, C=32`）：5 个宽度组合下 `demo` 条件参数量恒为 **3424**、`direct` 恒为 **5152**，**参数名与形状逐条相同**；`cat` 则是 5 个互不相同的签名（3424/5152 与 cat 的 14144→10848/21888/9344/23744 形成对照）。而 `stage3_promoted(['demo','direct'])` 产出 18 条候选、**18 个互不相同的 `trial_id`**，`validate_trials` 一条不拦 ⇒ 同一个模型要跑 5 次完整 150 epoch 微调，差异只来自训练噪声，却被当成「宽度搜索结果」并排排名。随机搜索侧同源：`enumerate_random_space()` 的完整笛卡尔积是 2916 点，其中 `direct` 的 729 点只有 81 个互不相同的结构 | 新增唯一判定入口 `width_axes_live(arm)`（`experiments/condition_space.py:69-87`，取值直接来自 `models.conditioning.FUSIONS`，不另立一份臂名清单），三处按它取值：① `stage3_promoted` 只对生效臂铺 5 档，惰性臂只出锚点一档并在 `rationale` 写明原因（`:501-516`）；② `validate_trials` 增加**硬拒绝**（不是告警）——`mdd8` 分组 + 惰性臂 + 非锚点 `d_new`/`H_delta` 直接抛错，因为手写计划、旧计划文件或未来生成器改动都可能重新引入这些行，而它们 `config_hash` 互不相同、去重与复用判定拦不住（`:1106-1116`）；③ `enumerate_random_space` 对惰性臂只保留锚点宽度（`:963-966`），声明空间 2916→**2268**（`direct` 729→81）。④ 运行器侧把闸门接到 `--plan` 入口：`collect_trials()` 现在对导入计划也跑一次 `validate_trials`（`condition_runner.py:2389-2394`，告警打印、错误抛出）——此前该路径**一条语义校验都没有**，而 `load_plan`→`trial_from_dict` 只查「文件里的 `trial_id` 与内容是否自洽」（`condition_space.py:1035-1039`），查不出结构重复 ⇒ 旧口径导出的计划文件可以把同一个模型的 5 份副本直接送进训练；⑤ `write_initial_config()` 的 `plan_sizes_reference.stage1_3_trials_per_promoted_arm` 不再用假臂名 `'<arm>'` 算单值——③ 之后它会静默落到惰性臂分支、把生效臂的额度从 11 报成 7（低报搜索规模，违反 §八.6），改为按两类臂分别登记 `width_live_arm_cat_sum_mean=11` / `width_inert_arm_demo_direct=7`（`condition_runner.py:1778-1788`）。Dropout 与条件 LR 两轴**未动**：它们对 `demo`/`direct` 仍然真实生效（作用在数据与优化器上，不改结构）。`cat`/`sum`/`mean` 的计划一行未变 | 新增 `test_11_stage3_width_axes_are_live_only_for_fusions`（`tests/test_condition_training_workflow.py`，`@requires_merged`）：用真实 merged 统计量把 `width_axes_live` 的声明**钉在实测结构签名上**——生效臂必须 5 个不同签名且「实测隐藏宽度 == 声明 `H_delta`」（顺带覆盖 §五「不能声明 H=64 却实现为另一宽度」），惰性臂必须只有 1 个签名；再验计划侧（惰性臂 1 档宽度、生效臂 5 档、`trial_id` 无重复、Dropout/LR 各 2 档、`static_new`+`mask_only` 两个对照不因宽度收敛而消失）、闸门侧（`d_new=32` 与 `h_delta=128` 两种手改都被拒，生成器产出的计划与 `cat` 的非锚点宽度都放行）、空间侧（惰性臂无非锚点宽度点、`per_arm == {direct:81, cat/sum/mean:729}`、总数 2268、无完全重复点）⇒ `1 passed, 10 deselected in 5.05s`。CLI 实测三条路径均 `exit=0`：`--promote demo direct` 每臂由 11 条降到 7 条且只剩 `mdd_*_d16_H64` 一档宽度；`--promote mean cat` 仍是每臂 5 档；`--random 4` 打印「声明空间容量=2268」。④⑤ 由 `test_O_runner_plan_path_runs_the_semantic_gate`（旧口径计划文件里 `direct`+`d_new=32` ⇒ `ValueError`，消息含「不改变条件结构」；`cat`+`d_new=32` 照常放行且 `trial_id` 不变）与 `test_O_initial_config_artifact_carries_measured_phase_budget` 的**新增断言**（起点产物两类规模分别等于 `len(stage3_promoted(['cat']))`=11 与 `len(stage3_promoted(['demo']))`=7，且生效臂 > 惰性臂）钉住 ⇒ 连同原有的 `test_O_search_space_rejects_impossible_trials` 一起 `3 passed, 65 deselected in 3.69s`。全量回归 `354 passed, 11 skipped in 110.35s` |

顺带修掉一个**先前就存在**的登记表缺陷：`PROVENANCE_COLUMNS` 里 `source_checkpoint_sha256`
出现两次（一次来自 `cs.provenance_columns()`），CSV 镜像会写出两个同名表头列；改为
`list(dict.fromkeys(...))` 去重（`condition_runner.py:43-48`）。实测 `len(CSV_COLUMNS)=57`、
重复列名为空、`recipe_hash` 已在列。

实测（本轮真实跑过，命令与数字逐字记录；`PY=$HOME/anaconda3/envs/mamba/bin/python`）：

```bash
$PY -m pytest tests/test_condition_injection.py -q -k test_M   # 4 passed in 4.82s
$PY -m pytest tests/test_condition_fresh_gate.py -q             # 24 passed in 3.87s（新增 test_24）
$PY -m pytest tests/test_condition_injection.py -q -k "test_O or test_Q"  # 29 passed in 26.85s
$PY -m pytest tests/test_condition_injection.py \
      tests/test_condition_workflow.py tests/test_condition_fresh_gate.py -q
#   => 109 passed in 49.00s
$PY -m pytest tests/test_condition_training_workflow.py -q -k test_3
#   => 1 passed, 9 deselected in 21.78s（第 4 条：扫描口径由 5 个组合扩到 15 个组合）
$PY -m pytest tests/test_condition_injection.py -q -k test_P_shared_norm_generator
#   => 1 passed, 66 deselected in 3.47s（第 5 条新增测试）
$PY -m pytest tests/test_condition_training_workflow.py -q -k test_11
#   => 1 passed, 10 deselected in 5.05s（第 6 条新增测试）
$PY -m pytest tests/test_condition_injection.py -q \
      -k "test_O_runner_plan_path or test_O_search_space_rejects or test_O_initial_config"
#   => 3 passed, 65 deselected in 3.69s（第 6 条 ④⑤：--plan 闸门 + 起点产物两类规模）
$PY -m pytest tests/test_condition_activation.py -q   # 24 passed in 24.60s
$PY -m pytest tests -q          # 354 passed, 11 skipped in 110.35s（§0.9 时为 347 passed）
$PY -m experiments.condition_space --stages 3 --promote demo direct   # exit=0，每臂 7 条
$PY -m experiments.condition_space --stages 3 --promote mean cat       # exit=0，每臂仍 5 档宽度
$PY -m experiments.condition_space --stages 3 --promote mean cat --random 4 --sampling-seed 20261011
#   => exit=0，打印「声明空间容量=2268」（原 2916）
$PY -m experiments.condition_runner --stages 5 --promote demo direct --registry_dir /tmp/gatechk/demo_direct
#   => exit=0（非 --execute，只排程）；阶段 3 共 14 试次（每臂 7）；[validate] 告警 0 条
$PY -m experiments.condition_runner --stages 5 --promote mean cat --registry_dir /tmp/gatechk/mean_cat
#   => exit=0；阶段 3 共 22 试次（每臂 11）；[validate] 告警 0 条；两个临时目录里的
#      initial_condition_config.json 都写 status=search_not_run 与
#      {'width_live_arm_cat_sum_mean': 11, 'width_inert_arm_demo_direct': 7}
#      （/tmp 下的一次性目录，跑完即删，不进版本库）
bash -n scripts/Run_condition_study.sh    # 语法 OK
PYTHON="$PY" bash scripts/Run_condition_study.sh --dry-run --stages 0..5 \
  --study-id dr8b --study-root /tmp/dr8b/study --max-trials 200 --gpu-budget-hours 300
#   => exit=0；dry-run 未创建任何目录
```

> §0.9 实测块里 `pytest tests -q` 的 `347 passed, 11 skipped in 103.87s` 是**当时**的真实读数，
> 保留以留痕；本节六条（五条修复 + 一条测试口径加固）之后同一命令的最新读数是
> `354 passed, 11 skipped in 110.35s`（新增 7 个测试：`test_M_*` 两条 + `test_Q_same_budget_*`
> + `test_24_hc_family_*` + `test_P_shared_norm_generator_*` + `test_11_stage3_width_axes_*`
> + `test_O_runner_plan_path_*`；另有 `test_O_initial_config_artifact_*` **只增加断言**，
> 不改变测试个数；无删改、无放宽）。中途还有四次读数：`351 passed … in 100.29s`（第 4 条加固
> 之前）、`351 passed … in 110.57s`（第 4 条之后、第 5 条之前）、`352 passed … in 111.21s`
> （第 5 条之后、第 6 条之前）与 `353 passed … in 112.85s`（第 6 条 ①②③ 之后、④⑤ 之前）；
> `100.29s → 110.57s` 的跳变来自 `test_3` 由 5 个组合扩到 15 个组合，不是新增测试。
> 原有断言一条未删。

#### 0.10.1 仍未闭合（读到这里请不要把 §0.10 当成实验完成）

- **completed 试次仍为 0**：阶段 1–5 的正式学习试次一个都没跑，`best_condition_config.json`
  不存在，**最优条件注入方式仍未确定**。本节六条只是「训练出的权重能否被自己的快照 strict
  重建」「同预算排名能否被合法驱动」「共享 HC 起点是否确定」「死条件参数扫描是否真的扫到
  主干注入档」「`shared` 生成器是否还建死行」「结构惰性的宽度轴是否被当成真搜索维度」的
  工程与验收闭合。
- 第 2 条的 fail-closed 只在**带 `study_id`** 的行上生效；旧入口写出的历史行没有这些签名，
  按 §六.8 保持其原有行为（不新增拒绝分支）。本轮正式执行恒走 fresh 入口，因此不受影响，
  但这也意味着旧登记表**不能**被搬进本轮驱动晋级——`--fresh` 的目录隔离已经在做这件事。
- `recipe_hash` 覆盖 `merged_recipe()` 除 `train_epochs` 以外的全部键；若将来给 recipe 增删键，
  历史行与新行的 `recipe_hash` 会自然分成两组（这是期望行为，不是 bug），但需要在文档里说明
  为什么两组不可比。**尚未**为「recipe 版本号」单独建列。
- 第 3 条的「最大预算优先」是**确定性口径而非统计判断**：它只回答「多条完成行里哪一条是
  跑满预算的正式起点」，不评价哪条更好。低档 HC 起点已被别的 MDD 试次用过时，本修复
  **不会**回溯改正那些试次——它们靠 §八.7 的可比性签名（含 `source_checkpoint_sha256`）
  落进不同的同预算组，因而不会与新起点的试次并排排名或一起晋级。
- 第 4 条只加固了**测试口径**，模型一行未改：15 个组合 `dead=0` 是本次审计的实测结论，
  但它只覆盖已实现的三档注入（`heads` / `backbone` / `heads+backbone`）。OS-only / CPM-only
  仍未实现（`models/tfm.py:642-647`），这两档将来一旦实现，必须同步扩进 `test_3` 的扫描，
  否则又会退回「建了不扫」。另外 `test_3` 用的是 CPU + 随机张量的单次 backward，
  属合成前反向验收，**不是训练证据**；首次非零梯度/更新仍由 `test_condition_activation`
  的 ≤200 step 口径负责。
- 第 5 条改变了 `shared` 生成器身份表的**形状语义**：行数由 `num_layers` 变成 `len(blocks)`，
  因此 `shared` 的 `cond_norm.ident.weight` **不能跨 `backbone_scope` 复用**（`last1` 与 `all`
  形状不同，strict 加载会直接失败，不会静默错配）。本轮阶段 4 恒为「从同一个 H0 新建恒等
  条件模块」（§六.4），不做跨范围继承，故没有受影响的历史权重；但若将来要做「先 `last1`
  训练、再扩到 `all` 并继承身份行」，必须显式写迁移逻辑（按绝对 block 下标重排行），
  不能指望 strict 加载。另外两种生成器的**参数量口径不同**（`per_layer` 随范围线性增长、
  `shared` 近似恒定，实测数字见第 5 条），阶段 4 比较生成器时必须同时报告条件参数量，
  否则会把「参数更少」误读成「结构更好」。
- 第 6 条**改变了对外登记的搜索规模**，修复前导出的产物按定义不可比：`enumerate_random_space()`
  的声明空间由 2916 点降到 **2268** 点（`direct` 729→81），因此 `random_search_record()` 的
  `space_size` 与 `untried_hashes` 换了一套口径，旧的随机搜索记录**不能**与本轮并排比较，需要
  重新导出；每个惰性晋级臂的阶段 3 试次数由 11 条降到 **7** 条（`cat`/`sum`/`mean` 仍是 11 条）。
  两类旧产物的失效方式不同：旧**计划文件**里惰性臂的非锚点行哈希仍然自洽（`config_hash` 只由
  配置决定，与空间枚举无关），`load_plan` 照收，现在靠修法 ④ 的运行器闸门拒绝；旧**随机搜索
  记录**是口径变了，本身没有「拒绝加载」这一说。磁盘上
  `results/condition_search/initial_condition_config.json:98` 与
  `.tmp_cond/smoke_study/registry/initial_condition_config.json:28` 仍是修法 ⑤ 之前的单值 11，
  按 §一.8 属历史产物，**不改动**（本轮 fresh 执行会在新 `study_root` 内重新生成）。
- `width_axes_live()` 是**声明式规则**（取值直接来自 `models.conditioning.FUSIONS`），但由
  `test_11_stage3_width_axes_are_live_only_for_fusions` 用真实 merged 统计量把声明钉在实测结构
  签名上：将来若给 `direct` 接上依赖 `d_new` 的编码器、或给 `demo` 接上新增五字段，该测试会先红。
  那时必须**同时**改规则与测试，并把新的宽度轴语义写进方案，不能只放宽断言。
- §0.9.1 的其余条目原样保留：非 fresh 的 `--frozen_trial` 仍绕过 `_freeze_gate`；
  ~~`required_controls_present=[]` 仍不报缺失~~ **已闭合，见 §0.11**（旧文字保留以留痕）；
  Pareto 的时延目标仍回退 `wall_seconds`；
  模型侧只剩 **OS-only / CPM-only 注入范围未实现**（`models/tfm.py:642-647`）——原
  「`ConditionNormGenerator(mode='shared')` 的未用 `ident` 行与缺 `shared`+`last1/last2` 用例」
  已由本节第 5 条闭合、「`test_3` 只查 `conditioner.*`+`heads`」已由第 4 条闭合
  （两条旧文字均在 §0.9.1 保留以留痕）。

### 0.11 同日第七次追加：冻结门槛对「对照证据为空」是 fail-open（§八.8）（2026-10-09 晚）

§0.9.1 与 §0.10.1 都把「`required_controls_present=[]` 仍不报缺失」当成**已知但刻意不动**的旧行为
记着，理由是「fresh 路径由 `_freeze_gate` 自己兜住」。复核这条理由时发现它只对**一个**调用方成立，
而库函数自己写的契约是 §八.8 的「缺对照不能冻结」⇒ 按实测断裂修掉，不再留在「已知旧行为」清单里。

**实测 A/B**（同一份候选组、同一份 `comparisons`，只换对照证据参数）。旧行为不是推断出来的：用真实
模块源码在临时命名空间副本里把新分支文本回退成旧的一行判断后 `exec`，与现函数并排跑同一组输入
（不改磁盘、不改已导入模块；探针脚本一次性写在临时目录，不进版本库）：

| `required_controls_present` | 旧写法 | 新写法 |
|---|---|---|
| 5 条对照齐全 | `can_freeze=True`，`reasons=0` | `can_freeze=True`，`reasons=0` |
| 只给前 3 条 | `can_freeze=False`，`reasons=1` | `can_freeze=False`，`reasons=1`（消息不变） |
| 显式空列表 `[]` | **`can_freeze=True`，`reasons=[]`** | `can_freeze=False`，`reasons=1` |
| 漏传参数（走默认 `()`） | **`can_freeze=True`，`reasons=[]`** | `can_freeze=False`，`reasons=1` |

候选组是「3 个 seed 全 completed + 护栏齐全且有同被试基线 + 5000 次被试级配对 bootstrap + CI 支持
正向改善 + 相对改善过预注册门槛」的**本可冻结**一组，所以后两行的 `True` 不是「别的门槛已经拦住了」，
而是**整条对照检查被跳过**：旧写法 `if required_controls_present and missing_ctrl:` 的前半个条件在空
序列上直接短路，`reasons` 一条不加，而函数文档写着「缺护栏、缺对照、缺多 seed、bootstrap 未达标者
一律不能冻结」（`experiments/condition_study.py:682`）。默认值
`required_controls_present: Sequence[str] = ()`（`:678`）意味着**任何漏传该参数的新调用方都会静默拿到
「可以冻结」**——这正是 §八.8 要禁止的「缺对照提前冻结」，只是它藏在默认参数里而不是显式传空。

**修法**（`experiments/condition_study.py:734-746`）：把「空证据」与「部分缺失」拆成两个分支——空证据
（且预注册确有必需对照）报「没有提供预注册对照的实跑证据（`required_controls_present` 为空）：
缺对照不能冻结，需要 [...]」，部分缺失仍走原来的「缺少预注册对照的实跑结果：[...]」。原分支的条件与
消息文本一字未改，原有断言一条未删、未放宽。

**生产链路影响面（实测）**：`check_freeze_eligibility` 在生产代码里只有一个调用方
`condition_runner._freeze_gate`（`:1985-1988`），它自己先算 `_controls_present`、缺项时 prepend
「预注册对照缺实跑结果：[...]」（`:1981-1984`），再把 `can_freeze` 重算成 `not decision.reasons` ⇒
**fresh 执行路径此前也是 fail-closed**，本轮交付的命令不受这条缺陷影响。修的是**库层契约**：新调用方
（独立统计腿、复核脚本、将来接进 `condition_analysis` 的冻结复核）不必再自己复制一遍对照检查才安全。

**证据**：新增 `tests/test_condition_workflow.py::test_freeze_blocked_when_control_evidence_is_absent`
——先断言同一组在对照齐全时**确实可冻结**（防止「新分支把所有情况都判死」这种假 fail-closed），再断言
`[]` 与漏传两种情况都不能冻结且消息含「没有提供预注册对照」，最后断言部分缺失仍走原分支、消息不被新
分支吞掉。

```bash
PY=$HOME/anaconda3/envs/mamba/bin/python
$PY -m pytest tests/test_condition_workflow.py -q -k control_evidence
#   => 1 passed, 20 deselected in 0.02s
$PY -m pytest tests/test_condition_workflow.py tests/test_condition_analysis.py \
      tests/test_condition_fresh_gate.py -q
#   => 80 passed in 10.67s（三个直接调用冻结门槛的文件，一条未删未放宽；
#      test_condition_analysis.py:571 那条显式传 [] 的旧用例仍通过——它只断言 reasons 非空）
$PY -m pytest tests -q          # 355 passed, 11 skipped in 115.04s（§0.10 时为 354 passed）
```

> §0.10 实测块里的 `354 passed, 11 skipped in 110.35s` 是**当时**的真实读数，保留以留痕；本节之后
> 同一命令的最新读数是 `355 passed, 11 skipped in 115.04s`（只多这一个测试，无删改、无放宽）。

#### 0.11.1 仍未闭合（读到这里请不要把 §0.11 当成实验完成）

- **completed 试次仍为 0**，`best_condition_config.json` 不存在，**最优条件注入方式仍未确定**。本节
  只是「冻结门槛在缺对照证据时会不会放行」的库层闭合，不是训练结果。
- 新的 fail-closed 分支以「预注册里确有必需对照」为前提（`required` 为空则不报）：若某个 study 把
  `required_controls` 全登记为 `False`，空证据不会被判缺失——那是预注册口径本身没要求对照，不是门槛漏判。
- `_freeze_gate` 自算的 `missing_ctrl` 不按 `required_controls[k]` 的真假过滤（列出所有未出现的键），
  比库函数**更严**；两处都 fail-closed，因此不会互相放行，但这两套口径**没有合并**成一处。
- 非 fresh 的 `--frozen_trial` 仍完全绕过 `_freeze_gate`（§六.8 兼容，见 §0.9.1）；那条路径连库函数
  都不会被调用，本修复对它无效。
- §0.9.1 / §0.10.1 的其余条目原样保留：Pareto 的时延目标仍回退 `wall_seconds`；模型侧只剩
  **OS-only / CPM-only 注入范围未实现**（`models/tfm.py:642-647`）。

### 0.12 同日第八次追加：final test 不幂等，也不绑定冻结的那份 checkpoint（§八.9）（2026-10-09 晚）

顺着 §0.11 继续查同一条冻结/测试链路时发现：`_maybe_final_test_eval` 只校验「验证集那条 completed
了没」，**从不看 `@test_final` 是否已经存在**（`experiments/condition_runner.py:1634`）。§八.9 要求
「final test 执行幂等，绑定已冻结配置和 checkpoint 集合」，实测两条都不满足：

| 场景（登记表状态） | 旧行为（实测） | 新行为（实测） |
|---|---|---|
| 只有验证集 completed 行 | 评估 test 一次，登记 `running`+`completed` 两行 | 同（未变） |
| `@test_final` 已 completed，**同一** checkpoint | **再评估一次**（重复烧 GPU·h），再追加两行同 id 记录 | `evaluate_trial` 调用 0 次，不追加行，`results` 里标 `accounting='reused'` |
| `@test_final` 已 completed，绑的是**另一个** checkpoint | **不报错**，用新权重的 test 结果覆盖 `latest()` 视图里的旧结果 | `ValueError`：「final test 必须绑定同一 checkpoint（方案 §八.9）」，不评估、不登记 |
| `@test_final` 上一次是 `failed` | 重跑（期望行为） | 同（未变，§八.5：失败不是低分也不算跑过） |
| dry-run + `@test_final` 已 completed | 打印 `[dry-run] FINAL-TEST` 命令（谎报「会跑」） | 打印 `[reuse] … 幂等跳过`，`accounting='reused'` |

第二种场景不只是浪费预算：`Registry.append` 是追加式（历史行永不改写），因此重复执行会在
`registry.jsonl` 里留下**同一个 `trial_id` 的多份 test 指标**，`latest()` 只保留最后一份 ⇒ 报告读到
哪一份取决于执行次数，而 §八.9 要求 test 绑定「已冻结配置和 checkpoint 集合」。第三种场景更直接：
它允许「冻结的是 A 配置、登记表里 `A@test_final` 的指标其实来自 B 权重」这种错配被静默覆盖。

**修法**（`experiments/condition_runner.py:1650-1666`）：在 dry-run/execute 分支**之前**查
`registry.latest()[f'{trial_id}@test_final']`——已 completed 且 checkpoint 一致 ⇒ 打印复用说明、把该行
以 `accounting='reused'` 放进本次 `results`（沿用训练复用路径 `:1077-1079` 的既有约定，不新造口径）
后返回；checkpoint 不一致 ⇒ `ValueError` 拒绝，不覆盖既有 test 结果；`failed`/`interrupted`/缺行 ⇒
照旧评估。原有的三个前置校验（必须给 `--frozen_trial`、trial 必须在当前计划里、必须有 completed 的
验证集结果）与执行路径一行未改。

**证据**：新增
`tests/test_condition_injection.py::test_O_final_test_eval_is_idempotent_and_bound_to_frozen_checkpoint`
——用**真实 `Registry`**（JSONL 落盘 + 重新读盘）只 stub `evaluate_trial`，覆盖上表 5 个场景，并断言
第一次的调用签名是 `('test', <冻结权重路径>, full_guardrails=True)`（顺带钉住「test 评估开全护栏」）。

```bash
PY=$HOME/anaconda3/envs/mamba/bin/python
$PY -m pytest tests/test_condition_injection.py -q -k final_test_eval
#   => 1 passed, 68 deselected in 3.66s
$PY -m pytest tests -q          # 356 passed, 11 skipped in 115.19s（§0.11 时为 355 passed）
bash -n scripts/Run_condition_study.sh    # 语法 OK
PYTHON="$PY" bash scripts/Run_condition_study.sh --dry-run --stages 0..5 \
  --study-id dr8c --study-root /tmp/dr8c/study --max-trials 200 --gpu-budget-hours 300
#   => exit=0；dry-run 未创建任何目录（交付入口未被本修复影响）
$PY -m experiments.condition_runner --stages 5 --promote demo direct --registry_dir <临时目录>
#   => exit=0；阶段 3 仍 14 试次（每臂 7）；[validate] 告警 0 条
```

> §0.11 实测块里的 `355 passed, 11 skipped in 115.04s` 是**当时**的真实读数，保留以留痕；本节之后
> 同一命令的最新读数是 `356 passed, 11 skipped in 115.19s`（只多这一个测试，无删改、无放宽）。

#### 0.12.1 仍未闭合（读到这里请不要把 §0.12 当成实验完成）

- **completed 试次仍为 0**，`best_condition_config.json` 不存在，**最优条件注入方式仍未确定**；本节
  只是「test 评估重复执行/权重错配会不会被静默接受」的编排闭合，不是训练结果。
- 幂等判据是「`checkpoint_path` 字符串相同」，**不是**权重 SHA256 相同：同一文件被就地覆盖（本轮
  fresh 目录隔离下不会发生，但旧入口共享 `checkpoints/` 时可能发生）会被当成「同一份权重」而复用。
  登记行里已有 `source_checkpoint_sha256` 列，若要把判据升级到哈希，需要先在 test 评估行上也落该列。
- ~~「冻结配置」这里等价于「`--frozen_trial` 指定的那一条试次的那一份权重」；§八.9 说的
  「checkpoint **集合**」（最终配置 + 预先固定的直接对照一起评 test）目前仍是**单条**：本入口一次只评
  `--frozen_trial` 指定的那一条，直接对照的 test 评估要靠多次调用（每次一个 `--frozen_trial`），
  没有「一次调用评完整个冻结集合」的入口。~~ **已闭合，见 §0.13**（旧文字保留以留痕；实测旧路径一次
  调用只产生 1 条 `@test_final`、对照 0 条，而且 test 是在冻结门槛判定**之前**就被评掉的）。
- `results` 里的复用行沿用了 `accounting='reused'`，因此预算裁剪逻辑（`:973-977`）会把它当成
  「不花钱的保留项」；这是对的（本次运行确实没有新消耗），但那一行的 `gpu_hours`/`wall_seconds`
  仍是**上一次**执行的实测值，读摘要时要按 `accounting` 区分「本次花的」与「复用来的」。

### 0.13 同日第九次追加：test 在冻结判定之前就被评掉，且「冻结集合」只有一份权重（§八 / §八.9）（2026-10-09 深夜）

§0.12.1 留下的「一次调用只评 `--frozen_trial` 一条」不只是缺功能：顺着同一条链路实测还发现顺序错了。
`run()` 里 `_maybe_final_test_eval` 排在 `write_best_config` **前面**（`experiments/condition_runner.py:1057-1059`），
而预注册冻结门槛只在 `write_best_config` 里跑 ⇒ 门槛根本不通过的配置也会先评一次 test、登记
`@test_final`，之后才由 `write_best_config` 报「冻结门槛未满足」。留出集因此暴露给一个还没定案的决策，
而 §八 要求的是「**冻结后**一次性评估 test 上的最终配置及预先固定的直接对照，不再根据 test 调参」。

用 `tests/test_condition_fresh_gate.py::_full_rows` 那份合成登记表（3 seed 候选 + 同族 demo 基线 +
4 类对照；每条给一份自己的权重路径，否则「同权重对照」过滤会把整个对照集合吃掉）实测：

| 场景 | 旧行为（实测） | 新行为（实测） |
|---|---|---|
| 缺 bootstrap 产物 ⇒ `can_freeze=False` | `evaluate_trial` 调用 **1** 次，登记表出现 `T-cand-2024@test_final`，随后 `write_best_config` 才拒绝 | `TrialError`「预注册冻结门槛未满足，拒绝评估 test…」；`evaluate_trial` 调用 **0** 次；`@test_final` 行 **0** 条 |
| 门槛通过（5 类预注册对照都有 completed 行） | 一次调用只评 **1** 份权重（冻结配置自己），对照的 test 指标 **0** 条 | 一次调用评 **6** 份：`frozen` + 5 类对照；每条各绑自己的权重、登记 `<trial_id>@test_final`、带 `final_test_role`/`final_test_of` |
| 同一条命令再跑一次 | （§0.12 起）冻结配置那一条 reused | 6 条全部 `[reuse]`，`evaluate_trial` 调用 0 次，`accounting='reused'` |
| 某类对照与冻结配置是同一份权重（配置去重后阶段 2/3 共享权重） | 不适用（旧路径不看对照） | 该类对照不评第二次，打印「与冻结配置是同一份权重，test 只评一次」 |
| 某类预注册对照没有任何 completed 行 | 静默不评 | 打印 `⚠ 预注册对照没有可评 test 的 completed 行：[…]`，明确 test 表不完整 |

**修法**（全部在 `experiments/condition_runner.py`；§0.12 的幂等/绑定逻辑原样搬进 `_final_test_one`，
所以 §0.12 里 `:1650-1666` 那个行号是**当时**的读数，现在对应 `:1694-1741`）：

1. `_freeze_decision`（`:2097-2119`）：把「按 `family_config_hash` 聚同配置的多 seed 组 + 过 `_freeze_gate`」
   从 `write_best_config` 里抽出来，`write_best_config`（`:2122`）与 `_maybe_final_test_eval` 共用同一份判定，
   两处不会再各说各话；非 fresh 的旧入口仍返回 `decision=None`（§六.8 兼容性未变）。
2. `_maybe_final_test_eval`（`:1744-1780`）：解析出冻结行之后、动 test 之前先跑 `_freeze_decision`，
   `can_freeze=False` 直接 `TrialError`（`:1762-1772`）。
3. `_final_test_targets`（`:1643-1691`）：集合 = 冻结配置 + 预注册直接对照。对照**不是这里挑的**——名字来自
   本轮预注册 `required_controls`，命中规则复用冻结门槛的 `_CONTROL_PREDICATES`（`:1956`），所以「test 上
   跟谁比」在看到任何 test 数字之前就定死了；同一命中集合里取**验证集**主指标最好的那条（平手按
   `trial_id` 升序），因为挑最弱的对照会凭空放大候选的收益，而 test 从不参与这个选择。
4. `Runner.evaluate_trial` 的第一个参数改为 `Optional[Trial]`（`:852-859`）：对照行通常不在本次计划里
   （`--stages 5` 只排复核试次），而评估命令只由权重路径 + split + 护栏开关决定，协议由 §六.7 的
   strict 重建从 checkpoint 里读，本来就不依赖计划里的 `Trial`（函数体一字未改）。

`--evaluate_test_final` 的帮助文本、`--status` 的下一步提示（`:1579-1584`）和
`scripts/Run_condition_study.sh` 的收尾说明同步改成「一次调用评完整个冻结集合；fresh 下门槛不通过会先拒绝」。

**证据**：新增
`tests/test_condition_fresh_gate.py::test_25_final_test_covers_the_preregistered_control_set_after_the_gate`
（四段：门槛未过 ⇒ 拒绝且 0 次评估、0 条登记；门槛通过 ⇒ 角色集合 == `{frozen} ∪ required_controls`、
每条 test 行绑定自己那份权重、`final_test_of` 指回冻结试次、对照不在计划里也能评；重跑 ⇒ 全部
`reused`；同权重对照 ⇒ 集合里不重复）。原有断言一条未删、未放宽：直接调 `_freeze_gate` /
`write_best_config` 的 `test_6/7/8/19/20/21/23` 全部照旧通过。

```bash
PY=$HOME/anaconda3/envs/mamba/bin/python
PYTHONPATH=. $PY -m pytest tests/test_condition_fresh_gate.py -q -k test_25
#   => 1 passed, 24 deselected in 0.58s
PYTHONPATH=. $PY -m pytest tests/test_condition_fresh_gate.py \
  tests/test_condition_injection.py tests/test_condition_workflow.py -q
#   => 114 passed in 48.14s
PYTHONPATH=. $PY -m pytest tests -q                # 357 passed, 11 skipped in 112.93s
bash -n scripts/Run_condition_study.sh             # 语法 OK
PYTHON="$PY" bash scripts/Run_condition_study.sh --study-id dr9c --study-root <临时目录> --stages 3
#   => exit=0；dry-run 未创建 study 目录；收尾说明按新口径打印
PYTHONPATH=. $PY -m experiments.condition_runner --stages 3 \
  --checkpoint_dir <临时目录> --registry_dir <临时目录>
#   => exit=0；[validate] 告警 0 条
```

> §0.12 实测块里的 `356 passed, 11 skipped in 115.19s` 是**当时**的真实读数，保留以留痕；本节之后同一
> 命令的最新读数是 `357 passed, 11 skipped in 112.93s`（只多 `test_25` 一个测试，无删改、无放宽）。

#### 0.13.1 仍未闭合（读到这里请不要把 §0.13 当成实验完成）

- **completed 试次仍为 0**，`best_condition_config.json` 不存在，**最优条件注入方式仍未确定**；本节只是
  「留出集什么时候能评、一次评哪几份权重」的编排闭合，不含任何训练结果。
- 对照的解析口径与冻结门槛一致（`_CONTROL_PREDICATES`，不限 seed）：同一类对照有多个 seed 的 completed
  行时，进 test 表的是**验证集最好的那一条**，登记行带它自己的 `seed`。跨 seed 的对照 test 比较必须在
  报告里写明，不能当成同 seed 配对差。
- 顺序闸门只在 fresh（`cfg.fresh` 且 `cfg.study is not None`）下生效；非 fresh 的旧入口按 §六.8 保持原行为
  （先评 test，再由 `write_best_config` 判门槛）。本轮交付的所有命令都走 fresh 分支。
- 幂等/绑定判据仍是 `checkpoint_path` 字符串（见 §0.12.1），不是权重 SHA256。
- 阶段 5 的多 seed 复核矩阵只重训**晋级候选**（`stage5_multiseed`，`experiments/condition_space.py:595-649`
  ——行号是写 §0.13 时的读数），不为每个 seed 重建五类对照；因此 `--frozen_trial` 指向 seed≠2024 的复核试次时，
  `demographic_same_init_family` 这类要求「同一 HC 起点哈希」的对照可能整类找不到 completed 行（会打印上面的 ⚠），
  冻结门槛同样会拒绝。要按 seed 配齐对照，需要显式把对照试次也展开到复核 seed —— 这属于实验矩阵扩容，未做，
  等确认口径。
  **→ 本条已在 §0.14 闭合**：对照按 seed 展开（连同 §一.6 的逐 seed HC 起点重训）已进矩阵，入口默认
  `--review-scope full`。原文按留痕保留，不再代表当前行为。

### 0.14 同日第十次追加：阶段 5 缺诊断腿与逐 seed 对照、入口锁死弱复核口径、阶段 4 起点家族用错（§一.4 / §一.6 / §六.4 / §七阶段 4-5 / §八.8）（2026-10-09 深夜）

四处真实断裂，全部用只读探针实测复现（不训练、不写产物），修完再复测。

| # | 缺陷（修复前实测） | 后果 | 修法 |
| --- | --- | --- | --- |
| 1 | `condition_runner.main()` 的阶段 5 分支只调 `stage5_multiseed`，把 `generate_stages` 的 `max_stage` 压到 4，因此走不到它内部的 `review_diagnostics`。实测 `--stages 5 --promote demo direct`：阶段 5 只有 **4** 条试次，`kind=diagnostic` 计数 **0** | `all_missing_degradation` 只在 `kind='diagnostic'` 的行上实测，再由 `_guardrail_evidence` 按「同一权重 + 同一 seed」搬回复核训练行；没有诊断行 ⇒ 复核行护栏永远缺测 ⇒ `check_freeze_eligibility` 按「缺测≠通过」永远拒绝冻结。`best_condition_config.json`、final test 在**构造上**就到不了，不是「跑完再看」 | 阶段 5 排程改为 `deduplicate(trials + reviews + review_diagnostics(reviews))`；复测阶段 5 = **30** 条、诊断 **6** 条且逐条绑定被诊断权重的 `trial:<config_hash>` |
| 2 | 阶段 5 只重训**晋级候选**，不为每个 seed 重建五类直接对照（即 §0.13.1 最后一条） | `demographic_same_init_family` 要求对照行与候选行有**相同且非空**的 `source_checkpoint_sha256`；`full` 口径下每个 seed 的 HC 起点各自重训、哈希必然不同 ⇒ seed≠2024 的候选整类找不到对照；§八「各 seed 分别报告」也没有同 seed 对照可配对 | 新增预注册对照表 `REVIEW_CONTROL_KINDS` + `_stage5_controls()`，由 `stage5_multiseed(..., with_controls=True)` 展开；`generate_stages` 与 runner 都传 `True`。默认 `False` 只为保住「只列候选复核」的库层语义（`test_18` 原有断言未改一字） |
| 3 | 交付入口 `scripts/Run_condition_study.sh` 根本不暴露 `--review-scope`，runner 的默认是 `mdd_only` | fresh 一键入口只能跑「只换 MDD seed、HC 起点沿用 seed 2024」，把 §一.6「每个 seed 重新训练其 HC 起点」悄悄降级，且不打印任何提示 | 脚本增 `REVIEW_SCOPE="full"`（fresh 入口默认 full）+ 解析分支（只接受 `full｜mdd_only`）+ 阶段 5 显式传 `--review-scope`；选 `mdd_only` 时打两行 `[warn]` 明说这不是完整两阶段复核 |
| 4 | `stage4_injection` 用 `fam = _hc_family(d_demo, cond_dim)`（**H1**）当注入位置比较的起点，文档串还写着「`heads` 位置与阶段 3 的锚点配置完全相同，靠 `trial_id` 去重自动复用、不重复训练」 | 直接违反 §一.4 / §六.4 / §七阶段 4「在共同本轮 H0 初始化族比较注入位置」「阶段3的 H1 起点 heads 权重不能作为阶段4 H0 起点 heads 对照复用」；而本文件 §0.10 第 5 条已经写下「本轮阶段 4 恒为『从同一个 H0 新建恒等条件模块』」——代码与自述不一致 | 起点改 `_hc_none_family()`（H0）；H0 无条件 HC 没有条件统计量可继承，故 `path_slots` 只留 `pretrained_weight`、删掉 `condition_inherit_stats`（与 `stage2_mdd_screen` 无条件臂同一口径）；`compared_against` 由「阶段 3 的 H1 锚点」改为**同家族**的 `mdd_<arm>_heads_last1`（heads 参照自己那条为空，不跨家族指） |
| 4b | `stage5_multiseed` 在 `full` 口径下补逐 seed HC 基线时恒取 `arm='demo'` | 第 4 条改完后，H0 家族的候选会把一条**带三字段条件头**的 HC 权重登记成 `hc_none#seedN`，与 `_stage5_controls` 补出的真 H0 撞成同名家族的两条不同权重 ⇒ 家族解析按 §0.10 第 3 条只能 block | 按被复核试次要求的家族选臂：`hc_none` → `arm='none'`，`hc_demo_*` → `arm='demo'`；找不到就抛错，不静默换一条 |

对照构造口径（写死在代码里，不看结果增删）：同 seed、同 HC 起点（`requires_family` 与 `pretrained_weight`/`condition_inherit_stats` 槽位一致）、同非条件配方（`merged_recipe()` 逐键相同，§八.7）；只改**条件信息**、不改注入位置（`inject`/`backbone_scope` 沿用候选）；容量类对照（`static_new`/`mask_only`）保留候选的 `(d_new, h_delta, dropout_p)`，而 `demo`/`direct` 是惰性宽度轴、回落到 `ANCHOR_D_NEW`/`ANCHOR_H_DELTA`（否则 `validate_trials` 按「mdd8 + 惰性臂 + 非锚点宽度」直接拒）；`arm='none'` 的无条件辅助基线 `cond={}`、起点是 H0 家族，初始化差异写进 `rationale`（§七阶段 5「明确标注初始化差异」）；`role='control'` 使 `TRIM_PRIORITY` 里 control(1) 优先于 review(4)，预算裁剪不会先砍对照，而 `review_diagnostics` 只诊断 `role='review'`，不给对照白烧评估预算。

实测（只读探针，累积计划 = 阶段 1..4 + 阶段 5，`--promote demo direct --position backbone`）：

```text
# 第 4/4b 条修完后
review_scope=full      plan=65  stage4=5  stage4_families=['hc_none']  blocked=0
                       duplicate_family_providers={}   validate_trials 告警 0
                       providers={hc_demo_d16_C32:1, hc_none:1,
                                  hc_none#seed2025:1, hc_none#seed2026:1}
review_scope=mdd_only  plan=63  blocked=0  validate_trials 告警 0

# stage4_injection(['cat'], scope_rung='first') 逐条读数
requires_family  = ['hc_none'] * 3
path_slots keys  = ['pretrained_weight']          # 不再有 condition_inherit_stats
compared_against = [[], ['mdd_cat_heads_last1'], ['mdd_cat_heads_last1']]
```

脚本 dry-run（`--dry-run`，无 `--execute`/无 `--device`，不启动任何训练）两种口径都 `exit=0`，`[CMD]` 行分别带 `--review-scope full` / `--review-scope mdd_only`；`bash -n scripts/Run_condition_study.sh` 通过。

回归：`tests/test_condition_fresh_gate.py` 新增 `test_26`（阶段 5 逐 seed 对照 + 诊断）与 `test_27`（阶段 4 共享 H0、阶段 3/4 的 heads 身份不得互相去重、逐 seed H0 基线只有一种身份）；该文件 `27 passed in 3.99s`，三个条件测试文件 `116 passed in 48.45s`，全套 `PYTHONPATH=. pytest tests -q` = **`359 passed, 11 skipped in 116.09s`**（§0.13 是 357，只多 `test_26`/`test_27` 两条，原有断言一条未删、未放宽）。

第 4 条把阶段 4 的起点换成 H0 之后，正式执行时每条注入位置试次都要走「源权重里没有任何条件 bundle」这条迁移分支，因此又补了**模型级**验收 `tests/test_condition_injection.py::test_K2_h0_start_builds_new_identity_condition_modules`（真实 `NeuroTwinTFM`，`dim=16, layers=2`，三个注入档各跑一遍）：

- `migrate_pretrained_checkpoint(..., arch_policy='require_match')` 不报错，`missing=[]`、`skipped=[]`；
- `conditioning_param_names(model)` 的每一个键都落在 `new_initialized` 里 —— 没有任何条件参数是从 H0 权重里拿的（`cond_names - new_initialized == ∅`）；
- H0 `state_dict()` 的每个键在迁移后**逐位相同**（`torch.equal`）；
- 恒等/零初始化下，带条件模型的 OS 与 CPM 输出对 H0 无条件输出的最大误差 `< 1e-6`（§六.3 口径），且**全缺失条件**与**有效观测条件**两种批次都成立；
- 一次 backward 后条件参数 `grad` 全部非 `None`，且至少一个非零（不是死模块）。

读数：`pytest tests/test_condition_injection.py -q -k "K2 or K_migration"` = `2 passed, 68 deselected in 12.08s`；加入 `test_K2` 后全套 = **`360 passed, 11 skipped in 112.96s`**（比上一条读数只多 `test_K2` 一个测试）。

`test_K2` 直接调迁移器，绕过了正式执行时排在迁移器**之前**的两道 CLI 装配层判定，所以又补了两处验收把这条链闭合（`tests/test_condition_fresh_gate.py::test_28_h0_source_passes_the_cli_task_signature_gate`，以及 `test_K2` 里新增的一条断言）：

- **任务口径闸门**：`main.py` 在 finetune 装配时先跑 `check_pretrain_config_compat(args.pretrained_weight, dims, ...)`，源权重 args 快照的任务口径签名与当前运行不一致就 `RuntimeError`。`test_28` 不训练、不读数据，而是拿**运行器真实吐出的命令行**（`cr.build_train_command` + `cr.fresh_init_flags`）过 `main.parse_args()`，再按闸门自己的口径逐键比对。实测 H0 与 H1 保存的签名完全相同（`{task_mode: next_timepoint, model_arch: tfm, in_window: 1, pred_window: 1, seq_len: 64, patch_len: 4, cpm_horizon: 8}`，起点家族不同不改变非条件任务口径），阶段 4 三条注入位置命令的期望值与之**逐键一致**，即闸门会放行；同一测试还断言每条命令带 `--init_source current_study_hc_checkpoint`、`--source_trial_id`、`pretrained_weight` 就是 H0 家族槽位、`condition_inherit_stats` 为空、`eval_after_train is False`（搜索期不评 test），并**故意把 `seq_len` 加 1 构造一次失配**，确认这条比对不是空转。
- **bundle 分支**：`main.py` 的两条 layout 闸门只在 `load_condition_bundle(pretrained_weight)` 读得到条件协议块时才生效；H0 是无条件权重，必须返回 `None` 才会跳过（否则会被误判成「同组同臂但布局冲突」而直接 `RuntimeError`，阶段 4 一条都跑不起来）。`test_K2` 现在对刚存出的 H0 权重断言 `load_condition_bundle(path) is None`。

读数：`pytest tests/test_condition_fresh_gate.py -q` = `28 passed in 3.94s`；`pytest tests/test_condition_injection.py -q -k "K2"` = `1 passed, 69 deselected in 4.79s`；全套 = **`361 passed, 11 skipped in 112.40s`**（比上一条只多 `test_28`，`test_K2` 是加断言不是加测试；原有断言一条未删、未放宽）。

#### 0.14.1 仍未闭合（不要把本节当成实验完成）

- **completed 试次仍为 0**，`best_condition_config.json` 不存在，**最优条件注入方式仍未确定**。本节全是编排/矩阵层面的闭合，不含任何训练结果。
- 第 4 条让阶段 4 的 `heads` 位置不再复用阶段 3 的试次，因此**阶段 4 的训练条数增加**（原去重省掉的那条现在必须真训）；这是 §七阶段 4 要求的代价，不是回归。§5 的实测预算表按旧口径估的阶段 1→4 数字因此偏低，等正式跑时以本轮实测画像为准。
- 每 seed 的直接对照按 `(seed, arm, control)` 去重后是 **7** 条而非 10 条：两个晋级候选同为 `demo` 臂（只差 `inject`/`backbone_scope`），它们的无条件辅助基线配置完全相同，被 `deduplicate` 按 `config_hash` 合并 —— §一.11 的合法去重，不是漏排。
- 从 H0 起点新建条件模块这条链现在**四层都有验收**：模型级（`test_K2`：条件参数全部归
  `new_initialized`、非条件权重逐位继承、OS/CPM 输出与 H0 等价 `<1e-6`、backward 后条件参数
  有非零梯度、`load_condition_bundle(h0) is None`）、CLI 装配级（`test_28`：H0/H1 的任务口径
  签名相同且与阶段 4 每条命令逐键一致，闸门放行；来源声明与不评 test 也在同一条里钉住）、
  以及**端到端真跑**（§0.15：真实 `main.py` + 真实 merged 数据跑完 H0 预训练→条件 MDD 微调
  →独立 strict 重建评估→置换/全缺失诊断，四条全部退出码 0）。
  仍未做的是**正式配方**下的真跑：§0.15 用的是阶段 0 的 tiny 结构与 1 epoch，只证明链路通，
  不产生任何可排名的结果。
- 幂等/绑定判据仍是 `checkpoint_path` 字符串（§0.12.1），不是权重 SHA256。
- OS-only / CPM-only 注入档未实现（`models/tfm.py:642-647`），阶段 4 的位置比较只覆盖 heads / backbone / heads+backbone 三档。

### 0.15 同日第十一次追加：H0→条件 MDD 端到端真跑（真实 merged 数据 + 独立评估 + 诊断）（§六.4 / §七阶段 4-5 / §九）

§0.14.1 上一条留的口子（「只比签名，没真跑」）已经用真跑堵上。四条命令全部退出码 0，
study `h0smoke20261009`，**全部产物写在 `/tmp/h0_smoke/`**，`checkpoints/` 与仓库内一个字节都没动；
结构用阶段 0 的 tiny 档（`tfm_dim 64 / tfm_layers 2 / cpm_horizon 4 / context 16`）、1 epoch、
`--is_probe`，所以**这一节只证明链路能跑通，不产生任何可排名、可晋级的结果**。

| 腿 | 命令入口 | 结果 |
|---|---|---|
| ① HC H0 预训练 | 真实 `main.py --mode pretrain --condition_arm none` | 退出码 0，26 s，`Trainable Parameters: 241680` |
| ② MDD 条件微调（H0 起点） | 真实 `main.py --mode finetune --condition_arm cat --cond_inject backbone --cond_backbone_scope last1`，**不带** `--condition_inherit_stats` | 退出码 0，`Trainable Parameters: 26800`，`Val next MAE: 0.5004` |
| ③ 独立 strict 重建评估 | `python -m experiments.evaluate_variant --splits val --no_rollout --no_fc --save_arrays` | 退出码 0，`val_next_mae=0.5004308896594577`（与训练循环内 0.5004 一致），`n_params=268480`，逐被试 CSV 225 人 |
| ④ 置换/掩码/全缺失诊断 | `python -m experiments.condition_diagnostics --split val --n_boot 200 --max_batches 4` | 退出码 0，9 个场景全部实跑，8 条配对比较各带 `ci95` 与 `p_two_sided` |

关键日志原文（含竖线，故不放进表格单元格；表格里转义的竖线会在 `<code>` 里留下反斜杠）：

- ①：`[condition] arm=none：绕过条件通路（不构建编码器，也不构建 FiLM / legacy HAMD 数值调制）`；
  `tfm_patho_mode=none | OS FiLM: off | CPM FiLM: off | 主干条件 RMSNorm: off`；
  `[split] 复用版本化切分 manifest: data/subject_split_pretrain.json`，`train_subjects=1778 / val=222 / test=222`。
- ②：`[fresh] MDD 起点校验通过：study=h0smoke20261009 source_trial=phase0_smoke_h0 sha256=4c3e5188e440 seed=2024`；
  `[fresh][warn] 起点是阶段 0 探针/smoke 权重…不得进入正式排名、晋级或 best 配置`；
  `[migrate] base_best.pt（arch_version=5）| loaded/copied=72 | new_initialized=35 | zero_expanded=0 | skipped=0 | missing=0`；
  `条件参数=14128` + `主干条件 RMSNorm scope=last1 generator=per_layer blocks=[1] 额外参数=12672`。
- ③：`val_next_pcc_spatial=0.7807235855526394`；`val_persistence_mae=0.5175459190209707`；
  `per_subject_metrics_val.csv` 写出 225 名被试。
- ④：9 个场景 = `identity` / `perm_new_joint` / `perm_value_keep_mask` / `mask_only_inference` /
  `all_missing_repr` / `perm_HAMD_HAMA_joint` / `perm_HAMD` / `perm_HAMA` / `bypass_condition`；
  分层表给出 `Dataset` / `HAMA_observed` / `n_observed_fields` 三轴的人数与不确定性。

**这一节真正钉住的四件事**

1. **H0 起点 + 条件 MDD 的组合在真实数据上不炸**。此前 `test_K2`（模型级）与 `test_28`（CLI 装配级）
   都是不加载数据的验收；阶段 4 的三条注入位置命令第一次真的走完「读 merged → 拟合 → 建条件模块
   → 迁移 → 训练 → 存权重」。迁移报告 `new_initialized=35 / missing=0 / skipped=0` 与 §六.2 的口径一致：
   H0 没有条件键，所以条件模块整体归「新建」，没有任何键被静默跳过。
2. **参数量三条路径互核**：$241680$（①的 H0 可训练参数，即非条件主干）$+\ 26800$（②的条件参数）
   $=\ 268480$（③独立重建后的 `n_params`）。②里 `26800 = 14128 + 12672`，即编码器 + 主干条件
   RMSNorm 生成器，与 `param_report()` 的实测算法同源。
3. **`inject=backbone` 的「模型生效 `tfm_patho_mode=film` 但两个头 FiLM 都 off」不是缺陷**。
   日志那行容易误读，实际语义是：`head_cond_dim` 只在 `cond_inject ∈ {heads, heads+backbone}` 时才等于
   `cond_dim`（`models/tfm.py:642-647`），backbone 档两个头都不建 FiLM，条件只从主干 AdaRMSNorm 进。
   ③④两条都在这份权重上跑通，说明 strict 重建与诊断都没有任何一处假设「`tfm_patho_mode=film` ⇒
   `one_step_head.film` 存在」：重建走 `build_model_from_args`，它按快照显式转发
   `cond_inject / cond_backbone_scope / cond_norm_generator / cond_gate_*`（`evaluate_variant.py:145-149`）；
   冻结/解冻与优化器分组走 `conditioning_module_prefixes`，按 `is_conditioning_adapter` 标记取**模块路径**
   前缀（`train/optim.py:329-341`），不写死头名。`inference.py` 的定型入口对任何带条件臂的权重都会在
   `build_model_from_args` 处因缺 `condition_bundle` 直接报错，不会被冒名成「最终 FiLM 主权重」。
4. **`condition_activation.json` 的 `all_activated=true` 在 backbone 档是「按配置该有的桶都激活」**：
   backbone 桶 `first_nonzero_grad_step=1 / first_param_update_step=1 / first_grad_max=1.5127e-3 /
   first_param_update_max=5.0e-5 / in_optimizer=true`（参数 `cond_norm.gen.L1.{weight,bias}`）；
   encoder 桶 `first_output_change_step=1 / first_nonzero_grad_step=2`；cpm 桶 `status=not_built`
   ——注入位置是 backbone，CPM 头本来就不建条件模块，「没建」与「建了不激活」在记录里是分开的。

**顺带得到的 §一.14 正面证据**：第一次跑②时我把 `--checkpoint_dir` 指到了 study 目录**之外**，
`verify_current_study_hc` 立刻 `ProvenanceError: 路径 …/probe/phase0_smoke_h0/base_best.pt 不在本轮
study 目录 …/study 之内 —— 疑似历史/外部权重`。把 checkpoint 目录改回 study 目录内的
`phase0_probe`（交付脚本 `scripts/Run_condition_study.sh` 用的就是这个位置）才通过。
也就是说起点校验按**路径归属 + SHA256** 判，不按文件名判。

**两处必须写清的口径限制（不是缺陷，但引用读数时会踩）**

- **PCC 两个口径不相等，MAE 相等**：训练循环打印的 `Val next PCC(sp)=0.7697` 是**逐 batch 空间 PCC 的
  meter 平均**（`main.py:333`），③的 `0.7807` 是**拼接全 split 后**再算的空间 PCC。PCC 是非线性统计量，
  批平均 $\ne$ 全量；MAE 是线性的，所以两边都是 0.5004。排名与护栏只读③这一口径
  （`read_metrics` → `per_subject_metrics_<split>.csv` + `metrics_<split>.json`），符合 §八「以被试为统计单位」。
- **诊断里的 `ci_excludes_zero=true` 不能当「条件有效」读**：本次 `identity_minus_all_missing_repr` 的
  `mean_diff = -7.35e-07`、`ci95 = [-8.00e-07, -6.69e-07]`、`p_two_sided = 0.0050`。差异只有
  $7\times10^{-7}$（相对量级 $\approx 1.5\times10^{-6}$），是 1 epoch + 近零初始化的数值噪声，
  但配对 bootstrap 对**逐被试同号的确定性微差**几乎必然给出「区间不含 0」。
  产物里已带 `identity_is_smoke_checkpoint_hint` 明说这点；控制流也不吃这个标志：
  预注册护栏 `all_missing_degradation` 用的是 `delta_vs_identity / identity_mae` 这个**相对幅度**
  （`condition_runner.py:394-397`），冻结判定的主指标门槛用 `relative_mae_improvement_min` + 配对 CI
  （`condition_study.py:757-770`），$1.5\times10^{-6}$ 的"改善"远低于 1% 门槛，只会**追加**一条阻塞理由。
  `missing_robustness = -delta_vs_identity` 会进 Pareto 支配判定，而 `_pareto_front` 用的是裸浮点 `<`
  （无容差，`condition_study.py:804-832`）：噪声量级的鲁棒性差会改变前沿成员。**Pareto 表是报告件、
  不参与 `can_freeze`**，所以不影响冻结结论；但正式跑时若各候选的 `all_missing_degradation` 都落在
  噪声量级，应把这一轴读成「未区分」，不要据它排序。

**这节没有证明的**：completed 学习试次仍为 **0**，`best_condition_config.json` 仍不存在，
**最优条件注入方式仍未确定**；②的权重带 `is_probe`，按 §一.7/§七阶段 0 不得冒充正式 HC 权重；
④用的是 `--n_boot 200 --max_batches 4`（诊断队列 128 名被试，其余 194 个 anchor 被显式剔除并计数），
正式跑必须用默认 `n_boot ≥ 5000`、`max_batches=0`（全量）。

### 0.16 同日第十二次追加：编排层的三道闸门实测（计划防篡改 / 预算 fail-closed / 起点记录）（§八.2 / §八.6 / §八.9）

§0.15 绕过了 runner（直接调 `main.py`／`evaluate_variant`／`condition_diagnostics`），所以这一节单独
核 runner 自己的编排层。全部是 **dry-run 或只读**，没有启动任何训练，产物都在 `/tmp/runner_smoke/`。

1. **计划文件防篡改（§八.2 的入口侧证据）**。`condition_space --stages 1 --json` 导出的阶段 1 计划
   是 2 条试次（`hc_none` / `hc_demo`，`train_epochs=150`、`tfm_dim=256`、`eval_fc=true`）。把每条的
   `recipe` 手工改成 1 epoch + tiny 结构后交给 `--plan`，`trial_from_dict` 直接拒绝：
   `ValueError: 计划文件与当前代码的哈希不一致：s1_hc_none#f3f1075452c1 vs s1_hc_none#08ef0f42240a；
   说明搜索空间常量或配方已改动，旧计划须重新生成（禁止混用两套 id）`。
   也就是说**改配方就会改身份**，「30 epoch 的计划冒充 150 epoch」在加载计划这一步就不可能悄悄发生。
   我没有为了跑通而重算哈希去绕过这道校验——绕过它就等于把这条证据作废。
   代价是：runner 没有「tiny 档」训练配方开关（`--cost_preset` 只影响**成本核算**，`--ladder` 又要求
   至少两档），因此**无法廉价地用真实训练跑一遍 runner 的阶段循环**；这一层仍是待用户正式执行时验证的项。
2. **预算闸门 fail-closed（§八.6）**。不给 `--max_trials` 时它的默认值是 `0`（`condition_runner.py:2563`），
   runner 报 `执行次数=2 / 上限 0` 并把 2 条试次全部裁掉（`为放进预算被裁掉的试次：2 个`）——
   即使 `--gpu_budget_hours 100` 也不会执行任何训练。缺实测画像时它不猜数字，而是明确写
   `没有实测画像，无法核算 GPU·h：先运行 experiments/condition_budget.py --device cuda:N 实测 HC/MDD 探针
   （期望位置 <study_root>/analysis/budget_profile.json）`，同时声明「次数上限仍生效」。
   交付脚本在这一层之上再加一道：`scripts/Run_condition_study.sh:270-271` 对 `--execute` 跑正式阶段
   强制要求显式正整数 `--max-trials`，否则 `die`。`--unlimited` 被脚本映射成「不传 `--ladder`」的单档
   完整预算（runner 本身没有 `--unlimited` 开关，脚本注释里逐条核对过 argparse），且与 `--ladder` 互斥。
3. **起点记录不伪造结论（§八.9）**。fresh dry-run 会把搜索起点写进本轮自己的 registry：
   `[initial] 搜索起点记录已写入 <study_root>/registry/initial_condition_config.json（status=search_not_run）`；
   目录里**没有** `best_condition_config.json`。最终报告同时打印「未覆盖的搜索范围（不得声称已搜索）」
   清单（外层宽度家族、illness 的 log1p／HAMD 条目 vs 总分、真 AdaLN 与条件低秩、未补测的 rollout/FC
   护栏、未排期的置换诊断），排名段落写的是 `completed=0`，没有把 pending 当成低分或当成结果。

两次 dry-run 退出码都是 0；`study.json` 与 `registry/` 都建在给定的 `/tmp` study 根内，
`results/condition_search/`、`checkpoints/` 与仓库内文件均未被读写。

### 0.17 同日第十三次追加：修复冻结判定跨 ladder 档位混算（§八.2 / §八.7 / §八.8）

**这是一个真实缺陷，不是文档问题**，此前所有测试都没有覆盖到它。

- 现象：`condition_runner._freeze_decision` 只按 `family_config_hash` + `status=completed`
  + 主指标有限 聚「同配置多 seed 组」。而 `family_config_hash` 取自 Trial 的
  `merged_recipe()`，`--ladder` 的档位覆盖只写进命令行与登记行的
  `train_epochs`/`ladder_rung`（`_row_base` 里 `train_epochs = rung if rung is not None
  else recipe['train_epochs']`），所以 30/90/150 epoch 的三批行**共享同一个 family hash**。
- 后果：低档位行会进入冻结排名、`ranked`、`pareto` 与 `deterministic_tiebreak`；
  更危险的是 `_freeze_gate` 的 `arm=demo` 基线取 `min(...)`、以及 `_controls_present`
  找对照证据时也不看档位——一条**更差的 30 epoch 基线**会把候选的「相对改善」抬高，
  可能把没有信息增益的配置推过预注册的 1% 实践门槛。这正是方案 §八.2
  「30 epoch 不能复用为 90/150 epoch」与 §八.7「同预算比较校验…训练资源」要禁的事。
- 原有的 `write_best_config` 里确有一道 `rung < max(cfg.ladder)` 检查，但它只查
  **被冻结那一行**，且依赖「本次调用恰好传了同一个 `--ladder`」；单独跑冻结/评 test
  的那次调用不带 `--ladder` 时它完全不生效，也管不到基线与对照。

改动（3 个文件，全部是收紧，没有放宽任何原有断言）：

1. `experiments/condition_runner.py` 新增 `_BUDGET_COLUMNS` 与 `_row_budget(row)`：
   按 `train_epochs` → `planned_epochs` → `ladder_rung` → `rung` 取第一个有限值；
   四者皆无（诊断行、旧入口行）返回 `None`，语义是「无法证明预算」而**不是** 0，
   调用方对 `None` 一律不过滤（§六.8 旧入口兼容）。
2. `_freeze_decision` 先按 family hash 聚组，再按候选行自己的档位切一刀，
   被切掉的行以 `excluded_budgets` 传给 `_freeze_gate`；`_freeze_gate` 的
   `arm=demo` 基线、`_controls_present` 的对照证据都只在**同档位**行里找。
   排除说明写进新的 `FreezeDecision.notes`（不阻断冻结——ladder 本来就会留下淘汰行），
   并落进 `best_condition_config.json` 的 `freeze_gate.notes` / `freeze_gate.train_epochs`。
3. 新增一条**阻断**理由：候选行的档位 ≠ `DEFAULT_RECIPE['train_epochs']` 时不许冻结
   （「本轮没有在登记预算上跑完该配置」）。这条不依赖 `cfg.ladder`，
   因此单独跑 `--frozen_trial` 也拦得住。
4. `experiments/condition_study.py` 的 `FreezeDecision` 增加 `notes: list[str]`
   （带默认值，向后兼容；`reasons` 与 `notes` 分开，避免「为了过门槛而删说明」）。

证据（全部实跑，非语法检查）：

- `tests/test_condition_fresh_gate.py` 新增 `test_29/30/31`，`31 passed`；
  29 放一条主指标 0.001 的 30 epoch 行进同家族，断言它**不在**冻结组里且出现在 notes；
  30 刻意把 `cfg.ladder` 清空，证明拒绝来自新的预算一致性检查而非旧的 rung 检查；
  31 把 demo 基线与四类对照全部只留在 30 epoch 档，断言 150 epoch 候选报「缺基线
  （同训练资源档位）」+「对照缺实跑结果」，并把它们补回同档位后门槛立刻放行。
- **有牙校验**：把 `cr._row_budget` 临时替换成恒返回 `None`（等价于修复前），
  三条新测试全部失败；换回后全部通过。
- 全量回归：`pytest tests/ -q` → `364 passed, 11 skipped in 124.53s`
  （修复前为 361 passed；净增 3 条即本次新增，无失败、无跳过变化）。

仍未覆盖：这三条是**合成登记行**上的控制流验收，不是训练证据；
`--ladder` 真实跑出的多档位 registry 还没在真训练上产生过（见 §0.16 的说明）。

### 0.18 同日第十四次追加：冻结判定继续收紧到「整套非条件配方」（§二.2 / §八.7 / §八.8）

§0.17 只堵住了 epoch 档位这一个维度，留下的口子是**它自己的**：登记行的
`recipe_hash` 列（`_row_base` 写 `merged_recipe()` 去掉 `train_epochs` 的指纹，
覆盖 batch/lr/架构/采样/冻结档）此前只被 `same_budget_ranking` 当作可比性证据，
**从未与预注册值比对过**，`_freeze_gate` 也只看 epoch。于是一个在「便宜配方」
（例如 smoke 用的小 batch、短 context）上跑完、并且基线与对照也用同一套便宜配方
配套跑完的 study，可以完整通过冻结门槛被写成 `best_condition_config.json` —— 
而方案 §二.2 要求非条件配方保持固定，§八.9 要求只有真实完整复核才能产生 best。

改动（2 个文件，全部是收紧；`condition_study.FreezeDecision` 不变）：

1. `experiments/condition_runner.py` 新增 `preregistered_recipe_hash()`：
   与 `_row_base` **同一口径**算 `DEFAULT_RECIPE` 去掉 `train_epochs` 的指纹，
   避免「门槛比的值」和「登记写的值」各算一套。
2. 新增 `_same_training_recipe(row, cand)`：候选行有 `recipe_hash` 证据时要求两侧
   指纹相同，再叠加 §0.17 的 epoch 档位比较；候选自己没登记该列（旧入口行）就不按
   配方过滤，保持 §六.8 的既有行为。`_freeze_decision` 的分组、`_freeze_gate` 的
   `arm=demo` 基线、`_controls_present` 的对照证据（形参由 `budget=` 改成 `like=`）
   全部改用这一把刀；被排除的行仍写进 `FreezeDecision.notes`，并注明其中有多少行
   连配方指纹都不同。
3. 新增第二条**阻断**理由：候选行 `recipe_hash` 非空且 ≠ 预注册指纹时不许冻结
   （「只能作为显式登记的协议差异参与对照比较」）。
4. `best_condition_config.json` 的 `freeze_gate` 段新增 `recipe_hash`、
   `preregistered_recipe_hash`、`recipe_matches_preregistration` 三个字段，
   事后能回答「这个 best 是在哪套非条件配方、哪个预算档上跑出来的」。

证据（全部实跑，非语法检查）：

- `tests/test_condition_fresh_gate.py` 新增 `test_32`，该文件 `32 passed`：
  先断言 `runner._row_base(...)['recipe_hash'] == preregistered_recipe_hash()`
  （口径一致的锚点），再断言全套行都带上预注册指纹时冻结照常放行、五类对照齐备；
  然后把候选换成另一套配方指纹，断言冻结组只含同配方行、拒绝理由含
  「非条件配方指纹 ≠ 预注册值」；最后把基线与对照也搬到那套配方上，断言
  `decision.reasons` **只剩这一条**（`len == 1`）—— 即此场景下唯一挡住冻结的就是
  预注册一致性检查，`write_best_config` 抛 `TrialError` 且不落盘 best 文件。
- **有牙校验**（`/tmp/teeth_recipe.py`，临时改内存里的实现，不改仓库文件）：
  把 `_same_training_recipe` 退化成只比 epoch → 测试失败；把 `_freeze_gate` 输出里
  「非条件配方指纹」那条理由剥掉 → 测试失败；两处换回 → 通过。
- 全量回归：`pytest tests/ -q` → `365 passed, 11 skipped in 112.44s`
  （§0.17 时为 364 passed；净增 1 条即本次新增，无失败、无跳过变化）。

仍未覆盖：`test_32` 依旧是**合成登记行**上的控制流验收，不是训练证据。
另外，`recipe_hash` 只证明「命令行/配方字段一致」，不证明训练进程真的按该配方跑完
（那要靠 §0.16 的任务口径闸门 + 真跑日志）；`--ladder` 与非默认配方都还没有在
真训练上产生过 registry。

### 0.19 同日第十五次追加：给「资源口径」开一个受控且永远冻不结的口子（§二.2 / §八.7 / §八.9）

§0.18 之后，编排层出现了一个**自我锁死**的局面：要验证阶段循环能在真实数据上端到端
跑通（依赖解析 → 训练 → 评估 → 诊断 → 晋级 → 冻结门槛），就必须有一套便宜的资源口径；
但仓库里没有任何 CLI 能产生这种口径 —— 手改导出的计划会被 `trial_from_dict` 的哈希
校验拒绝（§0.16 第 1 条），`--ladder` 又要求至少两个档位且只会**缩短** epoch。
结果就是「编排链路从未在真实数据上跑完整圈」这条阻塞项没有任何合法的解除方式。

本次补的就是这个口子，并且让它**只能通向 smoke、通不到部署配置**：

1. `experiments/condition_space.py` 新增三张表：`RECIPE_OVERRIDE_ALLOWED`（6 个资源/
   调度键：`train_epochs`、`warmup_epochs`、`freeze_backbone_epochs`、`patience`、
   `batch_size`、`num_workers`）、`RECIPE_OVERRIDE_MIN`（各自下限）、
   `RECIPE_KEY_CATEGORY`（其余 43 个键 → 类别名：架构 / 预测目标 / 损失 / 采样与增强
   分布 / 评估口径 / 优化超参）。两表**互斥且并集覆盖** `DEFAULT_RECIPE` 全部 49 键，
   由 `test_33` 断言，所以将来新增配方键要么被显式归类、要么被报成「未分类」，
   不会掉进缝里。
2. 新增 `parse_recipe_overrides()`：拒绝信息按**类别**点名（不是「不支持」而是
   「它属于架构，方案 §二.2 要求非条件 backbone 宽度/层数保持固定」）；与默认值
   相同的覆盖按 no-op 丢弃，避免无谓地把一次正式运行标成非正式。
3. 新增 `apply_recipe_overrides()`：把覆盖写进 `Trial.recipe`（**参与 `config_hash`**），
   于是去重、复用、同预算分组、`recipe_hash` 登记列全都能区分便宜口径与预注册口径
   —— 与 §八.2「30 epoch 不能复用为 90/150 epoch」是同一把刀。跨键自相矛盾
   （`warmup_epochs >= train_epochs`、`freeze_backbone_epochs >= train_epochs`）
   失败即退出。两个调用方都在**阶段 5 复核/对照/诊断试次全部生成之后**才调用它，
   否则后加的行会留在预注册档位上，形成一张内部资源不一致的计划。
4. 新增 `plan_recipe_deviation()`：`--plan` 与 `--stages` 两条来源共用的**唯一事实
   来源**（读试次真正会用什么，不信命令行声称改了什么）。三种情况拒绝：非资源维度
   偏离、试次之间偏离不一致、recipe 里有当前代码不认识的键。`load_plan` 再核对
   计划级元数据 `recipe_overrides` 与 `formal` 是否与试次实际配方一致。
5. `experiments/condition_runner.py`：`ExecConfig.recipe_overrides` 由
   `plan_recipe_deviation(trials)` 从试次反填；`_freeze_gate` 增加**第三条**阻断理由，
   只看本次运行自己声明的口径（前两条都依赖登记行里有 `recipe_hash` / 档位列，
   缺列就拦不住）；`best_condition_config.json` 的 `freeze_gate` 段新增
   `run_recipe_overrides`；`last_schedule_summary.json` 新增
   `recipe_overrides` / `formal` / `formal_note`；`--plan` 与 `--recipe-override` 互斥
   （否则计划文件里的口径声明会与实际训练不符）。
6. `scripts/Run_condition_study.sh` **不加**这个开关：一键正式入口不该能悄悄产出
   非正式 study。要 smoke 就直接调 `condition_space` / `condition_runner`。

顺带修掉一处自己写错的语义：原先 `freeze_backbone_epochs >= train_epochs` 的报错说
「条件模块永远不会解冻」。按 `main.py:953-961` 与 `train/optim.py:set_finetune_stage`，
条件模块在冻结阶段**本来就参与训练**，真正不发生的是 `epoch >= freeze_backbone_epochs`
这个解冻分支 —— 主干全程不解冻，于是 `backbone` / `heads+backbone` 注入位置的联合训练
与 §六.6「解冻保留 Adam 状态」路径都不会被执行。报错文本已改为这个准确后果；同时该
闸门改为**只在计划里真的有 `mode='finetune'` 试次时**才校验，因为 `cli_flags` 对
pretrain 行根本不渲染 `--freeze_backbone_epochs`，纯 HC 计划（`--stages 1`）缩短 epoch
属于正当用法，不该被误伤。

证据（全部实跑）：

- `tests/test_condition_fresh_gate.py` 新增 `test_33`–`test_36`，该文件 `36 passed`：
  三张表的覆盖/互斥、6 类禁改键的类别化拒绝、类型与下限拒绝、no-op 丢弃；
  覆盖后 `trial_id` 与 `family_config_hash` 全部改变而身份仍两两不同、非条件固定项
  逐键不动、`cli_flags` 真的带上覆盖值、pretrain 行不渲染冻结档但 recipe 仍记录偏离；
  真实导出 `--stages 2` 便宜计划（`formal=false`）与正式计划（`formal=true`）后
  往返 `load_plan`，再对元数据说谎、`formal` 说谎、篡改 `trial_id` 三种计划断言拒绝，
  对旧格式（无元数据 dict / 裸 trials 列表）断言仍可读（§六.8）；
  runner 侧断言 `--plan` + `--recipe-override` 抛 `TrialError`、
  登记行完全合法（连 `recipe_hash` 都缺列）时冻结仍被第三条理由拒绝、
  `write_best_config` 抛错且不落盘、`write_run_summary` 记 `formal=false` 且原有字段
  未被挤掉。
- **有牙校验**（改内存/临时改文件后逐条还原，最终 `grep MUTATION` 为空）：
  `_freeze_gate` 忽略运行级口径 → `test_36` 失败；`plan_recipe_deviation` 不查计划内部
  一致性 → `test_35` 失败；`load_plan` 不核对元数据 → `test_35` 失败；
  覆盖只写 `rationale` 不写 `recipe`（即不参与哈希）→ `test_34` 失败；四处还原 → 全通过。
- 全量回归：`pytest tests/ -q` → `369 passed, 11 skipped in 113.18s`
  （§0.18 时为 365 passed；净增 4 条即本次新增，无失败、无跳过变化）。
- dry-run 实测（不启动任何进程）：
  `python -m experiments.condition_runner --stages 2 --recipe-override train_epochs=2
  warmup_epochs=1 freeze_backbone_epochs=1 batch_size=8 --registry_dir /tmp/… --device cuda:1`
  打印五行 `[非正式]` 横幅，8 条 TRAIN 命令全部带 `--train_epochs 2 --batch_size 8
  --warmup_epochs 1`（finetune 行另带 `--freeze_backbone_epochs 1`），而 `tfm_dim 256`、
  `context_max 64`、`sampling_seed 20261009`、`cond_dropout_seed 20261008`、`lr_*`
  等非条件值逐字保持预注册值。

仍未覆盖：这条口子解除了「无法产生便宜口径」的阻塞，但**真实的端到端 smoke 还没跑**
（下一步：在空闲 GPU 上执行一次 `--execute`，产出真 registry / 逐被试评估 / 诊断产物，
再把证据补进本节）。在此之前，§0.15 的结论不变：runner 的阶段循环从未驱动过一次
完整配方的真训练；`--ladder` 与任何非默认配方也都还没有在真训练上产生过 registry。

### 0.20 同日第十六次追加：fresh 编排其实跑不完阶段 2——来源校验发生在填值之前（§一.12 / §八.5）

§0.19 收尾时按它自己写的「下一步」去跑真实 dry-run，第一次就把编排打崩了。这不是
便宜口径带来的新问题，而是 fresh 模式一直存在的**顺序缺陷**，只是此前所有验收都停在
合成登记行上，从没让 runner 真的走完一次 `--stages 2` 的排程：

```
File "experiments/condition_runner.py", line 1166, in run_trials
    results.append({**runner._row_base(trial, rung), 'status': 'pending', …
File "experiments/condition_runner.py", line 761, in _row_base
    row = cs.decorate_row(self.cfg.study, row, kind=trial_kind(trial), …
File "experiments/condition_study.py", line 647, in decorate_row
    raise ProvenanceError('MDD registry 行必须记录 source_trial_id。')
```

根因：`_row_base` **先**调 `decorate_row` 校验来源，`execute_trial` 才在返回后
（原 `:854-855`）把 `paths['__source_trial_id__']` 写进行里。于是每一个 fresh MDD 行
在被装饰的瞬间都是「缺来源」——dry-run 打印到第 3 条 MDD 命令就抛异常退出（exit=1），
执行模式连 `status='running'` 都登记不了。换句话说：**§0.15 说的「阶段循环从未驱动过
一次完整配方的真训练」，此前即使想跑也跑不到阶段 2 的第二个试次。**

顺带查出同一条链路上的第二个洞：`resolve_paths` 对「家族起点在本次排程里、但还没跑完」
的情况返回占位串 `'<family:…（本次排程中会先训练，执行时按序解析）>'` 且 `blocked=None`。
dry-run 里这是诚实的（打印出来就是告诉用户它会解析成什么），但**执行模式**下阶段 1 的
HC 一旦失败，阶段 2 的 MDD 就会拿着这个占位串当 `--pretrained_weight` 去启动训练，
同时 `--source_trial_id ''`——正好是 §一.3 禁止的「MDD 不是从本轮真实 HC 起点开始」。

改动（2 个文件，全部是收紧或修正顺序，没有删掉任何原有断言）：

1. `experiments/condition_runner.py::_row_base` 新增 `paths` 形参与 keyword-only
   `require_source`（默认 `True`）：fresh 分支在调 `decorate_row` **之前**就从
   `paths['__source_trial_id__'] / ['__source_sha256__']` 落位两列来源。
2. 9 个 `_row_base` 调用点全部改为传入已解析的 `paths`；`execute_trial` 尾部那两行
   重复赋值删掉（避免「装饰时一套、事后又一套」的口径分叉），只留一行注释说明来源由
   `_row_base` 负责。
3. `experiments/condition_study.py::decorate_row` 新增 keyword-only
   `require_source: bool = True`。默认严格分支**原样保留**
   `raise ProvenanceError('MDD registry 行必须记录 source_trial_id。')`；只有调用方
   显式声明「这是非证据行」时才允许来源为空。哪些行是非证据行有明确边界：
   dry-run 打印行与 `blocked` 行**从不写 registry**，`failed` 行则**必须**能写进去
   ——否则一次失败会把整轮编排打崩，连「失败过」这个事实都留不下（§八.5）。
   忘记传参就落到严格分支，不会静默放行。
4. `resolve_paths` 的 `family:` 与 `trial:` 两个前向引用分支加上执行模式闸门：
   `cfg.execute` 为真而起点仍未完成时改为 `blocked`，理由里点名 §一.3/§八.5；
   dry-run 保持占位串不变。

证据（全部实跑）：

- dry-run 从 `exit=1` 变成 `exit=0`：同一条命令
  （`--fresh --stages 2 --execute` 去掉、`--recipe-override train_epochs=2 …`）打印
  8 条 `[dry-run] TRAIN` + 8 条 `[dry-run] EVAL`（2 个 HC pretrain + 6 个 MDD
  finetune），末尾正常写出 `initial_condition_config.json（status=search_not_run）`
  与「未覆盖的搜索范围」清单，stderr 为空。
- `tests/test_condition_fresh_gate.py` 新增 `test_37`，该文件 `37 passed`
  （§0.19 时为 36）。它把「修好之后必须同时成立的两件事」钉在一起：非证据行不再打崩
  编排（`require_source=False` 时来源为空、行仍带本轮 `study_id`/`init_source`），
  而证据行照旧强校验（同样缺来源的 `paths` 按默认口径装饰必须抛 `ProvenanceError`）；
  另外断言来源已解析时两列在装饰前就位，以及执行模式下未完成的家族起点被 block 且
  `paths` 里拿不到任何占位权重路径。
- **有牙校验**（`/tmp/teeth_prov.py`，只改内存不改仓库文件，4 处变异各自独立失败）：
  装饰前不落位来源 → `test_37` 失败；`require_source` 被忽略（一律严格）→ 失败；
  强校验被拿掉（一律放行）→ 失败；执行模式又接受占位起点 → 失败；未变异 → 通过。
- 全量回归：`pytest tests/ -q` → `370 passed, 11 skipped in 114.45s`
  （§0.19 时为 369 passed；净增 1 条即本次新增，无失败、无跳过变化）。

仍未覆盖：`test_37` 依旧是合成登记行上的控制流验收。真实 `--execute` smoke
（`--stages 2`，2 epoch 便宜口径，隔离 `/tmp` study 目录，GPU1）已启动并在写
registry，其结果与逐被试产物在本节写就时尚未跑完，**不得**据此声称阶段 2 已验收。

### 0.21 同日第十七次追加：真实 `--execute` fresh 编排跑完了，并当场揪出三处来源登记缺陷（§一.12 / §八.4 / §八.7）

**这次运行是什么**：`scripts/Run_condition_study.sh` 的 fresh 执行分支，真实 `main.py`、真实
merged 数据、新 `study_id=smoke_exec_20261009`、独立输出目录（临时目录，未触碰
`checkpoints/` `results/` `data/`），`--stages 2 --execute --device cuda:1`，并用
`--recipe-override` 把资源维度压到 `train_epochs=2, warmup_epochs=1, batch_size=8,
freeze_backbone_epochs=1` —— 所以它是**工程 smoke，不是正式实验**（`formal=false` 由
`last_schedule_summary.json` 自己标注，`formal_note` 明确这些权重不得当正式 HC 家族起点）。

**编排结果（实测，7 次训练启动 + 1 个 blocked）**

| 试次 | 状态 | 墙钟(s) | GPU·h | 被试级 OS MAE | 人数 | 条件参数 | `rho_conditioning` | `init_source` |
|---|---|---|---|---|---|---|---|---|
| `s1_hc_demo` | completed | 817.46 | 0.227 | 0.12240 | 222 | 37200 | 0.006134 | `random_initialization` |
| `s1_hc_none` | **failed** | 787 | — | — | — | — | — | `random_initialization` |
| `s2_mdd_demo` | completed | 665.71 | 0.185 | 0.1143544 | 225 | 37200 | 0.006134 | `current_study_hc_checkpoint` |
| `s2_mdd_direct` | completed | 663.95 | 0.184 | 0.1143526 | 225 | 38928 | 0.006417 | 同上 |
| `s2_mdd_sum` | completed | 727.00 | 0.202 | 0.1143489 | 225 | 43824 | 0.007218 | 同上 |
| `s2_mdd_mean` | completed | 712.01 | 0.198 | 0.1143568 | 225 | 43824 | 0.007218 | 同上 |
| `s2_mdd_cat` | completed | 702.66 | 0.195 | 0.1143579 | 225 | 47920 | 0.007887 | 同上 |
| `s2_mdd_none` | **blocked** | 0 | 0 | — | — | — | — | — |

合计消耗 `gpu_hours≈1.19`。逐被试文件齐备：`registry/eval/` 6 个目录，各含
`per_subject_metrics_val.csv`（3774–3825 行）与 `metrics_val.json`。
**`best_condition_config.json` 没有生成**（冻结门槛按 `formal=false` 拒绝，`initial_condition_config.json`
照旧在位）—— 方案 §八.9 的反向检查成立。

**上一轮修的两道闸门在真实执行里确实开了**：`s1_hc_none` 崩溃后，依赖 `hc_none` 家族的
`s2_mdd_none` 被按 §一.3/§八.5 登记为 blocked，理由是
「执行模式不接受占位权重路径——MDD 只能从本轮真实训练出的 HC checkpoint 开始」。
它没有被静默训成占位命令，也没有被当成「一个分数差的臂」进入排名。

**三处缺陷（都只可能在真实执行里暴露，合成测试全部看不见）**

1. **`source_checkpoint_sha256` 每条 fresh MDD 行都是空串**。`resolve_paths` 读
   `prov['checkpoint_sha256']`，而 `verify_current_study_hc` 把当场算出的摘要放在
   `_file_sha256`（内嵌块里的 `checkpoint_sha256` 是**来源行**记的值，语义不同）。键名对不上
   ⇒ §一.12 要求的「按 SHA256 追溯起点权重」在真实链路里静默失效，而权重文件内嵌块明明带着
   `4ca790c5ef8523b2…`。改为读 `_file_sha256`，算不出来就 block，不登记不可追溯的行。
2. **14 行 registry 的 `split_hash` 全为空，`study.json` 也是空** ⇒ §八.7 的同预算校验把
   每一条已完成行都判为「缺可比性证据」，阶段 2 直接拒绝晋级：
   `以下已完成行缺少同预算可比性证据（split_hash/init_source/recipe_hash）`。
   根因是行里只抄 `study.split_hash`，而这个字段从来没人按 main.py 的口径填。权威值只有训练
   进程知道（`main.split_hash(dl, args)` 用实际装载名单算 16 位语义指纹，且会覆盖 `_study_provenance`
   里的同名字段），所以改成**跑完后从权重内嵌块读回**，并新增 `split_sha256`（划分文件摘要，
   与 `dedup_signature` 同口径）；读不到、为空一律 `TrialError`，绝不写空值让下游静默失去可比性。
3. **来源核验的两个口径错误**（同一处代码，收紧而不是放宽）：
   * `expect_split_hash=self.cfg.study.split_hash` 拿 manifest 来源标识去比 main.py 的语义指纹，
     是不同量：真实 smoke 里它为空 ⇒ 校验空转；一旦非空（测试用 `'split-sha'`）又会**误拒**
     合法的本轮起点。改为比**来源行自己记录**的 `split_hash`（同一口径）。
   * registry 每次执行会写 `running` + 终态两行，旧匹配逻辑取**第一条** trial_id 命中行 ⇒
     永远停在 `running` 上。加了下述状态校验后，这个顺序会误判合法完成的 HC 起点，因此改成取
     **最后一行**。
   * 「文件内容对得上」曾被当成「试次完成了」。`s1_hc_none` 崩溃前已存过 `base_best.pt`，
     那份**孤儿权重**的内嵌 study 块完全合法，旧逻辑会让它通过来源核验被当成本轮起点。
     现在来源行的 `status` 非 `completed` 即拒（§八.4/§八.5）；`status` 列缺失不判，
     合成测试里不带该列的行照旧可校验（未删任何原断言）。

**验证证据**

- 新增 `tests/test_condition_fresh_gate.py::test_38`：用真实 `torch.save` 权重 + `file_sha256`
  断言 ① `paths['__source_sha256__']` 等于当场算出的 64 位摘要且落到行上，② `running`+`completed`
  两行时合法起点**不被**误拒，③ 后一行为 `failed` 时抛 `ProvenanceError` 且消息含 `§八.4`，
  ④ 文件被改动仍按 SHA256 拒（原断言保留）。该文件 `38 passed in 3.87s`。
- 变异检验（每处都改回旧行为，跑完立即还原，`restored byte-identical: True`）：

  | 变异 | 结果 |
  |---|---|
  | M1 来源 SHA 读回不存在的键 | `1 failed` |
  | M2 去掉「来源行必须 completed」 | `1 failed` |
  | M3 取第一行而不是最后一行 | `1 failed` |
  | M4 `expect_split_hash` 退回 `study.split_hash` | `1 failed` |

- 全量回归：`pytest tests/ -q` → **`371 passed, 11 skipped in 116.90s`**（上轮基线 370 passed，
  新增 1 条，无既有测试转红、无断言删除）。
- §八.7 在**真实产物**上的重放（`_same_init_family(s2_mdd_demo, s2_mdd_cat)`）：修复口径前
  `False`（`split_hash=''`、`source_checkpoint_sha256=''`），按权重内嵌值补齐后 `True`；
  跨队列的 `(s1_hc_demo, s2_mdd_demo)` 仍为 `False`（cohort 前置生效）；重算的来源摘要
  `4ca790c5ef8523b2` 与训练进程自己写进权重的值逐位一致 —— 两条独立入口对齐。

**真实失败与残余问题（不当作已修好）**

- `s1_hc_none` 的失败是训练进程侧的 `RuntimeError: Pin memory thread exited unexpectedly`
  （`main.py:1091 evaluate_tfm` 的第二个 epoch 验证阶段，`num_workers=4` + `pin_memory=True`
  的 torch 多进程/fd 传递问题，`returncode=1, wall=787s`）。**不是条件链路的缺陷**，
  本轮也没有改任何非条件配方；但它是正式运行时真会撞上的运维风险，重跑该臂应按资源维度处理
  （`--retry_failed` 重开，或对单试次降 `num_workers`——这是白名单资源键，不改模型语义）。
- `[promote]` 的拒绝消息把 `s2_mdd_none` 也列进「以下**已完成**行…」，而它的登记状态是
  blocked（`registry.csv` 7 行 / `registry.jsonl` 14 行里都没有它）。这次没有造成错误排名
  （结果是拒绝晋级），但措辞与行状态不符，属于 §八.5「失败/中断不得混进已完成证据」的表述面，
  留作下一步核对，不改判据。
- 本节仍**不是**阶段 2 的验收通过：`formal=false` 的 2-epoch 权重不能当正式起点，MAE 差异
  （0.1143489–0.1143579，跨度约 `8.9e-6`）在 2 epoch、单 seed、无 bootstrap、无护栏的条件下
  毫无解释力。**在用户正式跑完阶段 1–5 之前，不得声称已确定最优注入方式或实验完成。**
- §0.2 里 `tests/test_condition_benchmark.py::test_non_finite_timing_and_empty_step_count_raise`
  的时序敏感加固**仍未完成**，该条目继续按「已实现但未验证」计。

### 0.22 同日第十八次追加：性能画像的时序敏感测试加固为确定性断言（§二.3「已实现但未验证」清账）

§0.2 一直挂着一条「`test_non_finite_timing_and_empty_step_count_raise` 在与其他进程并发时
失败过 1 次、单独复跑通过 ⇒ 需要加固成确定性断言后才算稳定」。本轮把它读透并修掉了，
根因不是「机器慢」，而是**两条与断言无关的随机/墙钟依赖**：

1. `bench.build_case(_specs()[0], device='cpu')` 不带 `seed` ⇒ 模型权重与合成 batch 落在
   **全局 torch RNG 的当前位置**上。测试顺序、此前哪些测试跑过、并发时的导入交错都会移动这个
   位置，于是「同一条测试」检查的其实是不同的数值轨迹。
2. `build_profile(..., warmup=1, n_steps=1)` 用**真实墙钟**跑一遍，只为了拿一个结构再去
   `median_ms = float('nan')` 验证「NaN 不许落盘」。一条纯逻辑断言被绑到了计时源上。

**改法（收紧，不是放宽）**：固定 `seed=20261008`；把计时源换成常量假时钟
（`monkeypatch.setattr(bench, '_timed', lambda fn, dev: 1.0)`），并**当场新增**三条精确断言
—— `samples_ms == [1.0]`、`median_ms == 1.0`、`n_steps == 1 且 warmup_steps == 1`。
原有 12 个 `pytest.raises` 异常断言（`n_steps`/`warmup`/NaN/inf/空样本/样本数不一致/
`write_profile` 拒 NaN/三处 `NonFiniteTimingError`）逐条保留、顺序不变；测试末尾仍然把 `_timed`
改回 `float('nan')`，所以「真实测量路径遇到非有限值必须报错」这条覆盖也没有丢。
AST 清点：`assert` 语句 2 条（新增）、`pytest.raises` 12 处（原样）。

**压测证据**

- 单条测试连续 20 次独立进程运行：**20/20 通过**（此前会偶发失败）。
- 并发压力：同时起 8 个 `pytest tests/test_condition_benchmark.py` 进程（整机负载 + 导入交错），
  结果 `12 passed` × 7（129.8–154.5 s）与 `12 passed`（77.4 s）——**8/8 全通过，0 失败**。
- 全量回归：`pytest tests/ -q` → `371 passed, 11 skipped in 114.74s`，与 §0.21 基线完全一致。
- 画像 CLI 本身未改：`experiments/condition_benchmark.py` 一行没动，只改测试；真实墙钟画像仍由
  `test_profile_json_roundtrip_records_reproducibility` 等其余用例继续覆盖（它们本来就通过）。

**§0.2 状态更新**：该条目从「已实现但未验证」移到「已实现且已验证（测试确定性已加固）」，
旧文字保留并就地标注 → §0.22。

**仍然未验证 / 未实现（不要当成已完成）**

- 真实 `--execute` fresh 编排只在 `formal=false` 的 2-epoch 资源档上跑通（§0.21），
  **正式阶段 1–5、≥5000 次被试级配对 bootstrap、3 seed 复核、冻结后一次性 test 全部仍未运行**，
  由用户手动执行；在此之前不得声称已确定最优条件注入方式。
- 正式运行仍可能撞上 `pin_memory` 数据加载器崩溃（§0.21 的运维风险），重跑路径是
  `--retry_failed` 或对该试次单独降 `num_workers`（白名单资源键），本轮未改任何非条件配方
  （→ §0.24：`--retry_failed` 在 fresh 入口被拒，本轮正式运行的合法路径是
  `--rerun-interrupted`（入口脚本）/ `--rerun_interrupted`（runner），整试次从头再训）。
- OS-only / CPM-only 注入仍未实现（`models/tfm.py:642-647`），真正的 AdaLN、条件门控低秩、
  `heads+backbone` 的 last2/all 扩展都在「未排期」清单里；`[promote]` 消息把 blocked 试次
  称为「已完成行」的措辞问题也仍未改（判据本身正确，是拒绝晋级）
  （→ §0.23：核实后发现这不是措辞问题，而是会让阶段 3+ 永远无法晋级的真实缺陷，已修复）。

> **§0.23–§0.25 的历史性说明**：这三节修的都是 `experiments/condition_runner.py` /
> `experiments/condition_budget.py` / `scripts/Run_condition_*.sh` 这套对比编排层，
> 它们已于 §0.26 全部物理删除。记录本身保留（含当时的实测数字与变异测试结论），
> 但**不要再按这些节去找对应文件或对应当前行为**；其中与模型/数据协议相关的结论
> （臂语义、可比性口径、blocked/failed 不等于低分等）仍然适用于现在的代码。

### 0.23 同日第十九次追加：晋级护栏被非 completed 行污染，阶段 3+ 原本永远无法开工（§八.5 / §八.7 / §八.8）


**触发点**：§0.22 收尾时把 `[promote]` 的「已完成行」当成措辞问题挂着。本轮按提示词要求
「发现编排无法完整运行就继续修复，不要停在建议后续实现」，用真实产物重查这条消息，
结论是判据本身也不对。

**证据（全部来自本轮 `/tmp/nt_smoke_exec` 的真实 execute 产物，非合成表）**

1. smoke 日志第 41 行原样打印：
   `阶段 2 晋级臂推导：以下已完成行缺少同预算可比性证据（split_hash/init_source/recipe_hash）`，
   点名 5 个 `completed` 臂**和 1 个 `s2_mdd_none#ab364e8efe39`（blocked）**，而当时 5 个臂都带真实
   验证集 MAE。同一次运行还打印「阶段 2 排名不足，暂不能确定晋级臂」——排名其实不缺，是护栏误伤。
2. `_row_base`（`experiments/condition_runner.py:813`）把 `split_hash` 写成
   `cfg.study.split_hash`，该字段 CLI 从不填充；只有 `execute_trial` 成功尾部才从权重内嵌块回写
   真值（§0.21 的修复）。因此 **`blocked` / `failed` / `running` 行天然缺 `split_hash`**。
3. 生产调用链传的是 `Registry.rows`（append 全量日志，**含每次训练先写的 `running` 影子行**），
   不是 `latest()`。真实登记表 14 行 = `completed` 6 + `running` 7 + `failed` 1。
   把 5 个 completed 行的 `split_hash` 按其**真实权重文件**内嵌值（`f1f36cbbede3ca7e`）回写后重放，
   修复前判据仍拒绝 6 行：5 个 `running` 影子行 + 1 个 `blocked` 行。
   **含义：即使一次失败都没有，`validate_promotion` 也会在任何阶段 3+ 执行前退出**——
   §0.21 的 split_hash 修复单独并不足以让正式实验跑起来，每个试次的 `running` 影子行都会自己
   把自己的阶段判死。这是交付给用户手动执行时会直接撞上的硬阻塞。

**修复（不放宽任何护栏）**

- 新增 `ranking_eligible(row)`：`status == 'completed'` 且带验证集主指标，才是「参与排名/晋级」的行。
- `same_budget_ranking` 的内联判据改为调用它（行为等价，只是把口径收一处）。
- `assert_comparability_evidenced` 只对 `ranking_eligible` 的行要求证据；逐行明细带真实 `status`，
  消息改为「以下参与排名的行（status=completed）缺少同预算可比性证据」，并说明
  `blocked/failed/running/interrupted/pruned` 行不进入排名、不因缺签名而阻断晋级。
  两处调用点（阶段 2 晋级臂推导、阶段 N 胜出配置推导 `_stage_completed`）自动共用该口径。
- 理由：非 completed 行**永不成为第一名**，也就永不可能是「把不可比配置冻进后续阶段」的那一行；
  对它们要证据不增加任何保护，只把 §八.5「失败不等于低分」变成「失败即全盘停摆」。
  `completed` 行缺证据仍然硬拒（§八.7 的牙齿保留）。

**真实产物重放结果**：同一批 11 行（5 completed + 5 running 影子 + 1 blocked）→
`stage2_top_arms` 得出前两名 `['sum', 'direct']`，`validate_promotion` 核对通过，
可比组键里带真实 `split_hash=f1f36cbbede3ca7e`、`init_source=current_study_hc_checkpoint`、
`init_seed=2024`、`requires_family=hc_demo_d16_C32`、`recipe_hash=972ae016…9c0d`；
参与排名行数 5/11。把 completed 行的 `split_hash` 抹空后仍然拒绝，消息点名 `status=completed`。

**新增测试的牙齿（4 个变异，全部 CAUGHT；`tests/test_condition_injection.py::test_Q_unfinished_rows_never_poison_the_promotion_guard`）**

| 变异 | 预期 | 实测 |
| --- | --- | --- |
| M1 护栏重新检查全部行（回到修复前） | CAUGHT | 1 failed |
| M2 `ranking_eligible` 恒为真 | CAUGHT | 1 failed |
| M3 消息不再注明真实 status（回到「已完成行」措辞） | CAUGHT | 1 failed |
| M4 `COMPARABILITY_REQUIRED` 清空（真放宽护栏） | CAUGHT | 1 failed |

M3 在第一轮曾 **survived**——因为行明细元组里本来就含字符串 `completed`，
断言 `'completed' in msg` 被无关文本满足；已把断言收紧到 `'status=completed' in msg` 后 CAUGHT。
这条记录说明「消息措辞」类断言容易写出假阳性，变异测试是唯一能发现它的手段。

**回归**：`tests/test_condition_injection.py` 全量 71 项通过；
全仓 `pytest tests -q` → **372 passed, 11 skipped in 112.69s**（§0.21 基线 371 + 本条新增 1）。
变异脚本每次改前断言目标串唯一、跑完按 SHA256 还原源文件并校验一致。
所有临时产物留在 `/tmp/nt_promote_replay`、`/tmp/nt_smoke_exec`，未写入 `checkpoints/`、
`results/`、`data/`，未提交、未推送。

**仍然未验证（不要当成已完成）**

- 这条修复只证明**控制流程**能走到阶段 3+，不是训练证据；正式阶段 1–5 仍由用户手动执行，
  在此之前不得声称已确定最优条件注入方式。
- 未做实机 `--execute --stages 3` 端到端（需完整预算与 GPU·h），阶段 3/4/5 的晋级臂、
  冻结宽度、位置推导仍只有合成 + 真实产物重放级别的验证。
- `Registry.rows` 含 `running` 影子行这一事实未改（其他消费者如 `--status` 视图仍看到全部行）；
  本轮只统一了「谁参与排名」的口径，未重构登记表读取语义
  （→ §0.24：`--status` 视图与残留 `running` 判定都按新状态口径改了，见下）。

### 0.24 同日第二十次追加：blocked 冒充 pending + fresh 没有任何合法重跑路径（§八.5 / §八.6 / §一.9 / §八.8）

**触发点**：用户要求收尾并给出人工执行正式实验的命令。交付前按「代码是否真的能跑完整轮」
重查，而不是把 §0.23 当成终点。查出两处会让正式运行中途卡死的缺陷，都已修复。

#### 缺陷 A：被拒绝的试次登记成 `pending`，且不落盘

- `STATUSES` 里根本没有 `blocked`（旧值 `pending/running/completed/failed/pruned`），
  两处 blocked 写入点都把行写成 `status: 'pending'` 且**从不** `registry.append`。
- 真实 execute smoke 的日志原样为证：
  `状态计数：completed=6 failed=1 pruned=0 pending=1 running=0`，
  而那个 `pending=1` 正是被拒绝的 `s2_mdd_none#ab364e8efe39`
  （`[blocked] …: 执行模式不接受占位权重路径——MDD 只能从本轮真实训练出的 HC checkpoint 开始`）。
- 后果：① 报告把「依赖不成立」读成「还没排到」；② 进程退出后「拒绝过、为什么拒绝」
  在登记表里查无此条，候选空间被缩小却无迹可查（§八.6）。

**修复**

- `STATUSES` 加入 `blocked`；新增 `_blocked_row()` 统一两处写入：状态写 `blocked`、
  `failure_reason` 以 `blocked:` 开头、**执行模式追加登记**（dry-run 保持零副作用）。
- 最终报告状态计数加 `blocked=`，并新增「被拒绝清单」段（带 trial_id 与解锁办法）；
  `--status` 只读视图同样列出「被拒绝启动」清单，并给失败清单补上重跑指引。
- `orphans_running` 由「看最新状态」改为「最后一次 `running` 之后没有终态行」：
  `blocked` **不算**终态，否则「上次崩溃留下 running + 这次因缺本轮 HC 权重被拒绝」
  会被读成已处置，`--adopt_orphan`/`--rerun_interrupted` 的提示就此消失。

#### 缺陷 B：fresh 入口拒绝全部三个恢复开关 ⇒ 正式运行一次中断就彻底死路

探针 `/tmp/nt_blocked_mutate/probe_recovery.py`（真实代码路径，dry-run，零 subprocess）实测：

1. `build_exec_config` 在 fresh 下拒绝 `--retry_failed`、`--adopt_orphan`、`--resume`（§一.9，
   这三条禁令**本身是对的**，本轮一条没放宽）。
2. 编排进程被 Ctrl-C 打断后登记表留下 `status=running`；再次排程时
   `run_schedule` 拦停：「中断残留需要先处置：…**传 --adopt_orphan**…**或传 --resume**」——
   而这两个开关正是第 1 条里 fresh 拒绝的。**fresh 用户没有任何合法出路**，
   只能换新 `study_id` 把已烧的几十 GPU·h 全部重烧。
3. 某试次因 `pin_memory` 崩溃登记为 `failed` 后，提示「重跑需 `--retry_failed`」，fresh 同样
   拒绝 ⇒ 该臂**永不重跑**；若它是阶段 1 的 HC 家族起点，阶段 2+ 全部 `blocked`。
   这正是 §八.8 禁止的「失败静默缩小候选空间」。

**修复：补一条 fresh 安全的重跑开关，而不是放宽禁令**

- runner 新增 `--rerun_interrupted`（`ExecConfig.rerun_interrupted`），入口脚本新增
  `--rerun-interrupted`。它只做两件事：把残留 `running` **追加**登记为 `failed`
  （`failure_reason` 以 `rerun_interrupted` 开头并写明「不读 training_state、不加载历史权重」），
  以及让 `failed` 试次**整试次从头再训**。
- 为什么不违反 §一.1/§一.9/§一.14：它不恢复任何训练状态、不引入任何权重路径——起点仍由
  `fresh_init_flags` 现场重建（HC=`random_initialization`，MDD=`current_study_hc_checkpoint`
  且只能在**本轮 study 的 registry** 内解析；`Registry` 在装载阶段就拒绝异 `study_id` 行）。
- 默认关闭：重跑要再烧一份预算，不能由编排器自作主张。
- `cs.assert_command_fresh` 的 `FORBIDDEN_ORCHESTRATOR_FLAGS` 未改动（它校验的是 `main.py`
  训练命令，新开关不进入训练命令）；fresh 的拒绝消息现在指出合法替代路径。

**验证**

- 新增 4 条带断言的测试：`test_O_blocked_trial_is_labelled_blocked_and_stays_in_the_log`、
  `test_O_running_orphan_survives_a_later_blocked_row`、
  `test_Q_status_view_names_blocked_reasons`、
  `test_39_fresh_has_a_legal_rerun_path_for_interrupted_and_failed`
  （后者第 1 步先复现三条禁令仍生效，再验新通路）。
- 变异测试 **15/15 CAUGHT**，每次改前断言目标串唯一、跑完按 SHA256 还原并校验：

| 变异 | 结果 |
|---|---|
| M1 blocked 退回 `pending` | CAUGHT |
| M2 执行模式不再追加登记 | CAUGHT |
| M3 残留判定退回「只看最新状态」 | CAUGHT |
| M4 把 `blocked` 当成进程终态（平账） | CAUGHT |
| M5 `STATUSES` 里没有 `blocked` | CAUGHT |
| M6 dry-run 也写登记表 | CAUGHT |
| M7 报告计数去掉 `blocked=` | CAUGHT |
| M7b 报告去掉「被拒绝清单」 | CAUGHT |
| M8 `--status` 视图去掉拒绝清单 | CAUGHT |
| N1 中断闸门不认 `--rerun_interrupted` | CAUGHT |
| N2 复用判定不认 `--rerun_interrupted` | CAUGHT |
| N3 拒绝消息不再指出合法替代 | CAUGHT |
| N4 命令行值没接进 `ExecConfig` | CAUGHT |
| N5 **放宽 §一.9**（让 `--retry_failed` 在 fresh 可用） | CAUGHT（禁令原样生效） |
| N6 处置行写成 `blocked` 而非 `failed` | CAUGHT |

- 第一轮 M7 曾以 **SyntaxError** 形式「被抓住」，那不能算验证——它证明不了断言能识别语义退化。
  已把变异改成语法合法的写法重跑（去掉 `blocked=` 但保留合法 f-string），确认是断言本身在拦。
- 入口脚本 `bash -n` 通过；`--dry-run --rerun-interrupted --stages 2` 实测打印的 runner 命令
  带 `--rerun_interrupted`、`assert_no_forbidden` 不误拦、且 `.tmp_cond/study/` 未创建（零副作用）。
- 回归：`pytest tests -q` → **376 passed, 11 skipped in 117.35s**（§0.23 基线 372 + 本条新增 4）。
  所有临时产物留在 `/tmp/nt_blocked_mutate`，未写入 `checkpoints/`、`results/`、`data/`，未提交、未推送。

**仍然未验证（不要当成已完成）**

- 新恢复路径只在**登记与排程层**验证：没有真跑一个正式试次、中途 kill 掉、再用
  `--rerun_interrupted` 重训（需要真实 GPU·h），所以「重跑后的权重确实来自随机初始化/本轮 HC」
  仍只有 §0.21/§0.23 那层的间接证据 + 代码路径推理。
- `blocked` 行不带 `dedup_key`/`split_hash`（结构上不可能带），因此永不参与复用与排名——
  这是设计，但它意味着**跨轮统计**只能靠 `failure_reason` 文本读，未做结构化拒绝码列。
- OS-only / CPM-only 注入仍未实现（`models/tfm.py:642-647`）；真正的 AdaLN、条件门控低秩、
  `heads+backbone` 的 last2/all 仍在「未排期」清单。
- **正式阶段 1–5、≥5000 次被试级配对 bootstrap、3 seed 复核、冻结后一次性 test 全部仍未运行**，
  由用户手动执行；在此之前不得声称已确定最优条件注入方式或实验已完成。

### 0.25 同日第二十一次追加：交付「正式完整实验」的一条命令启动器；旧的两个训练脚本本轮不可用（§七）

**触发点**：用户要「跑真实完整训练微调实验的脚本」。先把仓库里已有的候选查清，再决定要不要新写。

**事实核查（grep，不是推断）**

| 文件 | 实测 | 本轮可用性 |
|---|---|---|
| `scripts/Pretrain_HC_tfm.sh` | `condition` / `study` / `device` / `init_source` 关键字 **0 命中** | 非条件基线 runbook，不构成本轮实验 |
| `scripts/Finetune_MDD_tfm.sh` | 同上 0 命中；`:22` 把 `--pretrained_weight` 指向 `checkpoints/neurotwin_tfm_pretrain/base_best.pt` | **本轮禁止使用**：那是本轮之前已存在的历史权重，违反 §一.3/§一.14 |
| `scripts/Condition_search_full.sh` | 不带 `--fresh`／`--study_id`，默认读写共享 `results/condition_search` | 本轮禁止 `--execute`（§0.9.1） |

⇒ 本轮唯一合法路径仍是 `scripts/Run_condition_study.sh`（恒开 `--fresh`）。缺的不是「另一条训练命令」，
而是**少打一次字 + 全程留痕**，所以新增一个薄启动器而不是第二条入口。

**新增 `scripts/Run_condition_full.sh`（启动器，不复制任何闸门）**

- 只做三件事：拼参数、把 `--execute` 的全程输出 `tee` 进 `<study_root>/logs_formal/formal_<ts>.log`、
  追加一行 `formal_manifest.tsv`（时间/study_id/study_root/device/两个预算闸门/解释器/`git rev-parse --short`/
  完整命令；**不含**数据内容或凭据）。
- 五个必填值缺任一即非零退出：`--study-id`、`--study-root`，以及 `--execute` 时的
  `--device`、`--max-trials`、`--gpu-budget-hours`。没有设备默认值，也没有预算默认值。
- 只接受登记过的开关并**原样透传**（`--stages`、`--ladder`、`--unlimited`、`--seeds`、
  `--review-scope`、`--split-manifest`、`--run-tests`、`--rerun-interrupted`）；未知参数直接 die，
  避免把拼错的参数当成默认行为。`--resume`/`--adopt_orphan`/`--retry_failed` 不在接受集合内，
  fresh 的三条禁令不能从启动器这一侧漏出去。
- 默认 `--stages 0..5`；`pipefail` 已开，入口非零退出原样传播（实测桩入口 `exit 3` ⇒ 启动器 `rc=3`，
  且不再打印收尾「下一步」横幅）。
- 收尾只**打印**需要人确认的三步（`--status` 只读视图、`--frozen_trial` 冻结、冻结后
  `--evaluate_test_final`），一律不代劳。

**验证**

- `bash -n` 通过；`--stages 0..5` 与 `--stages 3 --rerun-interrupted` 两种 dry-run 都 `rc=0`，
  且 `.tmp_cond/study/` 未被创建（零副作用）。
- 五类缺参 + 未知参数 + `--study-root results/condition_probe`（受保护目录，由入口拒）共 **7 条异常路径全部 rc=1**。
- 变异测试：`P3`（放进 `--retry_failed`）/`P4`（自己拼训练命令）/`P5`（恢复开关没真的透传）/
  `P6`（dry-run 也建目录）在真入口下 **4/4 CAUGHT**；`P1`（给 `--device` 塞默认值）/`P2`（给预算塞默认值）
  **不能拿真入口做变异**——闸门一旦被拆，那条命令就会真的开始烧 GPU·h；改用 `/tmp` 桩入口验证：
  原文件 `rc=1` 拒绝，变异后 `rc=0` 并把 `--device cuda:0` / `--max-trials 9999` 转发出去，
  证明这两条检查是承重的。
- **P5 第一版是空断言，值得记下来**：原来写的是「`--rerun-interrupted` 出现在输出里」，
  而入口脚本的收尾说明本来就印着这个开关名 ⇒ 变异后仍然「通过」。已改成只断言
  **被转发的那一行命令**（启动器的 `将要执行` 行 + runner 的 `[CMD]` 行），P5 随即 CAUGHT。
- 回归：`pytest tests -q` → **377 passed, 11 skipped in 114.09s**（§0.24 基线 376 + `test_40` 1 条）。
  新增测试 `tests/test_condition_fresh_gate.py::test_40_formal_launcher_is_a_thin_passthrough_and_keeps_every_gate`。
  临时产物都在 `/tmp`（已清理），未写入 `checkpoints/`、`results/`、`data/`，未提交、未推送。

**仍然未验证（不要当成已完成）**

- 启动器的 `--execute` 分支只用**桩入口**验证（日志、manifest、失败传播）；没有用它真跑过一次阶段 0
  之后的正式训练——那是本次交付给你手动执行的内容。
- manifest 里的 `git` 字段在工作区脏（本轮就是脏的）时仍是真实 HEAD 短 sha，但不含未提交差异；
  要逐字节复现正式运行仍需配合 `initial_condition_config.json` 与各试次的 `cond` 字段。
- §0.24 的「仍然未验证」全部继续有效（正式阶段 1–5、bootstrap、3 seed、留出 test 均未运行）。

### 0.26 同日第二十二次追加：选定单一注入方式（cat），移除对比与搜索全链（§一 之外的口径变更）

**决定**：用户要求「现在就选一种，其余清除干净，不再跑对比」。选定 **MDD 侧
`--condition_group mdd8 --condition_arm cat`**，HC 侧配套 `hc3/demo`。

> **口径必须先说清**：本轮**没有**任何一次臂间对比的真实运行数据，所以 `cat` 的地位是
> **选定方案（selected default）**，不是「实测最优」。文档、产物与任何结论表述都只能按
> 前者写；`is_best` / 「已确定最优条件注入方式」这类说法在本轮依然禁止。

**为什么 HC 侧不是 cat（结构约束，不是选择）**：`hc3` 分组只有共享三字段，没有新增组可融合，
`build_condition_encoder` 对 `hc3 + FUSIONS` 直接抛错（`models/conditioning.py:596`）。
所以「只用一种注入方式」在两个阶段落到不同臂上：HC=`demo`，MDD=`cat`；
`none` 保留为「结构性不建条件通路」的取值（非条件路径与旧权重 strict 重建需要它能被表达）。

**物理删除（未跟踪文件，不可恢复，已逐条确认 `git ls-files` 命中 0）**

| 类别 | 删除 |
|---|---|
| 编排与搜索 | `experiments/condition_runner.py`、`condition_space.py`、`condition_budget.py`、`condition_analysis.py`、`condition_benchmark.py`、`condition_cohort.py`、`migrate_condition_checkpoint.py` |
| 入口脚本 | `scripts/Run_condition_study.sh`、`Run_condition_full.sh`、`Condition_search_full.sh` |
| 测试 | `tests/test_condition_{fresh_gate,analysis,benchmark,cohort,migration,migrate_checkpoint,training_workflow}.py` |
| 测试内裁剪 | `tests/test_condition_injection.py` 删掉 33 个编排/晋级测试 + 6 个失去调用方的辅助函数（71 → 42 个） |

**保留（有据可查，不是手软）**：`experiments/condition_study.py` —— `main.py:441` 的
fresh 权重闸门依赖它，删了就没有「本轮 HC→MDD 起点」的证明能力；
`experiments/condition_diagnostics.py` —— 主指标（被试级 OS MAE）与单模型消融的定义所在；
`tests/test_condition_{injection,workflow,activation}.py` —— 覆盖的是保留下来的
schema/编码/迁移/strict 重建/梯度激活行为。

> **上面这段保留清单在同日第三次口径变更里被部分推翻（§0.27）**：用户要求把
> `condition_study.py`、`condition_diagnostics.py` 与 `.tmp_cond/` 一并删除，并选择
> 「整体移除」fresh 闸门。此处原样保留，是为了留下「当时为什么留着它们」的判断依据；
> 现状以 §0.27 为准。`tests/test_condition_activation.py` 与
> `tests/test_condition_injection.py`（38 用例）仍在。

**删掉的东西必须有替代，否则链路断裂**（三处都补了）

1. **随机来源登记处**：`SAMPLING_SEED/DROPOUT_SEED` 原来只活在 `condition_space.py` 里。
   迁到 `utils/condition_schema.py`（新增 `MODEL_SEED/SAMPLING_SEED/DROPOUT_SEED`），
   `models/conditioning.py` 的 `dropout_seed` 默认值改为引用常量（原来是重复字面量），
   `main.py` 的 `--cond_dropout_seed` 默认值同源于常量，拒绝消息改指新登记处。
   `PERM_SEED` 属置换诊断，随对比工具链删除。
2. **study 身份建立**：`study.json` 原先由编排器顺手创建，而 `load_study` 只读不建
   （缺文件即拒）。给 `experiments/condition_study.py` 加了命令行入口
   `python -m experiments.condition_study --init_study --root ... --study_id ...`
   （只写身份与四个子目录）；实测重复 init 无 `--allow_existing_study` 会被拒、
   跨 study 复用被拒、非空无 `study.json` 目录被拒。
3. **正式训练入口**：`scripts/Pretrain_HC_tfm.sh` / `Finetune_MDD_tfm.sh` 升级为默认 fresh
   （要 `STUDY_ID/STUDY_ROOT`，产物只落本轮目录，`LEGACY=1` 才回到旧口径）；
   微调脚本原来把 `--pretrained_weight` 默认到
   `checkpoints/neurotwin_tfm_pretrain/base_best.pt`（本轮之前已存在的历史权重，
   §一.3 禁止），现在改为在本轮目录里查 `checkpoints/hc_demo*/base_best.pt`，
   找到 0 个报错、找到多个**拒绝猜**并要求 `HC_WEIGHT=` 显式指定。

**模型侧的臂收敛**：`ARMS=('none','demo','cat')`、`FUSIONS=('cat',)`，
删除 `direct_canonical_vector`/`direct_canonical_width`/`_probe_direct_width` 与
`pool_phi` 的 sum/mean 分支（签名从 4 参数降到 2 个）；`delta_style` 的 `linear`
**保留但不再被任何臂使用**（旧快照的 strict 重建仍要能声明它）。
`condition_study.default_preregistration` 的 `required_controls` 从 5 项减到 2 项
（`static_new_value_shared_mask`/`mask_only_retrained`/`direct_canonical_linear`
依赖已删除的臂，留着会让任何门槛永远卡在一个无法补齐的清单上）——
这会改变 `preregistration_hash`，因为预注册口径确实变了。

**验证**

- `pytest tests -q` → **171 passed, 11 skipped in 47.61s**（删除前基线 377；净减 206 条全部属于
  已移除的对比/编排能力）。保留侧 `import main / evaluate_variant / condition_study /
  condition_diagnostics` 全部通过。
- 新写的 `test_I_cat_unknown_row_is_trainable_and_removed_arms_are_refused` 里
  **发现并修掉一条既有空断言**：旧 `test_I_..._sum_mean_mask_it_out` 的合成批次根本没有缺失的
  类别字段，「cat 的 Unknown 行可学习」那段 `if col.any()` 从未触发（我加的
  `assert saw_missing` 防空转守卫当场把它抓出来了）。现在显式逐行制造缺失，断言真实生效。
- 变异测试：`X1` 把 `direct` 放回 `ARMS`、`X2` 让 `pool_phi` 重新接受 `sum` —— 两条都
  **CAUGHT**，源文件按 SHA256 校验还原。
- 脚本层：两个脚本 `bash -n` 通过；在临时 study 里提取真实拼装命令行，HC 侧
  `--condition_group hc3 --condition_arm demo --init_source random_initialization
  --sampling_seed 20261009 --cond_dropout_seed 20261008` 齐备并**实测通过**
  `main.fresh_study_gate`（返回 kind=hc、schema=`condition_v1_20261008`）；
  MDD 侧命令行过 argparse 后，`fresh_study_gate` 对伪造起点抛
  `ProvenanceError`（证明闸门是活的、脚本没开后门），多候选起点报错也实测到。
- README 的快速开始已按 fresh 双脚本改写，并标注 MoE/RECC 四条命令的脚本早已不存在。

**这样做的代价（必须随附，不能省）**

- 「条件的增益是否超过人口学基线 / 超过原始线性拼接 / 超过无条件」这三问在本轮
  **结构上无法回答**了：对照臂与产出它们的编排器一起删除。留下的 `demo`/`none` 只是
  取值，不是同预算对照实验。
- §八.9 的 ≥5000 次被试级配对 bootstrap 需要两份可配对的权重；现在只有一条训练路径，
  该统计腿除非将来再引入第二个模型，否则不会自动产生。
- `condition_study.check_freeze_eligibility` 与冻结门槛仍在模块里，但唯一的调用方
  （编排器）已删除 ⇒ 目前是**无人调用的库函数**。保留它是因为它是 kept 模块的公开 API，
  删除属于另一项决定；这条如实记为「已就位但未接线」。
- 全部删除都是不可逆的（那些文件从未进过版本库）。若之后要恢复对比能力，需要重写
  编排器、预算画像与晋级护栏，而不是改几个开关。

### 0.27 同日第二十三次追加：删除 study/diagnostics 模块与 fresh 身份闸门，清空 scratch（§一.3/§一.12 的代码保证消失）

**决定**（用户 2026-10-10 指示）：删除 `experiments/condition_diagnostics.py`、
`experiments/condition_study.py` 与整个 `.tmp_cond/`。追问「删掉 condition_study 之后
main.py 的 fresh 来源闸门怎么处理」，用户选 **整体移除（彻底删）**；`.tmp_cond/` 选 **全删**。

**这条决定的真实代价（不是措辞问题，是保证级别的下降）**

- §一.3「MDD 只能引用本轮新训出的 HC 权重」与 §一.12「按 SHA256 追溯起点权重」**不再有代码保证**：
  `main.py` 不再读取/校验权重内嵌来源、不再拒绝历史权重、checkpoint 不再写 `meta['study']`。
  「HC 从随机初始化训练」现在只由「HC 脚本不传 `--pretrained_weight`」这一事实维持。
- §一.9 的「产物只落本轮目录」强制同样消失：目录改由 `CHECKPOINT_DIR` 决定，默认回落仓库共享的
  `checkpoints/`（`main.py:801-808` 的目录自增保证不会覆盖已有权重）。
- 主指标定义与诊断场景（被试级 OS MAE、置换 / mask-only / 全缺失 / 绕过、配对 bootstrap）随
  `condition_diagnostics.py` 一起删除；此后仓库内**没有**这些计算的可执行实现，只有文档描述。

**删除清单（三个目标都未跟踪 ⇒ 不可恢复，已逐个 `git ls-files` 确认为 0）**

| 对象 | 规模 | sha256 前 16 位（删除前） |
|---|---|---|
| `experiments/condition_study.py` | 974 行 | `65bc888945968781` |
| `experiments/condition_diagnostics.py` | 577 行 | `f0dc69ad4acab046` |
| `tests/test_condition_workflow.py` | 405 行 | `fa91ea49e9b9d37a` |
| `.tmp_cond/` | 2.2 GB（111 个 `.pt` + 日志/探针产物/scratch 脚本） | 整目录删除 |

`tests/test_condition_injection.py` 另裁掉 4 个只测 diagnostics 的 `test_O_*` 与失去调用方的
`_diag_batches`（42 → 38），并删掉两段已无内容的分节横幅；`test_K2_*` 的 docstring 里指向
「阶段 4 / `stage4_injection`」的措辞改为描述模型级不变量（**断言一条未删未放宽**）。

**`main.py` 的实际改动（1679 → 1538 行）**

- 删 `load_study_registry_rows()`、`fresh_study_gate()` 及 `main()` 里的调用点。
- 删 CLI：`--init_source / --study_id / --study_root / --trial_id / --source_trial_id / --is_probe`
  （`--is_probe` 的唯一消费者就是那道闸门，留着即死开关）。
- 删 checkpoint 内嵌 `meta['study']` 块与 `args._study_provenance`；激活探针 payload 去掉
  `study_id / trial_id / init_source` 三个键。
- **保留** `--condition_inherit_stats`：它属于 §四 统计量按位继承，与 study 身份无关
  （实现是 `main.py:177 load_condition_stats_payload`，不引用被删模块）。
- **把随机来源分离校验从闸门里救出来**，改成无条件的 `main.check_seed_separation()`：
  这条 §四 校验原先寄生在 fresh 闸门内部，闸门一删就会静默消失。实测三种情形 ——
  `--seed` 与 `--sampling_seed` 同值 ⇒ `ValueError`；Namespace 缺 `cond_dropout_seed` ⇒ `ValueError`；
  三个登记值（2024 / 20261009 / 20261008）⇒ 通过。如实记其副作用：`python main.py --mode pretrain`
  若沿用两个历史默认值（都是 2024）现在会被拒 —— 这是把既有撞车摊开，而不是静默把两条随机流并成一条。

**两个脚本的新形态（fresh/legacy 双分支没了）**

- `STUDY_ID / STUDY_ROOT / LEGACY` 全部去掉，条件注入与种子读取变成无条件附加。
- 新增 `CHECKPOINT_DIR`（默认 `$ROOT/checkpoints`）承接「想隔离就指到本轮目录」这件事 ——
  这是原先由 `STUDY_ROOT` 顺带提供的能力，不保留它就没有不写共享目录的跑法。
- MDD 仍在 `"$CHECKPOINT_DIR"/hc_demo*/base_best.pt` 里找起点：0 个报错、多个**拒绝替你猜**并要求
  `HC_WEIGHT=` 显式指定；但不再校验该路径是否属于「本轮 study」，也不核对权重来源。

**验证**

- `pytest tests -q` → **146 passed, 11 skipped in 46.25s**（删除前 171；净减 25 = 21 条 workflow
  用例 + 4 条 diagnostics 用例，一条未放宽）。
- `import main` 通过；`hasattr(main, 'fresh_study_gate')` 为 False；函数清单逐名比对：
  只少 `fresh_study_gate` / `load_study_registry_rows`，只多 `check_seed_separation`。
- 旧开关变成硬错误：`main.py --init_source random_initialization --study_root …` →
  argparse `exit=2`「unrecognized arguments」（不是静默忽略）。
- 两个脚本的**真实拼装命令行**用临时探针脚本抽出后过 `main.parse_args()` +
  `check_seed_separation()`：HC 112 个 token、**无** `--pretrained_weight`、带
  `--condition_group hc3 --condition_arm demo --sampling_seed 20261009 --cond_dropout_seed 20261008`；
  MDD 124 个 token、带 `--condition_group mdd8 --condition_arm cat` 与
  `--condition_inherit_stats <同一个 HC 权重>`。两条都 `pretrained_weight` 语义正确、无残留 study 键。
- 三条拒绝路径实测 exit=1：无起点 / 命中 2 个候选起点 / `HC_WEIGHT` 指向不存在的文件。
  探针脚本与临时目录已删除，`ls scripts/` 只剩两个正式脚本；两脚本 `bash -n` 通过。

**顺带查出的「已就位但未接线」（本轮未自行处理，需另行决定）**

- `train/training_state.py` **没有任何调用方**（原消费者是已删除的编排器与 fresh 闸门），
  而 `main.py --save_training_state` 也只剩 argparse 定义 —— 训练循环里没有任何代码读它，
  即其 help 承诺的「额外落盘 optimizer/scheduler/scaler/EMA/随机状态」**目前不兑现**。
  删除它属于另一次授权，这里只如实登记为 help 与实现不符的死开关。
- `experiments/` 现在只剩 `evaluate_variant.py`；条件链路的现存实现是
  `utils/condition_schema.py` + `models/conditioning.py` + `models/tfm.py` 的注入点 + 两个脚本。

## 1. 一句话状态

> **当前（2026-10-10 第三次口径变更，见 §0.27）**：条件注入方式仍是**选定**的
> MDD 侧 `cat` + HC 侧 `demo`（**选定方案，不是实测最优** —— 用户不跑对比，所以不存在臂间证据）。
> 同日两次删除的净结果：搜索/编排/预算/晋级/冻结工具链（§0.26）＋
> `experiments/{condition_study,condition_diagnostics}.py` 与 fresh 身份闸门（§0.27）全部移除，
> `.tmp_cond/` 已清空删除。正式入口仍是两个脚本
> `scripts/Pretrain_HC_tfm.sh` → `scripts/Finetune_MDD_tfm.sh`，但**不再需要**
> `STUDY_ID`/`STUDY_ROOT`/`LEGACY`，改为可选 `CHECKPOINT_DIR`（默认 `$ROOT/checkpoints`）。
> `results/condition_search/` 与 `configs/final_model.json` 保持原样未动；本轮**不会**产生
> `best_condition_config.json`，也不得声称实验已完成或已确定最优注入方式。
> **保证级别的变化必须一起读**：HC 随机初始化、MDD 起点来自本轮 HC、test 只评一次这三件事
> 现在都由操作顺序维持，不再由代码校验（§0.27 代价清单）。
> 下面几段是删除前的历史记录，保留以便追溯，其中提到的
> `Run_condition_study.sh` / `Run_condition_full.sh` / `Condition_search_full.sh` /
> `condition_runner.py` / `condition_study.py` / `condition_diagnostics.py` 等文件**现已不存在**。

工程实现、合成验收、短 smoke 与搜索工具链**已完成并可用**，阶段 1→5 已收敛成
**一条命令**（`scripts/Condition_search_full.sh`，默认 dry-run）；**条件注入搜索本身尚未运行**
（0 个完成的学习试次），因此没有任何"最优条件方法"的结论可报告。
`results/condition_search/` 里现在只有 `initial_condition_config.json`（预注册起点，
`status=search_not_run`、`is_best=false`）与预算画像/计划；**没有** `best_condition_config.json`
——该文件只允许由 `--frozen_trial` 在真实 completed 且跑满预算的配置上生成。

> **本轮（§一 硬约束）请改读下面这条，上面那句只描述旧口径**：§一 要求全部从头重训、
> 禁止旧 registry / 旧拟合统计量 / 旧预算画像参与本轮，因此本轮唯一合法的一键入口是
> **`scripts/Run_condition_study.sh`（恒开 `--fresh`，所有产物落在新 `--study_root` 内）**；
> 阶段 5 排程之后会自动接一条只读统计腿（被试级配对 bootstrap + Pareto）。
> 上面提到的 `scripts/Condition_search_full.sh` **不满足本轮约束**：实测
> `scripts/Condition_search_full.sh:38` 把 `REG_DIR` 默认到共享目录
> `results/condition_search`（那里已有旧 `budget_profile.json` 与 `plan_stage12.json`），
> 且该脚本从不传 `--fresh`／`--study_id`（`:130`、`:158`、`:188`）⇒ dry-run 无害，但
> **`--execute` 会读写旧登记目录与旧画像，违反 §一.7 / §一.8**。旧脚本按 §六.8 的兼容性
> 保留不删，但本轮正式执行不得用它；已登记到 §0.9.1。

| 项 | 状态 |
|---|---|
| 数据/协议/编码/注入/优化器/checkpoint 链路 | 完成，验收 A–Q 全绿 |
| 搜索空间、预算实测、编排器（dry-run 默认）、诊断工具 | 完成 |
| 阶段 1→5 一键排程（`--auto_promote` 反推晋级臂/宽度/位置） | 完成（dry-run 全链路验证通过） |
| **本轮 fresh 一键排程 + 冻结链路（`Run_condition_study.sh`，阶段 5 后接统计腿）** | **完成（dry-run `--stages 0..5` 实测 exit=0、不落目录；闸门见 §0.9）** |
| **被拒绝（blocked）试次的状态口径与持久痕迹（§八.5/§八.6）** | **完成（`blocked` 进状态机、执行模式追加登记、报告与 `--status` 分列；`orphans_running` 改为「最后一次 running 之后无终态行」。9/9 变异 CAUGHT，见 §0.24）** |
| **fresh 的合法中断/失败重跑路径（§一.9 × §八.8）** | **完成（新增 `--rerun_interrupted` / 脚本 `--rerun-interrupted`：追加登记 failed + 整试次从头再训，三条旧禁令原样保留并由变异 N5 证明；6/6 变异 CAUGHT。仅登记/排程层验证，未实机中断重跑，见 §0.24）** |
| **正式完整实验的一条命令启动器 `scripts/Run_condition_full.sh`（§七）** | **完成（薄启动器：五个必填值强制、只透传给唯一入口、执行日志+manifest、失败码原样传播；4/4 真入口变异 CAUGHT + P1/P2 用桩入口证明承重。⚠ 旧的 `Pretrain_HC_tfm.sh`/`Finetune_MDD_tfm.sh` 无任何条件/fresh 参数，且后者的 `--pretrained_weight` 指向历史权重，本轮禁止使用，见 §0.25）** |
| **训练快照 ↔ 独立 strict 重建往返（§六.7，含显式 `condition_arm=none`）** | **完成（探针先复现 10 个 unexpected keys 的失败，修后两侧键集一致、strict 干净；见 §0.10 第 1 条）** |
| **同预算可比性校验（§八.7：split / 初始化家族 / 来源权重 / 非条件配方）** | **完成（分组键加可比性签名，缺证据的 fresh 行 fail-closed 不驱动晋级；见 §0.10 第 2 条）** |
| **共享 HC 起点的确定性解析（§一.4/§一.12）** | **完成（多条完成行按最大训练预算选取并告警，平手或无预算证据一律 block；见 §0.10 第 3 条）** |
| **死条件参数扫描覆盖三个注入档（§三「backbone-only 是否建立了永不使用的条件参数」）** | **完成（探针实测 15 个组合 `grad is None`=0；`test_3` 口径由 5 个组合扩到 15 个并加防自欺断言。属合成前反向验收，不是训练证据；OS-only/CPM-only 未实现故不在覆盖内，见 §0.10 第 4 条）** |
| **`shared` 生成器不再建死行（§三「永不使用的条件参数」，逐元素判据）** | **完成（身份表按被选中 block 建行；6 个模型级组合恒等起点 + strict 干净；实测条件参数量 `per_layer` 3168/6336/12672、`shared` 3944/3952/3968，见 §0.10 第 5 条）** |
| **惰性宽度轴不再被当搜索维度（阶段 3「不生效的 `d_new`/`H_delta` 不得反复训练」）** | **完成（实测结构签名：`demo` 恒 3424、`direct` 恒 5152 条件参数，参数名与形状逐条相同；`width_axes_live` 三处取值 + `--plan` 闸门 + 起点产物按两类臂报数；声明随机空间 2916→2268，见 §0.10 第 6 条）** |
| **冻结门槛缺对照证据时 fail-closed（§八.8「缺对照不能提前冻结」）** | **完成（A/B 实测：旧写法在 `[]` 与漏传参数两种情况下都返回 `can_freeze=True`、`reasons=[]`，即整条对照检查被短路跳过；现两种情况都不能冻结，对照齐全/部分缺失的原行为与消息一字未改。生产 fresh 路径此前已由 `_freeze_gate` 兜住，修的是库层契约，见 §0.11）** |
| **final test 幂等且绑定冻结的那份 checkpoint（§八.9）** | **完成（实测：同一条命令跑两次会重复评估 test 并追加同 id 行，已登记 test 结果绑另一权重时被静默覆盖；现同一权重 ⇒ 跳过并标 `accounting='reused'`，不同权重 ⇒ `ValueError` 拒绝，`failed` 后仍可重跑。判据是 checkpoint 路径而非 SHA256，「一次调用评完整个冻结集合」当时仍缺，见 §0.12；该缺口已在 §0.13 闭合）** |
| **final test 在冻结判定之后执行，且一次评完冻结集合（§八「冻结后一次性评估 test 上的最终配置及预先固定的直接对照」/ §八.9）** | **完成（实测：旧路径把 test 评在冻结门槛判定「之前」，留出集在配置尚未决定时就被消费；且一次调用只登记 1 条 `@test_final`，五类预注册直接对照 test 指标 0 条。现门槛不通过直接 `TrialError`、0 次评估、0 条登记；门槛通过后一次调用评完 `{冻结配置} ∪ {required_controls}`，对照集合按预注册名 + `_CONTROL_PREDICATES` 定死（不由 test 数字挑选），各自幂等、各自绑定自己那份权重并登记 `final_test_role`/`final_test_of`；全套 357 passed, 11 skipped，见 §0.13）** |
| **阶段 5 复核矩阵带齐逐 seed 直接对照与诊断腿（§一.6 / §七阶段 5 / §八.8）** | **完成（实测：runner 从不排 `review_diagnostics`，阶段 5 只有 4 条试次、`kind=diagnostic` 0 条 ⇒ `all_missing_degradation` 永远缺测 ⇒ 冻结门槛按「缺测≠通过」永远拒绝，`best_condition_config.json` 与 final test 在构造上就到不了；且不逐 seed 重建五类对照，`full` 口径下 `demographic_same_init_family`（要求同一 HC 起点哈希）整类找不到 completed 行；入口还根本不暴露 `--review-scope`，只能吃 runner 的弱默认 `mdd_only`。现诊断 0→6 条且逐条绑定被诊断权重、对照按 seed 展开、fresh 入口默认 `--review-scope full`（选 `mdd_only` 打 warn 明说不是完整两阶段复核）；累积计划 blocked=0、`validate_trials` 告警 0；全套 359 passed, 11 skipped，见 §0.14）** |
| **注入位置比较共享同一个本轮 H0 起点（§一.4 / §六.4 / §七阶段 4）** | **完成（实测：`stage4_injection` 原以 H1 家族 `hc_demo_d16_C32` 为起点，并靠 `config_hash` 去重复用阶段 3 的 H1 heads 试次 —— 正是 §七阶段 4 禁止的「阶段3的 H1 起点 heads 权重不能作为阶段4 H0 起点 heads 对照复用」，也与本文件 §0.10 第 5 条的自述相矛盾。现 `requires_family=['hc_none']*3`、`path_slots` 只剩 `pretrained_weight`（H0 无条件统计量可继承）、`compared_against` 改指同家族的 `mdd_<arm>_heads_last1`，阶段 3/4 两条 heads 身份不再互相去重。连带修掉 `stage5_multiseed` 恒取 `arm='demo'` 补逐 seed 基线的问题：否则 H1 权重会冒名 `hc_none#seedN`，与真 H0 撞成同名家族的两条身份、家族解析只能 block。模型级验收 `test_K2`（真实 `NeuroTwinTFM`，三个注入档各一遍）实测：条件参数全部归 `new_initialized`、非条件权重逐位继承、OS/CPM 输出与 H0 起点误差小于 1e-6（全缺失与有效观测两种条件批次都成立）、backward 后条件参数有非零梯度；CLI 装配层由 `test_28` 用运行器真实命令行过 `main.parse_args` 实测：H0 与 H1 保存的任务口径签名相同且与阶段 4 三条命令逐键一致（`check_pretrain_config_compat` 放行），来源声明为 `current_study_hc_checkpoint`、不评 test，并另构造一次 `seq_len+1` 失配确认比对不空转；全套 361 passed, 11 skipped。见 §0.14 第 4/4b 条）** |
| 短 smoke（真实 merged 数据 + 真实结构，1–2 epoch） | 完成（HC + MDD 双臂）；跑 smoke 的编排器已删除，smoke 权重留在 `.tmp_cond/` |
| 搜索编排器的**执行链路**端到端 smoke | ~~完成~~ **已随编排器删除（2026-10-10，§0.26）**：`test_O_runner_execute_end_to_end_reuses_completed_trials` 所在文件被裁掉 |
| 阶段 1–5 正式搜索 | **已取消**（用户 2026-10-10 决定不跑对比，改为选定单一方案 `cat`，§0.26）；不是"待运行" |
| test 留出评估 | **口径变了（不再是人工一次性闸门）**：`main.py:1149-1159` 让 finetune 结束就自动评 test 并落盘 `<save_dir>/eval/metrics_test.json`；本轮无搜索 ⇒ 无"挑赢家再看 test"的选择性偏差，但重复微调＝重复看 test。要保留人工闸门需显式 `--no_eval_after_train`，见 §7.1 第 3 条 |
| **正式训练入口（当前唯一）：`Pretrain_HC_tfm.sh` → `Finetune_MDD_tfm.sh`** | **就绪、未实机训练**：2026-10-10 第三次变更后不再需要 `STUDY_ID`/`STUDY_ROOT`/`LEGACY`，起点串接靠 `CHECKPOINT_DIR`（默认 `$ROOT/checkpoints`）下的 `neurotwin_tfm_pretrain*/base_best.pt`（目录名 2026-10-11 由 `hc_demo` 改为此名），0 个/多个候选都拒绝；真实拼装命令行已过 `main.parse_args()` + 随机来源分离校验，但 **完整 150 epoch 训练从未在本轮跑过**，见 §7.1 |
| **来源证明（HC 随机初始化 / MDD 继承本轮 HC / SHA256 追溯）** | **已无代码保证**（§0.27）：`fresh_study_gate` 与 `condition_study.py` 一起删除，`main.py` 不校验权重来源、不拒历史权重。仅随机来源分离（§四）改由 `main.check_seed_separation()` 无条件保住 |
| **臂间/对照增益结论** | **无法给出**：对照臂与产出它们的编排器一起删除，留下的 `demo`/`none` 只是取值，不是同预算对照实验（§0.26 代价清单） |

## 2. 实施后的完整数据流

### 2.1 数据来源与清洗（唯一入口）

```
data/Rest-meta-MDD-V1V2-Merged-HC.xlsx   → hc3  : [Sex, Age, education]                K=3
data/Rest-meta-MDD-V1V2-Merged-MDD.xlsx  → mdd8 : [Sex, Age, education,
                                                    episode, illness, medication,
                                                    HAMD, HAMA]                         K=8
```

- 只读这两张 merged 表；其它 V1/V2 表即使存在也不会被读取来补齐、覆盖编码或构造 mask
  （验收 C 用"改名/移走"实验证明）。
- `HAMA` 只取列名恰为 `HAMA` 的列；条目列 `HAMA_i`/`*HAMA_i` 与 `HAMATotal` 一律忽略，
  不做回退（验收 C）。
- `medication` 允许在同一 merged 行内从 `medication` / `On medication?` / `OnMedication`
  取可解析值；映射后仍冲突 → 报错停止该配置；不新增"来自哪一列"的特征（验收 D）。
- `site` / `Dataset` / `ID` / 诊断标签永不进入条件向量；`Dataset`、`site` 只用于划分审计
  与结果分层，`ID` 只用于匹配。

有效性判定（验收 A/B）：**任何可解析的有限数值都是有效观测**，包括 `0`、`-1`、`-9999`；
绝不按临床范围、符号或词表把数字判成缺失。只有 `episode` 与 `medication` 做码值替换
$T(-1)=2$、$T(-9999)=0$（只换码，mask 恒为 1，且幂等）。真正缺失 = 空单元格 / `None` /
`NaN` / `pandas.NA` / `NaT` / 空或纯空白字符串 / 大小写与空白归一后的
`null|none|nan|na|n/a|<na>|nat`；不可解析与 `Inf` 判缺失并记录原因。判定不使用
`if not value` 这类真值测试，数字字符串 `"-1"`、`"0"` 仍是数字。

### 2.2 划分冻结 → train-only 拟合（顺序不可交换）

```
有效影像队列 → 冻结 subject 级划分（manifest / --split_file）→ 拟合连续与类别统计量
```

- 因缺少临床数据而被删人的做法已移除：HC 的 27 名 merged 无行被试仍保留影像样本，
  条件按"全字段缺失"编码（验收 F）。
- 分层键显式规定：MDD 用 `HAMD`，**不再跟随 `fields[0]`**；分层键缺失进预注册的
  `MISSING_BIN=-1` 桶做确定性分配。
- 模型 seed、Condition Dropout seed、划分 seed 三者分离（验收 J）。
- 连续字段（Age / education / illness / HAMD / HAMA）在 train 有效观测上取
  $\mu_j=\operatorname{median}$、$s_j=\operatorname{IQR}/1.349$，IQR 退化时回退 std，
  再回退 1；先填补后归一化，避免 $0\cdot\mathrm{NaN}$；缺失位输出 $z=0, m=0$。
  整列全缺失 → `fitted=false, n_valid=0`，不报错也不伪造统计量（验收 E）。
- HC→MDD 继承：Age / education 的 $\mu,s$ 与类别词表逐位继承 HC-train，只在 MDD-train
  上拟合 illness / HAMD / HAMA；用"整表重拟合"会改写继承统计量，已被数值证明并被拒绝
  （验收 E）。

本轮真实数据的实测值（train 被试上拟合，写进 checkpoint 元数据）：

| 字段 | $\mu$ | $s$（IQR/1.349） | $n_{\text{valid}}$（train） |
|---|---|---|---|
| Age | 29.000 | 16.308 | 1759 |
| education | 14.000 | 3.706 | 1754 |
| illness | 12.000 | 37.806 | 1473 |
| HAMD | 22.000 | 5.930 | 1803 |
| HAMA | 19.000 | 8.895 | 453 |

类别词表：`Sex=[1,2]`、`episode=[0,1,2]`、`medication=[0,1,2]`，各带**独立** Unknown
（内部索引 0，与有效 `code_0` 不同，验收 B）。哨兵实测：`episode` 原始 `-1` 226 个、
`medication` 原始 `-1` 379 个，替换后归零且 mask 仍为 1。

### 2.3 编码、融合与条件向量

> **2026-10-10 口径变更（§0.26）**：本节按 2026-10-09 的实现写，其中 `sum` / `mean` 融合算子与
> `direct` 规范向量**已从代码删除**，当前 `FUSIONS=('cat',)`、`ARMS=('none','demo','cat')`。
> 下面的 $I_{\text{sum}}$ / $I_{\text{mean}}$ / `direct` 数值只作为删除前的实测记录保留；
> 现役只有 `cat`（$I_{\text{cat}} = C + 5d_{\text{new}} + 5 = 32+80+5 = 117$，
> $\Delta E$ 参数量 9632，$H_{\text{delta}}=64$、`delta_style=mlp`）。

独立编码（每字段一套参数，验收 I）：

$$
e_j = F_j([z_j, m_j]) = W_2\,\operatorname{GELU}(W_1 [ z_j, m_j ] + b_1) + b_2 \in \mathbb{R}^{d}
$$

类别字段 $e_j = E_j(q_j)$，`Unknown` 是每字段自己的第 0 行。融合：

$$
c_{\text{demo}} = E_{\text{demo}}([e_{\text{Sex}}, e_{\text{Age}}, e_{\text{education}}, m_{\text{demo}}])
$$

$$
c_{\text{MDD}} = c_{\text{demo}} + \Delta E_f([c_{\text{demo}}, \Phi_f(e_N, m_N), m_N])
$$

$\Phi_{\text{cat}}$ 拼接、$\Phi_{\text{sum}}$ 掩码求和、$\Phi_{\text{mean}}$ 掩码均值
（分母 clamp 到 1，全员缺失时 $\Phi=0$ 但保留完整 mask）。$\Delta E$ 末层零初始化 ⇒
起点 $c_{\text{MDD}} = c_{\text{demo}}$（验收 K）。此外 `direct` 基线把新增五字段的
规范 $[z,m]$ / one-hot+Unknown / mask 直接过一个**线性** $\Delta E$，并登记容量差异。
`sum`/`mean` 的严格掩码会把 Unknown 行乘 0 消掉，`cat` 的 Unknown 则真的可学 —— 这条差异
由验收 I 的双向测试固定。

输入宽度解析式（由构造断言校验，不手算）：

$$
I_{\text{cat}} = C + 5 d_{\text{new}} + 5,\qquad
I_{\text{sum}} = I_{\text{mean}} = C + d_{\text{new}} + 5
$$

$$
N_{\Delta} = I H + H + H C + C \quad (\text{mlp}),\qquad N_{\Delta} = I C + C \quad (\text{linear})
$$

实测（$d_{\text{new}}=16, C=32, H=64$）：$I_{\text{cat}}=117$、$I_{\text{sum}}=I_{\text{mean}}=53$，
$\Delta E$ 参数量 cat 9632 / sum=mean 5536 / direct 1728 / demo 0。

### 2.4 注入

第一阶段只保留两个已有头部位置，两头共享 $c$ 但生成器参数独立（验收 H）：

$$
h' = (1 + \gamma_\ell(c)) \odot h + \beta_\ell(c)
$$

- One-Step：$z_g = A_{\text{eff}} z_t$，$h = T([z_t, z_g])$，调制 $h$ 后
  $z_{\text{next}} = z_t + h'$；
- CPM：查询 $q_{i,h} = e^{\text{ROI}}_i + e^{\text{horizon}}_h$ 先调制，再对历史 token 做
  交叉注意力（**不是** One-Step 式残差，写法与实现都对齐方案 §7）。
- HC 预训练现在也能建立 / 拟合 / 前反向条件通路（`--condition_arm` 不再被预训练一刀切关闭）。
- `condition=None`（绕过）≠"条件全缺失"（一个真实缺失表示），两者输出差 0.0493，已被固定。
- 主干候选 `AdaRMSNorm`（条件 RMSNorm，**不叫 AdaLN**）以 `(1+γ)⊙RMSNorm(h)+β` 实现，
  零初始化，注入范围 `last1/last2/all`，生成器 `per_layer/shared`；真 AdaLN 需要配套
  非条件 LayerNorm 对照，属未排期项。门控残差与门控低秩已实现并恒等起步，但未排期搜索。
- `arm=none` 参照现在同时下发 `--tfm_patho_mode none`：否则 `main.py` 默认的 legacy 单标量
  HAMD FiLM 仍会构建，那个"无条件臂"其实是被旧条件调制过的模型（本轮发现的真实缺陷，已修
  `models/tfm.py` 使两头的条件宽度同源）。

### 2.5 Condition Dropout

$$
m^{\text{eff}}_{ij} = m^{\text{obs}}_{ij} \cdot r_{ij},\qquad
r_{ij} \sim \operatorname{Bernoulli}(1 - p_j)
$$

被屏蔽的连续字段用 train 统计量填值并置 $z=0, m=0$；类别字段回到自己的 Unknown。
原本缺失的格子永远不会变成有效；统计量在 dropout 之前于 raw train 观测上拟合；推理关闭；
同一被试跨窗口一致、按 $(\text{seed}, \text{epoch})$ 可复现（确定性 64 位整型混合，键位
$(\text{epoch}\!\ll\!32)\,|\,(\text{subject}\!\ll\!8)\,|\,\text{field}$，不依赖 Python hash）。
融合初筛阶段 $p=0$，晋级臂才比较 $p\in\{0,0.1,0.2\}$（验收 J）。

> **2026-10-10**：「阶段/晋级」的说法随编排器一起失效（§0.26）。当前实现保留 $p$ 本身
> （`--cond_dropout_p`，`main.py:1288` 默认 0.0），两个训练脚本都不传它 ⇒ 本轮实际
> 配置就是 $p=0$；$p>0$ 与 $\{0.1,0.2\}$ 的比较**不会**在本轮发生。

### 2.6 迁移与 checkpoint

`ARCH_VERSION` 4 → 5。显式迁移器（非宽松加载）报告 loaded / copied / new_initialized /
zero_expanded / skipped / missing，并对同宽度但字段顺序变化的权重直接拒绝（验收 K）。
checkpoint 保存统计量、mask 语义、类别字典、`layout_hash`、`schema_version`、
merged 文件哈希、`split_hash`；`experiments/evaluate_variant.py` 重建时 `strict=True` 并
复核布局与来源哈希（验收 M）。

## 3. 修改与新增文件

> **下面的「新增」是 2026-10-09 的状态。**2026-10-10 两次删除之后的**现存**条件相关文件是：
> `utils/condition_schema.py`（835 行）、`models/conditioning.py`（806 行）、
> `models/tfm.py` / `models/tfm_backbone.py` 的注入点、`tests/test_condition_injection.py`
> （38 用例）、`tests/test_condition_activation.py`，加上 `scripts/Pretrain_HC_tfm.sh` 与
> `scripts/Finetune_MDD_tfm.sh`；`experiments/` 只剩 `evaluate_variant.py`。
> 已删除：§0.26 的 7 个 `experiments/` 模块 + 3 个 `scripts/*.sh` + 7 个测试文件，
> 以及 §0.27 的 `experiments/condition_study.py`、`experiments/condition_diagnostics.py`、
> `tests/test_condition_workflow.py` 与整个 `.tmp_cond/`。
> `results/condition_search/` 里的画像/计划属于旧口径产物，现存代码不读不写。

新增：

- `utils/condition_schema.py` — 字段白名单、merged 内列映射、清洗与码值替换、类别协议、
  mask、`layout_hash`、`schema_version`、文件哈希。
- `models/conditioning.py` — 独立特征编码器、融合 $\Phi_f$、`E_demo` / $\Delta E$、
  `direct` 规范向量、Condition Dropout、AdaRMSNorm / 门控残差 / 门控低秩候选、参数量公式断言。
  （2026-10-10：`direct` 规范向量与 `sum`/`mean` 融合分支已删除，臂空间收敛为
  `ARMS=('none','demo','cat')`、`FUSIONS=('cat',)`，§0.26。）
- `experiments/condition_space.py` — 预注册常量、阶段 1–5 试次生成、依赖阻塞、按配置哈希去重、
  随机搜索记录、参数量实测。
- `experiments/condition_budget.py` — HC/MDD 分别实测的**分段代价探针**（epoch 网格 ×
  两档样本预算 × 冻结/解冻两套日程，反演 $O,V,T,T_f,R$）、显存采样、评估与诊断成本实测、
  结构指纹、GPU·h 核算与预算闸门。
- `experiments/condition_runner.py` — dry-run 默认编排器、追加式 registry（JSONL+CSV）、
  复用/断点/失败与剪枝状态机、同预算逐淘汰、留出评估、机器可读排程摘要、
  **晋级臂核对**（阶段 3+ 的执行必须由阶段 2 的同预算验证排名决定，否则拒绝执行）、
  **`--auto_promote`**（阶段 3+ 的晋级臂 / 阶段 3 胜出的冻结宽度 / 阶段 4 第一档胜出的
  注入位置全部按 `config_hash` 从登记表反推，整条链路不需要手填）、
  **`--status`**（只读：登记表状态 + 同预算排名 + 下一步可直接粘贴的命令）。
- `scripts/Condition_search_full.sh` — 阶段 1→5 完整搜索的一键排程脚本（默认 dry-run；
  `--execute --device` 才启动进程；前置检查解释器/merged 表/画像/设备存在性与同卡并发；
  每阶段日志落 `results/condition_search/logs/`；按 `config_hash` 复用所以可断点续跑）。
  最终 test 留出**不自动化**，脚本只打印待执行命令。
- `experiments/condition_diagnostics.py` — 被试级置换 / mask-only / 全缺失 / 绕过场景、
  被试级配对 bootstrap、来源与缺失分层。
- `tests/test_condition_injection.py` — 验收 A–Q（62 项）。
- `results/condition_search/{budget_profile.json,plan_stage12.json,initial_condition_config.json}`
  — 实测画像、阶段 1+2 预生成计划、锚点配置说明（不含性能数值）。
  **没有** `best_condition_config.json`：搜索还没跑，任何"最优配置"文件都不应存在；
  它只能由 `condition_runner --frozen_trial` 在真实 completed 试次上生成。
  阶段 1–3 的预算粗估原先放在 `plan_stage123.json` 里，但那份计划里的晋级臂
  （`cat`/`sum`）是手填示例而非验证结果，已从交付目录移出到
  `.tmp_cond/ILLUSTRATIVE_plan_with_handtyped_promote.json`。

修改：`utils/dataloader.py`、`models/tfm.py`、`models/tfm_backbone.py`、`models/__init__.py`、
`train/optim.py`、`main.py`、`experiments/evaluate_variant.py`、`analysis/next_point_eval.py`、
`models/pathology.py`、`utils/common.py`、`scripts/Pretrain_HC_tfm.sh`、
`scripts/Finetune_MDD_tfm.sh`、`TASK_HANDOFF.md`。

另外**只动了两处** `inference.py`（它属于另一条工作线，未入库）：显式传权重路径时不再被
默认主权重闸门拦截、定型入口的版本要求改从 `configs/final_model.json` 的结构签名读取
（详见 §4.1 第 15 条）。除这两处外没有触碰它的其它逻辑。

未改动（按用户规范）：非条件主干宽度 / 层数 / 任务定义 / 预测目标 / 损失组成 / 数据增强 /
采样配方；`checkpoints/`、原始 xlsx 与既有结果目录；`configs/final_model.json` /
`tests/test_final_model.py`（另一条工作线的未提交文件，只读未改）。

## 4. 验证

> **本节的数值是 2026-10-09 删除前的记录**（`146 passed` / 条件注入验收 62 项等 ——
> 注意这个「146」与下面当前值同数但口径完全不同，勿混用）。
> 2026-10-10 两次删除后的当前状态：`pytest tests -q` → **146 passed, 11 skipped in 46.25s**
> （§0.26 后为 171，§0.27 再减 25 = 21 条 workflow 用例 + 4 条 diagnostics 用例）。
> 另外本节命令里的解释器路径属于旧记录，当前请用装了 torch 的 ≥3.10 解释器，
> 例如 `PYTHON=$HOME/anaconda3/envs/mamba/bin/python -m pytest tests -q`。
> §4.1 逐条缺陷里点名的文件（`condition_budget.py`、`condition_diagnostics.py`、
> `condition_runner.py` 等）**绝大多数已删除**，条目描述的是当时的修法与证据；
> 现存文件清单以 §0.26 / §0.27 的删除表与 §3 顶部说明为准。

```bash
# 全量（CPU 即可）
/home/user/anaconda3/bin/python -m pytest tests/ -q
# 只看条件注入验收
/home/user/anaconda3/bin/python -m pytest tests/test_condition_injection.py -q
```

结果：`146 passed, 11 skipped, 0 failed`（全量 49.28 s，纯 CPU）；其中条件注入验收
`tests/test_condition_injection.py` **62 passed**（A–Q 全部字母都有对应行为测试）。
11 项 skip 全部来自 `tests/test_final_model.py`，原因都是同一句「checkpoints/ 主权重不在本机
（权重不入版本库）」——即定型权重的端到端用例在本机只能验证到路径与版本闸门这一层。

之前 `tests/test_final_model.py` 的 3 项失败也已消除——但它们**不是纯环境性问题**：
定型入口在拿到显式权重路径后仍然先去解析默认主权重，测试传的临时权重因此照样撞
`FileNotFoundError`（见 §4.1 第 15 条）。权重本身仍不在本机（不入版本库），那部分用例
现在正确地报告"缺权重"而不再是"路径逻辑撞墙"。

### 4.1 本轮实测发现并修好的真实缺陷

不是"顺手改风格"，都是跑出来的错：

1. `experiments/condition_budget.py`：显存探针的 `nvidia-smi` 格式串写错
   （`nounit,mb`），采样恒为 0 MiB —— 会把「没测到」报成「很省显存」。改为
   `-i <idx> --format=csv,noheader,nounits`、按进程树归属，采样失败置 -1 并显式说明。
2. `experiments/condition_budget.py`：`probe_recipe` 按固定名单**删除**结构键、指望预设补齐，
   而 `structure` 预设没写 `tfm_patch_len/cpm_horizon/cpm_layers`，于是真实结构探针实际用了
   `main.py` 的 CLI 默认值（patch 8）。第一次实测的每任务代价因此属于另一个模型，并被
   HC 权重的任务口径闸门报错暴露。现在预设只覆盖它显式给出的键，并对 `structure` 断言
   结构键与正式配方逐值一致；画像新增 `arch` 结构指纹，`estimate_trial_cost` 在漂移时报错，
   画像 `version` 升到 2（旧画像拒绝混用）。
3. `experiments/condition_budget.py`：MDD 的固定开销由 1↔2 epoch 差分推导时为
   $-11.99\ \text{s}$。负开销物理上不存在，当时先按 0 截断并在画像里写"MDD 成本是下界"——
   **这个处置是不充分的**：它把一个口径错误伪装成一个"保守数"，而诊断试次仍然拿被截断后的
   开销报价。根因与正确解法见下面第 12 条。
4. `experiments/condition_runner.py`：`evaluate_trial` 的产物目录写成
   `Path / 'eval' / f"…" + ('_test_final' if …)`，`/` 比 `+` 结合更紧 → 执行模式必然
   `TypeError`（dry-run 走不到，因此长期潜伏）。已拆开拼接。
5. `experiments/condition_budget.py`：`load_profile('')` 把空串解析成当前目录并尝试读取，
   `IsADirectoryError`。现在空路径 / 非文件一律按「没有画像」处理。
6. `experiments/condition_diagnostics.py`：被试级条件表按 `max(subject_index)+1` 稠密分配，
   分批采集会让未采集到的被试越界或静默读到全零行（伪装成"全字段缺失"）。改为紧凑表 +
   `rows_for` 查不到即显式剔除并计数；跨窗口一致性校验里 `np.array_equal([n,K],[K])`
   恒为 False，会把完全一致误判成协议违规，已改成逐行比较。
7. `models/tfm.py`：`arm=none` 只关掉字段级编码器，CPM 头仍按 `cond_dim` 构建 legacy 标量
   FiLM，导致"无条件臂"其实被旧条件调制（且与 One-Step 头不一致）。现在两头条件宽度同源，
   `--condition_arm none` 同时下发 `--tfm_patho_mode none`。
8. `experiments/condition_runner.py`：`--adopt_orphan` 只打印"继续"却没有真的追加登记，
   承诺的行为没实现；`--resume` 的帮助文本写"跳过这些试次"而实现是重跑。现已实现
   `adopt_orphan` 追加 `failed` 行（保留原 `running` 历史行），并把 `--resume` 说明改成与
   行为一致的"容忍残留并本次重跑"。
9. `experiments/condition_runner.py`：护栏清单里的 FC 逐边 PCC 没有登记列，评估产物里其实
   已有 `fc.model.edge_pcc_mean` / `fc_mae`。已补两列（未开 `--eval_fc` 的试次保持 null，
   报告写"未测"而不是补数）。
10. `experiments/condition_diagnostics.py`：分层只用了 `Dataset`，方案 §12 要求
    `Dataset`/`site` 都在；已补齐两个审计列（仍只用于分层，不进条件向量）。

11. `experiments/condition_budget.py`：`estimate_trial_cost` 的 `return` 之后残留了整段旧实现
    （160 行不可达代码）。它不会被执行，但会让人以为成本还有「单一每任务秒数套满 150 epoch」
    的口径。已删除，成本折算与探针反演改为共用同一个 `PhaseFit.predict_wall` 公式。
12. `experiments/condition_budget.py`：**v2 的分段模型不可辨识**，因此把负的固定开销截成 0
    并不是"数据有噪声"，而是模型写错了。`main.py:788` 用 0 基 epoch 判断
    （`epoch >= freeze_backbone_epochs` 才解冻），所以 `train_epochs=1` 的探针**整段都在冻结期**，
    而 2-epoch 探针的第 2 个 epoch 才解冻；用 $W(1,s)$ 反推「一个解冻 epoch」必然把冻结期的
    便宜算成固定开销为负。同时它也没有分离每 epoch 的循环内验证 $V$，$T$ 与 $V$ 完全纠缠
    （旧画像的 `seconds_per_task` 其实是 $(V+tT)/t$）。
    v3 改成「epoch 网格 1/2/3/4 × 两档样本预算 4/12 × 冻结/解冻两套日程」的 18 次运行
    （每队列 8 次主网格，finetune 另加 2 次整段冻结），显式识别
    $O,V,T,T_f,R$（$R$ = 解冻那一刻重建 optimizer / 重新 compile 的一次性开销），
    并保留四组自由度自检：两档样本预算回代出的 $V$ 必须一致、逐 run 残差在容差内、
    固定开销 $O$ 的多路估计必须一致、$0<T_f<T$；任一不满足就**报错不出数**。
    另有一条**独立的测量路径**复核 $V$（从探针日志里 `[Val-TFM]` 进度条的自身耗时统计，
    与墙钟差分完全不同源，见 §5「独立复核」）。
    合成网格自检（现已固化成 `test_O_phase_fit_recovers_all_five_cost_components`，见第 17 条）
    在实现阶段就抓出了我自己的一个识别式错误
    （全冻结网格跨档位做差忘记除以 epoch 数，会把 $T_f$ 高估一倍，进而触发 $T_f<T$ 断言失败）。
13. `experiments/condition_budget.py`：`profile_eval` 接收 `data_root` / `split_file` 参数却
    从不转发（`evaluate_variant` 的口径来自 checkpoint 的 `config.args` 快照自省）。
    「参数存在但不起作用」会让探针被误认为跟随当前配方；已删掉这两个参数并在 docstring 里
    写清口径来源。
14. `experiments/condition_budget.py`：诊断试次的成本原先用「只测主指标的 val 评估」秒数近似，
    而真实诊断（置换/掩码 + FC 对照 + bootstrap）比它贵。现在用**排程器实际会跑的那条命令**
    实测 `diagnostic.seconds`，画像里没有这个字段就报错而不是近似。
15. `inference.py`：定型主模型入口有两处真实缺陷（不只是"权重不在本机"）。
    (a) 显式传入权重路径时仍然先解析默认主权重，导致临时权重的用例也撞 `FileNotFoundError`
    —— 传了路径却被默认路径的闸门拦住；现在只有 `ckpt_path=None` 才要求默认主权重存在，
    显式路径只按自身做校验。(b) `ARCH_VERSION` 已被条件注入实验线抬到 5，定型入口原先拿
    全局常量当要求，于是 v4 定型权重被自己的闸门拒绝。现在定型入口的版本要求取自
    `configs/final_model.json` 的结构签名（隔离版本闸门），实验线的版本推进不再改变定型契约；
    代价是**将来重新定型主模型时必须更新该签名**，否则入口会继续认旧版本。
16. `experiments/condition_budget.py`：`run_probe` 在调用方配方之上**又合并一次**
    `PROBE_PRESETS[preset]`，于是探针网格里逐格设置的 `train_samples_per_subject` 与
    `freeze_backbone_epochs` 被预设静默回滚——名为 `s12` 的运行实际跑成 `s4`
    （子进程命令行可见），两档任务数因此相同，「按任务量差分分离 $V$ 与 $T$」直接退化。
    现在预设只合并一次，且每格运行完成后断言子进程命令行确实带着本格的
    `--train_samples_per_subject/--freeze_backbone_epochs/--train_epochs`。
17. `experiments/condition_budget.py`：**识别式本身写错过一次**——全冻结网格跨样本档位做差
    时漏除 epoch 数，会把 $T_f$ 高估一倍。是合成网格自检（已知 $O,V,T,T_f,R$ 生成墙钟再反解）
    当场抓出来的；该自检已固化成 `test_O_phase_fit_recovers_all_five_cost_components`。
18. `experiments/condition_budget.py`：pretrain 分支原先断言「解冻一次性开销必须 ≈0」，
    而实测里 epoch 之间存在 1.6 s 量级的升温/降频漂移，这个断言会把**可用**画像判死。
    改为：$V$/$T$ 只取稳态 epoch 之差，漂移作为诊断量登记（`epoch_drift_seconds`）并做容差
    检查；finetune 的 $R$ 保留**带符号**估计值，报价取 $\max(R,0)$ 并注明是否低于漂移分辨力。
19. `experiments/condition_budget.py`：`tasks_per_epoch_at` 原先只有单档实测、对正式配方的
    12 样本/被试做线性外推。现在探针在 4 与 12 两档都实测（12 正好是 auto 上限），
    正式试次落在**实测点**上；只有档位外才外推并显式标注。
20. `experiments/condition_runner.py`：`same_budget_ranking` 对字典键做
    `for k, v in sorted(groups, ...)` 解包，任何调用都必然 `ValueError: too many values to
    unpack`——即报告里的排名与新的晋级核对一执行就会崩。由新增测试抓出并修复。
21. `experiments/condition_budget.py`：MDD 探针的 HC 权重按**预设**解析
    （`_budget_HC_<preset>_…`）。原先写死 `structure`，用 `tiny` 预设做网格验证时会拿
    另一套结构的权重去微调，错误在很下游才暴露。缺该预设权重时立即报错说明原因。


短 smoke（真实 merged 数据，真实结构，只跑 1–2 epoch，产物在 `.tmp_cond/`，不写 `checkpoints/`）：
HC `arm=demo/hc3` 与 MDD `arm=cat/mdd8` 均完成训练 + 严格重建 + 评估 + 诊断，参数量
6,064,880 与 6,075,600，条件参数 37,200（$\rho=0.0061$）与 47,920（$\rho=0.0079$）。

## 5. 实测预算表（分段代价，两条独立路径互核）

> **本节整体属于 2026-10-09 的历史实测记录**：产生这些数的画像工具
> `experiments/condition_budget.py` 与 `results/condition_search/budget_profile.json` 的
> 读写方（编排器）已在 2026-10-10 删除（§0.26），本轮**没有**重测画像的脚本，
> §一.8 要求的「本轮 study 内重测画像」因此处于**无法执行**状态。
> 下表可用来判断量级（表 3 的实测单价：HC 一整轮 **6.11 GPU·h**、MDD 一整轮 **5.95 GPU·h**，
> 150 epoch 上限配方），**不可**当作本轮的预算结论，
> 更不可用它反推「跑满 N 小时还剩多少试次」——阶段概念已不存在。

画像 `results/condition_search/budget_profile.json`：**version=3**，`device=cuda:1`
（NVIDIA RTX A6000，探针期间**单卡独占**），`measured_at=2026-10-09T02:14:28+0800`。
结构指纹绑定 `num_rois=116 / tfm_dim=256 / tfm_layers=4 / tfm_heads=4 / tfm_patch_len=4 /
cpm_horizon=8 / cpm_layers=2 / context_max=64 / eval_context_length=64`；这些键任一变化，
`estimate_trial_cost` 会因指纹不匹配拒绝换算（必须重测），而不是给一个近似数。

**测量设计**（`experiments/condition_budget.py: profile_cohort`）：每队列 8 次运行
（epoch 网格 1/2/3/4 × 两档样本预算 4/12），finetune 另加 2 次「整段都冻结」
（`freeze_backbone_epochs=2`，运行内不解冻，因此不含解冻一次性开销）。反演模型：

    W(n, t, f) = O + n·V + n_fr·t·T_f + (n − n_fr)·t·T + R·[n > n_fr],  n_fr = min(f, n)

$V$ 与 $T$ **只取稳态 epoch 之差** $W(4)-W(3)$（两档各一次，相减得 $T$，回代得 $V$）：
这条差分不含冻结 epoch、不含解冻事件、也不含 epoch 1 的预热，是唯一没被一次性项污染的途径。
旧口径用 $W(2)-W(1)$ 当「一个 epoch」，而 `main.py:788` 的 0 基冻结判断让 $n=1$ 探针整段
都在冻结期——这就是当初 MDD「固定开销 −11.99 s」的来源。

**$R$ 的口径要说清（也是"截 0 是否可靠"的答案）**：$R$ 与「epoch 间代价漂移」在这批数据上
**不可分离**——它按 $R=W(2)-W(1)-\Delta$ 定义，即第 2 个 epoch（第一个解冻 epoch）相对稳态
epoch 多出来的那部分，画像同时登记带符号值 `epoch_drift_seconds`（MDD 两者同为 2.92 s）。
所以 2.92 s 是「解冻一次性开销 + 首个解冻 epoch 的预热/重编译」的**合计上界**，不是纯
optimizer 重建代价。让它可用的是**稳态差分**（$V/T$ 完全不经过这些一次性项），而不是
$\max(\cdot,0)$ —— 截 0 只是避免报出非物理的负开销，本身不提供任何可靠性；因此这里
$R$ 只付一次、占整轮 <0.02%，报价对它不敏感，且画像把带符号漂移如实留在 `epoch_drift_seconds`
字段里供复核。

### 表 1｜分段代价（实测反演）

| 队列 | 任务数/epoch（4 与 12 样本/被试） | $T$ 解冻 s/任务 | $T_f$ 冻结 s/任务 | $V$ 每 epoch 循环内验证 s | $O$ 固定启动 s | $R$ 解冻一次性 s | 一个 epoch 总代价 s（s=4 / s=12） | 显存峰值 MiB | 参数量 |
|---|---|---|---|---|---|---|---|---|---|
| HC pretrain（`arm=demo`, hc3） | 7112 / 21336 | 0.004836 | 不适用（无冻结期） | 43.23 | 5.41 | 不适用（无解冻事件；实测漂移 +1.57） | 77.6 / 146.4 | 13498 | 6,064,880 |
| MDD finetune（`arm=cat`, mdd8） | 7212 / 21636 | 0.004811 | 0.002500 | 41.83 | 13.56 | 2.92 | 76.5 / 145.9（冻结期 59.9 / 95.9） | 7800 | 6,075,600 |

两个数直接决定「能不能忽略」：

- **冻结期单价是解冻期的 52%**（0.0025 vs 0.0048 s/任务），所以真实配方的前 10 个 epoch
  明显更便宜；用一个混合单价套 150 epoch 会同时错报训练与冻结阶段。
- **每 epoch 的循环内验证与训练同量级**：s=4 档上一个 epoch 是「训练 34 s + 验证 43 s」。
  验证在 `main.py` 的 epoch 循环里每轮都跑（含 free rollout 与 FC），按正式评估口径实测。

### 表 2｜评估与诊断耗时（三种口径，互不可替代）

| 队列 | 主指标 val 评估 s（每试次选优） | 完整评估 rollout+FC s（晋级候选护栏） | 一次置换诊断 s（含 FC/bootstrap） |
|---|---|---|---|
| HC | 21.22 | 71.74 | 不适用（诊断只挂在 MDD 臂上） |
| MDD | 24.23 | 76.04 | 105.80 |

评估口径来自 checkpoint 的 `config.args` 快照（`evaluate_variant` 自省重建模型），不是探针参数；
诊断耗时是**用排程器实际会跑的那条命令**实测的——用主指标评估近似会少算 FC 部分（这里差 4 倍）。

### 表 3｜正式配方下单个试次的整轮成本（按阶段拆开，不用一个混合单价）

`train_epochs=150`（配方上限）、`train_samples_per_subject=0` → auto 上限 12，
12 样本档的任务数是**实测档位**（不外推）。单位 GPU·h（= 单卡占用小时）：

| 队列 | 冻结期训练<br>（前 $n_{fr}$ epoch × $T_f$） | 解冻期训练<br>（其余 epoch × $T$） | 循环内验证<br>$n\cdot V$ | 固定启动 $O$ + 解冻一次性 $R$ | 独立 val 评估 | **合计/试次** | 验证占比 |
|---|---|---|---|---|---|---|---|
| HC pretrain（$n_{fr}=0$） | — | 4.30 | 1.80 | 0.002 + — | 0.006 | **6.11** | 29.5% |
| MDD finetune（$n_{fr}=10$） | 0.15 | 4.05 | 1.74 | 0.004 + 0.001 | 0.007 | **5.95** | 29.3% |

- MDD 的前 10 个 epoch 只花 0.15 GPU·h（540 s），而后 140 个 epoch 花 4.05 GPU·h——
  把两段的 $T_f/T$ 混成一个单价会同时错报这两段（旧口径正是这样）。
- 「合计」列与 `--plan` 折算、`last_schedule_summary.json` 的 `budget` 字段同源
  （同一个 `PhaseFit.predict_wall`），报告与核算不会有第二套数。

### 表 4｜阶段折算

| 计划 | 执行次数 | GPU·h（各卡占用时长之和） |
|---|---|---|
| 阶段 1+2（2 HC 基线 + 6 MDD 臂） | 8 | **47.9** |
| 阶段 1+2+3（假想只晋级 1 臂） | 20 | **113.4** |
| 阶段 1+2+3（假想晋级 2 臂） | 32 | **178.9** |
| 阶段 1→4（2 臂 + 注入位置第一/第二档） | 36 | **202.8** |
| **阶段 1→5 完整链路（2 臂，`review_scope=mdd_only`）** | **40** | **226.6** |
| 阶段 1→5 完整链路（2 臂，`review_scope=full`，每个种子各重训 HC 家族） | 42 | 238.8 |
| 单个置换诊断试次（不训练） | 1 | **0.029** |

阶段 1→5 的构成（2 臂口径）：`hc_baseline 2 / candidate 28 / control 4 / diagnostic 2 / review 4`
—— 容量对照与信息对照都计费，不是免费附件。

阶段 1+2 的成本构成（GPU·h）：训练 33.79 / **循环内验证 14.06（29.3%）** / 固定启动 0.03 /
解冻一次性 0.005 / 独立评估 0.05。近三成预算花在每 epoch 的验证上——旧口径把这个数
混进每任务单价，所以当时回答不了「验证到底贵不贵」。

> 表 4 的「1 臂 / 2 臂」是**容量粗估用的假想臂数**（用占位臂名 `<arm>` 折算，不是晋级结论）。
> 真实臂名由阶段 2 的验证集排名给出，而 `--auto_promote` 就是按这条规则从登记表反推的，
> 所以脚本排出来的实际计划与这张表的口径一致；执行模式下排程器还会再核对一次（§6.4）。

### 独立复核（两条不同的测量路径）

1. **$V$ vs 日志里的验证进度条**：$V$ 由墙钟差分反演，而 `[Val-TFM]` 进度条的自身耗时是完全
   独立的内部测量。HC：反演 43.23 s vs 进度条均值 41.0 s（区间 40–42，n=20）；
   MDD：41.83 s vs 42.04 s（区间 42–43，n=24）。结果已写进画像的 `bar_crosscheck`，
   `--check_profile_bars` 可随时复跑，不一致就报错而不是继续报价。
2. **$O$ 的多路估计**：HC {5.89, 3.16, 7.19}、MDD {11.96, 11.95, 14.54, 15.79} s，
   一致性通过（$O$ 每轮只付一次、误差不随 epoch 放大，所以这里容差刻意宽于 $V$/$T$）。
3. **逐 run 残差**：MDD 最大 2.23 s；HC 在 s=12 的 n≥3 运行上有 −14.78 s（见下面的限制）。
4. **跨批复现性**：同一配置的独立两批探针，HC 1-epoch 墙钟 84.7 s vs 83.5 s（−1.4%）。

### 口径限制（引用这些数时必须带上）

- HC 的 −14.78 s 残差与 $O$ 的多路散布指向**升温/降频**：探针期间该卡实测到过 86 °C、
  SM 时钟从 2100 降到 1725 MHz。$V$/$T$ 取的是稳态（最后两个 epoch 之差），即降频后的速率，
  所以 150-epoch 报价方向上偏保守（HC 的 s=12 档影响约 1%）。
  **这些数是单卡独占实测值**：同卡并发会显著拉长墙钟，不能套用本表。
- epoch 数按配方上限 150 计，`patience=20` 的早停只会更便宜 → 表 3/4 是**上界**。
- 表 4 的阶段 1→5 已经包含注入位置（阶段 4）与多种子复核（阶段 5，`review_scope=mdd_only`）；
  **仍未计入**的是：最终 test 留出、晋级候选的完整评估（表 2 第 2 列，每个候选 71.7/76.0 s）、
  外层宽度家族（`C`/`d_demo` ≠ 锚点）与表示法 A/B（§7.2）。每多一个 seed 复核约
  6.11 + 5.95 GPU·h/配置；`review_scope=full` 还要按种子各重训一次 HC 家族基线。
- 这批探针自身花费约 1.6 GPU·h（18 次运行）：它是一次性测量成本，没算进搜索预算，
  但也不该被说成「凭空知道要多久」。
- `tiny` 预设的结构与正式配方不同（缩小版主干），只用于打通流程与快速回归，
  **不能拿来报价**；`--cost_preset structure` 是唯一可用于核算的预设。

## 6. 完整跑实验的执行方式

> **§6.1–§6.7 已全部失效（2026-10-10，见 §0.26 与 §0.27）**：这一节写的搜索/预算/晋级/冻结/诊断
> 命令依赖 `scripts/Condition_search_full.sh`、`experiments/condition_runner.py`、
> `experiments/condition_budget.py`、`experiments/condition_analysis.py`、
> `experiments/condition_diagnostics.py` 等文件，它们**都已删除**，
> 命令原样敲会直接「找不到模块」。下面逐条保留只为追溯当时的口径
> （预算数字、auto_promote 反推规则、冻结门槛条件等都仍是有效的设计记录，
> 但 §6.7 那条 `python -m experiments.condition_diagnostics …` 连**模块**都没了，
> 被试级置换 / mask-only / 全缺失场景在本仓库已无实现）。
> 当前的执行方式只有一条：`scripts/Pretrain_HC_tfm.sh` → `scripts/Finetune_MDD_tfm.sh`
> （见 §7.1 与 README 快速开始）。

### 6.1 一条命令排完整条链路（默认 dry-run，不启动任何进程）

```bash
# 只看每个阶段将要跑什么（不落进程）
PYTHON=/home/user/anaconda3/bin/python bash scripts/Condition_search_full.sh

# 真跑阶段 1→5（必须显式给执行卡）
PYTHON=/home/user/anaconda3/bin/python bash scripts/Condition_search_full.sh \
  --execute --device cuda:1
```

脚本按 阶段 1+2 → 3 → 4 → 5 顺序调用排程器，**阶段 3 起一律 `--auto_promote`**：晋级臂、
阶段 3 胜出的 `d_new/H_delta/dropout_p`、阶段 4 第一档胜出的注入位置，全部按
`config_hash` 从登记表反推，整条链路没有一处需要手填臂名。要点：

- 执行闸门（`--max_trials/--gpu_budget_hours`）默认给得**很宽**（500 次 / 5000 GPU·h）：
  本脚本的目标是跑完，不是省。三个闸门仍然必须显式存在，脚本用宽松值而不是"取消闸门"，
  这样失控展开（例如误加外层家族）仍会被拦下。
- 前置检查：解释器 ≥3.10 且有 torch（本机 `python3` 是 3.8，所以要给 `PYTHON=`）、
  两张 merged 表、预算画像存在、`--device` 指向真实卡、卡上已有计算进程时告警
  （画像秒数是**单卡独占**实测值）。缺哪项就停在哪项。
- 每阶段日志写 `results/condition_search/logs/`；中断后重跑同一条命令即续跑
  （completed 配置按 `config_hash` 复用，不重训）。
- 只想跑某一阶段：`--only 12|3|4|5|status`。

### 6.2 只看进度与下一步命令（只读，不建目录、不起进程）

```bash
PYTHONPATH=. /home/user/anaconda3/bin/python -m experiments.condition_runner \
  --status --registry_dir results/condition_search \
  --budget_profile results/condition_search/budget_profile.json
```

打印：按阶段/状态计数、失败清单、残留 running、同预算可比组排名、下一阶段需要的输入
（以及推不出来时缺哪一条证据）、可直接粘贴的下一步命令。

### 6.3 预算核算（把预生成计划折算成 GPU·h 并判断是否放得下）

```bash
PYTHONPATH=. /home/user/anaconda3/bin/python -m experiments.condition_budget \
  --plan results/condition_search/plan_stage12.json \
  --profile_out results/condition_search/budget_profile.json \
  --gpu_budget_hours 60 --max_trials 10 --data_root ./data
```

### 6.4 画像复核与重测（结构键或队列一旦变了必须重测）

```bash
# 复核（不重测，秒级）
PYTHONPATH=. /home/user/anaconda3/bin/python -m experiments.condition_budget \
  --check_profile_bars --profile_out results/condition_search/budget_profile.json \
  --device cuda:1
# 重测（每队列 8–10 次短运行，单卡独占，约 1.5–2 GPU·h）
PYTHONPATH=. /home/user/anaconda3/bin/python -m experiments.condition_budget \
  --refresh_profile --device cuda:1 --data_root ./data \
  --probe_epochs_series 1 2 3 4 --probe_sample_levels 4 12 \
  --checkpoint_dir ./.tmp_cond/budget_ckpt --profile_out <画像路径>
```

### 6.5 不用脚本时的等价手工命令（阶段 3 起同样用 `--auto_promote`）

```bash
# 阶段 1+2：2 个 HC 基线 + 6 个 MDD 臂（8 次执行，实测 47.9 GPU·h）
PYTHONPATH=. /home/user/anaconda3/bin/python -m experiments.condition_runner \
  --stages 2 --execute --device cuda:1 \
  --max_trials 500 --gpu_budget_hours 5000 \
  --data_root ./data --checkpoint_dir ./checkpoints/condition_search \
  --registry_dir results/condition_search

# 阶段 3 / 4 / 5：只换 --stages，其余同上，并一律加 --auto_promote
... --stages 3 --auto_promote --execute --device cuda:1 ...
... --stages 4 --auto_promote --execute --device cuda:1 ...
... --stages 5 --auto_promote --execute --device cuda:1 ...
```

`--promote <臂> <臂>` 的手填写法仍然保留，但执行模式会拿登记表核对它是否等于阶段 2
同预算验证集的前 `PROMOTE_TOP_K=2` 名，不符即退出；`--auto_promote` 与手填冲突时同样报错，
不会静默偏向任何一个。本文档**不给**可以直接跑的 `--promote cat sum` 之类示例：`cat`/`sum`
只是搜索空间里的臂标识，没有任何一次真实验证支持它们优于其他臂。

推导规则（`experiments/condition_runner.py: derive_stage_inputs`）：

- 只认**唯一**的同预算组（`stage, train_epochs, cohort`）；有多个组就停下要求你统一预算，
  不跨预算挑"看起来最好"的那个。
- `failed`/`pruned`/缺验证主指标/`control`/`diagnostic` 行永不作为胜出者。
- 阶段 4 的第二档只有第一档三个位置**全部** completed 才展开（没跑完的位置不等于输）。
- 阶段 3 胜出的 `lr_scale` 不是默认值时会打印警告：阶段 4/5 不展开该轴，因此不会自动带上，
  要带就必须显式 `--cond-override lr_scale=…`，而那会使已跑完的第一档不再可比、需要重测。

### 6.6 最后一步留给人：冻结配置 + test 留出一次评估

测试集只在配置冻结后评估一次（方案 §10：测试集不参与选择），所以 `--frozen_trial`
**不由脚本自动填**。跑完阶段 5 后，从 `--status` 的同预算排名里确认 trial_id，再执行：

```bash
PYTHONPATH=. /home/user/anaconda3/bin/python -m experiments.condition_runner \
  --stages 5 --auto_promote --frozen_trial <确认过的 trial_id> --evaluate_test_final \
  --full_guardrails --execute --device cuda:1 \
  --max_trials 5 --gpu_budget_hours 20 \
  --data_root ./data --checkpoint_dir ./checkpoints/condition_search \
  --registry_dir results/condition_search
```

只有该 trial_id 是 completed 且跑满预算的配置时才会生成
`results/condition_search/best_condition_config.json`；否则排程器拒绝生成并说明原因，
此时合法产物仍然只有 `initial_condition_config.json`。

### 6.7 单独跑诊断（已训练权重上的信息/容量检验，不训练）

```bash
PYTHONPATH=. $PY -m experiments.condition_diagnostics \
  --ckpt <finetuned_best.pt> --split val --device cuda:N \
  --out_dir <本轮 study 内的 diagnostics 子目录>
```

## 7. 真正阻塞的项 vs 只是"未搜索的选项"

必须区分这两类，否则会把「还没试」误读成「还不能跑」。

### 7.1 还需要人做的事：按顺序启动训练（脚本已就绪）

**没有 dry-run／排程／预算闸门**：编排器与预算画像工具（§0.26）、study 身份与 fresh 来源闸门
（§0.27）都已删除，所以下面第一条命令**就是真训练**。

1. **两条命令，顺序固定**（HC 必须先跑，两者用同一个 `CHECKPOINT_DIR`）：

   ```bash
   PY=$HOME/anaconda3/envs/mamba/bin/python      # 必须是装了 torch 的 ≥3.10 解释器
   export CHECKPOINT_DIR=runs/run1010        # 可选；默认 $ROOT/checkpoints

   # ① HC：随机初始化 → $CHECKPOINT_DIR/neurotwin_tfm_pretrain/base_best.pt
   PYTHON="$PY" bash scripts/Pretrain_HC_tfm.sh

   # ② MDD：起点自动取 ①的产物 → $CHECKPOINT_DIR/neurotwin_tfm_finetune/finetuned_best.pt
   PYTHON="$PY" bash scripts/Finetune_MDD_tfm.sh
   ```

   - 可选环境变量（两个脚本共用）：`CHECKPOINT_DIR`、`DEVICE=cuda:1`（不给则沿用 `main.py`
     的历史默认，`main.py:472-473`）、`DATA_ROOT`；微调脚本另有 `HC_WEIGHT=<完整路径>`。
   - `CHECKPOINT_DIR` 不给就写进仓库共享的 `checkpoints/`。**想本轮隔离就自己指一个空目录** ——
     这个决定权现在完全在人手里，代码不再强制（§0.27）。上面示例用 `runs/…`，因为
     `.gitignore` 已忽略 `runs/` 与 `checkpoints/`（实测 `git check-ignore -v runs/run1010` 命中）；
     随手起一个新目录名（如 `.tmp_run/`）不会被忽略，别用 `git add -A`。
   - 解释器选错的失败点比较晚：`scripts/Pretrain_HC_tfm.sh:50` 只导入
     `utils/condition_schema.py` 读常量，本机默认 `python3`（3.8.10，实测无 torch）
     这一步也能过，真正报错要到 `main.py` 导入 torch 时。用 `PYTHON=` 显式指定即可。

2. **重跑会产生平行目录，起点会撞多个候选**：`main.py:801-808` 在
   `checkpoint_dir/<name>` 已存在时把新目录自增成 `neurotwin_tfm_pretrain2`（不覆盖旧权重）。
   此后微调脚本的起点查找会命中多个 `neurotwin_tfm_pretrain*/base_best.pt`，脚本
   **拒绝替你猜**并要求显式指定（`scripts/Finetune_MDD_tfm.sh:47-61`）：

   ```bash
   PYTHON="$PY" CHECKPOINT_DIR=runs/run1010 \
     HC_WEIGHT=runs/run1010/neurotwin_tfm_pretrain2/base_best.pt \
     bash scripts/Finetune_MDD_tfm.sh
   ```

   2026-10-11 改名前留下的 `checkpoints/hc_demo`、`checkpoints/mdd_cat` 是训练产物，
   **不替人删**；但微调脚本的自动查找已只认 `neurotwin_tfm_pretrain*/base_best.pt`，
   要复用旧目录必须显式 `HC_WEIGHT=<旧路径>`。⚠️ 现在没有任何代码能证明你选的这份
   就是本轮训出来的 HC —— 内嵌来源校验随 `condition_study.py` 一起删掉了（§0.27）。

3. **test 评估发生在每次微调结束时**：`main.py:1149-1159` 是 finetune 的自动闭环，会把 test
   指标写进 `<save_dir>/eval/metrics_test.json`。本轮只有一个选定配方、没有搜索，因此不构成
   「跑很多组再挑一个看 test」的选择性推断；但**重复跑第 ② 步就是重复消费留出集**。
   要保留人工闸门就用等价的手工命令并加 `--no_eval_after_train`（`main.py:1223-1226`；
   完整命令行见 README 快速开始），或事后手动复评
   `python -m experiments.evaluate_variant --ckpt <finetuned_best.pt> --out_dir <目录>`。

4. **随机来源分离是现在唯一由代码强制的协议项**：`main.check_seed_separation()`
   （`main.py:414`，在 `main()` 开头无条件调用）要求 `--seed` / `--sampling_seed` /
   `--cond_dropout_seed` 两两不同，登记值是
   `utils/condition_schema.py` 的 `MODEL_SEED=2024 / SAMPLING_SEED=20261009 / DROPOUT_SEED=20261008`。
   实测：同值 ⇒ `ValueError`；Namespace 缺任一 ⇒ `ValueError`。
   副作用是 `python main.py --mode pretrain` 若同时沿用两个历史默认值（`--seed` 与
   `--sampling_seed` 默认都是 2024）会被拒；两个脚本与 README 命令行都显式传分离值，不受影响。

**已经从链路上断开、需要人另行决定是否重建的事**（不是"还没跑"，是"没有实现体了"）

- **配置冻结**：`--frozen_trial`／`--evaluate_test_final` 随编排器消失，`check_freeze_eligibility`
  所在模块也已在 §0.27 删除 ⇒ 仓库内不再有冻结门槛的任何实现。
- **`best_condition_config.json`**：不会由本轮产生；任何下游都不应假设它存在。
- **主指标与诊断场景**：被试级 OS MAE 的实现定义、置换 / mask-only / 全缺失 / 绕过场景、
  ≥5000 次被试级配对 bootstrap 全部随 `condition_diagnostics.py` 删除（§0.27）。
  `experiments/evaluate_variant.py` 仍能做常规的 val/test 指标评估，但那不是上述诊断。
- **来源证明**：见本节第 2 条末尾的 ⚠️。
- **`--save_training_state` 与 `train/training_state.py`**：CLI 开关存在但训练循环不读它，
  模块也没有任何调用方 ⇒ help 承诺的落盘行为不兑现（§0.27 已登记，删不删等你决定）。


### 7.2 不阻塞的项（已定；「未搜索」这一类本轮已无法再搜）

- **HAMD/HAMA 总分与条目列的取舍：已定，不是待确认项**。`HAMA` 只取列名恰为 `HAMA` 的列，
  条目列 `HAMA_i` / `*HAMA_i` 与 `HAMATotal` 一律不读（§2.1 已写死并由验收测试覆盖）；
  HAMD 同理，且 MDD 的分层键固定为 `HAMD`，不跟随 `fields[0]`。剩下的「用总分还是条目做
  条件」原本属于搜索选项 —— 编排器已删除（§0.26），本轮**不会**再对它做 A/B，
  当前实现按合并表的固定字段顺序取总分列。
- **`episode` / `medication` 的词表语义：不阻塞预测实验**。当前码值来自数据实测
  （`episode=[0,1,2]`、`medication=[0,1,2]`，含哨兵替换后的 0/2；`Sex` 只有 `level_1/level_2`），
  每个类别字段都带独立 Unknown 索引。即使临床字典给出的标签与数字含义不同，改变的也只是
  **one-hot 各轴怎么读**，取值集合、缺失判定与训练/评估链路都不变，因此不需要等字典就能跑。
  只有当字典里出现数据中没观测到的新码值时，才需要 `--condition_extra_codes` 登记
  （`main.py:1275`，两个脚本都不传它 ⇒ 默认空串 ⇒ 不启用）。
- **`illness` 的 log1p、HAMD 条目 vs 总分等表示法 A/B**：原先列在搜索报告的 `not_covered`
  字段里；那份报告由已删除的编排器产出，**现在没有这个字段可读**。当前行为是写死的：
  连续字段一律走 robust-z（`median` / `IQR÷1.349`，`utils/condition_schema.py:462-465`），
  `illness` **没有** log1p 预处理，类别字段一律走码值词表 + 独立 Unknown。
  这属于「已实现且固定」，不再是待搜索选项。

## 8. 解释边界（写结论时必须带上）

- 参数量一致不等于函数容量一致；`static_new` 对照同时取消了新增字段"取值"与"缺失模式"的
  个体差异；推理时 mask-only ≠ 用 mask-only 重训。
- 两种置换指标的差值不等于"缺失信息的贡献"（逐字段置换打破相关结构）。HAMD 与 HAMA 高度
  相关，联合与单字段置换必须一起报告，不解释为各自的独立因果效应。
- $\gamma_\ell(c)$ 前的 $\lambda_\ell$ 可被 $W_\ell$ 吸收，不是临床贡献度量。
- 条件能预测脑信号 ≠ 疾病机制 ≠ 疗效。
- 搜索规模本身带来选择性推断偏差；最终比较只在冻结后的独立 test 上做一次。
  **2026-10-10 修正**：本轮没有搜索，也就没有「挑赢家再看 test」这一类偏差来源；但
  「只做一次」这条现在**不再是自动保证** —— `main.py:1149-1159` 在每次微调结束时自动评 test，
  重复微调就是重复消费留出集（§7.1 第 3 条）。
- 机制参考（FiLM、DiT 的 AdaLN、LoRA）只支持机制可实现，不证明在本任务上的收益；
  本项目的条件 RMSNorm 与条件门控低秩属于待验证候选。
