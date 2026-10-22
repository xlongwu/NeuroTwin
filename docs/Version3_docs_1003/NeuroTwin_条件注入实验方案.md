# NeuroTwin 条件注入实验方案 v2（冻结编码协议 · 阶段 0 文件改动与验收清单）

- 版本：v2（2026-10-08），取代 `docs/NeuroTwin_条件注入实验方案_20261008.md`（v1，保留作核查证据存档）
- 本轮范围：**仍未修改任何模型、损失、脚本或配置；未启动训练**。v2 只做：编码语义核实、协议冻结、引用与数字修正、阶段 0 工程清单。
- 推进顺序（与用户指示一致）：① 冻结字段与编码协议 → ② v1 修订为 v2 → ③ 实现并验收阶段 0 → ④ 最小初筛（HC 2 臂 + MDD 3 臂）。门控残差 / 条件门控低秩 **推迟到初筛结果之后**再决定是否开工。

---

## 1. v1 → v2 变更清单（逐条，含原因）

| # | v1 | v2 | 原因 |
|---|---|---|---|
| 1 | HAMD 主臂改用 `HAMD17 = Σ前17条目`，并加 `m_HAMD_scale_anon` 来源指示列（K=18） | **删除**该重建与指示列；正式输入 = 原始 `HAMD` 总分（K 减 1） | 用户裁定：字段范围严格限定，编码来源列属越界风险；口径差改为**报告层**按 `Dataset` 分层 + 待确认项 |
| 2 | `K_MDD = 18`，episode 3 列 | **`K_MDD = 17`**（episode 3 列，HAMD 块由 3 列降为 2 列） | 变更 1 的直接结果 |
| 3 | 完整样本数 1488（朴素 coalesce） | 改为**映射依赖的三档**：5 字段 1839 / 7 字段（推断映射）1413 / 7 字段（保守映射）1289；并规定敏感性队列必须在看到结果前二选一 | 新映射把 V1 的 `medication=0`（源 `-9999`）判为缺失，覆盖下降 |
| 4 | 分组编码 `e_bg/e_course/e_med/e_sym` + `ΔE` 融合 | 初筛**取消分组分支**，改为唯一迁移路径 `c_MDD = c_HC + ΔE(v_new, c_HC)`（线性 / MLP 两臂） | 减少结构自由度与迁移混杂；分组列为初筛之后的候选 |
| 5 | 阶段 3 一次铺开 6 种方法（直连 / C32 / C64 / 分组 / 门控残差 / 门控低秩） | 初筛只跑 3 个 MDD 臂，统一 `C=32` | 用户裁定：不铺开全部候选；统一宽度消除容量与迁移混杂 |
| 6 | 参数量表按 `C∈{K,32,64}` 混排 | 重算为 `C=32` 单宽表（§4） | 与变更 5 一致 |
| 7 | `episode` 保守映射作为主案 | **降级**：保守映射使 `m_episode` 成为强来源代理（缺失行中 V2 仅 17.3%，基线 56.7%）→ 改为"两条候选映射 + 掩码-来源耦合量化 + 正式启用需字典确认" | 新量化证据（§2.3） |
| 8 | `illness` identity 与 log1p 作为等价 A/B | **预选 log1p 为主案**，identity 降为消融 | 依据是数值条件数而非预期收益：raw 的 `max|z|=14.03`，log1p 的 `max|z|=2.12`，skew 2.74→0.18 |
| 9 | 引用 `inference.py:161-171`、`train/optim.py:206-229`、`models/__init__.py:6-7` 等 | 精确化为 `inference.py:160-173`、`train/optim.py:206-219`、`inference.py:131-138`、`experiments/evaluate_variant.py:79-101` 等（§9） | 逐条回读原文核对 |
| 10 | 队列 2195 需"限制旧划分"但未给实现路径 | 给出具体机制：新增冻结的 `cond_v2` 划分文件 + `--split_file` 显式入口（§5、§6） | 现无 `--split_file`，`utils/site_stratified_split.py:8` 的说明与实际不符 |

---

## 2. 冻结的字段与编码协议（唯一 schema）

### 2.1 字段范围（不再变动）

- **HC**：`Sex`、`Age`。
- **MDD**：`Sex`、`Age`、`education`、`episode`、`illness`、`medication`、`HAMD`（原始总分）。
- 禁止进入模型：`HAMA`/`HAMATotal`、条目分、`MedicationName/Dose`、`EverMedication`、`FirstEpisode`、`If first episode?`、`On medication?`、`Epi`、`AO`、`CurrentEpisodeDuration`、`TreatmentResponsive`、`FamilyHistory`、`CGIS`、`RRS`、`WSCT`、`Other/Other2`、`Dataset`、诊断标签、ID。
  其中 `Epi / AO / OnMedication / MedicationName` **只允许出现在数据审计与核实脚本中**（本文 §2.3 的取证依据），不得成为模型输入。
- 缺失掩码 `m_i` 与上述字段的合法编码/变换属于预处理，不算新增临床字段。

### 2.2 Schema 表（列布局由此表自动顺序生成，禁止手写索引）

| 组 | 字段 | 类型 | 有效性规则（`m_i=1` 的条件） | 生成列 | 列数 |
|---|---|---|---|---|---|
| demo | `Age` | 连续 | 18 ≤ Age ≤ 90（HC 实测 9–67 / MDD 11–77；边界待确认，超界判无效） | `z_Age`, `m_Age` | 2 |
| demo | `Sex` | 匿名类别 | 取值 ∈ {1,2}（`0`/其他 → unknown 列） | `sex_1`, `sex_2`, `sex_unk` | 3 |
| new | `education` | 连续（年） | 0 ≤ edu ≤ 26 | `z_edu`, `m_edu` | 2 |
| new | `illness` | 连续（月，见 §2.3-C1） | 数值可解析且 0 ≤ x ≤ 600；`'？'/'[]'/'-'` → 无效 | `z_log1p(ill)`, `m_ill` | 2 |
| new | `episode` | 类别 | 见 §2.3-C2：**需字典确认** | `epi_首发`, `epi_非首发`, `epi_unk` | 3 |
| new | `medication` | 类别（当前用药状态） | 见 §2.3-C3：**需字典确认** | `med_用药中`, `med_未用药`, `med_unk` | 3 |
| new | `HAMD` | 连续（原始总分） | 1 ≤ HAMD ≤ 58（可解析即可） | `z_HAMD`, `m_HAMD` | 2 |

派生宽度：`K_HC = 5`（demo 块），`K_MDD = 17`。
**前缀性约束**：`v_MDD[0:5] ≡ v_HC`（同一字段、同一列序、同一统计量），这是 §3 迁移能按列复制的前提。
连续性字段的 `[z, m]` 成对相邻；类别字段的 unknown 列固定在块尾。布局由 `SCHEMA` 列表推导，任何字段增删只能改 `SCHEMA` 并重算 `layout_hash`。

统计量口径（train-only，与 v1 §4.2 一致）：`z = (x̃ − median_train∧m=1)/s`，`s = IQR/1.349`，IQR 退化 → std，再退化 → 1；`m=0 ⇒ z=0`；`illness` 先 `log1p` 再取 train-only 中位数/IQR。

### 2.3 本次核实得到的语义证据（新增，决定哪些映射可进正式实验）

**C1 `illness` 单位 = 月（实证支持，可进正式实验）**
- V2 中同时有 `Age / AO / illness` 的 938 行：`corr(illness, Age−AO) = 0.887`，最小二乘斜率 **10.74 ≈ 12** → 若 illness 为月、(Age−AO) 为年，斜率应为 12。
- 不可能性检验：`illness/12 > Age` 违例 **0/938**（月口径自洽）；`illness > Age` 违例 **626/2253**（年口径不可能）→ 排除"年"。
- 非整数结构：118 个非整数值中 **72.9% 乘 30 后为整数** → 与"按天数 ÷30 折算成月"一致；(0,1) 区间取值恰为 `7/30, 9/30, 14/30, 28/30, 0.5, 0.7, 0.75, 0.996`。
- 结论：**按"月"处理**，并以 `log1p` 压制右尾（`max|z|` 14.03 → 2.12）。残留待确认：个别站点（`S21/IS018/S16/S22/IS006`）的非整数集中与 `.04/.76/.96` 结尾仍无解释 → 记录但不阻塞实验。

**C2 `episode`：合并列存在跨源语义冲突（需字典，暂不入正式实验）**
- V2 与 `Epi`（发作次数，仅审计用）交叉表：`episode==1(首发)` ↔ `Epi∈{0,1}`（781/827 在 `Epi==1`）；`episode==0` ↔ `Epi≥2`（180@2, 93@3, …，仅 8 例在 `Epi==1`）→ V2 侧 `0=非首发, 1=首发` 语义**自洽**。
- `episode==2`（19 例）与 `Epi∈{1,2,3,4}` 交叉重叠（4 例 `Epi==1`、9 例 `Epi==2`…），既不与首发也不与非首发单调对应，且这 19 例的 `AO` 全部缺失 → **第三级别含义无依据**。
- V1 侧画像（n=975）：`ep=-1` 组 illness 中位数 **72.0 月**、illness 缺失 2.9%、HAMD 19.22、med 分布 0.22/0.39/0.39；`ep=0`（源 `-9999`）组 illness 中位数 **7.0 月**、illness 缺失 **31.2%**、med 的 66% 也落在"源 `-9999`"列；`ep=1` 组 illness 中位数 8.0 月。
  → 判读：`-9999` 是**跨列共现的"未记录"哨兵**（与 illness 缺失强相关，且 `P(med==-1|ep==-1)=0.224 < P(med==-1|ep!=-1)=0.348`，说明 `-1` 不是同行级缺失）；V1 的 `-1` 更像**"长病程/非首发"的真实类别**（病程 6 年、记录完整），而不是缺失。
- 由此两条候选映射：

```
MAP-保守: 首发={ep==1}（两源一致）；非首发={V2 ep==0}；unknown={V1 ep∈{0,-1}, V2 ep==2, NaN}
          → cov 0.735（V1 0.455 / V2 0.948）  ← m_episode 中"缺失行 V2 仅 17.3%"→ 掩码成为来源代理，不可接受为主案
MAP-推断: 首发={ep==1}；非首发={V1 ep==-1} ∪ {V2 ep==0}；unknown={V1 ep==0(源-9999), V2 ep==2, NaN}
          → cov 0.828（V1 0.671 / V2 0.948）；缺失行中 V2 占 17.3%→（同上，仍偏 V1）
```
- **v2 裁定**：`episode` 属"映射依据不足"，**保留为待确认项，不进入正式七字段实验**；正式实验先跑 `SCHEMA-5F`（Sex/Age/education/illness/HAMD）。字典到位后一次性解锁 `episode`（并采用当时确认的映射，不再自行推断）。

**C3 `medication`：两种源列需要统一，且合并表的重映射方向可疑（需确认，暂不入正式实验）**
- 结构事实：合并 `medication` 列 **V1 100% 有值 / V2 100% 缺失**；`OnMedication` **V2 有值 / V1 全缺**；二者同时非空的行数 = 0 → 两列互斥，直接用合并列会让"缺失掩码"退化为数据集 one-hot。
- V2 侧语义可实证：`OnMedication × MedicationName` 交叉表给出 `1→有药名 369/562(66%)`、`0→有药名 138/433(32%)`、`NA→25/283(9%)` → `OnMedication` 确为**"当前是否用药"**（`0` 且有药名者属既往用药），与规范要求的"统一后的当前用药状态"一致。`EverMedication` 与 `OnMedication` 存在 136 例逻辑矛盾 → 不采用 `EverMedication`，不外扩用药字段。
- V1 侧 `-1` 的含义可用画像推断：V1 `med==-1` 组 Age 34.0 / illness 中位 9.5 月 / **77.3% 首发**；V2 `OnMedication==0` 组 Age 29.8 / illness 中位 9.0 月 / **80.4% 首发** → 两者高度相似，提示 **V1 的 `-1` ≈ "未用药"**；而 V1 的 `0`（源 `-9999`，66% 与 episode 未记录共现）≈ **"未记录"**。
  → **这与合并表当前的重映射方向相反**（合并把 `-9999→0`、保留 `-1`），使"未记录"占据了本应属于"未用药"的取值 0。
- 建议映射（待确认）：`MAP-med: V1 {-1→未用药, 1→用药中, 0(源-9999)→unknown} ∪ V2 OnMedication {0,1}` → cov **0.719**（V1 0.640 / V2 0.779），缺失行中 V2 占 44.6%（基线 56.7%）→ 耦合明显弱于合并列的完全共线，但仍需报告。
- **v2 裁定**：同 `episode`，列为待确认项，不入正式实验；核实问题是"V1 `-1` 与 `-9999` 的原始码表含义"。

**C4 `HAMD`（按用户裁定使用原始总分；风险转入报告层）**
- V1 `HAMD == Σ(*HAMD1..17)` 完全一致 486/486；V2 中 `Σ(HAMD1..24)` 完整行 482/482 与总分一致，但 `(HAMD − Σ_{1..17}) == Σ_{18..24}` 仅在 **51.4%** 的行成立（差值中位 1、均值 3.24、最大 20）→ V2 条目与总分**内部不自洽**。
- 分组差异：V1 均值 21.03(SD 6.88)、V2 22.78(SD 7.80)，差 1.75 ≈ **0.23 合并 SD**；另有 35 行 HAMD ≤ 3（14 行 =1）。
- 处理：原始 `HAMD` 进模型（robust_z，train-only），**不加任何来源指示列**；改为①结果按 `Dataset` 分层报告、②待确认项记录量表版本口径、③若分层显示 HAMD 效应方向在两源相反，则该字段收益判定为不可信（预注册）。

**C5 `Sex` 匿名编码（可进正式实验，不必等待标签解释）**
- HC 影像匹配队列 `Sex` 仅 {1,2}（train/val/test = 716/1043、100/116、110/110），MDD 亦仅 {1,2}（+1 例缺失）→ one-hot{1,2,unknown} 安全，模型只学"类别 A/类别 B"的差异。
- 约束：结论与图表措辞一律写 `Sex level 1 / level 2`，**不得**写"男/女"；全仓无数据字典可支撑方向。

### 2.6 证据分级与正式实验准入

| 字段 | 证据状态 | v2 准入 |
|---|---|---|
| `Age`, `education`, `Sex` | 直接可解析、覆盖 ≥99.8%，无需语义推断 | ✅ 正式 |
| `HAMD` | 原始总分可用；量表口径混用属已知风险，转报告层控制 | ✅ 正式（附分层要求） |
| `illness` | 单位"月"由 §2.3-C1 实证支持 | ✅ 正式（log1p） |
| `episode` | 第三级别含义无依据；两源 `0` 语义冲突 | ⛔ 待确认，不入正式实验 |
| `medication` | 需统一两源列；V1 哨兵映射与合并表相反 | ⛔ 待确认，不入正式实验 |

→ **`SCHEMA-5F`（当前可执行正式协议）** = `Sex, Age, education, illness, HAMD`，`K = 5 + 6 = 11`，`v_new` 列数 = 6。
→ **`SCHEMA-7F`（待确认解除后启用）** = `K = 17`，`v_new` 列数 = 12。
两协议共用同一 demo 前缀、同一迁移路径、同一 `C=32`；`7F` 只是在 `v_new` 上多 6 列并新增两个类别块，因此**不重跑 5F 的任何臂**，`7F` 相对 `5F` 的差即为 `episode + medication` 的边际信息检验。

---

## 3. 唯一迁移路径（初筛采用；消除方法与迁移的混杂）

```
共享编码器（HC 训练，MDD 继承）:
  v_demo ∈ R^5          → c_HC = E_demo(v_demo) ∈ R^32
  E_demo = Linear_2( GELU( Linear_1(v_demo) ) )      C = 32

MDD 增量融合（新增，输出层零初始化）:
  v_new ∈ R^{K-5}       → c_MDD = c_HC + ΔE( [v_new ; c_HC] ) ∈ R^32
  ΔE_linear = W_out [v_new ; c_HC] + b_out                              （初筛臂 M2）
  ΔE_mlp    = Linear_2( GELU( Linear_1([v_new ; c_HC])) )               （初筛臂 M3）
  两者末层 W=0, b=0  ⇒  训练起点 c_MDD ≡ c_HC

双头 FiLM（继承 HC 权重，禁止重新清零）:
  [γ_ℓ; β_ℓ] = W_ℓ c + b_ℓ ,  W_ℓ ∈ R^{512×32},  ℓ ∈ {OS, CPM}
  h' = (1+γ) ⊙ h + β    （位置与 v1 §6 完全一致，本轮不换位置、不进 backbone）
```

规则（全部写成可执行校验）：

1. **继承不重零**：`E_demo`、`W_OS/b_OS`、`W_CPM/b_CPM` 从 HC checkpoint **原样复制**；`W_new` 新增列零初始化；HC 已训练的 `b` 段保持不变。若 HC 为无条件初始化（INIT-A，见规则 4），这些模块按"新建 + FiLM 零初始化"处理，此时初始行为等价于 HC 无条件模型（同一等价校验）。
2. **统计量继承**：`Age` 的 `center/scale` 沿用 **HC-train** 拟合值（保证 `E_demo` 输入含义不变）；`education/illness/HAMD` 由 **MDD-train** 拟合。重新用 MDD 拟合 `Age` 尺度 → 单独消融臂，且必须放弃规则 1 的等价校验。
3. **初始行为等价校验（硬验收）**：同一 HC batch 下
   `max|x̂_HC(c) − x̂_MDD,init(c)| < 1e-6`（One-Step raw 与 CPM raw 两路都要过），
   以及 `‖ΔE 输出‖_∞ = 0` 在初始化时逐位成立。
4. **初始化族预注册（不看结果再选）**：
   - 若 HC 条件臂（H2）相对 H0 在验证集上通过 §7 效应门槛 → MDD 三臂统一用 **INIT-B**（H2 胜出的 checkpoint）；
   - 否则统一用 **INIT-A**（HC 无条件 checkpoint，MDD 侧新建 `E_demo`+FiLM，FiLM 零初始化）。
   两族各自内部三臂完全同初始化、同预处理、同训练设置 → 满足"同一 HC checkpoint"的要求，且不存在事后挑选。
5. **不引入**：分组专家、RECC 耦合损失、softmax 竞争门控、backbone 注入、`future_control`（`cpm_intervention` 保持 False）。

术语纪律：`education/illness` 等"仅在 MDD 阶段加入"的字段属**群体特异新增条件**，不得因加入阶段而称病理因素；机制/因果类表述一律禁用（人口学收益不解释为机制，调整 `HAMD`/`medication` 后的变化不解释为疗效）。

---

## 4. 参数量与状态字典（v2 重算，`C=32` 统一）

FiLM 两头 = `2×(512×32 + 512) = 33,792`；`E_demo = (5×32+32) + (32×32+32) = 192 + 1056 = 1,248`。
`ΔE` 的输入宽度 = `|v_new| + C`（5F: `6+32=38`；7F: `12+32=44`），不是 `K`：

- `ΔE_linear(5F) = 38×32 + 32 = 1,248`；`ΔE_linear(7F) = 44×32 + 32 = 1,440`
- `ΔE_mlp(5F) = 1,248 + (32×32+32) = 2,304`；`ΔE_mlp(7F) = 1,440 + 1,056 = 2,496`

| 臂 | 条件模块 | 条件参数量 | 模型总参数量 |
|---|---|---|---|
| H0 HC 无条件 | 无 | 0 | 6,027,680 |
| H2 HC `Sex+Age` 编码器+FiLM | `E_demo` + 2×FiLM | **35,040** | 6,062,720 |
| M1 MDD 仅人口学（D0） | 同 H2（继承） | 35,040 | 6,062,720 |
| M2 MDD 5F 线性增量+FiLM | + `ΔE_linear(5F)` | 36,288 | 6,063,968 |
| M2 MDD 7F 线性增量+FiLM | + `ΔE_linear(7F)` | 36,480 | 6,064,160 |
| M3 MDD 5F MLP 增量+FiLM | + `ΔE_mlp(5F)` | 37,344 | 6,065,024 |
| M3 MDD 7F MLP 增量+FiLM | + `ΔE_mlp(7F)` | 37,536 | 6,065,216 |
| 现状归档（C=1，单 HAMD） | 2×FiLM | 2,048 | 6,029,728 |

要点：

- **M1↔M2↔M3 的参数量差只有 0–2,304（≤0.038% 主干）**，因为 `C=32` 已统一、FiLM 宽度不变 → 三臂比较几乎是容量匹配的，差异主要来自"新增列是否被使用"而不是"多了一堆参数"。这正是 v2 相比 v1（`C∈{11,17,32,64}` 混排）的关键改进。
- 静态等容量对照 `S1`：架构与 M2 完全相同，但把 `v` 替换为 **train 列均值常向量 v̄**（对全体被试相同、不参与梯度）→ 参数量逐位相同，个体差异为 0。`条件信息收益 = M2 − S1`。
- state_dict buffer：`PathologyNormalizer` 仍用于连续块（`input_dim = n_cont`），buffer 元素数 = `2·n_cont + 64·n_cont + 8·n_cont + n_cont + 1` → HC `n_cont=1` 为 76（与现状一致）；5F `n_cont=4` 为 **301**；7F 仍 `n_cont=4` → 301（`episode/medication` 是类别块，不占 normalizer）。注意 `quantiles/knot_*` 在 `robust_z` 模式下也照样分配（现状行为，不改），因此 buffer 形状必须进 schema 哈希。
- 定型件不动：`configs/final_model.json`（`total 6029728 / film_total 2048`）、`inference.py:125-173` 三道闸门、`tests/test_final_model.py:29-65,78-81,101,196-204` 保持原样，实验一律写独立 `configs/condition_v2_*.json`。

---

## 5. 队列与划分冻结（修正样本数）

| 项 | HC | MDD |
|---|---|---|
| 临床表行数 | 2445（唯一 ID 2445，重复 0） | 2490（唯一 ID 2490，重复 0） |
| 影像（BOLD+SC）被试 | 2222 | 2253 |
| **条件实验队列** | **2195**（imaged ∧ 有人口学；剔除 `S4-2-0001..0024`、`IS005-2-0057..59`） | **2253**（全部 imaged；缺失用掩码不删人） |
| 由限制得到的划分 train/val/test | **1759 / 216 / 220** | 1803 / 225 / 225（不变） |
| 分层键 | `Age` 分 5 箱（HC 无 HAMD；现 pretrain 走纯随机分支 `dataloader.py:1049-1054`） | **保持 `HAMD`**（覆盖率 1.000）→ 现 manifest 成员归属无需重排 |

字段覆盖率（v2 映射，MDD）：

```
train(n=1803) Sex .999 Age .999 edu .999 illness .817 HAMD 1.000 | ep(保守 .828)
val  (n= 225) Sex 1.000 Age 1.000 edu .996 illness .804 HAMD 1.000
test (n= 225) Sex 1.000 Age 1.000 edu 1.000 illness .831 HAMD 1.000
```

完整样本敏感性队列（预先声明其一，不得看结果后改）：`5F 全有效 = 1839`；`7F 保守映射 = 1289`；`7F 推断映射 = 1413`。v2 默认 **1839（5F）**，待确认项解除后再声明 7F 队列。

实现要求（当前缺失）：新增 `--split_file`，读取本研究的冻结划分（含 `cohort_ids + train/val/test + 字段值快照 sha256`），由现有 manifest **限制（restrict）**而非重排得到，确保与归档结果的 train/val/test 成员可比。同时把 `fields/encoding_version/layout_hash` 加入 `dataloader.py:988-990` 的指纹，使换编码必然触发校验失败而非静默复用。

---

## 6. 阶段 0 文件改动清单（逐文件、逐函数；本轮不执行）

### 6.1 数据与预处理

| 文件:位置 | 改动 |
|---|---|
| `utils/dataloader.py:17-51` `load_clinical_scores` | 拆成 `read_clinical_raw()` + `apply_condition_schema()`；**删除整表 `fillna(mean)`/`fillna(0)`**（`:38-44`），缺失只保留 NaN 与 `valid` 位；`drop` 不再在划分前删人（`:231-233`） |
| `utils/dataloader.py:155-170` | 新增 `SCHEMA` 驱动的 `condition_fields` 解析；pretrain 分支允许读 HC 表（`--clinical_file` 已存在，`:125`/`main.py:870`，需按 mode 校验默认与列存在性）；越界字段 fail-fast |
| `utils/dataloader.py:231-247` `_build_index` | `subject_to_score` 的分层键由 `fields[0]`（`:245-247`）改为显式 `--stratify_field`（MDD=`HAMD`，HC=`Age` 分箱）；队列改为"imaged ∧ 条件字段可读" |
| `utils/dataloader.py:387-389, 517-521` | 返回三元组：`cond_raw [n_cont]`、`cond_valid [n_cont]`、`cond_code [n_cat]`（整型索引），替换单一 `pathology_score`；保持旧键向后兼容（旧臂仍可用） |
| `utils/dataloader.py:982-1030, 1046-1075` | manifest 指纹加入 `fields/encoding_version/layout_hash/cohort_hash`；支持 `--split_file` 与 restrict 语义 |
| `utils/dataloader.py:1180-1196` | 泛化为 `get_train_condition_table()`（train-only 拟合来源），并暴露类别词表与 `v̄` 常量 |
| 新增 `utils/condition_schema.py` | `SCHEMA`（字段/类型/边界/词表/列序）、`build_layout()`、`layout_hash()`、`encode(vocab, stats, raw)`、字段白名单与 `assert_field_allowed()` |

### 6.2 模型

| 文件:位置 | 改动 |
|---|---|
| `models/pathology.py` | 新增 `ConditionEncoder`（持有 `PathologyNormalizer(input_dim=n_cont)` + 类别 one-hot + `E_demo` + `ΔE`）；从 HEAD 恢复 `AdaLNConditioner`/`LowRankDelta` 仅作 §8/§9 推迟阶段的参考实现 |
| `models/tfm.py:415-420, 438-449` | 允许 `pretrain_mode` 在显式开关下建条件模块（默认关闭，现行为不变）；`cond_dim` 由编码器 `out_dim=C` 给出而非 `normalizer.out_dim` |
| `models/tfm.py:481-485, 582-586` | `_prepare_condition` 接三元组；把"pretrain 收到条件即 `raise`"改为"pretrain 且**未建**模块才 raise" |
| `models/tfm.py:177-184, 286-291` | 调制位置与公式**不动**（本轮只换 `c` 的来源与宽度） |
| `models/__init__.py:21` | `ARCH_VERSION` 4 → **5**（依 `:6-7` 自有规则：参数集合/语义变化必须递增） |

### 6.3 迁移与训练闸门

| 文件:位置 | 改动 |
|---|---|
| `train/optim.py:206-219` | 保留 shape 过滤但**强制清单**：`loaded / copied / zero_expanded / skipped / missing` 全量打印并写入 ckpt meta；未归类键 → `raise`；`main.py:451` 不再丢弃返回值 |
| `train/optim.py:84-107, 121` | `pretrain_config_signature` 加入 `cond_fields/cond_encoding_version/cond_dim/tfm_patho_mode/layout_hash` |
| 新增 `train/optim.py:expand_condition_columns()` | 按 `layout_hash` 做 `W[:, :K_old] ← W_old`、新列置 0、`b` 不重置；含 `E_demo` 与 `normalizer` buffer 的同类扩展 |
| `train/optim.py:307-324`（尤其 `:317-321` 硬编码白名单） | 解冻集合改为"非条件模块全集 − 显式冻结清单"，消除"新增顶层模块被永久静默冻结" |
| `main.py:289-295` | 取消无条件强制 `none`，改由 `--cond_enable_in_pretrain` 决定；`--tfm_recc_mode` 仍强制 off（初筛不涉及专家） |
| `main.py:263-281, 465-466` | 拟合入口改为 `fit_condition_encoder()`（train-only；HC 也调用）；打印 `layout_hash`、每字段 `median/s/n_valid` |
| `main.py:313-317` | 默认设备不再硬编码 `cuda:1`；`--device` 必填或由环境变量决定 |
| `main.py:468-469` | 增加逐组参数量打印（`adapter` / `cond_encoder` / `film_os` / `film_cpm`） |
| 新增 CLI | `--cond_schema {5F,7F}`、`--cond_hidden 32`、`--cond_method {demo_film, delta_linear, delta_mlp}`、`--cond_static`（S1 常量条件）、`--cond_permute`、`--split_file`、`--stratify_field`、`--cond_enable_in_pretrain` |

### 6.4 评估与诊断

| 文件:位置 | 改动 |
|---|---|
| `analysis/next_point_eval.py:727, 1132-1141, 812-828` | 去掉"条件为标量"假设（`reshape(-1)` 会把 `[N,11]` 摊平成 `11N`）→ 改存 `cond_vectors [N,K]` + `valid`；`_hamd_association` 改为按字段矩阵；CSV 增加 `dataset/site` 列仅用于**事后分层** |
| 新增 `analysis/condition_diagnostics.py` | L1 敏感度（固定 BOLD/SC 改 `c`）、L2 使用度（`‖γ‖_∞/‖β‖` 与置换后输出漂移）、置换（整组/新增组/逐字段；`perm_value_keep_mask` 与 `perm_joint_value_mask` 分别定义）、掩码-only 臂、`Dataset` 分层 |

---

## 7. 阶段 0 验收清单（可执行判据；合成数据 + 少量 step，不启动完整训练）

| ID | 验收 | 判据（全部机器可判） |
|---|---|---|
| `P0-1` 无数据泄漏 | train-only 统计量 | 篡改 val/test 的字段值 → `center/scale/词表/v̄` 逐位不变；`load_clinical_scores` 不再在任何路径用全体统计量填补（grep 断言） |
| `P0-2` 队列与划分冻结 | 限制而非重排 | 1759/216/220 与 1803/225/225 的成员 ⊇/⊆ 归档 manifest 一致；`layout_hash/cohort_hash` 写入 meta；换字段触发 fail |
| `P0-3` 布局唯一来源 | schema 自动生成 | `build_layout(SCHEMA)` 与 `layout_hash` 一致；人为交换两列 → 加载报 `layout_hash` 不匹配 |
| `P0-4` HC 条件通路打通 | pretrain 建模块 | `mode=pretrain ∧ --cond_enable_in_pretrain` → `cond_dim=32`、`E_demo/FiLM` 存在、前反向不报错、传入三元组不 raise |
| `P0-5` 双头注入生效 | One-Step 与 CPM 各自独立 γ/β | `f(c=0) − f(cond=None) > 1e-6`（证明"喂零 ≠ 无条件"）；两头的 `W_ℓ` 梯度非同一张量 |
| `P0-6` 迁移初始行为等价 | §3 规则 1/3 | `max|x̂_HC − x̂_MDD,init| < 1e-6`（OS 与 CPM 两路）；`‖ΔE_out‖_∞ == 0`；`b_OS/b_CPM` 与 HC 权重逐位相等 |
| `P0-7` 无静默跳过 | §6.3 | 构造 `C=32→C=32` 但字段顺序变化 → 必须 `raise`；构造 5F→7F → `zero_expanded` 计数正确且 `skipped==∅` |
| `P0-8` 新模块确实被训练 | 冻结期 | `set_finetune_stage(unfrozen=False)` 后可训练集合 == 条件模块全集（逐名比对）；`get_param_groups()` 含 `adapter` 组且 numel == 记录值；不带标记的新模块 → 测试应失败（防"永久静默冻结"回归） |
| `P0-9` 条件路径可激活 | ≤200 step 合成训练 | `‖W_film‖` 由 0 变 >1e-3；记录 OS 与 CPM **首次非零 step**（已知 OS 受 `decode` 零初始化门控，`tests/test_tfm.py:175-179` 断言其首步梯度为 0）；若 OS 在 200 step 内未激活 → 判为通路问题，不得进入初筛 |
| `P0-10` checkpoint 严格重建 | 保存→加载 | `load_state_dict(strict=True)` 无 missing/unexpected；`center/scale/词表/fitted` 与编码 buffer 形状逐位一致；重建后同 batch 输出逐位一致 |
| `P0-11` 参数量与记录一致 | §4 表 | 各臂 `cond_params` 与 §4 数值逐位相等（35,040 / 36,288 / 37,344（5F）；36,480 / 37,536（7F））；总参数量匹配 |
| `P0-12` 评估侧多字段安全 | §6.4 | `K=11` 时 `cond_vectors.shape[0] == n_anchors`，CSV 行数 = 被试数；旧 `hamd` 列语义明确（字段 0）或移除 |
| `P0-13` 字段白名单 | 越界拦截 | 传入 `HAMA/HAMATotal/HAMD3/MedicationName/Epi/AO/Dataset/OnMedication` → `raise`，错误信息含被拒字段名 |
| `P0-14` 回归不破坏 | 现有测试 | `python -m pytest tests/ -q -p no:cacheprovider`；基线 **3 failed / 79 passed / 13 skipped**（3 个失败是 `inference.py:96` 先解析权重路径、权重已迁至 `checkpoints/Version3_weights_and_results_1003/`，属环境路径而非逻辑）→ 新增测试只增不减，且不引入新失败 |
| `P0-15` 定型件隔离 | 不受影响 | `configs/final_model.json` 与 `inference.py` 未被修改；无条件旧臂（`C=1`）在新代码下输出与归档逐位一致（bit-exact 回归） |

阶段 0 全部通过后才进入初筛；`P0-1..P0-8` 任一失败即停止并回到 §6 修实现，不得用"结果看起来还行"绕过。

---

## 8. 最小初筛矩阵（阶段 0 之后；每格 1 个预指定种子）

| 阶段 | 臂 ID | 设置 | 参数量 | 说明 |
|---|---|---|---|---|
| HC | `cond_hc_none` | 无条件 | 0 | 复用归档 `base_best.pt`（sha `667bf1ab…`）前先核 `config.args`；不一致则重跑 |
| HC | `cond_hc_demo_enc32` | Sex+Age → `E_demo(C=32)` → 2×FiLM | 35,040 | H2 |
| MDD | `cond_mdd_demo` (M1/D0) | 仅人口学条件（**不等于全关**） | 35,040 | 同初始化族 |
| MDD | `cond_mdd_5f_linear` (M2) | 5 字段线性增量融合 + FiLM | 36,288 | 7F 解锁后同名加 `_7f` 变体（36,480） |
| MDD | `cond_mdd_5f_mlp` (M3) | 5 字段小型 MLP 增量融合 + FiLM | 37,344 | 同上（7F: 37,536） |
| 对照 | `cond_static_const` (S1) | M2 架构 + `v ← v̄`（train 列均值常向量） | 36,288 | 与 M2 **逐位同容量**，去个体差异 |
| 对照 | `cond_all_none` | 全条件关闭（不建模块 / `cond=None`） | 0 | 与 M1 严格区分命名 |
| 诊断 | `diag_perm_newgroup` | 新增组（edu/illness/HAMD）被试级联合置换 | — | 纯推理 |
| 诊断 | `diag_perm_maskonly` | 只留掩码/unknown 列，观测值置 train 中位数 | — | 纯推理；掩码收益不得计入临床信息 |
| 诊断 | `diag_perm_field` | 逐字段置换 | — | 报告须标注"制造不合理组合，不可作独立贡献解释" |

约束与规则：

- M1/M2/M3 同 §3 规则 4 预注册的**同一 HC checkpoint**、同一预处理、同一训练配方（§v1 12.1 的全清单），只在 `ΔE` 形式与字段集上不同。
- 置换单位为**被试**，同一被试所有窗口共享同一置换条件；`perm_value_keep_mask` 与 `perm_joint_value_mask` 分别报告。
- 统计：被试级配对 + bootstrap（≥5,000 次重采样被试）95% CI；主指标 = 验证集 one-step 被试级 MAE（`dynamics_mode=discrete` 下它仍是 checkpoint 判据，故与本方案的选优同数；2026-10-15 起 `graph_ode` 路线的判据已改为验证集 `metric_mae_dynamics_h1_N_subject`——同一次前向的 Graph-ODE 轨迹在 H1..N × 全部 ROI 上的逐锚点 MAE，只纳入 `future_mask` 具备完整 N 步真值的锚点，先被试内平均再跨被试等权，N=`--dynamics_metric_horizon` 默认 8——若改用 graph_ode 执行本方案，晋级须按该多步指标核对，不能再默认与本主指标同值）；护栏 = CPM H8 MAE、rollout H16 MAE、`variance_ratio`、FC edge PCC（恶化 >2% 不得晋级）；效应门槛 ≥1% 且 CI 不跨 0 且 ≥2/3 种子同向（依据历史噪声量级：`tfm_no_cond` vs FiLM −0.04%、MoE 各臂 0.10–0.29%）。
- 结果必须按 `Dataset(V1/V2)` 分层报告（§2.3-C4 的 HAMD 口径要求）；分层显示两源效应方向相反时，判该字段收益不可信（预注册）。
- 1 个种子只用于可行性与量级判断；**晋级方法**再用 ≥3 个预先指定种子复核（在结果可见前写定种子）。
- 预算：8 次训练 × ≈5.5 GPU·h ≈ **44 GPU·h**（实测单卡 150-epoch 微调 246–797 分钟，中位约 330 分钟，`tfm_baseline`=327）；诊断与评估为纯推理 ≈6 次 × 0.4 h。
- 门控残差（v1 §8）与条件门控低秩（v1 §9）**本轮不做**，其开工条件：初筛中 M2 或 M3 通过效应门槛且 `M − S1` 显著 > 0（即收益来自条件信息而非容量）。

---

## 9. 引用与数字修正对照（v1 中被替换的表述）

| v1 表述 | v2 更正 | 依据 |
|---|---|---|
| `K_MDD = 18`（含 `m_HAMD_scale_anon`） | `K_MDD = 17`；`SCHEMA-5F` 下 `K = 11` | 用户裁定删除来源列 |
| 完整样本 1488 | 5F=1839 / 7F 保守=1289 / 7F 推断=1413（默认 1839） | §5 实测 |
| 分组切片 `bg[5:7] course[7:13] med[13:16] sym[16:18]` | 删除；`v_new = v[5:K]`（5F: `v[5:11]`） | §3 唯一迁移路径 |
| 参数量"39,152 / 36,000 / ≈49k" | 重算：35,040（H2/M1）/ 36,288 / 37,344（5F M2/M3）/ 36,480 / 37,536（7F） | §4 |
| `episode` 保守映射为主案 | 降为待确认；主案只含证据充分字段 | §2.3-C2 + 掩码-来源耦合量化 |
| `illness` identity 与 log1p 等价 A/B | log1p 为主案（`max|z|` 14.03→2.12） | §2.3-C1 |
| `inference.py:161-171` | `inference.py:160-173`（参数量闸门）；FiLM 闸门 `:131-138`；strict 加载 `:142-152`；`is_fitted` `:155-158`；权重路径先解析 `:96` | 逐行回读 |
| `train/optim.py:206-229` | `:206-219`（过滤 207-208、skipped 217、load 219）；`_iter_conditioning_modules:283-295`；`set_finetune_stage:307-324`（白名单 317-321）；`get_param_groups:327-366`；`count_trainable_params:393`；`pretrain_config_signature:84-107`；`check_pretrain_config_compat:121`；`save_backbone_weights:68-81`；`BACKBONE_KEY_PREFIXES:18-20` | grep 定位 |
| `experiments/evaluate_variant.py:86-105` | `:79-101`（`build_model_from_args` + `inspect.signature` + `参数快照缺少必填构造参数`） | grep 定位 |
| "`models/pathology.py` 有 AdaLN/LoRA" | 澄清：**当前工作树无**（仅 `PathologyNormalizer`）；两者存在于 git HEAD 旧版（`HEAD:models/pathology.py:198-265`、`:267-317`），随 legacy 主干删除 | 已直接读取 HEAD 版本 |
| "HAMD 是 17/24 混用总分" | 精确化：V1 = 17 条目和（486/486）；V2 的 24 条目和与总分一致（482/482）但 `总分−Σ_{1..17} == Σ_{18..24}` 仅 51.4% 成立 → 条目层不自洽；V1/V2 均值差 1.75 = 0.23 SD | §2.3-C4 实测 |

---

## 10. 待确认项（阻塞 7F 解锁）

| # | 需向数据字典/采集方确认 | 当前可执行分支 | 解锁效果 |
|---|---|---|---|
| 1 | V1 `If first episode?` 的 `-1` 与 `-9999` 各代表什么（本方案据病程 72 月 vs 7 月、记录完整度推断 `-1`=非首发、`-9999`=未记录） | `episode` 不入模型；`SCHEMA-5F` 跑初筛 | 7F 启用，并直接检验 `episode` 边际信息 |
| 2 | V2 `FirstEpisode=2`（19 例，`AO` 全缺）含义 | 同上（若启用则归 unknown） | 类别词表是否需第四类 |
| 3 | V1 `On medication?` 的 `-1 / 1 / -9999`（本方案据首发率 77.3% vs V2 的 80.4% 推断 `-1`=未用药、`-9999`=未记录） | `medication` 不入模型；**绝不使用合并 `medication` 列**（其与 `Dataset` 完全共线） | 7F 启用统一"当前用药状态" |
| 4 | `HAMD` 量表版本口径（17 项 vs 24 项，及条目不自洽的 51.4%） | 使用原始总分 + `Dataset` 分层报告 | 若为混用需决定是否限用单一子量表（属字段内变换，仍是预处理） |
| 5 | `illness` 五站点的非整数值集中（`.04/.76/.96`）是否有额外换算 | 已按"月"+log1p 处理，截断策略记录在案 | 可选改为显式单位换算 |
| 6 | `Age/education` 的合法边界（HC 出现 `Age=0`、`Sex=0` 各 1–3 例，均不在影像队列） | §2.2 边界（18–90 / 0–26）为暂定值 | 若有源表约定则替换 |
| 7 | 合并表 10-08 改动与 09-26 冻结划分的关系 | **强制双哈希**：`cohort_hash + 字段值快照 sha256` 写入 registry 与 ckpt meta；若分层所用 HAMD 分布已变 → 声明这是新划分并重跑所有臂 | 结果可比性的唯一保障 |

---

## 11. 自检（v2）

| 项 | 状态 |
|---|---|
| 字段是否越界 | 未越界；`Epi/AO/OnMedication/MedicationName` 仅用于审计取证，白名单测试 `P0-13` 强制拦截 |
| 预处理是否 train-only | 是，且 `P0-1` 以"篡改 val/test 值而统计量不变"为判据；明确删除现状的划分前整表填补 |
| HC 条件是否真正接入 | 给出 `P0-4` 的具体打通路径（建模块 + 放宽 raise + 读表 + train-only 拟合 + 队列 2195 与划分 1759/216/220） |
| 迁移是否保持预期初始行为 | `ΔE` 末层零初始化 ⇒ `c_MDD ≡ c_HC`；FiLM/编码器继承不重零；`P0-6` 以 `max|Δ|<1e-6` 验收 |
| 新增参数是否实际训练 | `P0-8`（标记 + 参数组 + numel 逐名比对）与 `P0-9`（激活）双重验收 |
| 不同方法是否公平 | 统一 `C=32` → M1/M2/M3 容量差 ≤0.03%；同一预注册初始化族；护栏与门槛在看结果前写定 |
| 条件信息与额外容量是否区分 | `S1` 常量条件臂与 M2 **逐位同参数量**；`条件收益 = M2 − S1`；另有置换与掩码-only 诊断 |
| 缺失模式混杂是否检查 | 量化 `P(V2 | 缺失)`：illness 0.692 / episode 保守映射 0.173 / medication 合并列 1.000（故禁用）→ 掩码-only 与来源分层为硬要求 |
| 测试集是否保持独立 | 全部选择规则只在验证集；配置冻结后才跑 test；`P0-15` 保证定型件与旧臂不受影响 |
| 是否具备实施与验收路径 | §6 文件级清单 + §7 十五条可执行判据 + §8 预算 44 GPU·h |
| 公式是否与调用链一致 | `z_t → z_g=A_eff z_t → h=T([z_t,z_g]) → h'=(1+γ)⊙h+β → z_{t+1}=z_t+h' → Δ̂=Decode(z_{t+1}) → x̂_norm=x_{t,norm}+Δ̂ → raw`；CPM 为查询调制后交叉注意力直接输出完整状态（无残差锚点）；未混淆三个空间 |
| 是否已修改代码/启动训练 | **未修改任何代码与配置，未启动训练**；审计脚本置于 `/tmp/cond_audit/`，未入库 |
