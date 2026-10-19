# NeuroTwin-TFM 引入 Looped SC-Guided Transformer 的详细研究与实施方案

> 版本：V1.0 · 2026-10-08（上海时间）  
> 适用仓库：<https://github.com/xlongwu/NeuroTwin>；适用分支：当前默认分支的 **NeuroTwinTFM / TFM-only** 实现  
> 任务：HC 预训练 + MDD/HAMD 病理条件微调；next-timepoint 和 CPM 多步预测  
> 本文性质：**拟实施研究方案，不表示仓库已经修改或模型效果已经验证**。所有数值阈值、搜索范围和验收门槛，除标为“代码现状”外均为研究设计起点。

## 0. 执行摘要与推荐路线

**核心判断：优先把 TFM 的 4 个独立 `TFMEncoderBlock` 改造成可配置的参数共享循环主干，而不是恢复已移除的 GraphODE 或将每个时间点做 RNN 式循环。** 引入 Loop 的研究目标有三个：①同等计算深度下提升预测精度/参数效率；②允许根据预算选择循环深度；③建立可审计的“逐轮表征与预测修正”分析，而不是将内部计算步当成真实 BOLD 时间步。

推荐顺序：

1. **P0 基准审计**：冻结现行 TFM 的数据协议、参数量、时间、结果与代码版本。
2. **P1 结构筛选**：比较 4 层非共享、1 层共享×4、2 层共享×2、1+共享×2+1、同参数量单次模型；仅保留真正有收益的方案。
3. **P2 循环条件化/稳定性**：添加虚拟迭代时间 `τ`、步长 `Δτ`、小幅残差更新与过平滑诊断。
4. **P3 弹性深度**：参考 LoopFormer 做 full/shortcut 双轨迹训练与一致性约束；独立验证不同预算。
5. **P4 自适应停止**：先做离线停止规则和验证集阈值搜索，再考虑在线动态停止。
6. **P5 神经可解释性与临床分析**：逐轮预测修正、SC 依赖、功能网络富集、跨种子稳定性、删除/置换忠实度、HAMD 协变量分析。
7. **P6 MDD 微调与拓展**：保留现有 HAMD→FiLM 路径作主实验；与 RECC/MoE 设计组合仅为后续独立实验，不能作为 Loop 必须依赖的组件。

**Go/No-Go 总原则**：如果固定 4 次参数共享循环在 3 种种子、严格相同 subject split 下不能达到验证集预测指标非劣且显示明确效率优势，则停止 P3/P4，不因可视化“好看”而继续叠加模块。解释性结论必须通过扰动忠实度和稳定性测试。

---

## 1. 已核对的当前代码基线（实施前重新对照 Git commit）

### 1.1 真实结构（重要：不是 Legacy GraphODE）

当前 `models/tfm.py` 中 `NeuroTwinTFM` 的基础数据流：

```text
BOLD context [B,F,1,K]
   │  F=116（AAL116），K∈[16,64]
   ▼
Context-only RevIN（由 --norm 控制）
   │
   ├── SoftAnatomicalPrior(context, SC) → A_eff [B,F,F]（只算一次）
   ▼
TemporalPatchEmbed → tokens H0 [B,F,P,D]
   ▼
TFMEncoderBlock θ1 → θ2 → θ3 → θ4（独立参数）
   ▼
final_norm → context [B,F,P,D]
   ├── OneStepDynamicsHead：最后 patch token、A_eff 图混合、状态转移、x_t 残差锚点
   └── CPMHorizonHead：ROI×horizon 查询，对历史 token 做 cross-attention，直接输出 H 步
   ▼
HC：通用动力学；MDD：HAMD 经 PathologyNormalizer → One-Step/CPM FiLM
```

输入输出约定：

- `x`: `[B,F,1,K]`；`sc`: `[B,F,F]` 或受支持的广播形式；`pathology_score`: `[B,C_raw]` 或 `None`。
- `forward` 主预测 `[B,F,1,1]`，附加字典中 `cpm_pred` `[B,F,H]`、`sc_prior` 等。
- 预训练 **HC**，微调 **MDD**。当前真实可执行条件分支是 **FiLM**；仓库中的 RECC/MoE 目前是一份方案文档，而不是应在本计划中假设可调用的模块。
- `SoftAnatomicalPrior` 的 `A_eff` 在主干各层及 One-Step 头复用；不用每轮重算或让各轮各有 SC 估计，否则同时改变图学习与循环机制。
- `TFMEncoderBlock` 是 PreNorm 的 temporal self-attention → SC-guided ROI attention → FFN 三个残差子层；`SCGuidedROIAttention` 中通过逐头 `β log(ε+A_eff)` 注入 SC 软偏置。

**脚本基线**（实际训练请以 `scripts/*.sh` 为准，CLI 默认与脚本可能不同）：

| 项目 | 预训练/微调脚本参考设置 | 备注 |
|---|---:|---|
| `tfm_dim` | 256 | 需被 `tfm_heads` 整除 |
| `tfm_layers` | 4 | 当前 4 组独立权重 |
| `tfm_heads` | 4 | temporal 与 ROI attention |
| `tfm_patch_len` | 4 | CLI 单独默认值为 8；**不要混用** |
| `tfm_ff_ratio` | 4 | FFN 维度 1024 |
| `context_min/max` | 16/64 | 动态 context、每批分桶 |
| `cpm_horizon` | 8 | 原头暂不改 |
| `lambda_one / lambda_cpm` | 1.0/1.0 | 保持现行 `TFMDualLoss` |
| `lambda_pcc` | 0.1 | ROI 维 spatial PCC |
| `batch_size / epochs` | 32 / 150 | 先短跑后完整训练 |
| `seed` | 2024 | 扩展至多种子 |
| AAL atlas | 116 ROI | 网络映射需读 `data/AAL116.xlsx` 的“对应网络”列，缺失时不推断 |

相关文件和现有公共接口：

| 文件 | 现在的职责 | 预计修改 |
|---|---|---|
| `models/tfm_backbone.py` | `TFMEncoderBlock`, attention, patch embedding | 新增不破坏旧签名的可选诊断/更新封装 |
| `models/tfm.py` | `NeuroTwinTFM`, `_encode`, One-Step/CPM | `encoder_mode`, `loop_cfg`, `encode_with_budget`, `return_loop_info` |
| `models/tfm_loop.py`（新增） | 无 | 权重共享调度、循环门控、步长条件化、诊断 |
| `train/losses.py` | `TFMDualLoss` + graph 正则 | `LoopAuxLoss` 独立实现；原损失不改定义 |
| `main.py` | CLI、训练/验证、优化器/保存 | 新增 loop 配置、辅助训练调用、log |
| `train/optim.py` | EMA、冻结解冻、参数组、检查点 | 兼容检查、精确权重映射、`ARCH_VERSION`/签名 |
| `experiments/evaluate_variant.py` | checkpoint→模型自省构建与统一评估 | loop 参数重建、预算覆盖、解释导出 |
| `analysis/next_point_eval.py` | next-state / CPM / rollout / FC / subject metrics | 循环对比指标接入，延续原口径 |
| `analysis/loop_interpretability.py`（新增） | 无 | 提取逐轮状态/预测、忠实度、统计与 CSV/NPZ |
| `tests/test_tfm*.py` | 基础 TFM 测试 | 新增 `tests/test_tfm_loop.py` 与集成测试 |

### 1.2 当前 One-Step 与 CPM 的关键差异

One-Step 是下一 TR 的**残差预测**，用 `x_t` 为锚点：

$$z_g=A_{eff}z_t,\qquad \Delta z=f_{one}([z_t;z_g]),\qquad \widehat{x}_{t+1}=x_t+D(z_t+\Delta z).$$

CPM 直接从 `[B,F,P,D]` 的全部历史 patch token 交叉注意力解码出 `[B,F,H]`。在迭代比较中，**One-Step 与 CPM 都必须读同一个最终循环表示**；如果只改 One-Step，无法判断多步能力来自主干还是头部。

当前 `one_step_head.transition` 最后一层与 `decode` 做了零初始化，初始预测等于 persistence：新的 Loop 初始化不得破坏此断言。

### 1.3 关键协议约束

1. **严格按被试切分**，复用/冻结 split manifest；禁止从不同训练轮次重新抽独立 train/test 名单，禁止 windows 级泄漏。
2. 同样的 `F, K, patch_len, CPM H, norm, SC prior`，同一训练/验证样本索引、anchor 与评估 cutoffs。
3. 不改动源 BOLD、SC、HAMD 标注和 checkpoint；新实验写独立 `name`/目录。
4. 评价计算预算既记录模型参数量，也记录真实执行 FLOPs、GPU latency、显存与训练时长。**参数减少不意味着时延减少**。
5. `--compile` 在当前变长 K 路径会被训练入口跳过；Loop 的 Python 层动态 early exit 可能增加图断裂，先以 eager 为基准。

---

## 2. 研究假设与必须验证的主张

- **H1 参数共享有效性**：同为 4 次 Block 更新，shared model 相比独立 4 层预测不显著变差，同时参数量明显减少；是否更好为待验证。
- **H2 分阶段更新**：迭代步条件使模型内部表征不会过早静止/发散，更多循环可提供可控的额外收益，而非循环越多越好。
- **H3 弹性计算**：训练过预算 `1..4` 后，较低预算的性能优于 naïve 截断 4 次模型；P4 仅在真实推理节省可测时成立。
- **H4 可解释性**：迭代更新对应的 ROI/网络重要性在种子/重采样之间稳定，重要节点的 targeted intervention/permutation 会显著改变输出，而不是只显示大的 attention 权重。
- **H5 临床关联**：迭代深度/网络重要性与 HAMD 的关系即便存在，也需控制年龄、性别、站点、头动等可用混杂因素；未有可用协变量时只报告探索性关联。

**明确不主张**：`r=1,2,3` 是实际神经生理时间；中间隐藏状态一定对应可观测 FC；attention 本身构成因果解释；刺激节点扰动自动对应真实脑刺激结果；高 HAMD 患者“必定需要更多模型思考”。

---

## 3. 方案空间：先实验比较，而非提前押注一种结构

设 patch 后表示 `H^(0)∈R^{B×F×P×D}`，`A_eff∈R^{B×F×F}`，`D=256`，`P=ceil(K/p)`（具体 padding 语义沿用 `TemporalPatchEmbed`）。令 `B_θ(H,A)` 为当前完整 `TFMEncoderBlock`。

### 3.1 原始非共享 4 层（U4）

$$H^{(l+1)}=B_{\theta_l}(H^{(l)},A_{eff}),\quad l=0,1,2,3.$$

- 基准配置 `loop_mode=none`、`tfm_layers=4`；**数值与已有模型必须一致**。

### 3.2 完全共享 1×4（T1R4，第一重点）

$$H^{(r+1)}=B_\theta(H^{(r)},A_{eff}),\quad r=0,1,2,3.$$

- 仅一组参数，执行 4 次；如果只做 `ModuleList([same_module]*4)` 会导致 state_dict/加载语义混乱，建议单独持有 `self.loop_block` 并显式循环。
- 模型结构对照时不得把预训练的 `blocks.0` 复制后又声称“公平复现原 U4”；正式的 U4 vs T1R4 应分别从随机初始化完整 HC 训练。

### 3.3 两层单元重复 2 次（T2R2）

$$H^{(r+1)}=B_{\theta_2}(B_{\theta_1}(H^{(r)},A),A),\quad r=0,1.$$

- 保留较强的局部层间多样性，但仍可减少部分参数；总有效 Block 数仍为 4。

### 3.4 Prelude–Loop–Coda（PLC，候选）

$$H^{(0)}=B_{pre}(E(X),A),\qquad H^{(r+1)}=B_{loop}(H^{(r)},A),\qquad H^*=B_{post}(H^{(R)},A).$$

- 设 `pre=1, post=1, R=2`，总 Block 执行数=4。
- 目的：避免同一个 block 同时负责输入初始化、收敛修正、输出读出；缺点是相对 T1R4 参数量更多。

### 3.5 参数匹配单次/深层对照（P-Match）

- 调整 **独立层网络的层数或 D/FFN 比率**，使参数量近似 T1R4；真实可执行 Block 数、FLOPs 和 latency 记录，不能宣称同时“参数与 FLOPs 匹配”。
- U4 与 T1R4 是主要 **FLOP/执行深度匹配**；P-Match 主要检验参数共享是否优于同参数预算下的一次前向结构。
- 如果需要严格 FLOP-matched，使用 profiler 微调 D/层数以将误差压到 ±5%，报告差值。

### 3.6 路线清单

| ID | 主干 | 执行 Block 数 | 独立 Block 参数组 | 核心问题 |
|---|---|---:|---:|---|
| B0 / U4 | 独立×4 | 4 | 4 | 当前基准 |
| B1 / T1R1 | 共享×1 | 1 | 1 | 深度减少的下界 |
| B2 / T1R2 | 共享×2 | 2 | 1 | 较小计算量 |
| B3 / T1R4 | 共享×4 | 4 | 1 | 完全共享是否有效 |
| B4 / T1R6 | 共享×6 | 6 | 1 | 过循环、收益饱和检测（主要推理/训练对照） |
| B5 / T2R2 | 两层单元×2 | 4 | 2 | 部分共享是否更合理 |
| B6 / PLC | 1+共享×2+1 | 4 | 3 | 输入/输出层是否应解耦 |
| B7 / P-Match | 同参数量非共享 | 待测 | 待测 | 参数预算影响 |
| B8 / U6 | 非共享×6 | 6 | 6 | 单纯增加计算深度的上界 |

优先做 **B0, B1, B3, B5, B6**；B4/B7/B8 仅在需要判断机制时追加，避免第一轮组合爆炸。

---

## 4. 推荐的 Looped SC-Guided Block 细节

### 4.1 保留已验证的 TFMEncoderBlock 作为核心

原 block 内部：

$$\begin{aligned} T_r&=H_r+\mathrm{Drop}(\mathrm{TempAttn}(\mathrm{RMSNorm}(H_r)))\\ R_r&=T_r+\mathrm{Drop}(\mathrm{SCROI}(\mathrm{RMSNorm}(T_r),A_{eff}))\\ B_\theta(H_r,A)&=R_r+\mathrm{Drop}(\mathrm{FFN}(\mathrm{RMSNorm}(R_r))). \end{aligned}$$

`SCROI` 的注意力得分仍按当前代码处理：

$$S^{(r)}_{b,p,h,i,j} =\frac{\langle q^{(r)}_{b,p,h,i}, k^{(r)}_{b,p,h,j}\rangle}{\sqrt{d_h}} +\beta_h\log(\epsilon+A_{eff,b,i,j}).$$

循环下 **每轮 `A_eff` 固定**，注意力的 `Q,K,V` 根据 `H_r` 重新计算，因此 SC 对每轮的结构偏置相同、状态依赖注意力不同。

### 4.2 Step-conditioned residual loop（阶段 P2）

为避免权重共享使每轮完全同态、又不额外创建一组 Block 参数，建议添加小型条件网络：

$$\tau_r=\sum_{j<r}\Delta\tau_j,\quad c_r=[\tau_r,\Delta\tau_r],\quad [\gamma_r,\beta_r]=g_\phi(c_r)\in\mathbb R^{2D}.$$

推荐 `g_phi: Linear(2,64)→SiLU→Linear(64,2D)`，最后一层权重/偏置全零（identity-at-init）。对更新候选的输入做调制：

$$\widetilde H_r=\mathrm{RMSNorm}(H_r)\odot(1+\gamma_r)+\beta_r,\quad U_r=B_\theta(\widetilde H_r,A)-\widetilde H_r.$$

最终状态更新：

$$H_{r+1}=H_r+\alpha_r\,\Delta\tau_r\,U_r.$$

说明：

- `α_r` 设正值，推荐 `α_r=softplus(a_r)`，`α` 初值 1.0；如固定循环不做归一化时间，**对照 B3 可保持 `Δτ=1`**，以免将步幅改变误认为共享收益。
- 弹性路径使用 `Δτ_r=1/R`，保证 `ΣΔτ=1`；**固定共享 B3 与弹性共享 E1 的公式需要分开记录**，不能默默互换。
- 可选更简单的设置：不执行候选输入调制，直接在 `(B(H)-H)` 上逐通道 FiLM；需单独消融，不与其它因素同时调整。
- `α` 可在 `[0.05, 1.5]` 有界化，建议通过 `α=0.05+1.45 sigmoid(a)` 以避免负更新或过大步长；请按照选定参数化计算精确初始化 logit。

### 4.3 参数共享、梯度与训练稳定性

1. 共享模块一个实例：循环调用会自动累计来自各次展开的梯度，无需在每轮手动 `detach()`。
2. `R≤4` 首选全 BPTT；`R=6/8` OOM 后再评估 checkpointing，**不要默认截断反传**，否则各方案优化问题不同。
3. 保留原 dropout=0.1；解释阶段 `model.eval()`、禁用随机 dropout。对 full/shortcut consistency，需将随机性视为训练噪声并测方差，必要时在 consistency 的 paired path 使用确定性 dropout 设置。
4. 每轮监测 `||H_r||`、`||H_{r+1}-H_r||`、梯度全局范数、`A_eff`/attention 熵、ROI 对比度与是否 NaN/Inf。
5. ROI 过平滑测度（token 和通道规范化之后计算）：

$$C^{(r)}_{ROI}=\frac{2}{F(F-1)}\sum_{i<j} \operatorname{cosine}(\bar H^{(r)}_i,\bar H^{(r)}_j),\quad \bar H_i^{(r)}=\operatorname{mean}_{p}H^{(r)}_{i,p,:}.$$

`C_ROI` 接近 1 可作为塌缩**警报**而非严格证明。还需看 rank/effective rank、ROI 间差异方差以及预测误差，不允许用单一余弦判定过平滑。

### 4.4 统一读出（critical）

每种主干必须通过同一个接口：

```python
context = final_norm(h_final)            # [B,F,P,D]
z_t = context[:, :, -1, :]
delta = one_step_head(z_t, A_eff, cond)
cpm = cpm_head(context, roi_embed, cond, future_control)
```

为了辅助损失和诊断，再提供 `decode_from_tokens(context_r, A_eff, cond, ...)`，让中间循环共享同一组**最终读出头**；不为每轮新建 decoder，否则迭代结果不可比较。

---

## 5. 明确实现接口与伪代码（Codex 可直接按模块实施）

### 5.1 新建 `models/tfm_loop.py`

建议接口：

```python
@dataclass(frozen=True)
class TFMLoopConfig:
    mode: str = "none"                 # none | tied | grouped | plc
    rounds: int = 4                    # 每组共享单元重复次数
    shared_group_layers: int = 1       # tied=1，grouped=2
    prelude_layers: int = 0            # plc=1
    coda_layers: int = 0               # plc=1
    use_step_condition: bool = False
    loop_step_mode: str = "unit"       # unit | normalized
    residual_init: float = 1.0
    max_rounds: int = 4
    trace_mode: str = "off"            # off | stats | states

class TFMLoopedEncoder(nn.Module):
    def forward(self, tokens, adj, *, budget=None,
                schedule=None, trace_mode="off"):
        # return (states, trace); states = [B,F,P,D]
        ...
```

`trace_mode=off` 默认不创建每轮列表，避免增加显存；`stats` 只返回小标量/ROI 均值；`states` 在 `no_grad/eval` 下导出各轮张量，不在常规训练里长期留存图。

一个模型可以在**初始化时**选定 `mode`；不要在前向中动态创建 Module。`budget` 只控制已配置且已训练支持的循环次数；`mode=none` 强制禁止 `budget` 非法覆盖。

### 5.2 LoopedEncoder 伪代码

```python
def run_loop(h0, adj, *, R, scheduler, trace=False):
    h = h0
    trace_dict = {"updates": [], "states": []} if trace else None
    for r in range(R):
        tau, dt = scheduler(r, R)    # unit: (r,1); elastic: (r/R,1/R)
        hin = condition(h, tau=tau, dt=dt) if step_condition else h
        block_out = shared_block(hin, adj)
        update = block_out - hin
        h_new = h + residual_gate(r) * dt * update
        if trace:
            trace_dict["updates"].append((h_new-h).detach())
            trace_dict["states"].append(h_new.detach())
        h = h_new
    return h, trace_dict
```

**实现关键**：B3 的 naïve tied 版本必须用 `h=shared_block(h, adj)`，不要套入 `condition`, `dt`, `residual_gate`；P2 则显式新 mode/flag 计算 gated update，避免将多个修改绑定成一个无法解释的实验。

### 5.3 `NeuroTwinTFM` 接口扩展

构造新增配置字段（名称可以统一采用 CLI 的 `tfm_loop_*` 别名）：

- `loop_mode='none'`
- `loop_rounds=4`
- `loop_shared_group_layers=1`
- `loop_prelude_layers=0` / `loop_coda_layers=0`
- `loop_use_step_condition=False`
- `loop_step_mode='unit'`
- `loop_max_rounds=4`
- `loop_residual_init=1.0`
- `loop_trace_mode='off'`

新增方法：

```python
def encode_with_budget(self, x_grid, sc_matrix, *, budget=None, trace_mode="off"):
    """返回 context, sc_info, loop_trace；一次构建 A_eff。"""

def forward(self, x, sc_matrix, pathology_score=None,
            future_control=None, *, loop_budget=None, return_loop_info=False):
    """原来的 (pred, aux_info) 不变；附加字段仅 opt-in。"""
```

强约束：默认 `none + return_loop_info=False` **逐位或在明确浮点容差内保持与原实现一致**；`loop_budget` 非法值必须抛出清晰错误；`trace` 不影响最终输出。

### 5.4 主干模块与字典键

推荐参数命名：

- 原版：保留 `blocks.N.*`，原检查点键不变。
- tied：使用 `loop_encoder.shared_block.*`。
- grouped：使用 `loop_encoder.shared_group.0.*`, `.1.*`。
- plc：`loop_encoder.prelude.*`, `.shared_block.*`, `.coda.*`。

构造时不用原先固定 4 个 `blocks` 再“跳过”，否则 checkpoint 包含无用参数，模型参数量统计失真。

### 5.5 HC→MDD checkpoint 兼容策略

- 同一 `loop_mode` HC checkpoint → 同一结构 MDD finetune：严格映射，确保共享权重真正被加载。
- 原 U4 checkpoint → 新 T1R4：**不是严格一一同构**。仅为快速 warm-start 额外提供显式迁移实验（例如从 `blocks.0` 初始化共享 block 或各层权重平均）；日志保存 mapping、源/目标 keys、加载比例、遗失键清单。它不能替代正式 from-scratch 公平比较。
- Loop 相关字段加入 `config.args` 与结构签名，包括 `mode/group/R/pre/post/step_condition/step_mode`；语义不同的同形 shape 权重不得默认加载。
- `train/optim.py` 的 `BACKBONE_KEY_PREFIXES` 需纳入 loop 主干；`set_finetune_stage()` 对共享块是否冻结要做单元测试。
- `ARCH_VERSION` 依项目约定调整，**不要为共享结构直接使用 `warn` 略过不匹配**。
- `experiments/evaluate_variant.py` 的 checkpoint rebuild 需支持 CLI alias `tfm_loop_*` 与构造参数的映射；旧 checkpoint 缺少字段时明确回退 `mode=none`，新结构字段缺失应失败，而不是悄悄初始化随机权重。

### 5.6 训练入口、loss 和 debug

在 `main.py` 增加参数（**以下均为待实现选项，不是当前可运行 CLI**）：

```bash
--tfm_loop_mode {none,tied,grouped,plc}
--tfm_loop_rounds 4
--tfm_loop_group_layers 1
--tfm_loop_prelude_layers 0
--tfm_loop_coda_layers 0
--tfm_loop_use_step_condition False
--tfm_loop_step_mode {unit,normalized}
--tfm_loop_max_rounds 4
--tfm_loop_training {fixed,elastic}
--tfm_loop_shortcut_weight 0.10
--tfm_loop_consistency_weight 0.05
--tfm_loop_trace_mode {off,stats,states}
--tfm_loop_halt_mode {off,offline_threshold,online_threshold}
--tfm_loop_halt_min_rounds 2
--tfm_loop_halt_tol 0.01
```

只在 `tfm_loop_mode != none` 时启用新逻辑；早期实验保持 `TFMDualLoss`、SC 正则及优化器配置不变。

---

## 6. 训练目标：基础、逐轮监督、Shortcut 和稳定性

### 6.1 保持现行双头主损失

设 `m_{b,h}∈{0,1}` 表示未来第 h 步有效，`w_h=γ^{h-1}m_{b,h}`。保持 `train/losses.py` 的现行实现为正式定义（具体广播和归一化代码为准）：

$$\mathcal L_{one}=\mathrm{Huber}(\widehat x_{t+1},x_{t+1}) +\lambda_{pcc}\,[1-\mathrm{PCC}_{ROI}(\widehat x_{t+1},x_{t+1})],$$

$$\mathcal L_{CPM}= \frac{\sum_{b,h}w_{b,h}\mathrm{Huber}(\widehat X_{b,:,h},X_{b,:,h})} {\sum_{b,h}w_{b,h}+\epsilon} \quad (\text{保留原有 CPM spatial PCC 实现}),$$

$$\mathcal L_{base}=\lambda_{one}\mathcal L_{one}+\lambda_{CPM}\mathcal L_{CPM} +\mathcal L_{graph}.$$

### 6.2 逐轮输出辅助监督（可选 A1）

使用**同一读出头**将第 r 次状态转换为 one-step + CPM 预测，得到 `ŷ^(r)`。最初仅对 one-step 加辅助损失，避免每轮运行昂贵 CPM：

$$\mathcal L_{deep}=\frac{1}{R-1}\sum_{r=1}^{R-1} \omega_r\mathrm{Huber}(\widehat x^{(r)}_{t+1},x_{t+1}),\quad \omega_r=\frac{r}{R-1}.$$

建议 `λ_deep∈{0,0.05,0.1}`，首选 0；确认 loop 有效再调。若解码成本较高，可离线逐轮分析而不参与训练。

### 6.3 弹性 Loop：Full+Shortcut（P3）

取最大预算 `L=4`，完整路径采用 `Δτ=1/L`；随机 shortcut 预算 `S∼Uniform{1,2,3}`，第一版使用等间隔 `Δτ=1/S`，随后可再尝试 Dirichlet 步长（避免一开始引入路径采样差异）。两个路径复用**同一输入、同一个 `A_eff`、同一组读出头**：

$$H^L=\operatorname{Loop}(H^0,A,L),\qquad H^S=\operatorname{Loop}(H^0,A,S).$$

$$\mathcal L_{elastic}= \mathcal L_{base}(\widehat Y^{L}) +\lambda_{short}\mathcal L_{base}(\widehat Y^{S}) +\lambda_{cons}\mathcal L_{cons}.$$

建议：`λ_short=0.10`，`λ_cons=0.05` 起步；后者消融 `{0,0.05,0.10}`。一致性 **使用规范化的预测张量**（训练时在 RevIN inverse 前），不把不同个体绝对 BOLD 幅值混在一起：

$$\mathcal L_{cons}= \mathrm{Huber}(\widehat x^{S}_{t+1,norm},\mathrm{sg}(\widehat x^{L}_{t+1,norm})) +\eta\sum_{h=1}^{H}\widetilde w_h \mathrm{Huber}(\widehat X^{S}_{h,norm},\mathrm{sg}(\widehat X^{L}_{h,norm})), \quad\eta=0.5.$$

`sg` = stop-gradient；在辅助路径仅生成一次短预算结果，防止每个短预算同时训练而扩成 4 倍成本。原 LoopFormer 在语言模型上采用 full/shortcut/consistency 的形式；这里是**针对 BOLD 预测的改造，不把语言结果视为医学验证**。

**必要对照**：`elastic without cons`、`elastic with cons`、`fixed R=4`、`naïve truncated R=1/2/3`；否则无法判断“弹性改善”来自 shortcut loss 还是 step conditioning。

### 6.4 稳定性辅助项（失败后再加，不默认全开）

**幅值防爆**：

$$\mathcal L_{upd}=\frac{1}{R}\sum_r \max(0,\rho_r-\rho_{max})^2, \quad\rho_r=\frac{\|H_{r+1}-H_r\|_F}{\|H_r\|_F+\epsilon},\;\rho_{max}=0.5.$$

推荐 `λ_upd=0` 默认；出现明显发散才测 `1e-4,1e-3`。不要强加 $\|\Delta H\|\to 0$，否则模型可能学会完全不修正。

**ROI 多样性保护（仅过平滑时启用）**：

$$\mathcal L_{div}=\max(0, C^{(R)}_{ROI}-c_{max})^2, \quad c_{max}\in\{0.80,0.90\}.$$

优先通过结构、步幅、归一化解决；此项只有在观测塌缩后再尝试，并验证不伤害真实功能网络信息。

### 6.5 总损失与建议阶段

$$\boxed{\mathcal L= \mathcal L_{base} +\lambda_{deep}\mathcal L_{deep} +\mathbf 1_{elastic}(\lambda_{short}\mathcal L_{short}+\lambda_{cons}\mathcal L_{cons}) +\lambda_{upd}\mathcal L_{upd}+\lambda_{div}\mathcal L_{div}.}$$

| 阶段 | `deep` | `short` | `cons` | `upd` | `div` |
|---|---:|---:|---:|---:|---:|
| P1 基础结构 | 0 | 0 | 0 | 0 | 0 |
| P2 step-conditioned | 0 | 0 | 0 | 0 | 0 |
| P3 elastic | 0 | 0.10 | 0.05 | 0 | 0 |
| P4 若发散 | 可选 | 同 P3 | 同 P3 | 1e-4 | 仅确认塌缩时 |

---

## 7. Adaptive Halting：不要把“小更新”直接等价于“高质量预测”

### 7.1 更新收敛度

$$q^{(r)}_b= \frac{\|H^{(r)}_b-H^{(r-1)}_b\|_F} {\|H^{(r-1)}_b\|_F+\epsilon}.$$

建议可选再加预测变化 `p_b^(r)`（需要每轮 one-step decoder；仅在诊断/离线阈值阶段使用）：

$$p_b^{(r)}=\frac{\|\widehat x^{(r)}_{b,t+1}-\widehat x^{(r-1)}_{b,t+1}\|_2} {\|\widehat x^{(r-1)}_{b,t+1}\|_2+\epsilon}.$$

归一化分母接近 0 的场景需采用 `max(denom, eps_floor)`，`eps_floor` 可由训练集分位数确定。不能仅因 `q≈0` 就停，因为**停止更新也可能代表陷入坏的固定点**。

### 7.2 分两步实施

**P4a 离线阈值分析**：验证集上运行完整 `R_max=4`，缓存每样本各轮 `q_r,p_r`、每轮预测误差；在离线阶段拟合阈值，再模拟 `R_stop`，不修改前向控制流。验证可接受后进入 P4b。

**P4b 在线停止**：

$$R_{stop}(b)=\min\{r\ge 2:\ q_b^{(r)}<\tau_q \text{且最近两个可用步骤满足稳定条件}\},$$

强制 `2≤R_stop≤4`；如果阈值未达到则跑满。`τ_q` 搜索 `{0.005,0.01,0.02,0.05}` 只是初始尺度，实际用验证集 `q` 分布校准，**不是承诺阈值可跨实验复用**。离线附加 `p_r` 约束作增量消融。

实现时为了真实省时：如同 batch 的样本部分 early exit，必须压缩 active indices/子 batch，并保持终止样本输出不再改变。仅 mask 更新但仍计算所有样本的 attention 并不能节省主要计算。分桶/动态分支带来的 kernel 和控制流开销须真实测量，可能比固定 4 次更慢。

### 7.3 选择指标

在验证集上定义 `MAE_adapt / MAE_R4 ≤ 1.005`（示例非劣容忍度 0.5%），同时实际 median GPU latency 至少下降 15% 才保留在线自适应方案；否则保留固定/按预算推理模式。查看 HC/MDD 分组、不同 HAMD/站点的误差，避免效率来自对某一群体的系统性提前停止。

`R_test>R_train_max`（如 5/6/8）仅作外推压力测试，**不得声称必然有性能增益**；主要推理预算限制为训练覆盖的 `1..4`。

---

## 8. 脑功能可解释性：需要真正可验证的分析协议

### 8.1 解释对象分级

- **级别 A 计算过程**：每轮隐藏状态变化、输出修正，是直接可观测的模型内部行为。
- **级别 B 预测相关解释**：ROI/网络扰动导致预测误差或输出变化，体现模型对该信息的依赖。
- **级别 C 神经生物学关联**：结果与 AAL116 预定义网络、HAMD、SC 和真实 FC 的关系；是统计发现，仍需独立数据验证。
- **级别 D 因果/刺激解释**：除非有干预标签或明确因果识别策略，否则只称 counterfactual model response，不称真实干预效果。

### 8.2 Loop Update Map（优先交付）

每轮 ROI 统计：

$$U^{(r)}_{b,i}= \sqrt{\frac{1}{PD}\sum_{p,d}(H^{(r)}_{b,i,p,d}-H^{(r-1)}_{b,i,p,d})^2}.$$

为避免被高范数 ROI 主导，额外报告相对幅值：

$$\widetilde U^{(r)}_{b,i}= \frac{U^{(r)}_{b,i}} {\sqrt{\frac{1}{PD}\sum_{p,d}(H^{(r-1)}_{b,i,p,d})^2}+\epsilon}.$$

按预定义功能网络 $\mathcal N_k$ 聚合：

$$U^{(r)}_{b,k}=\frac{1}{|\mathcal N_k|}\sum_{i\in\mathcal N_k}U^{(r)}_{b,i}.$$

注意：`U` 是**内部更新幅度**不是“生物学重要性”；须与下一条输出忠实度配对。

### 8.3 与预测输出绑定的逐轮修正

同一套 `OneStepHead` 和 `CPMHead` 读出中间 `final_norm(H^(r))`：

$$\widehat x^{(r)}_{t+1}=D_{one}(H^{(r)},A_{eff}),\qquad \widehat X^{(r)}_{1:H}=D_{CPM}(H^{(r)}).$$

可计算：

$$V^{(r)}_{b,i}=|\widehat x^{(r)}_{b,i,t+1}-\widehat x^{(r-1)}_{b,i,t+1}|,$$

$$E^{(r)}_{b}=\mathrm{MAE}(\widehat x^{(r)}_{b,t+1},x_{b,t+1}).$$

分析 `E^(r)` 是否单调下降（**不应假设一定单调**），出现改善/恶化的 loop 区间，解释是否“先大范围、后局部”必须是**先验未固定的待检验假说**。如每轮完全由最终读出头读取，需在训练中确认中间读出不是明显 OOD；可用可选 deep supervision 验证。

### 8.4 SC 依赖与忠实度

不能简单把 ROI attention 权重当解释，因为注意力不等同于重要性。建议三组配套实验：

1. `A_real`：被试真实 SC 生成 `A_eff`；
2. `A_shuffled`：在相同脑区排列协议下破坏 SC 的跨 ROI 关系，最好保留度/权重直方图作为辅助控制；
3. `A_identity` / `A_functional_only`：消融跨 ROI 的 SC 引导（如果图模块支持该模式，需要明确数学语义）。

定义在固定训练完成模型上的 **SC 依赖敏感性**：

$$D_{SC}^{(r)}=\mathrm{MAE}(\widehat Y^{(r)}_{A_{shuffle}},Y) -\mathrm{MAE}(\widehat Y^{(r)}_{A_{real}},Y).$$

**必须区分**：推理阶段 SC 置换是 OOD 干预，可显示依赖但不足以证明某个连接具有因果脑机制；进一步在训练/评估中加入 SC null 模型作结构先验有效性的检验。不能用 `corr(A_eff,Attention)` 证明 SC 被“发现”，因为 SC 直接作为注意力 bias 输入，容易形成循环论证。

### 8.5 ROI 删除/置换忠实度（核心）

针对每轮的 `U` 排序和最终置换重要性排名，使用相同样本、相同 mask、相同扰动强度：

$$\mathrm{Faithfulness@k}= \frac{\mathrm{MAE}(f(X_{\setminus Topk}),Y)-\mathrm{MAE}(f(X),Y)} {\mathrm{MAE}(f(X),Y)+\epsilon}.$$

对比 `Top-k`、`Bottom-k`、随机 ROI 三种集合，建议 `k∈{5,10,20}`；优先采用**输入 ROI 的训练集匹配置换/掩蔽**，且明确该操作破坏真实脑区联合分布，避免因非生理扰动夸大效果。

`U` 与 feature importance 的 Spearman / Kendall 相关只作为补充；显著性用**被试为重采样单位**。对逐 ROI 差异执行 BH-FDR 控制。

### 8.6 功能连接 FC 解读的特殊约束

当前 CPM horizon=8 TR，**8 个时间点不足以支持稳定的静息态功能连接估计**。因此：

- 循环每步可以画 `predicted BOLD delta`, `latent ROI affinity`, `attention bias/weights`，但**不能把单个 loop hidden state 的 cosine similarity 命名成实测 FC**。
- 只有在从完整预测序列（例如自由 rollout 的 32/64 TR 或与真实数据可比的更长序列）估计 Pearson FC 时，才报告“预测 FC”；详细说明 TR、长度、是否重叠窗口、自相关和质量控制。
- 用**相同长度真实 BOLD**作对照，报告 edge-wise MAE、within/between network FC error 和 Fisher-z 相关，并检查短序列敏感性。
- `SC–FC consistency` 不能是越大越好的一维绝对指标：结构连接和功能连接不是相同测量，需报告对应受试者经验分布以及置换 null。

### 8.7 可解释性可信度测试

- 随机初始化模型 vs 训练模型：内部更新图是否显著不同。
- 随机标签/破坏预测目标训练的负对照：重要性图是否依赖任务训练。
- 3–5 个随机种子、subject bootstrap：Top-k ROI Jaccard、网络层 Spearman、ICC（若适用）。
- 同一受试者邻近窗口：稳定性 vs 时间敏感性（过度稳定也可能代表只学恒定空间偏置）。
- 没有 `AAL116.xlsx` 正确映射时，跳过网络名称、只保留 ROI ID；不手工猜 DMN/FPN/VAN 标签。

### 8.8 临床相关与 HAMD 条件响应

对 MDD subject 层级：

$$U^{(r)}_{s,k}=\beta_0+\beta_H\cdot HAMD_s +\beta_a Age_s+\beta_g Sex_s+\beta_m Motion_s +\mathrm{SiteEffects}_s+\epsilon_s.$$

协变量只有在实际数据存在且样本量足够时纳入；站点多时可考虑 site fixed/random effect，模型选取事前固定。多次网络×迭代检验做 FDR；没有临床独立外部集时用“探索性关联”，不称脑疾病机制证据。

若希望分析条件化作用，可在已训练的 MDD FiLM 模型上**固定输入 BOLD/SC 仅改变 HAMD**，画输出变化曲线，解释为“模型条件敏感性”，不是对某人疾病严重程度进行真实干预。

### 8.9 论文用图与输出文件

| 图/文件 | 横纵轴/主要内容 | 研究意义 |
|---|---|---|
| Figure A | loop depth×ROI/network 的 `U^(r)` 热图（组级） | 每轮表征更新分布 |
| Figure B | loop depth×验证误差/更新幅度/ROI similarity | 是否有有效的渐进修正 |
| Figure C | 每轮 one-step 与 CPM horizon 预测 vs ground truth | 迭代如何影响预测，不只是特征 |
| Figure D | 原始/打乱 SC 的性能变化 + null 区间 | SC 依赖忠实度 |
| Figure E | Top-k vs random/bottom-k ROI 置换误差 | 解释是否对应真实模型决策 |
| Figure F | 每个受试者 R_stop 分布 + 速度/误差权衡 | 自适应计算，而非直接临床因果 |

建议 `results/loop_eval/<run_id>/` 写出 `metrics.json`, `per_subject.csv`, `per_loop_metrics.csv`, `roi_update_by_loop.npz`, `network_update_by_loop.csv`, `sc_ablation.csv`, `faithfulness.csv`, `bootstrap_ci.json`, `figures/*.png` 和 `run_manifest.json`。可选 NPZ 可能包含逐被试医学数据，遵循本地访问控制，不公开上传。

---

## 9. 系统化实验矩阵与筛选算法

### 9.1 公平性矩阵

每一个对比必须锁定：`dataset_version`、`train/val/test_subject_ids`、SC 处理、context 采样种子、ROI 顺序、patch_len、输入 BOLD 归一化、基础损失、预测头、训练 epoch 上限/早停、optimizer/scheduler、batch、AMP/EMA。**如果替换主干使总参数量不同，应清晰报告，而不是当作同容量比较**。

两个观察维度：

- **计算深度匹配**：U4 vs T1R4/T2R2/PLC（每样本约 4 次相同级别 encoder block；实际 attention 调用数+FLOPs 以 profiler 核实）。
- **参数匹配**：T1R4 vs P-Match；比较同等参数预算的预测质量，算力预算另外单列。

### 9.2 筛选阶段 0：验证环境与基线

- B0 运行原脚本 HC→MDD，固定 2024 seed，独立 val/test；记录基线指标。
- 采集保存：`git_sha`, 实际命令、`args.json`, split manifest checksum, CUDA/PyTorch版本, trainable/total params, peak GPU mem, throughput, wall-clock train/infer。
- 对原 `none` 模式跑行为回归：相同初始化/状态字典及 eval 输入，one-step 和 CPM 输出一致；无条件预训练时拒绝 HAMD。

### 9.3 筛选阶段 1：架构候选（先 HC，再 MDD）

第一轮短跑候选：`B0, B1, B3, B5, B6`，每组 `seed=2024`，最多 30–50 epoch **只用于可行性过滤**；不能据短跑宣布结构优越性。删除出现稳定发散、过平滑、参数/显存大幅超预算的方案。

对剩余 2–3 个候选执行完整 HC 150 epoch + 对应 MDD 150 epoch；种子 `{2024,2025,2026}`，**按每个 seed 对齐数据划分/采样协议**。建议固定同一主 split 且仅变初始化/采样 RNG，另做跨 split 重复实验来测泛化；不要不说明就让每个 seed 自动产生不同 test subjects。

**候选保留逻辑**：

1. `val one-step MAE` 主指标；
2. `val CPM H1/H4/H8 MAE` 和 rollout 分层误差，要求没有严重退化；
3. 对 persistence/trend/AR(1) 的增益；
4. 参数/FLOPs/latency；
5. 训练稳定性（失败数/梯度/过平滑）。

示例保留门槛：相对 U4 one-step MAE ≤+1%，CPM 平均 MAE ≤+2%，且参数减少至少 20%，或 one-step/CPM 在被试级有可重复提升。门槛是资源控制规则，不是论文统计显著性结论。

### 9.4 筛选阶段 2：step conditioning 与稳定性

在阶段 1 最优 shared 结构上按顺序比较：

| ID | 变更 | 其它固定 | 问题 |
|---|---|---|---|
| C0 | naïve tied | 当前结构 | 起点 |
| C1 | block-level residual gate | 无 step condition | 共享深度是否需要缩放 |
| C2 | + loop step embedding (`r/R`) | `Δτ=1` | 轮次信息是否有益 |
| C3 | + `(τ,Δτ)` time-aware | normalized step | 轨迹参数化是否更好 |
| C4 | + deep supervision | `λ_deep=0.05` | 中间输出是否更可信 |

`C2` 与 `C3` 不同于“只换一个可学习 embedding”：后者也改变积分步幅，故必须同时报告 `C1+normalized dt` 无 condition 的对照。

### 9.5 筛选阶段 3：弹性深度

固定 winner 的架构/损失：

| ID | 训练预算 | shortcut supervision | consistency | 推理预算 |
|---|---|---|---|---|
| E0 | 固定 R4 | 无 | 无 | R1,2,3,4（截断 OOD） |
| E1 | 采样 R1–4 | 有 | 0 | R1–4 |
| E2 | full R4 + sample S1–3 | 有 | 0.05 | R1–4 |
| E3 | 同 E2 | 有 | 0.10 | R1–4 |
| E4 | E2，随机非均匀步长 | 有 | 0.05 | R1–4（步长需合法） |

每个预算都输出：one-step MAE、CPM H1/H4/H8、模型 FLOPs/latency、分位数误差、退化样本比率。推荐“质量-延时 Pareto frontier”，不要只用一个加权综合分随意宣布赢家。

### 9.6 筛选阶段 4：Adaptive Halting

同一个 E2 checkpoint 测：`fixed2/fixed3/fixed4`、随机停步（相同平均预算）、`q` 阈值、`q+p` 双阈值、可选小型学得的 halting head。**先离线选择，之后冻结阈值；只在锁定 test 上一次确认**。学习式 head 是额外容量，应报告其参数和监督目标，不能与纯阈值算法混在一起。

### 9.7 临床阶段的三组关键比较

- `HC-U4 → MDD-FiLM-U4`（真正基线）。
- `HC-Loop → MDD-FiLM-Loop`（Loop 对病理迁移是否有帮助）。
- `HC-Loop → MDD-Loop-without-FiLM`（病理条件通路的贡献）。

保留与真实仓库相同的 `PathologyNormalizer` train-only 拟合、FiLM 冻结/解冻、SC prior；如果将来 RECC/MoE 实施，增加一套“with vs without MoE”的独立因子试验，不改变本计划核心结论。

### 9.8 最终测试与统计

- 选型只用 `train/val`；test 完整冻结，最终候选和全部 baseline 统一跑一次，避免从 test 反复选模型。
- 预测统计以**被试**为单位，计算每人均值后总体 bootstrap（推荐 2,000 次 subject bootstrap，95% percentile/BCa 取其一，明确方法）；数据若有站点，建议 stratified subject bootstrap；不可对高度相关的 window 当独立观测做 t-test。
- 成对差值（`Loop−U4`）报告均值、95% CI、样本量、效应量；多个指标/网络/ROI 比较按 BH-FDR 处理。
- 同时报告每 seed 的结果，均值±标准差，不因单 seed 表现优越宣称统计稳健。
- 对多步预测固定 CPM H4/H8，rollout H4/H8/更远 horizon 依现有评估配置，对长 rollout 的有效真值 mask 使用原实现。

---

## 10. 如何明确判断“最优路线”

**采用门槛筛选 + Pareto + 预注册优先级，不能边看 test 边更改权重。**

阶段门槛：

1. **Correctness gate**：shape、因果性、无泄漏、权重加载、grad、EMA、旧 checkpoint、identical baseline 全通过。
2. **Quality gate**：验证 MAE 和 CPM 关键 horizon 不显著劣于 B0，且相对 trivial baseline 有正增益。
3. **Efficiency gate**：有真实参数/FLOP/latency 中至少一个明显收益，且代价透明；Adaptive 路线要求至少 15% 实测 latency 降低且 MAE 非劣示例 ≤0.5%。
4. **Explainability gate**：Top-k ROI 的预测误差增量高于随机组、SC 真/假扰动形成可复现差异，种子/被试稳定性可接受；不要求逐轮更新必定单调，也不要求某个特定网络必须显著。
5. **Clinical gate**：HC→MDD 迁移成立，HAMD 分组无异常偏差；相关解释中的混杂因素被记录。

若多个候选都可接受，以 **主指标 one-step MAE** 作为首要质量判据，再看 **CPM/rollout** 的安全约束，随后比较 latency 与参数、解释稳定性。可保留两个 Pareto 点：高精度方案（论文 SOTA 竞争）与低成本方案（部署）；不强求一个单一冠军。

建议**退出机制**：

- P1 未通过 → 停止，文档化“TFM 对 weight tying 不适合”的负结果，转向仅循环 decoder/SC residual 局部改造（新假设另立方案，不混用实验）。
- P2 step conditioning 不优于固定共享 → 保留 naïve tied、取消 P3 的复杂 trajectory conditioning，或单独验证弹性必要性。
- P3 无弹性收益 → 固定 R 最优版本，不进入 halting。
- P4 提前停止没有真实加速 → 保留预算可控 `R=2/3/4`，不部署 per-sample early exit。
- P5 解释不能通过忠实度/随机化测试 → 只报告过程诊断，不作神经机制解读。

---

## 11. 研发阶段任务包（每阶段可由 Codex 自主完成、验证、报告）

### Phase 0：实验与接口审计

**任务**：锁定 git SHA；读 `AGENTS.md`、现有 README、两份训练脚本；运行现有 TFM test；核实 `--tfm_patch_len` 脚本=4，CLI=8；核对已有 train/test split manifest、保存结果和评估日志。建立 `experiments/loop/README.md` 与 `configs/loop/`（目录仅建议）；不要改任何模型代码。

**验收**：baseline `none` 参数量/损失/latency/设备与固定 split 可复现；完整清单保存于 `run_manifest`。无法取得真实数据/GPU 时只做 smoke test 并明确未验证训练结果。

### Phase 1：权重共享架构和严格回归

**任务**：新增 `tfm_loop.py`，实现 tied/grouped/plc；保留老 `blocks` 逻辑；实现 `--tfm_loop_mode`；扩展 checkpoint 构造、冻结逻辑、测试。`none` 行为保持不变。

**验收测试**：

- 输入 K=16/17/32/64；`[B,F,1,K]→[B,F,1,1]` 与 CPM `[B,F,H]`。
- B3 同一个 `id(shared_block)` 在所有迭代一致，且共享层权重只有一份。
- 同一模型同一 eval 输入两次 bitwise/容差一致；trace 开关不改输出。
- ROI attention 的 patch/ROI 轴没有错位；替换后的模型仍只有历史 causal tokens，扰动未来 target 不影响当前 prediction。
- 所有 trainable 参数有应有梯度；`loss.backward()` 无 in-place autograd 错误；AMP/EMA 可运行。
- `mode=none` 和原实现输出一致；预训练在 one-step 的初始化仍等于 persistence。

**交付**：可复制命令、参数量与 Block 调用数表、测试报告；未改动已有 check­point。

### Phase 2：基线模型筛选

**任务**：使用统一 HC 预训练/验证协议比较 B0/B1/B3/B5/B6；后将前 2–3 个候选做 3 seeds 完整 HC→MDD 训练。缓存每实验签名和 checkpoint，不覆盖任何已有输出。

**验收**：含训练曲线、验证集各指标、被试级差值、效率与异常案例；产出 `architecture_selection.md` 给出进阶/退出决策理由。

### Phase 3：条件化与弹性 Loop

**任务**：对 Winner 加 `StepConditioner`、有界 update scale；分拆 fixed unit 与 normalized trajectory；在 `train/losses.py` 增独立 `LoopAuxLoss`；训练 E0–E3。

**验收**：各预算均有梯度；full 与 shortcut 共用同个 `A_eff`、patch embedding、读出头；stopgrad 只作用 teacher；null consistency 权重下行为不变；统计额外训练成本。

### Phase 4：自适应停止与效率

**任务**：离线收集收敛轨迹/误差，验证集选阈值，在线 active-batch early exit；测固定与动态 latency，在适合的 GPU 上 warmup 后 CUDA event 测 p50/p90 和吞吐。

**验收**：早停模型结果严格等于针对每个样本选择对应预算时的输出（容差内）；active-batch 不打乱 subject 或 sample 顺序；阈值/收益都通过验证门槛。

### Phase 5：神经解释与论文图

**任务**：提取 `U^(r)`、逐轮共享 decoder 预测、SC 置换、ROI Top-k perturbation、网络聚合、bootstrap CI、随机化 sanity checks；分 HC/MDD、HAMD 与站点汇总。完成 6 张主/补充图。

**验收**：没有“单 TR FC”“loop=生理时间”“attention=因果”错误命名；脑区映射有来源；每个 plot 能追溯到 subject IDs、checkpoint SHA、控制组和统计方法。

### Phase 6：联合评估与总结

**任务**：一次锁定 test；比较最佳 Loop 与 U4、trivial baseline；做剩余必要消融；整理最佳 checkpoint、可重复命令、参数表、指标表、解释图、失败案例；产出 `final_loop_report.md`。

**验收**：test 从未参与阈值/架构超参选择；所有增益按被试级给 CI；包含明确的失败条件与局限性。

---

## 12. 可直接用于开发的配置模板和执行命令

**说明：以下新增 `--tfm_loop_*` 参数为本方案的待实现 API；在 Codex 完成 Phase 1 前，直接运行会报 unknown arguments。**

### 12.1 配置模板（推荐放 `configs/loop/`）

```yaml
# loop_p1_tied4.yaml 结构示意；当前 main.py 仍使用 CLI，需先实现配置适配或转为脚本参数
model:
  backbone: tfm
  dim: 256
  layers_baseline: 4
  heads: 4
  patch_len: 4
  cpm_horizon: 8
  loop:
    mode: tied
    rounds: 4
    group_layers: 1
    prelude_layers: 0
    coda_layers: 0
    use_step_condition: false
    step_mode: unit
    trace_mode: off
training:
  batch_size: 32
  epochs: 150
  seeds: [2024, 2025, 2026]
  lambda_one: 1.0
  lambda_cpm: 1.0
  lambda_pcc: 0.1
  lambda_shortcut: 0.0
  lambda_consistency: 0.0
  lambda_deep: 0.0
  early_stopping: val_next_state_mae
```

### 12.2 训练方式建议

```bash
# 0. 当前完全无改造基线（现有真实命令）
bash scripts/Pretrain_HC_tfm.sh
bash scripts/Finetune_MDD_tfm.sh

# 1. Phase 1 实施后，在原脚本同一参数集合中仅额外追加：
#    --tfm_loop_mode tied --tfm_loop_rounds 4 --tfm_loop_training fixed
# 与 baseline 使用不同 --name、checkpoint_dir 子路径，不覆盖任何权重。

# 2. Phase 3 实施后新增：
#    --tfm_loop_training elastic --tfm_loop_step_mode normalized \
#    --tfm_loop_use_step_condition True \
#    --tfm_loop_shortcut_weight 0.10 --tfm_loop_consistency_weight 0.05

# 3. Phase 4 实施后：
#    使用统一 evaluate_variant 或其扩展入口遍历 --loop_budget 1/2/3/4，
#    并单独调用 offline_halting_eval / online_halting_eval。
```

### 12.3 运行批处理原则

- 每个候选都使用独立实验 `run_id=arch__trainmode__seed__split_hash`；同机器并行前确保 GPU、checkpoint、日志和缓存不共享可写路径。
- HC 预训练与 MDD 微调必须明确关联同一模型结构/种子，不能使用 U4 checkpoint 作为共享 Loop 的“无差别预训练”而隐瞒。
- 数据路径、显卡、最大资源在环境变量或现有 CLI 中配置；不得将用户个人绝对路径写入配置。
- 用一份 machine-readable experiment registry 记录实验状态 `queued/running/complete/failed`、父 checkpoint 与启动命令；不需为本课题额外引入复杂调度平台。

---

## 13. 关键工程细节与常见失败

1. **共享权重加载错位**：原来 `blocks.0...3`，新模型是 `shared_block`；shape 相同不代表语义相同，必须有可审计映射。
2. **误把 Loop 迭代当时序 step**：外部 TR `t` 与内部 refinement 变量 `τ` 用不同符号/日志键；每轮只更新同一历史输入的表征。
3. **SC 被重复计算**：`A_eff` 依赖归一化历史，从 `SoftAnatomicalPrior` 计算一次，循环中持久引用；若未来想做 per-loop graph update，作为新独立假设。
4. **注意力导出破坏高效前向**：现有 Temporal Attention/CPM 用 SDPA；研究用 attention weights 需可选诊断路径、关闭 dropout 并注意与 fused SDPA 输出数值校验，不在每次训练中保存 `[B,P,h,F,F]`。
5. **CPM 性能陷阱**：每轮同时跑全 CPM 会显著增加算量；P1 只最终头，P5 有选择抽样输出逐轮预测。
6. **固定 Loop 与 elastic Loop 混淆**：`Δτ=1` 与 `1/R` 改变有效步长，记录公式与实际参数；比较时额外做 matched step control。
7. **深度增加引发 over-smoothing 或 overshooting**：监控输出收益、ROI similarity、有效秩和 update ratio；发生负效应应允许停止增加 R。
8. **梯度路径被无意 detach**：仅日志/teacher consistency 使用 `detach`；循环主体的 `h_r` 始终可导。
9. **不恰当的早停加速声明**：单纯按 mask 将更新设 0，若 attention 仍计算整个 batch，不应宣称 FLOPs/时延节省。
10. **忘记冻结适配器**：MDD 冻结期间 FiLM (`is_conditioning_adapter=True`) 要可训练；新 step conditioner 是 HC backbone 一部分还是 MDD adapter，必须提前定义（本方案默认 backbone）并测试 optimizer 参数组。
11. **从验证集多次试探 test**：严格冻结 test；任何超参数或阈值选型都只看 val。
12. **解释标签不实**：AAL116 网络归属以加载结果为准；没有独立真实 FC 数据不应制作“逐轮真实 FC 演化”的论文主图。
13. **提前合并 RECC/MoE**：当前仓库只是设计方案；不把病理专家当已存在组件，也不把其效果归功于 Loop。

---

## 14. 测试清单（建议 `tests/test_tfm_loop.py`）

```text
[ ] test_none_mode_equivalence_and_checkpoint_reload
[ ] test_tied_weight_identity_and_parameter_count
[ ] test_grouped_plc_block_call_counts
[ ] test_all_context_lengths_and_cpm_shapes
[ ] test_causal_temporal_attention_no_future_leakage
[ ] test_sc_matrix_once_per_forward_and_broadcast
[ ] test_full_backward_reaches_shared_block_and_step_mlp
[ ] test_zero_init_persistence_and_hamd_film_identity
[ ] test_old_and_new_checkpoint_arch_guards
[ ] test_ema_and_freeze_backbone_with_shared_params
[ ] test_trace_off_stats_states_prediction_equivalence
[ ] test_uniform_schedule_sums_to_one
[ ] test_full_shortcut_consistency_stopgrad
[ ] test_budget_invalid_values_raise
[ ] test_halting_off_equals_fixed_max_budget
[ ] test_halting_active_batch_restores_original_order
[ ] test_eval_repeatability_no_dropout_noise
[ ] test_amp_and_cpu_smoke_no_nan
[ ] test_interpretability_maps_have_roi_mapping_or_skip
[ ] test_roi_perturbation_random_control_deterministic
```

示例测试命令：

```bash
python -m pytest tests/test_tfm_backbone.py tests/test_tfm.py tests/test_tfm_loss.py -q
python -m pytest tests/test_tfm_loop.py -q
python main.py --help | grep tfm_loop
```

测试必须包含 `torch.manual_seed`、小型输入 `(B=2,F=8,K=16,D=16)`、CPU smoke；GPU 专有性能/AMP 测试有条件时执行并记录 GPU 型号。

---

## 15. 参考资料及“引用什么、借鉴什么”

1. **Jeddi, A., Ciccone, M., Taati, B. (2026). LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation. ICLR 2026.** <https://arxiv.org/abs/2602.11451>；官方项目 <https://loopformer.github.io/>。借鉴 `(τ,Δτ)` 条件和 full/shortcut consistency，**不是**照搬语言 next-token 损失。
2. **Dehghani, M. et al. (2018). Universal Transformers.** <https://arxiv.org/abs/1807.03819>。共享深度与 adaptive halting 的经典参考。
3. **Zhang, Y., Feng, J., Qin, S. (2026). Recurrent Looped Transformer.** <https://arxiv.org/abs/2610.07591>。这是跨 token/time 的 hidden-state feedback，不作为本方案 P1 的深度 Loop 主路径。
4. **Jain, S., Wallace, B. (2019). Attention is not Explanation.** <https://aclanthology.org/N19-1357/>。提醒 attention weights 不能独立作为解释证据。
5. **Adebayo, J. et al. (2018). Sanity Checks for Saliency Maps.** <https://arxiv.org/abs/1810.03292>。解释需随机化与忠实度测试。
6. **NeuroTwin 现有实现与文档**：<https://github.com/xlongwu/NeuroTwin>，重点 `models/tfm.py`、`models/tfm_backbone.py`、`main.py`、`train/losses.py`、`analysis/next_point_eval.py`、`docs/Version3_docs_1003/NeuroTwin_TFM_RECC_详细应用方案.md`。

---

## 16. 给 Codex 的阶段执行总指令（可直接复制）

> 阅读项目根目录 `AGENTS.md`，并以本文件作为唯一 Loop 研究实施蓝图。首先核对当前 HEAD 与该方案所述 TFM-only 结构、训练脚本、入口及评估协议；如实现已发生变化，输出差异和影响，先完成必要审计。按 Phase 0 → Phase 6 顺序实施，默认只完成下一个**完整阶段**，不跨阶段加入尚未证实的模块；同一阶段内自行拆分合理工作包并执行，不必逐个文件向用户追问。任何模型/损失更动都设置默认关闭开关，保持原 TFM 行为回归。每阶段结束输出：实际修改文件、关键公式/参数、运行的测试命令与结果、未完成/不可验证事项、实验数据与决定下一阶段去留的 Go/No-Go 证据。不得覆盖已有 checkpoint、原始数据、split manifest，不得使用 test 集调参。P1 先实现结构与测试，不默认启动完整训练；长实验需要实际资源可用时再执行。严格区分当前已实现模块与仅在 `docs/` 规划的 RECC/MoE；仅研究 `models/tfm.py` 等当前 TFM 路线，不恢复 legacy GraphODE 架构。最优结构通过 subject-level val、计算效率和解释性忠实度联合选型，不能以单次种子或视觉效果作为结论。

---

## 17. 最终交付验收矩阵

| 交付物 | 最低要求 |
|---|---|
| 代码 | 默认 `none` 复现旧行为；tied/grouped/plc 至少 3 种主干；可选 step-cond；strict checkpoint |
| 训练 | HC/MDD 两阶段命令可运行；3 seeds；train-only normalizer；有效 split manifest |
| 预测实验 | U4、shared 结构、trivial baselines；one-step、CPM horizon、rollout；被试级统计 |
| 弹性实验 | 固定 vs elastic vs consistency；每预算 latency/质量曲线 |
| Halting | 离线验证阈值、上线前真实速度测试、误差非劣约束 |
| 解释 | Loop ROI/network updates、逐轮预测差值、SC null、Top-k perturbation、稳定性、负对照 |
| 医学解释 | 没有将内部 loop 等同真实时间或真实 FC；探索性 HAMD 关联含协变量可用性说明 |
| 复现 | config、代码 SHA、seed、split hash、checkpoint hash、原始 per-subject 结果与生成脚本 |
| 论文结果 | 含失败/负结果、限制与完整消融，而不是仅呈现最优模型 |

**建议最初的资源投入边界**：只做 P0+P1+P2 的最小可行结构比较，即 U4 vs T1R4 vs T2R2 vs PLC（不加 elastic、halting、MoE）。若已经发现主干共享没有任何质量或效率优势，及早停止，不让后续复杂机制掩盖本质问题。
