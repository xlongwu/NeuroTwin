# NeuroTwin 个体化脑动力学模型优化方案

> **版本**：v1.0  
> **制定日期**：2026-11-02  
> **项目**：[NeuroTwin](https://github.com/xlongwu/NeuroTwin)  
> **方案类型**：模型结构设计与训练策略优化  
> **状态**：待实施（设计方案，不代表已有实验结果）  
> **基线**：截至 2026-10-28 的 H1–8 研究主线；以当前 `main` 分支、实际 checkpoint 内嵌配置及数据划分为准。

---

## 1. 研究目标与核心问题

### 1.1 总体目标

建立面向个体化 rs-fMRI 的神经动力学模型，使模型不仅能预测下一时刻的 BOLD，而且能够保留**个体特异性（individual specificity）**、**可重复性（reproducibility）**与**疾病相关性（disease relevance）**的脑动态特征。

核心研究问题：

> 如何让从个体 rs-fMRI 中学习的神经动力学，不仅能够准确预测下一时刻的 BOLD，而且能够保留具有个体特异性、可重复性和疾病相关性的脑动态特征？

研究上区分四个概念：

- **预测能力**：给定真实历史，预测未来 BOLD 的误差与相关性。
- **动力学保真度**：自主生成轨迹是否保留动态 FC、频谱、时间相关与状态转移等特征。
- **个体化有效性**：更换被试的个体参数是否会改变预测；同一被试的不同观测段是否产生相近的稳定参数。
- **疾病相关性**：学习到的动力学差异是否在控制站点、人口学和其他混杂后，与 MDD 或临床量表存在独立关联。

**研究定位**：构建数据驱动、功能连接约束的个体化脑动力学数字孪生候选框架。不把预测能力直接等同于可辨识的神经机制，也不把模型内部扰动等同于真实刺激或临床治疗效应。

### 1.2 需要解决的三类结构性问题

1. **个体化主要体现在条件调制和初始状态**：现有共享 Graph-ODE 通过 Profile/临床条件的 FiLM 和标量扩散率调整个体计算过程，但缺少明确的个体化动力学算子参数化。
2. **Profile 缺少稳定性与区分性约束**：前 32 TR 得到的个体向量可能混合瞬时脑状态、噪声和站点差异；仅靠预测损失不足以保证它编码稳定的个体属性。
3. **疾病适配与疾病机制难以区分**：HC 预训练后对 MDD 进行微调，有利于迁移预测，却不保证模型权重变化对应可解释的病理动力学偏移。

### 1.3 当前项目基线及解释边界

当前 NeuroTwin 的关键实现为：

| 模块/任务 | 已有实现 | 对本方案的含义 |
|---|---|---|
| 时空编码器 | Temporal patch embedding、因果时间注意力、图引导 ROI 注意力 | 保留，不整体替换 Transformer |
| 个体输入 | BOLD 历史、最早 32 TR 校准、32 维 Profile、人口学/临床条件 | 复用既有输入与预处理 |
| 功能图 | 历史 `|FC|`、可学习图、低秩残差、Sinkhorn | 当前是功能图先验，**不是独立实测 SC** |
| 动力学 | 条件化 Graph-ODE，Heun RK2 | 在原向量场中加入低秩个体参数 |
| 当前主任务 | H1–8 多步预测及重建/FC/频域联合损失 | 新方案第一轮仍以 H1–8 为训练和评价边界 |
| MDD 适配 | HC 预训练；MDD 条件增量与微调 | 改造为显式病理动力学增量 |
| 当前测试结果 | HC H1 MAE 约 0.0906，H1–8 MAE 约 0.5344；H1 空间 PCC 约 0.9938、H8 约 0.4286 | 单步跟踪好于固定锚点无真值反馈的多步外推 |
| 既有长期实验 | 旧 H32 自由推进实验发现远期退化，相关实验已归档 | 只能作为历史诊断，不代表当前权重的长期结果 |

**注意**：当前 Profile 并非只承载人口学信息；图、当前历史表示与 Profile 均可以因被试不同而变化。真正的不足是它们尚未形成充分可分析、具有独立约束的个体动力学参数体系。

---

## 2. 总体架构

### 2.1 从条件化共享动力学到分层动力学

总体设计采用：

**Shared Healthy Dynamics + Individualized Dynamics Parameters + Pathological Dynamics Residual**

```text
个体 rs-fMRI BOLD ────────────────────────────────────────────┐
    │                                                       │
    ├─ 固定前缀校准、可见历史                                │
    │                                                       │
    ├─ Stable Profile Encoder ── p_s ── Hypernetwork ──┐     │
    │                       │                           │     │
    │                       └─ Stable/Contrastive Loss  │     │
    │                                                   │     │
    └─ Temporal–ROI Encoder ── z_s(t), M_s              │     │
           │                       │                    │     │
           └── History-FC / learned graph ── A_s ◄─────┤     │
                                                      │     │
         HC shared vector field f_HC  ◄───────────────┘     │
                        │                                   │
                Individualized Graph-ODE                    │
                        │                                   │
           ┌────────────┴────────────┐                      │
           │                         │                      │
     H1–8 BOLD trajectory      Dynamic signatures          │
           │                         │                      │
           │         MDD branch Δf_path(d_s) ◄─ Path Encoder┘
           │                         │
           └─────────────┬───────────┘
                         │
             HC reference / MDD deviation
```

模型拆为三个核心模块：

1. **IDG — Individualized Dynamics Generator**：从个体 Profile 生成区域参数、低秩动力学算子和个体图调制。
2. **SDRL — Stable–Dynamic Representation Learning**：分离稳定个体表征与瞬时状态，联合预测损失和跨窗口个体表征学习。
3. **PDD — Pathology-aware Dynamics Decomposition**：保留 HC 健康动力学作为基础，使用低秩病理增量表达 MDD 相关的候选动力学偏移。

三个模块的关系是 **IDG 定义个体动力学如何变化，SDRL 约束变化来自稳定的个体信息，PDD 描述在健康个体差异基础上的额外疾病偏移**。

### 2.2 符号与张量定义

| 符号 | 含义 | 建议维度 |
|---|---|---|
| $X_s$ | 第 $s$ 名被试的可见 BOLD | $[B,K,116]$ |
| $X_s^{\mathrm{cal}}$ | 不含预测目标的校准 BOLD 前缀 | $[B,32,116]$（外部布局） |
| $z_s(t)$ | 逐 ROI 潜在状态 | $[B,116,256]$ |
| $M_s$ | 历史记忆 | $[B,116,256]$ 或现有兼容布局 |
| $p_s$ | 稳定个体 Profile | $[B,32]$ |
| $c_s$ | 人口学/临床条件 | $[B,32]$ |
| $d_s$ | BOLD 推断的候选病理表征 | 建议 $[B,16]$ |
| $A_s$ | 个体有效功能耦合图 | $[B,116,116]$ |
| $g_s$ | 个体动力学低秩系数 | $[B,8]$ |
| $\kappa_s$、$\gamma_s$ | 区域扩散率、衰减/恢复系数 | 各 $[B,116]$ |

上述建议维度用于首轮实现，实际张量形状需与当前模型的 `tfm.py` 接口保持一致。

---

## 3. 模块一：Individualized Dynamics Generator（IDG）

### 3.1 设计动机

当前模型共享主要的神经残差权重，个体差异通过不同的初始 $z_s$、有效图 $A_s$、Profile/临床条件 FiLM 以及逐被试标量扩散率引入。该方式具有泛化和参数效率优势，但个体动力学差异的参数化不够明确。

IDG 的目标不是为每名被试学习独立模型权重，而是学习一个可泛化的映射：

$$
\Theta_s=H_\phi(p_s),\qquad p_s=E_{\mathrm{profile}}(X_s^{\mathrm{cal}}).
$$

其中 $H_\phi$ 是**跨被试共享**的参数生成器，输出该被试的小规模动力学参数 $\Theta_s$。

### 3.2 个体参数生成器

建议采用两层 MLP：

$$
H_\phi:\mathbb R^{32}\to\mathbb R^{128}\to\mathbb R^{248}.
$$

输出拆分为：

$$
\Theta_s=\{\kappa_s\in\mathbb R^{116},\ \gamma_s\in\mathbb R^{116},\ g_s\in\mathbb R^8,\ a_s\in\mathbb R^8\}.
$$

| 参数 | 功能 | 建议初始化及约束 |
|---|---|---|
| $\kappa_s$ | 逐 ROI 图扩散率 | 以当前标量扩散率为基准，加小幅个体修正，最终通过 softplus 保持为正 |
| $\gamma_s$ | 逐 ROI 恢复/衰减系数 | 初始为零或非常小；避免增加原模型的长程收缩问题 |
| $g_s$ | 个体低秩动力学调制系数 | 初始化令新增分支零输出，训练后可学习 |
| $a_s$ | 个体功能图低秩残差系数 | 控制新增图残差幅度并保持最终图非负/对称 |

不应直接生成完整 $256\times256$ 权重矩阵。低秩结构在参数量、计算开销与可分析性之间更合适。

### 3.3 个体化低秩动力学算子

定义共享基矩阵：

$$
U,V\in\mathbb R^{256\times r},\quad r=8.
$$

令个体算子为：

$$
\Delta W_s=U\operatorname{diag}(g_s)V^\top.
$$

在原共享神经残差 $R_\theta$ 基础上增加：

$$
R_s(z)=R_\theta(z,A_s,c_s,M_s)
+\alpha\,\Delta W_s\,\sigma(\operatorname{LN}(z)),
$$

其中 $\alpha$ 初始取 $0.05$。批处理时可避免显式构造 $[B,256,256]$ 矩阵，依次执行 $zV$、逐低秩维度乘 $g_s$、再乘 $U^\top$。

**与 FiLM 的区别**：现有 FiLM 也会改变向量场，不能说没有个体化动力学；新增低秩算子使这种变化具有显式、低维、可交换和可分析的参数形式。

### 3.4 区域扩散率与局部时间尺度

现有扩散率主要为被试级标量。建议升级为区域级：

$$
\kappa_{s,i}=\operatorname{softplus}(\kappa_{0,i}+[W_\kappa p_s]_i),\quad i=1,\dots,116.
$$

若增加逐 ROI 恢复项，可写为：

$$
-\operatorname{diag}(\gamma_s)z_s.
$$

**实施注意**：在现有模型出现长程动态衰减的背景下，$\gamma_s$ 不宜一开始赋予较大正值。优先实现区域扩散率和低秩算子；区域衰减作为同一模块中的可关闭选项，其是否启用由固定对照结果决定。

### 3.5 个体化功能图调节

当前图由历史功能图与共享学习图混合，仍应明确称为**功能耦合图**，而不是独立测量的解剖 SC。

推荐形式：

$$
A_s=\operatorname{Sinkhorn}\left(
\lambda_s A_{\mathrm{hist},s}
+(1-\lambda_s)A_\theta
+\Delta A(p_s)
\right),
$$

$$
\Delta A(p_s)=\operatorname{Sym}\left(
U_A\operatorname{diag}(a_s)V_A^\top
\right).
$$

其中 $\operatorname{Sym}(X)=(X+X^\top)/2$。在 Sinkhorn 之前延续代码已有的非负裁剪、对角自环与数值归一化；保留对外输出 `A_eff` 的接口。

第一轮可继续保留原图残差，实现 $\Delta A(p_s)$ 作为额外可开关分支，避免重复引入两套功能相近的图学习机制。最终保留的结构应去除无效重复分支。

### 3.6 新版 Graph-ODE

新版无外部控制输入时的向量场建议为：

$$
\boxed{
\frac{\mathrm d z_s}{\mathrm d t}
=-\operatorname{diag}(\kappa_s)L_{A_s}z_s
-\operatorname{diag}(\gamma_s)z_s
+R_\theta(z_s,A_s,c_s,M_s)
+\alpha\Delta W_s\sigma(z_s)
}
$$

BOLD 读出与求解器沿用原设计：

$$
\frac{\mathrm dq_s}{\mathrm dt}=W_yz_s+b_y,
$$

积分继续使用 Heun RK2，每 TR 两个子步。原有可选外部控制 $u$ 的门控项保留，但**不纳入当前无真实干预数据的核心训练与科学结论**。

**关键实现原则**：尽量保留现有共享网络、输入输出形状、统计量计算和求解器。参数生成器增加的个体参数在单次无观测 rollout 内固定；新观测进入后重新估计动态状态，是否更新 Profile 由独立策略决定。

---

## 4. 模块二：Stable–Dynamic Representation Learning（SDRL）

### 4.1 稳定个体表征与瞬时脑状态分离

当前 Profile 由同一扫描最早 32 TR 的共享编码器表示池化得到。短窗 BOLD 混合了稳定个体差异、瞬时状态和测量噪声；只使用预测损失很难确保 Profile 的稳定性。

建议明确区分：

$$
p_s=E_{\mathrm{stable}}(X_s^{\mathrm{cal}}),
\qquad
z_s(t)=E_{\mathrm{state}}(X_{s,t-K+1:t}).
$$

- $p_s$：在同一次扫描内相对稳定，用于生成动力学参数。
- $z_s(t)$：随时间演化，用于给定时刻的预测初始化。
- 两个编码器可共享大部分现有 Transformer，只在池化头/投影头上分支，以降低改造成本。

### 4.2 同被试跨窗口采样

训练时，对同一被试选择两个**非重叠且位于预测锚点之前**的窗口 $X_s^A$、$X_s^B$：

$$
p_s^A=E_{\mathrm{stable}}(X_s^A),\qquad
p_s^B=E_{\mathrm{stable}}(X_s^B).
$$

示例：TR 1–32 与 TR 33–64；仅对预测锚点足够晚的训练样本构造这一辅助任务。不得为了构造一致性正样本使用预测锚点之后的真值。

为避免训练/推理不一致，一部分训练样本始终以**最早 32 TR**作为主 Profile；第二个窗口只用于稳定性辅助损失。推理阶段仍使用最早 32 TR 提取 Profile。

### 4.3 一致性与主体对比损失

同被试窗口一致性损失：

$$
L_{\mathrm{cons}}
=\frac1N\sum_s\left[1-\cos(p_s^A,p_s^B)\right].
$$

仅用一致性损失容易将所有 Profile 推向同一常量，需要增加跨被试可区分性。建议采用主体对比损失：

$$
L_{\mathrm{subj}}=-\frac1N\sum_s
\log\frac{\exp(\operatorname{sim}(p_s^A,p_s^B)/\tau)}
{\sum_j\exp(\operatorname{sim}(p_s^A,p_j^B)/\tau)}.
$$

推荐 $\tau=0.1$；如使用归一化投影头，可将对比损失作用于投影后的向量，并保留原始 $p_s$ 作为动力学参数输入。

**训练采样要求**：

- 正样本：同一被试、不同可见历史窗口。
- 负样本：其他被试；尽量包含同站点、年龄相近和性别相同的样本，避免用站点/人口学捷径识别身份。
- 若缺少两段合法窗口，跳过该样本的对比损失，但仍执行正常预测训练。
- 不将 `subject_id` 作为推理输入，不建立逐被试 ID embedding。

### 4.4 可重复性的科学边界

只有一次扫描时，双窗口一致性反映的是**扫描内稳定性**，不能等同于跨访视 Test–Retest Reliability。真正的跨访视重复性需要可靠的同一受试者重复扫描或重复访视数据。

建议同时评价：

1. 同被试不同窗口的 Profile/个体参数相似度。
2. 跨被试 Profile 分离度和个体检索准确率。
3. **Profile 参数交换实验**：保持 BOLD 历史和其他条件不变，只替换为另一被试的 Profile，考察预测变化。
4. 若未来获得重复扫描，进一步报告 ICC、跨访视 Profile 与个体动力学参数稳定性。

Profile 交换指标：

$$
\Delta_{\mathrm{swap}}
=MAE(\hat X_{p_{\mathrm{other}}},X)
-MAE(\hat X_{p_{\mathrm{own}}},X).
$$

当 $\Delta_{\mathrm{swap}}>0$ 时，说明使用原被试 Profile 对该评价任务更有利；但这仍是**模型个体信息使用证据**，不自动等同于生物参数被准确辨识。

---

## 5. 模块三：Pathology-aware Dynamics Decomposition（PDD）

### 5.1 设计目的

当前 HC 预训练后 MDD 微调能够改善疾病群体适配，但全模型权重变化混合了普通域适配、个体差异和潜在疾病效应。

建议显式分为健康基础动力学和 MDD 病理候选残差：

$$
f_{\mathrm{MDD}}(z_s)=
f_{\mathrm{HC}}(z_s;p_s,A_s)
+\Delta f_{\mathrm{path}}(z_s;d_s).
$$

这里 $f_{\mathrm{HC}}$ 在健康数据上学习，$d_s$ 来自个体 BOLD 的病理候选表征，而不是直接从疾病标签或 HAMD 数值生成。

### 5.2 低秩病理残差

定义：

$$
d_s=E_{\mathrm{path}}(X_s),\qquad d_s\in\mathbb R^{16},
$$

$$
\Delta f_{\mathrm{path}}
=\alpha_d U_d\operatorname{diag}(g_d(d_s))V_d^\top\sigma(z_s).
$$

推荐病理低秩 $r_d=4$，并通过低容量、范数正则和 HC 样本作为参照，抑制病理分支吸收所有普通个体差异。

可选使用一个疾病相关门控 $m_s\in[0,1]$ 来控制病理分支激活。使用 HC 样本时将该分支限制在零附近，在 MDD 样本上学习适配增量。

**概念约束**：该分支表示“数据支持的 MDD 相关候选动力学偏移”，不是已被证明的真实病理机理，更不是因果效应。

### 5.3 临床标签与条件输入的分工

- HC 预训练继续使用已有基本人口学条件（性别、年龄、教育年限），保留现有缺失 mask 和训练集统计量。
- 疾病标签 `HC/MDD` 可作为训练病理表征的**辅助监督标签**，不能同时作为无条件病理发现的输入后再宣称识别疾病。
- 用 BOLD 学习的 $d_s$ 执行辅助疾病识别：

$$
L_{\mathrm{disease}}=\operatorname{BCE}(h(d_s),y_s).
$$

- 若使用 HAMD 作为外部关联或预测目标，不能在同一条分析路径中又把相同 HAMD 作为预测输入。临床回归可写为：

$$
L_{\mathrm{clinical}}
=\operatorname{Huber}(\widehat{\mathrm{HAMD}}_s,\mathrm{HAMD}_s).
$$

- 保留原临床条件预测分支，作为**条件化预测任务**；另建立不输入症状量表的 **BOLD-only 病理表征任务**。两种结果分别报告，避免信息泄漏和循环论证。

### 5.4 避免过早上复杂 MoE

当前临床条件对预测的历史贡献较弱，直接上多专家 MoE 可能增加容量，却不能保证个体或疾病信息被真正使用。

优先采用低秩病理残差：容量小、易解释、与 HC 基础模型关系明确。如果该模块已经产生可信的独立增量，再考虑更复杂的疾病异质性子群或专家路由。

---

## 6. 训练策略

### 6.1 总体路线

采用两阶段主线：

**阶段 I：HC 共享 + 个体化动力学学习**  
**阶段 II：MDD 病理动力学增量适配**

长程随机模拟作为完成 H1–8 个体化研究之后的独立研究扩展，不与首轮模型重构混在一起。

### 6.2 阶段 I：HC 个体化动力学预训练

**训练数据**：沿用已有 HC train 被试划分；每个训练样本使用合法的历史与预测锚点，增加同被试历史窗口配对。  
**初始化**：从当前 HC 模型权重加载已有模块；新增 IDG/SDRL 分支零输出或保守初始化。  
**训练范围**：共享主干、Profile、IDG、Graph-ODE 均可端到端优化；可以使用短暂的新增模块 warm-up，但无需分成多轮人工验收。

保留现有主预测目标：

$$
L_{\mathrm{base}}=L_{\mathrm{rec}}+L_{\mathrm{FC}}+L_{\mathrm{freq}}
$$

上式为结构性简写；在当前代码中，三项仍可沿用已实现的 Kendall 不确定性加权，而不应直接视为已固定等权。

新增时间差分损失：

$$
L_\Delta=\frac{1}{FH}\sum_{i,h}
\left|
(\hat x_{i,h}-\hat x_{i,h-1})
-(x_{i,h}-x_{i,h-1})
\right|.
$$

其中 $h=1$ 时 $x_{i,0}$ 和 $\hat x_{i,0}$ 均为最后一个真实历史观测点。

新增个体参数正则：

$$
L_{\mathrm{reg}}=
\mathbb E_s\left[
\|\Delta W_s\|_F^2+\|\Delta A(p_s)\|_F^2
+\|\kappa_s-\kappa_0\|_2^2
\right]
$$

推荐总目标：

$$
\boxed{
L_{\mathrm{HC}}=L_{\mathrm{base}}
+\lambda_\Delta L_\Delta
+\lambda_{\mathrm{subj}}L_{\mathrm{subj}}
+\lambda_{\mathrm{reg}}L_{\mathrm{reg}}
}
$$

**建议初值**：$\lambda_\Delta=0.1$，$\lambda_{\mathrm{subj}}=0.05$，$\lambda_{\mathrm{reg}}=10^{-3}$，主体对比温度 $\tau=0.1$。不同损失量级可能差异显著，实践中需按原始数值和梯度贡献调节；这些值不是已经证明最优的超参数。

如果新时间差分项与当前 FC/Freq 权重产生明显冲突，优先保留一个清晰的预测主目标与个体稳定性目标，不必无限叠加损失。首轮建议固定上述配方，按同一验证集标准完成模型对照。

### 6.3 阶段 II：MDD 病理增量适配

**初始化**：加载阶段 I 的 HC 个体化模型。  
**第一段训练**：冻结健康共享动力学与 IDG，训练 PDD 病理编码器、低秩残差及病理读出。  
**第二段训练（仅有必要时）**：以较小学习率解冻最后编码块或少量 Adapter，不直接整体改写 HC 动力学。

建议训练时保留一定比例的 HC 样本作为健康参照：HC 输入下病理增量应接近零；MDD 输入下学习额外候选偏移。避免仅在 MDD 样本上使残差分支吸收普通队列或扫描差异。

病理目标：

$$
L_{\mathrm{MDD}}
=L_{\mathrm{pred}}
+\lambda_d L_{\mathrm{disease}}
+\lambda_c L_{\mathrm{clinical}}
+\lambda_r L_{\mathrm{path-reg}}.
$$

建议起点：$\lambda_d=0.1$、$\lambda_c=0.05$、$\lambda_r=10^{-3}$。若没有独立、有效的临床目标，直接关闭 $L_{\mathrm{clinical}}$，不为补齐损失而虚构标签。

**疾病因果解释边界**：疾病辅助分类或临床评分关联最多证明模型提取了与疾病相关的统计表征；不应直接声称提取到病理因果机制。

### 6.4 训练采样与参数更新的对应关系

- 个体 Profile 每个训练样本由其合法校准前缀计算，不从未来目标中获取信息。
- SDRL 的两窗口任务仅对可以构造合法窗口的样本启用；主预测任务继续使用原历史窗口采样。
- 在一次固定锚点的自主 rollout 中，$p_s$、IDG 生成的个体动力学参数保持固定，$z_s(t)$ 连续演化。
- 有新观测的状态更新中，继续利用 `update_state()` 重新估计动态状态与图。是否重新校准 $p_s$ 作为另一个独立策略，不要在训练与评估时暗中混用。
- 训练/验证/测试继续按被试独立划分，保留原 test；站点混杂应在病理解释中控制。

---

## 7. 长程保真度：独立后续扩展

### 7.1 为什么不能只靠长程 MAE

现有预测目标主要依赖确定性点估计。对于不可完全预测的自发 BOLD 波动，点误差最优解会倾向条件均值；重复积分后容易出现动态收缩。较低 MAE 不保证长程 FC、PSD、FCD 或状态转移的真实性。

当前 H1–8 覆盖 16 秒（若 TR=2 s），其频率分辨率为：

$$
\Delta f=\frac{1}{8\times2}=0.0625\ \mathrm{Hz}.
$$

仅 8 TR 上的 FC 与频谱统计噪声较大，不适合直接承担完整的长程脑动力学真实性约束。

### 7.2 保留确定性预测，引入结构化随机创新

后续可以在确定性 Graph-ODE 上增加低秩、个体条件化的随机项：

$$
\mathrm dz=f_\theta(z,p_s,d_s)\,\mathrm dt
+B_\psi(z,p_s)\,\mathrm dW_t.
$$

对应两类输出：

1. **确定性预测**：以 MAE/RMSE 为主要终点，预测条件均值。
2. **随机动力学轨迹**：以多次自由样本的 FC、PSD、FCD、ACF 与概率评分评价统计保真度。

概率评价可考虑 CRPS、Energy Score、Variogram Score 等；原有方差头只输出对角边际方差，不能自动证明模型已经学习跨 ROI 的联合分布。

**实施决策**：暂不修改当前 H1–8 主线的训练协议来追求长程分布。待 IDG + SDRL + PDD 得到明确结果后，再作为独立实验重启较长时程训练与生成。

---

## 8. 评价体系和对照实验

### 8.1 固定对照组

| 配置 | 结构 | 目的 |
|---|---|---|
| **B0** | 当前 NeuroTwin H1–8 基线 | 参照模型 |
| **B1** | B0 + IDG | 评估显式个体参数化的收益 |
| **B2** | B1 + SDRL | 评估稳定表征约束的增量价值 |
| **B3** | B2 + PDD | 评估病理残差的预测与疾病相关增量 |

对照使用相同被试划分、目标窗口、主要评价协议及可比训练预算。新架构与旧 checkpoint 不能因参数名相似而视为同一模型；应保存配置和权重对应关系。

### 8.2 预测能力

| 维度 | 核心指标 | 主要协议 |
|---|---|---|
| 下一时刻预测 | H1 MAE、空间 PCC、相对 Persistence / AR 收益 | 固定锚点 H1 |
| 短程无真值反馈预测 | H1–8 MAE、逐 horizon 空间 PCC、时间差分误差 | 固定锚点 H1–8 |
| FC 保真度 | FC MSE、单锚点 FC 边 PCC | 固定锚点 H1–8；明确短窗限制 |
| 实际观测跟踪 | 滚动单步 MAE、完整序列 FC | 每 TR 输入更新后的真实历史，**不等于自由生成** |
| 后续长程扩展 | 时间 PCC、方差比、FCD、PSD、ACF、概率评分 | 独立长程自由生成协议 |

应先按被试聚合再跨被试等权汇总；滚动单步与固定锚点 H1–8 不应混为同一个多步性能指标。

### 8.3 个体化能力

- 同被试双窗口 Profile 的 cosine 相似度和参数一致性。
- 不同被试 Profile 的分离度及个体检索准确率。
- Profile 交换后 $\Delta_{\mathrm{swap}}$ 是否为正，以及其逐被试分布。
- IDG 生成的 $\kappa_s$、$g_s$、图残差统计和脑网络层面的个体异质性。
- 对比可选的零 Profile、均值 Profile、他人 Profile，以区分真实依赖与仅有参数变化。
- 控制站点与人口学信息后，个体差异是否仍然明显。

### 8.4 病理与临床相关性

- HC 健康参照下的 MDD ROI、网络和全脑预测偏离。
- 站点匹配或相应协变量控制后的 HC/MDD 动力学参数及病理残差差异。
- 与 HAMD/HAMA 的独立关联；若该评分曾作为模型输入，不能用同一路径作独立验证。
- 病理表征相对年龄、性别、教育、站点和静态 FC 描述量的增量价值。
- 健康负控与不同站点的敏感性分析，避免将站点效应当成疾病效应。

**解释原则**：B3 即使能够提高疾病分类 AUC，也不能单凭这一指标把病理残差解释为真实疾病机制。

---

## 9. 代码改造清单

### 9.1 推荐新增及修改文件

| 文件 | 修改内容 |
|---|---|
| `models/personal_dynamics.py`（新增） | `PersonalDynamicsGenerator`；输出区域参数、个体低秩系数、图调节系数 |
| `models/tfm_personal.py` | 稳定 Profile 投影头；双窗口表示接口，保持原 Profile 输出兼容 |
| `models/tfm_dynamics.py` | 支持逐 ROI 扩散率、可选局部恢复项、低秩个体动力学残差 |
| `models/graph_prior.py` | 可选 Profile-conditioned 低秩图修正，保留 `A_eff` 定义 |
| `models/pathology_dynamics.py`（新增） | 病理编码器、低秩病理残差、可选 HC 近零门控 |
| `models/tfm.py` | 集成 IDG/PDD，返回 `personal_params`、`pathology_params` 诊断字典 |
| `utils/dataloader.py` | 合法的同被试双窗口采样和标签/协变量传递 |
| `main.py` | HC + 个体训练、MDD 病理增量训练及联合损失 |
| `configs/neurotwin.json` | 在新实验配置中显式记录模块开关、维度、损失权重 |
| `analysis/` | Profile 交换、个体参数分布、疾病偏移、固定锚点预测等结果可视化 |

建议使用**新配置文件和新的 checkpoint 输出目录**，不覆盖当前已完成的 H1–8 正式结果；保留原模型作为 B0 对照。

### 9.2 建议初始参数

```yaml
model:
  num_rois: 116
  hidden_dim: 256
  profile_dim: 32
  personal_dynamics:
    enabled: true
    generator_hidden_dim: 128
    rank: 8
    residual_scale: 0.05
    regional_diffusion: true
    regional_decay: false        # 首轮默认关闭
    personalized_graph: true
    graph_rank: 8
  stable_representation:
    enabled: true
    window_length: 32
    contrastive_temperature: 0.1
  pathology_dynamics:
    enabled_in_hc_stage: false
    enabled_in_mdd_stage: true
    pathology_dim: 16
    rank: 4

training:
  stage_1: hc_personalized
  stage_2: mdd_pathology_adapter
  prediction_horizon: 8
  lambda_delta: 0.1
  lambda_subject: 0.05
  lambda_personal_reg: 0.001
  lambda_disease: 0.1
  lambda_clinical: 0.05
  lambda_path_reg: 0.001
```

此段是**建议配置草案**，不是当前仓库现有 JSON 解析器可直接运行的命令。实现时将其映射进实际配置系统，或新建实验配置，不要伪装成已存在的开关。

### 9.3 开发执行顺序

1. 在现有 Graph-ODE 上实现 IDG 个体参数生成和低秩残差，完成 B1 训练。
2. 增加 SDRL 合法窗口采样、对比训练与 Profile 交换评价，完成 B2 训练。
3. 将 PDD 作为 HC 基础上的独立病理增量模块，完成 B3 的 MDD 训练。
4. 按同一协议输出 B0–B3 的预测、个体、病理三组可视化结果，并进行科学解释。
5. 根据实际结果删去没有贡献的候选分支，保持代码结构精简；不要为每个实验增加额外复杂的审核框架。

每完成一个实际模块，即同步完成相应代码、配置与训练入口，再用真实模型实验检验，不采用大量独立“验证阶段”代替实施。

---

## 10. 预期研究贡献与可检验假设

### 10.1 核心创新方向一

**Profile-conditioned Low-rank Neural Dynamics**

> 从个体 BOLD 学习稳定 Profile，生成低秩神经动力学算子和区域耦合参数，使模型具有显式的 subject-conditioned dynamics，而不只是利用个体状态初始化与 FiLM 特征调制。

可检验假设：个体参数化不显著损害短程预测的前提下，能够提高个体依赖性、参数差异的稳定性，以及个体交换实验的区分度。

### 10.2 核心创新方向二

**Stable–Pathological Dynamics Disentanglement**

> 利用双窗口稳定表征学习约束健康个体动力学，并在健康基础向量场上学习低秩 MDD 残差，增强个体差异与疾病相关偏移的结构可分性。

可检验假设：在控制站点/人口学变量后，病理残差相对传统静态 FC 具有独立的疾病或症状相关信息。

### 10.3 尚不能声称的能力

- 未提供配对 dMRI 时，不能声称模型受真实个体解剖 SC 约束。
- 无独立跨访视数据时，不能把扫描内稳定性写作真正的 Test–Retest Reliability。
- 无真实干预标签与对照时，不能把模型控制输入的轨迹差解释为治疗反应或因果影响。
- 仅有 HC/MDD 区分效果时，不能宣称已经恢复了真实疾病机制。
- H1–8 模型的短程成绩不能直接证明长期自主生成的脑动力学真实性。

---

## 11. 最终建议与近期交付目标

**优先实施 IDG + SDRL，随后实施 PDD。** 不建议现阶段同时引入更大的 Transformer、复杂 MoE 和随机长程生成器，以免参数增长掩盖个体化机制的真实贡献。

近期应形成以下完整研究链：

```text
个体 rs-fMRI
    ↓
稳定个体 Profile + 当前脑状态
    ↓
个体化低秩动力学参数
    ↓
Graph-ODE 短程预测
    ↓
个体参数稳定性 / 参数交换检验
    ↓
HC 基础动力学 + MDD 病理候选残差
    ↓
站点控制后的病理差异与临床关联
```

最终判断标准不应只是 MAE 是否下降，而是：

1. **模型预测了什么**：下一 TR 和 H1–8 是否足够准确。
2. **模型记住了谁**：不同被试是否拥有有效、可区分且在同扫描内稳定的动力学参数。
3. **模型解释了什么**：这些参数或病理偏移是否携带超出普通静态 FC 的疾病相关信息。

这条路线在现有横断面 rs-fMRI 数据条件下可实施，并能为后续跨访视自适应、真实结构连接与受控虚拟干预研究奠定基础。

---

## 附：参考项目与研究入口

- 当前代码仓库：<https://github.com/xlongwu/NeuroTwin>
- 当前状态：<https://github.com/xlongwu/NeuroTwin/blob/main/docs/项目状态与工作范围.md>
- 实验结论与证据汇总：<https://github.com/xlongwu/NeuroTwin/blob/main/docs/实验结论与证据汇总.md>
- MINDy 相关研究：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7875185/>
- *Building digital twin brains at the limits of measurement*：<https://www.nature.com/articles/s44287-026-00320-8>

> **说明**：本文件是下一版模型研发方案。所列新增模块、数值超参数与预期贡献均为设计建议，尚未在 NeuroTwin 上实施或验证；已有模型表现与历史结果必须以实际 checkpoint 和评估协议为准。
