# NeuroTwin-TFM 引入病理专家与 Router–Expert Competence Coupling（RECC）的详细方案

> 适用范围：仅 `model_arch=tfm`  
> 不考虑 legacy NeuroTwin / `models/neurotwin_moe.py` / legacy MoE 路径  
> 当前基线：Temporal Patching + Causal Temporal Attention + SC-guided ROI Attention + One-Step / CPM 双头 + HAMD→FiLM  
> 目标：在不破坏 TFM 主干与 CPM 结构的前提下，引入一个局部的病理残差专家模块，并利用真实下一时刻 BOLD 监督 Router–Expert 对齐。

---

# 0. 方案摘要

当前 NeuroTwin-TFM 的病理条件化方式为：

```text
HAMD
  ↓
PathologyNormalizer
  ↓
FiLM
  ├─ One-Step Head
  └─ CPM Head
```

这种设计稳定、简单，但只有一个共享条件通路。它能回答：

> HAMD 是否会调制预测？

却无法回答：

> 不同病理状态 / 脑动力学状态是否需要不同的动态修正机制？

本方案建议将 **One-Step Head** 改造成：

```text
共享健康动力学转移
        +
病理专家残差 MoE
```

而 **TFM backbone、SC prior、CPM head 均保持原设计**。

核心结构：

```text
BOLD context
    ↓
TFM Backbone
    ↓
z_t , z_g
    ├───────────────→ Shared Dynamics Transition
    │                         ↓
    │                    Δz_shared
    │
    └─ HAMD + brain state → Router
                              ↓
                      Pathology Experts
                    E0 E1 E2 E3
                              ↓
                         Δz_path
                              ↓
                  z_(t+1)=z_t+Δz_shared+Δz_path
                              ↓
                         shared decode
                              ↓
                       x̂_(t+1)
```

然后进一步加入：

```text
所有 Expert 对同一样本分别预测
        ↓
利用真实 x_(t+1) 计算每个 Expert 的预测误差
        ↓
得到真实 competence distribution q
        ↓
监督 Router probability p
```

即：

$$
\boxed{
\text{真实未来脑状态}
\rightarrow
\text{Expert competence}
\rightarrow
\text{Router calibration}
}
$$

这比直接复制原 ERC 的 activation norm proxy 更适合 NeuroTwin-TFM。

---

# 1. 当前 TFM 结构与需要保留的设计

当前 `models/tfm.py` 的核心结构为：

```text
BOLD context [B,F,K]
    ↓
Context-only RevIN
    ↓
SoftAnatomicalPrior
    ↓
A_eff
    ↓
TemporalPatchEmbed
    ↓
SC-guided Temporal–Variate Block × N
    ↓
context tokens [B,F,P,D]
    ├─ One-Step Head
    └─ CPM Horizon Head
```

其中默认：

```text
F = 116
D = 256
N = 4
patch_len = 4
CPM horizon = 8
```

当前病理条件只在 finetune 阶段存在：

```text
raw HAMD
  ↓
PathologyNormalizer
  ↓
cond
  ├─ OneStepDynamicsHead.film
  └─ CPMHorizonHead.film
```

One-Step 当前：

$$
z_g=A_{eff}z_t
$$

$$
h=F_\theta([z_t,z_g])
$$

FiLM：

$$
h'=h\cdot(1+\gamma(c))+\beta(c)
$$

状态：

$$
z_{t+1}=z_t+h'
$$

预测：

$$
\hat{x}_{t+1}
=
x_t+
Decode(z_{t+1})
$$

当前方案中最重要的保留原则：

1. `TemporalPatchEmbed` 不改；
2. `TFMEncoderBlock` 不改；
3. `SoftAnatomicalPrior` 不改；
4. `One-Step` 的 HC shared transition 权重保留；
5. `CPM` 结构不改；
6. 现有 HC pretrain checkpoint 继续可复用；
7. pathology MoE 只在 MDD finetune 阶段创建；
8. RECC 只作用于病理专家 Router，不修改 TFM backbone 的基本结构。

---

# 2. 为什么 MoE 应该放在 One-Step Dynamics Head，而不是 TFM Backbone

不建议：

```text
TFMEncoderBlock
    ↓
MoE-FFN
    ↓
TFMEncoderBlock
```

原因有四点。

## 2.1 会破坏当前研究问题的清晰度

当前 TFM 已有四个明确创新维度：

```text
Temporal Patching
Causal Temporal Attention
SC-guided ROI Attention
One-Step + CPM
```

如果直接将 MoE 插入每个 Transformer block，会同时改变：

```text
representation learning
temporal modeling
ROI interaction
parameter capacity
routing mechanism
```

消融难以解释。

---

## 2.2 病理差异更适合建模为 shared dynamics 上的 residual

你当前整体研究逻辑本身就是：

```text
健康脑动力学
+
MDD 个体病理偏离
```

所以最自然的形式是：

$$
F_{MDD}
=
F_{shared}
+
\Delta F_{pathology}
$$

而不是：

$$
F_{MDD}
=
\text{一套完全不同的 Transformer backbone}
$$

---

## 2.3 HC 预训练权重可以完整保留

HC pretrain 时：

```text
pathology MoE = 不存在
```

MDD finetune 时再新增：

```text
router
experts
```

原：

```text
one_step_head.transition
one_step_head.decode
```

全部继续加载。

---

## 2.4 Expert 的生物学含义更明确

如果 Expert 放在 One-Step residual transition：

> Expert 表示“相对于共享脑动力学的某类病理状态转移修正”。

这比：

> Expert 是 Transformer 第 3 层某个 FFN 子网络

更容易做医学解释。

---

# 3. 推荐的新 One-Step Head

建议保留：

```python
OneStepDynamicsHead
```

现有字段：

```text
transition
decode
```

全部不改名。

新增：

```text
pathology_moe
pathology_mode
```

推荐：

```text
pathology_mode:
    none
    film
    moe
```

含义：

```text
none:
    无病理条件

film:
    当前 TFM baseline 行为

moe:
    新 pathology residual expert
```

默认仍：

```text
film
```

保证现有 checkpoint / baseline 行为不变。

---

# 4. Shared Dynamics 与 Pathology Residual 分离

在 `moe` 模式：

首先计算：

$$
z_g=A_{eff}z_t
$$

共享 HC 动力学：

$$
h_{shared}
=
F_\theta([z_t,z_g])
$$

$$
z_{shared}
=
z_t+h_{shared}
$$

这一部分不使用 HAMD。

然后病理分支：

$$
\Delta z^{path}
=
MoE(z_t,z_g,c)
$$

最终：

$$
z_{t+1}
=
z_{shared}
+
\Delta z^{path}
$$

预测：

$$
\hat{x}_{t+1}
=
x_t+
Decode(z_{t+1})
$$

即：

$$
\boxed{
z_{t+1}
=
z_t
+
\Delta z_{shared}
+
\Delta z_{path}
}
$$

这是整个方案最重要的结构。

---

# 5. 为什么不建议 One-Step FiLM 与 MoE 同时开启

如果：

```text
shared transition
   ↓ FiLM(HAMD)

同时

pathology MoE
   ↓ HAMD
```

则 HAMD 同时作用两次：

$$
F_{shared}(c)+\Delta F_{MoE}(c)
$$

病理效应来源难以区分。

因此第一版明确：

```text
tfm_patho_mode=film
    → One-Step 只用 FiLM

tfm_patho_mode=moe
    → One-Step 关闭原 FiLM
    → pathology condition 只进入 MoE
```

CPM 仍保留自己的 FiLM。

即：

```text
One-Step:
    film OR moe

CPM:
    film 保留
```

后期如果确实需要：

```text
film + moe
```

可以作为额外消融，但不应作为默认方案。

---

# 6. PathologyResidualMoE 设计

建议新建：

```text
models/tfm_experts.py
```

避免和 legacy MoE 共享代码。

包含：

```text
TFMPathologyRouter
TFMPathologyExpert
TFMPathologyMoE
```

---

# 7. Router 输入设计

当前 HAMD 默认只有一个标量。

如果只用：

$$
Router(c_{HAMD})
$$

Router 很容易退化成：

```text
低 HAMD → Expert 0
中 HAMD → Expert 1
高 HAMD → Expert 2
```

这会把复杂脑动力学过度简化成严重程度分箱。

因此 Router 必须同时使用：

```text
pathology condition
+
current brain state
```

推荐从：

```text
z_t [B,F,D]
z_g [B,F,D]
```

提取全脑状态描述。

---

# 8. 推荐 State Descriptor

计算：

$$
\mu_t
=
Mean_{ROI}(z_t)
$$

$$
\sigma_t
=
Std_{ROI}(z_t)
$$

$$
\mu_g
=
Mean_{ROI}(z_g)
$$

$$
\sigma_g
=
Std_{ROI}(z_g)
$$

拼接：

$$
s
=
[
\mu_t,
\sigma_t,
\mu_g,
\sigma_g
]
$$

因此：

$$
s\in\mathbb{R}^{4D}
$$

当前：

$$
D=256
$$

所以：

```text
state descriptor = 1024 dim
```

再和 HAMD condition：

$$
r=
[s,c]
$$

输入 Router。

---

# 9. Router 网络

建议：

```text
LayerNorm(4D + C)
    ↓
Linear → 128
    ↓
GELU
    ↓
Dropout 0.1
    ↓
Linear → E
```

默认：

```text
E = 4
```

得到：

$$
a\in\mathbb{R}^{B\times E}
$$

概率：

$$
p=
Softmax(a/T_r)
$$

第一版：

```text
router_hidden = 128
num_experts = 4
top_k = 2
temperature = 1.0
```

---

# 10. Expert 输入

每个 Expert 处理逐 ROI latent：

$$
u_i
=
[z_t,z_g]
$$

即：

```text
[B,F,2D]
```

不要直接让 Expert 再访问 raw BOLD。

原因：

```text
TFM backbone 已经负责 temporal representation
A_eff 已经负责结构信息
```

Expert 只需要回答：

> 在当前 latent brain state 上，需要怎样的病理动力学修正？

---

# 11. Expert 结构

每个 Expert：

```text
[z_t , z_g]
     ↓
Linear(2D, H_e)
     ↓
GELU
     ↓
FiLM(HAMD)
     ↓
Dropout
     ↓
Linear(H_e, D)
     ↓
Δz_i
```

默认：

```text
H_e = 256
E = 4
```

数学形式：

$$
v_i=
\phi(W^{(1)}_i[z_t,z_g])
$$

HAMD FiLM：

$$
\tilde v_i
=
v_i
\odot
(1+\gamma_i(c))
+
\beta_i(c)
$$

输出：

$$
\Delta z_i
=
W^{(2)}_i\tilde v_i
$$

---

# 12. Expert 初始化

病理专家不能在初始化阶段破坏 HC dynamics。

因此最后一层：

```text
Linear(H_e,D)
```

使用小随机初始化：

```text
std = 1e-3
bias = 0
```

而不是普通初始化。

这样：

$$
\Delta z_i\approx0
$$

于是 MDD finetune 初始：

$$
z_{t+1}
\approx
z_{shared}
$$

即从 HC dynamics 附近平稳开始。

不要所有 Expert 完全相同初始化。

可以：

```text
seed_i = base_seed + i * constant
```

确保：

```text
初始 correction 都接近 0
但微小方向不同
```

---

# 13. MoE 聚合

Expert 输出：

$$
\Delta z_1,\ldots,\Delta z_E
$$

Router：

$$
p_1,\ldots,p_E
$$

Top-K：

$$
S=\operatorname{TopK}(p)
$$

归一化：

$$
\bar p_i
=
\frac{p_i}
{\sum_{j\in S}p_j}
$$

最终病理修正：

$$
\Delta z^{path}
=
\sum_{i\in S}
\bar p_i
\Delta z_i
$$

默认：

```text
E = 4
K = 2
```

---

# 14. 训练早期 Router Warmup

直接从第一个 epoch 使用硬 Top-2 有风险：

```text
Expert 尚未学习
Router 随机选择
→ 部分 Expert 得不到梯度
→ 进一步变弱
→ Router 更不选择
→ collapse
```

因此建议：

```text
Epoch 0–4:
    dense soft mixture

Epoch 5+:
    Top-2 sparse mixture
```

前 5 epoch：

$$
\Delta z^{path}
=
\sum_i p_i\Delta z_i
$$

之后：

$$
\Delta z^{path}
=
\sum_{i\in Top2}
\bar p_i\Delta z_i
$$

新增：

```text
--tfm_moe_dense_warmup_epochs 5
```

---

# 15. 为什么 TFM 场景不需要原论文的 Router Prototype

原 ERC：

```text
Router prototype
    ↓
所有 Experts
    ↓
activation norm
```

用于解决：

```text
64/128/256 experts
×
海量 token
```

导致不能全部真实测试的问题。

你的 TFM 默认：

```text
4 experts
batch size ≈ 32
```

完全可以：

```text
每个样本
×
4 experts
```

全部预测。

因此：

$$
\boxed{
不使用 activation norm proxy
}
$$

而直接使用：

$$
\boxed{
真实 next-state prediction error
}
$$

---

# 16. Expert Candidate Prediction

当前 shared 状态：

$$
z^{shared}_{t+1}
=
z_t+h_{shared}
$$

第 $i$ 个 Expert 的 candidate：

$$
z^{(i)}_{t+1}
=
z^{shared}_{t+1}
+
\Delta z_i
$$

共享 decoder：

$$
\Delta\hat x_i
=
Decode(z^{(i)}_{t+1})
$$

最终 candidate：

$$
\hat x^{(i)}_{t+1}
=
x_t+
\Delta\hat x_i
$$

注意：

> 每个 Expert candidate 都包含 shared dynamics。

因此衡量的是：

> “在相同健康动力学基础上，哪个 pathology correction 最适合这个样本？”

不会错误地要求每个 Expert 重学完整脑动力学。

---

# 17. Competence Loss

真实 target：

$$
x^*_{t+1}
$$

第 $i$ 个 Expert：

$$
\ell^{Huber}_{b,i}
=
Huber(
\hat x^{(i)}_{b,t+1},
x^*_{b,t+1}
)
$$

沿 ROI：

```text
F=116
```

求平均，得到：

$$
L^{exp}\in\mathbb{R}^{B\times E}
$$

---

# 18. 可选 spatial PCC competence

后续可以加入：

$$
\ell^{PCC}_{b,i}
=
1-
PCC_{ROI}
(
\hat x^{(i)}_{b,t+1},
x^*_{b,t+1}
)
$$

组合：

$$
\ell_{b,i}
=
\ell^{Huber}_{b,i}
+
\lambda_{c,pcc}\ell^{PCC}_{b,i}
$$

推荐：

```text
V1:
lambda_c_pcc = 0

V2:
0.05 / 0.1
```

第一轮不建议复杂化。

---

# 19. Competence Distribution

不同样本预测难度不同。

先在 Expert 维标准化：

$$
\tilde\ell_{b,i}
=
\frac{
\ell_{b,i}-\mu_b
}{
\sigma_b+\epsilon
}
$$

再：

$$
q_{b,i}
=
Softmax
\left(
-\frac{\tilde\ell_{b,i}}{\tau_c}
\right)
$$

其中：

```text
tau_c = competence temperature
```

默认：

```text
1.0
```

搜索：

```text
0.5 / 1.0 / 2.0
```

---

# 20. RECC-V1：Expert Competence → Router

Router 正常概率：

$$
p
$$

构造：

$$
L_{RECC}
=
D_{KL}
(
sg(q)\Vert p^{cpl}
)
$$

这里：

```text
sg = stop-gradient
```

---

# 21. 为什么 coupling Router 输入必须 detach

RECC 的目标是：

> 校准 Router。

不是：

> 逼迫 backbone 改表示以满足 Router。

所以 coupling 分支必须：

```text
state descriptor.detach()
condition.detach()
    ↓
同一个 Router
    ↓
p_cpl
```

即：

$$
p^{cpl}
=
R(
sg(s),
sg(c)
)
$$

因此 RECC 梯度：

```text
只更新 Router 参数
```

不会直接进入：

```text
TFM backbone
SoftAnatomicalPrior
PathologyNormalizer
Expert
```

---

# 22. 正常 task loss 仍然训练 Experts

注意：

```text
RECC-V1 不训练 Expert
```

并不意味着 Expert 没有梯度。

正常预测：

$$
\hat x
=
x_t+
Decode(
z_{shared}+\Delta z^{path}
)
$$

TFMDualLoss 中 One-Step loss 会通过：

```text
mixture output
→ selected experts
```

正常反向。

所以职责是：

```text
TFM task loss
    → 学 Expert 能力

RECC
    → 学 Router 选择能力
```

这种分工最清晰。

---

# 23. RECC 权重

最终：

$$
L
=
L_{TFM}
+
\lambda_{RECC}L_{RECC}
+
L_{router-reg}
+
L_{graph}
$$

其中当前：

$$
L_{TFM}
=
\lambda_{one}L_{one}
+
\lambda_{cpm}L_{cpm}
$$

---

# 24. RECC Warmup

Early Expert competence 没有意义。

建议：

```text
Epoch 0–4:
    λ_RECC = 0

Epoch 5–14:
    线性增加

Epoch 15+:
    固定
```

默认：

```text
tfm_recc_start_epoch = 5
tfm_recc_ramp_epochs = 10
tfm_recc_weight = 0.05
```

搜索：

```text
0.01
0.05
0.1
```

---

# 25. Router Regularization

TFM 新 MoE 不建议直接继承 legacy 全套 MoE 机制。

第一版只保留必要项：

## 25.1 Load Balance

$$
L_{bal}
=
E
\sum_i
Importance_i
\cdot
Load_i
$$

避免 Expert collapse。

---

## 25.2 Router Z-Loss

$$
L_z
=
E[
\log\sum_i e^{a_i}
]^2
$$

抑制 logits 过度增大。

---

## 25.3 Entropy

可以保留非常小：

```text
1e-3
```

但不是必须。

如果 RECC 已经提供 soft target，过强 entropy 会与 competence specialization 冲突。

推荐默认：

```text
balance = 0.01
z = 0.001
entropy = 0.0005
```

---

# 26. RECC-V2：Top-2 Pair-Aware Coupling

当前最终使用：

```text
Top-K = 2
```

因此单 Expert competence 不是完整问题。

可能出现：

```text
E0 单独一般
E1 单独一般

但 E0 + E1
组合非常好
```

所以 V1 成功后加入 pair-aware coupling。

---

# 27. Pair Candidate

4 Experts：

$$
C_4^2=6
$$

组合：

```text
01
02
03
12
13
23
```

equal-weight latent correction：

$$
\Delta z_{ij}
=
\frac{
\Delta z_i+\Delta z_j
}{2}
$$

candidate：

$$
z^{(ij)}_{t+1}
=
z^{shared}_{t+1}
+
\Delta z_{ij}
$$

预测：

$$
\hat x^{(ij)}
=
x_t+Decode(z^{(ij)}_{t+1})
$$

计算：

$$
\ell_{ij}
$$

---

# 28. Pair Competence Distribution

$$
q^{pair}_{ij}
=
Softmax(
-\tilde{\ell}_{ij}/\tau_{pair}
)
$$

Router implied pair score：

$$
s_{ij}=p_i p_j
$$

归一化：

$$
\pi_{ij}
=
\frac{s_{ij}}
{\sum_{a<b}s_{ab}}
$$

损失：

$$
L_{pair}
=
D_{KL}
(
sg(q^{pair})
\Vert
\pi
)
$$

---

# 29. 为什么 Pair Candidate 使用 0.5 / 0.5

不建议：

$$
\Delta z_{ij}
=
\frac{
p_i\Delta z_i+p_j\Delta z_j
}{
p_i+p_j
}
$$

因为这样 competence target 又依赖当前 Router。

会造成：

```text
Router
→ candidate
→ competence
→ 再监督 Router
```

形成自反馈。

equal weight 能测量：

> Expert pair 自身是否互补。

更加干净。

---

# 30. TFM 病理条件的最终分工

推荐最终：

```text
TFM Backbone
    不直接使用 HAMD

One-Step
    pathology_mode = moe
    HAMD → Router + Expert FiLM

CPM
    继续 HAMD → FiLM
```

这样：

### One-Step

学习：

```text
局部 next-state pathology dynamics
```

### CPM

学习：

```text
病理条件下的多 horizon trajectory
```

二者职责不同。

---

# 31. 是否要让 CPM 也使用同一个 Router

第一版：

```text
不要
```

原因：

1. CPM 的 horizon query 机制与 one-step transition 不同；
2. 同一个 Expert 是否同时适合 t+1 和 t+8 未知；
3. RECC 本身已经是新变量；
4. 会极大增加消融复杂度。

后续如果 one-step RECC 成功，可以研究：

```text
horizon-aware experts
```

但不属于当前方案。

---

# 32. models/tfm_experts.py

建议新增。

包含：

```python
class TFMPathologyRouter(nn.Module)
class TFMPathologyExpert(nn.Module)
class TFMPathologyMoE(nn.Module)
```

---

# 33. TFMPathologyRouter 输出

建议返回：

```text
logits
soft_probs
routing_weights
topk_indices
```

训练早期 dense：

```text
routing_weights = soft_probs
```

后期：

```text
routing_weights =
topk_mask * soft_probs
再归一化
```

另外提供：

```python
def coupling_probs(
    z_t_detached,
    z_g_detached,
    cond_detached
)
```

只计算 full soft distribution。

---

# 34. TFMPathologyExpert 输出

输入：

```text
z_t [B,F,D]
z_g [B,F,D]
cond [B,C]
```

输出：

```text
delta_z [B,F,D]
```

不要直接输出：

```text
delta_x [B,F]
```

因为把专家放在 latent transition 层更符合：

```text
brain dynamics correction
```

的模型语义。

---

# 35. TFMPathologyMoE Forward

输出建议：

```text
delta_path_z
expert_delta_z
router_probs
router_weights
topk_indices
router_logits
```

其中：

```text
expert_delta_z [B,E,F,D]
```

训练 RECC 时保留。

若：

```text
tfm_recc_mode=off
```

仍然需要专家输出用于正常 mixture，但无需额外生成 candidate prediction。

---

# 36. models/tfm.py 改造

`OneStepDynamicsHead.__init__` 新增：

```text
pathology_mode
num_experts
top_k
expert_hidden_dim
router_hidden_dim
router_temperature
dense_warmup_epochs
```

---

# 37. OneStepDynamicsHead.forward

当前：

```python
h = transition(...)
h = FiLM(h)
z_next = z_t + h
return decode(z_next)
```

改成：

```text
h_shared = transition([z_t,z_g])
z_shared = z_t + h_shared

if pathology_mode == film:
    使用现有 FiLM
    z_next = ...

elif pathology_mode == moe:
    delta_path_z, moe_aux = pathology_moe(...)
    z_next = z_shared + delta_path_z

else:
    z_next = z_shared

delta_x = decode(z_next)
```

返回：

```text
delta_x
one_step_aux
```

而不是只返回 tensor。

---

# 38. Candidate Prediction

在 `moe` + diagnose/RECC 模式：

利用：

```text
expert_delta_z
```

一次性构造：

```text
[B,E,F,D]
```

candidate state：

$$
z_{cand}
=
z_{shared}.unsqueeze(1)
+
expert\_delta\_z
$$

共享 decode 向量化：

```text
[B,E,F,D]
  ↓ Linear(D,1)
[B,E,F]
```

得到：

```text
expert_pred_norm [B,E,F]
```

无需重复运行整个 Expert。

---

# 39. RevIN 空间处理

当前 TFM：

```text
context-only RevIN
```

最后预测会反归一化。

RECC candidate 必须与：

```text
target
```

处于同一空间。

推荐：

```text
expert_pred_norm [B,E,F]
    ↓
permute
[B,F,E,1]
    ↓
self.rev_norm(..., 'denorm', stats)
    ↓
[B,F,E,1]
    ↓
permute
[B,E,F]
```

即：

```python
cand4 = expert_pred_norm.permute(0, 2, 1).unsqueeze(-1)
cand_raw4 = self.rev_norm(cand4, 'denorm', stats=norm_stats)
expert_pred = cand_raw4.squeeze(-1).permute(0, 2, 1)
```

这样不需要手写 RevIN affine。

---

# 40. NeuroTwinTFM.forward aux_info

新增：

```text
tfm_router_probs
tfm_router_weights
tfm_router_logits
tfm_topk_indices
tfm_expert_pred
tfm_expert_delta_z_norm
```

仅在：

```text
pathology_mode=moe
```

时存在。

若：

```text
recc_mode=off
```

可以不返回：

```text
tfm_expert_pred
```

减少内存。

---

# 41. train/tfm_coupling.py

建议新建：

```text
train/tfm_coupling.py
```

不要继续堆进 `train/moe.py`。

包含：

```text
compute_tfm_router_regularization
compute_tfm_recc
compute_tfm_pair_recc
get_tfm_recc_weight
compute_tfm_coupling_diagnostics
```

---

# 42. compute_tfm_recc

输入：

```text
aux_info
target [B,F,1,1]
tau
pcc_weight
```

过程：

```text
expert_pred [B,E,F]
target → [B,F]

Huber
    ↓
expert_loss [B,E]

z-score across E
    ↓
q [B,E]

KL(q.detach || router_probs_coupling)
```

---

# 43. coupling_probs 的梯度边界

必须单元测试：

```text
RECC-only backward
```

结果：

```text
Router:
grad != 0

Experts:
grad == 0

TFM backbone:
grad == 0

SC prior:
grad == 0

CPM:
grad == 0
```

这条是核心验收项。

---

# 44. train/optim.py 改造

当前 TFM 冻结阶段：

```text
只训练 is_conditioning_adapter=True
```

新：

```text
TFMPathologyMoE
```

必须在 frozen stage 可训练。

推荐：

```python
class TFMPathologyMoE(nn.Module):
    is_conditioning_adapter = True
```

这样无需重写大量冻结逻辑。

---

# 45. 参数分组

当前 `get_param_groups` 已经按：

```text
adapter
moe
backbone
```

名字区分。

推荐给 TFM pathology MoE 单独 param group：

```text
tfm_pathology_moe
```

但第一版为了最小修改，也可以：

```text
模块名中包含 pathology_moe
```

并进入现有：

```text
moe
```

参数组。

更干净做法：

```text
group name = tfm_moe
lr = lr_peak
```

backbone：

```text
lr_peak * backbone_lr_scale
```

---

# 46. 两阶段 finetune

继续保留当前策略。

## Stage 1

前：

```text
freeze_backbone_epochs = 10
```

冻结：

```text
patch_embed
TFM blocks
final_norm
shared one-step transition
shared decode
SC prior
```

训练：

```text
Pathology MoE
Router
Expert
CPM FiLM
```

---

## Stage 2

解冻全部：

```text
backbone lr × 0.2
pathology MoE lr × 1.0
```

这样：

> 先让病理专家在固定 HC representation 上形成能力分工，再轻量调整 TFM representation。

---

# 47. 为什么 Stage 1 非常适合 RECC

Stage 1：

```text
shared dynamics fixed
```

因此不同 Expert competence 的差异主要来自：

```text
pathology experts
```

不会被 backbone 持续漂移污染。

所以建议：

```text
dense expert warmup: epoch 0–4
RECC start: epoch 5
```

这样正好：

```text
0–4:
专家先学

5–9:
冻结 backbone 下开始 RECC

10+:
联合微调
```

---

# 48. main.py 新 CLI

建议新增：

```text
--tfm_patho_mode
    none|film|moe
    default film

--tfm_num_experts
    default 4

--tfm_top_k
    default 2

--tfm_expert_hidden_dim
    default 256

--tfm_router_hidden_dim
    default 128

--tfm_router_temperature
    default 1.0

--tfm_moe_dense_warmup_epochs
    default 5
```

---

# 49. RECC CLI

```text
--tfm_recc_mode
    off|diagnose|r2e|pair

--tfm_recc_weight
    default 0.05

--tfm_pair_recc_weight
    default 0.0

--tfm_competence_temperature
    default 1.0

--tfm_competence_pcc_weight
    default 0.0

--tfm_recc_start_epoch
    default 5

--tfm_recc_ramp_epochs
    default 10
```

---

# 50. Router Regularization CLI

```text
--tfm_router_balance_weight
    default 0.01

--tfm_router_z_weight
    default 0.001

--tfm_router_entropy_weight
    default 0.0005
```

---

# 51. 配置约束

fail-fast：

```text
tfm_recc_mode != off
    requires tfm_patho_mode == moe
```

```text
tfm_top_k <= tfm_num_experts
```

```text
tfm_num_experts >= 2
```

```text
tfm_patho_mode == moe
    requires pretrain_mode == False
```

HC pretrain 时：

```text
tfm_patho_mode 自动 none
```

---

# 52. HC Pretrain 不需要任何 MoE

`scripts/Pretrain_HC_tfm.sh`：

```text
完全不加入 pathology expert
```

结构：

```text
TFM Backbone
+
Shared One-Step Head
+
CPM
```

这点非常重要。

不要在 HC 阶段预训练 Experts。

否则 Expert 会学习：

```text
健康亚型
```

而不是：

```text
病理 residual specialization
```

偏离设计目的。

---

# 53. Finetune 脚本

保留：

```text
scripts/Finetune_MDD_tfm.sh
```

作为当前 FiLM baseline。

新增：

```text
scripts/Finetune_MDD_tfm_moe.sh
scripts/Finetune_MDD_tfm_recc.sh
```

---

# 54. TFM-MoE 脚本

关键：

```text
--tfm_patho_mode moe

--tfm_num_experts 4
--tfm_top_k 2
--tfm_expert_hidden_dim 256
--tfm_router_hidden_dim 128
--tfm_router_temperature 1.0
--tfm_moe_dense_warmup_epochs 5

--tfm_recc_mode off
```

回答：

> 单纯引入 pathology experts 是否优于 FiLM？

---

# 55. TFM-MoE-RECC 脚本

在上面基础：

```text
--tfm_recc_mode r2e
--tfm_recc_weight 0.05
--tfm_competence_temperature 1.0
--tfm_competence_pcc_weight 0.0
--tfm_recc_start_epoch 5
--tfm_recc_ramp_epochs 10
```

回答：

> Router–Expert coupling 是否在 MoE 基础上进一步提高性能和 specialization？

---

# 56. 实验对照必须拆成四层

不要只比较：

```text
TFM baseline
vs
TFM + RECC
```

因为 RECC 需要先新增 MoE。

正确实验：

## A. TFM-FiLM

当前 baseline。

```text
One-Step FiLM
CPM FiLM
```

---

## B. TFM-MoE

```text
One-Step pathology MoE
CPM FiLM
RECC off
```

回答：

> 多专家结构本身是否有效？

---

## C. TFM-MoE + Diagnose

结构同 B。

只统计：

```text
Expert competence
Router mismatch
```

不加 RECC loss。

---

## D. TFM-MoE + RECC

```text
R2E coupling
```

回答：

> coupling 是否修复 Router–Expert mismatch？

---

## E. TFM-MoE + Pair-RECC

进一步：

```text
Top-2 pair coupling
```

回答：

> Expert collaboration 是否进一步改善？

---

# 57. 推荐 experiments/variants.py 新组

新增：

```text
G31_TFM_RECC
```

只包含 TFM。

---

# 58. 第一批实验

```text
tfm_film
tfm_moe
tfm_moe_diag
tfm_moe_recc
```

优先跑。

不要先 grid search。

---

# 59. 第二批

只有：

```text
tfm_moe_recc
```

有 positive signal 后再跑：

```text
weight 0.01
weight 0.10
tau 0.5
tau 2.0
pair-recc
```

---

# 60. 主预测指标

保持现有：

```text
one-step MAE
one-step RMSE
spatial PCC
delta_direction
```

---

# 61. Rollout

必须同时检查：

```text
H=1
H=2
H=4
H=8
H=16
```

因为 pathology MoE 直接修改：

```text
One-Step transition
```

短期提升可能导致：

```text
autoregressive instability
```

所以：

```text
rollout degradation
```

是核心验收指标。

---

# 62. CPM 指标

虽然 MoE 不进入 CPM，也必须检查：

```text
CPM H1–H8
```

原因：

> 联合微调时 backbone 会受到 One-Step MoE 与 CPM 双重梯度影响。

如果：

```text
one-step ↑
CPM ↓↓↓
```

说明新分支可能破坏共享 representation。

---

# 63. RECC 核心机制指标

必须新增。

---

# 64. Top-1 Agreement

$$
A_1
=
P[
argmax_i p_i
=
argmin_i \ell_i
]
$$

4 Expert 随机约：

```text
25%
```

重点比较：

```text
MoE baseline
vs
MoE + RECC
```

---

# 65. Top-2 Recall

$$
Recall_2
=
\frac{
|S^{router}_2\cap S^{oracle}_2|
}{2}
$$

更符合 Top-2 MoE。

---

# 66. Router Regret

$$
R_b
=
\ell_{b,argmax(p_b)}
-
\min_i\ell_{b,i}
$$

越低越好。

---

# 67. Probability–Competence Spearman

$$
\rho_b
=
Spearman(
p_b,
-\ell_b
)
$$

越高说明：

```text
Router probability
```

越能反映真实 Expert competence。

---

# 68. Competence Entropy

$$
H(q)
=
-\sum_iq_i\log q_i
$$

帮助判断：

> 某个样本是否真的存在明显专家偏好。

---

# 69. Oracle Best-Expert Gap

比较：

```text
actual MoE prediction
vs
oracle best single Expert
```

如果 gap 很大：

```text
Router 是瓶颈
```

---

# 70. Oracle Best-Pair Gap

比较：

```text
actual Top-2 Router pair
vs
oracle best equal-weight pair
```

用于判断：

> 是否需要 Pair-RECC。

---

# 71. Competence Matrix

定义：

$$
C_{ij}
=
E[
\ell_j
\mid
argmax(p)=i
]
$$

理想：

```text
对角线最低
```

它是 TFM-RECC 最重要的机制图之一。

---

# 72. Expert Specialization 分析

RECC 有效后分析：

```text
Expert 0 高概率样本
Expert 1 高概率样本
...
```

在以下变量上的差异。

---

# 73. HAMD

统计：

```text
mean
median
distribution
```

但不要直接解释成：

```text
MDD 临床亚型
```

推荐术语：

> pathology-conditioned dynamical regimes

---

# 74. Brain State Descriptor

分析各 Expert 对应：

```text
z_t statistics
z_g statistics
```

可以 PCA / UMAP，仅作为可视化。

---

# 75. ROI Correction

每个 Expert latent correction：

$$
\Delta z_i
$$

经 shared decode 后得到：

$$
\Delta x_i^{path}
$$

分析 116 ROI 的：

```text
mean absolute correction
signed correction
```

---

# 76. Network Level

按 AAL 网络标签聚合：

```text
DMN
FPN
VAN
limbic
visual
subcortical
...
```

看不同 Expert 是否对应不同网络级动力学偏离。

---

# 77. Subject-Level Routing Consistency

同一个 subject 多个 anchor：

```text
Router 是否始终类似？
```

定义：

```text
intra-subject routing similarity
```

但不要求完全稳定。

因为当前设计允许：

```text
同一患者
不同 brain state
→ 不同 expert mixture
```

因此更合理的问题是：

> intra-subject 是否比 inter-subject 更一致？

---

# 78. Shuffled-HAMD 负对照

当前项目已经有：

```text
shuffled-HAMD
```

RECC 后必须继续使用。

理想：

```text
true HAMD
    < MAE

shuffled HAMD
    > MAE
```

同时检查：

```text
Router distribution
```

是否随 HAMD shuffle 改变。

---

# 79. 额外负对照：State Shuffle

推荐新增：

```text
router state descriptor shuffle
```

保持 HAMD 不变，只打乱：

```text
z_t/z_g state descriptor
```

如果性能明显下降：

> Router 确实使用了动态状态，而不是只按 HAMD 分箱。

---

# 80. Router Input Ablation

建议三种：

```text
HAMD only
State only
HAMD + State
```

预期：

```text
HAMD + State
```

最好。

如果：

```text
HAMD only ≈ HAMD+State
```

说明 MoE 更像严重程度分箱，而非动态状态专家。

---

# 81. Expert 数量

第一版固定：

```text
E = 4
```

只有方法稳定后测试：

```text
E = 2
E = 4
E = 8
```

当前样本量下：

```text
8 experts
```

可能过度分裂。

所以默认不要超过 4。

---

# 82. Top-K

默认：

```text
Top-K = 2
```

后续：

```text
K=1
K=2
dense
```

作为 specialization–collaboration trade-off。

---

# 83. 与原 ERC 的对应关系

原 ERC：

```text
Router vector
    ↓
Expert activation
    ↓
对角占优
```

TFM-RECC：

```text
真实 MDD 样本
    ↓
所有 pathology experts
    ↓
真实 one-step prediction error
    ↓
competence distribution
    ↓
Router alignment
```

原：

$$
M_{ii}\gg M_{ij}
$$

TFM：

$$
C_{ii}\ll C_{ij}
$$

---

# 84. 对应论文创新点

如果最终有效，方法贡献可以表述为：

### Contribution 1

病理 residual experts：

> shared healthy dynamics 与 pathology-specific dynamics correction 解耦。

### Contribution 2

ground-truth competence coupling：

> 不依赖 activation proxy，直接利用未来 BOLD 监督 Expert competence。

### Contribution 3

state-aware routing：

> Router 同时依赖 clinical condition 与当前 latent brain state，而非 HAMD severity alone。

### Contribution 4

pair-aware expert collaboration：

> 面向 Top-2 pathology experts 的组合能力对齐。

---

# 85. train/losses.py 是否需要修改

不建议把 RECC 塞进：

```text
TFMDualLoss
```

保持：

```text
TFMDualLoss
```

只负责：

```text
One-Step
CPM
```

RECC 放：

```text
train/tfm_coupling.py
```

由 `main.py` 聚合。

这样：

```text
task objective
和
routing objective
```

解耦。

---

# 86. main.py 训练 loss

最终：

$$
L_{total}
=
L_{TFMDual}
+
L_{graph}
+
L_{router-reg}
+
\lambda_{recc}L_{recc}
+
\lambda_{pair}L_{pair}
$$

---

# 87. Checkpoint 选择

继续：

```text
val next-state MAE
```

不要用：

```text
包含 RECC 的 total validation loss
```

否则不同：

```text
λ_RECC
```

实验不可比。

---

# 88. 测试文件

建议新增：

```text
tests/test_tfm_experts.py
tests/test_tfm_recc.py
```

---

# 89. test_tfm_experts.py

至少：

### shape

```text
expert_delta_z:
[B,E,F,D]

router_probs:
[B,E]
```

### mixture

weights sum：

```text
=1
```

### top-k

非 Top-K：

```text
weight=0
```

### initialization

初始：

```text
|delta_path_z| 很小
```

---

# 90. test_tfm_recc.py

至少：

### Perfect agreement

```text
q == p
→ KL≈0
```

### Correct best expert

人为：

```text
loss=[0.5,0.1,0.8,0.4]
```

必须：

```text
argmax q == 1
```

### Uniform competence

```text
loss 全相同
→ q uniform
```

### Gradient isolation

RECC-only backward：

```text
router grad != 0
expert grad == 0
backbone grad == 0
```

### Pair count

4 Experts：

```text
6 pairs
```

### RevIN candidate

candidate denorm 必须和单独调用一致。

---

# 91. Backward Compatibility

必须保证：

```text
--tfm_patho_mode film
```

时：

```text
当前 tfm baseline 数值行为不变
```

新增模块不构建。

旧 checkpoint：

```text
正常加载
```

---

# 92. ARCH_VERSION

如果新增：

```text
tfm_patho_mode=moe
```

但：

```text
HC pretrain 结构没变化
```

不一定必须整体 bump 预训练架构版本。

但 finetune checkpoint config 必须保存：

```text
tfm_patho_mode
tfm_num_experts
tfm_top_k
...
```

evaluate_variant 必须从 checkpoint 重建相同结构。

---

# 93. experiments/evaluate_variant.py

需要让模型重建识别：

```text
tfm_patho_mode
tfm_num_experts
tfm_top_k
tfm_expert_hidden_dim
tfm_router_hidden_dim
```

否则 strict load 会出现：

```text
missing / unexpected keys
```

---

# 94. analysis 脚本

建议新增：

```text
analysis/analyze_tfm_experts.py
```

保存：

```text
subject_id
HAMD
router_probs
expert_losses
best_expert
top2_experts
router_regret
```

---

# 95. 输出文件

```text
tfm_expert_summary.json
tfm_expert_per_anchor.csv
tfm_expert_per_subject.csv
tfm_competence_matrix.csv
```

可视化：

```text
tfm_competence_matrix.png
tfm_router_hamd.png
tfm_router_regret.png
tfm_expert_network_profile.png
```

---

# 96. 第一阶段验收

先不看 RECC。

TFM-MoE 相比 TFM-FiLM：

```text
MAE 至少不明显下降
PCC 不明显下降
rollout 不出现明显爆炸
Router 不 collapse
4 Experts 都有实际使用
```

如果 TFM-MoE 本身明显差于 FiLM：

```text
停止 RECC
```

因为 coupling 无法挽救一个本身不适合的 Expert 架构。

---

# 97. 第二阶段验收

MoE + Diagnose：

希望看到：

```text
Router top1 agreement 不高
oracle gap 明显
router regret > 0
```

说明存在：

```text
Router–Expert mismatch
```

如果 Router 本身已经接近 oracle：

```text
RECC 必要性弱
```

停止进一步复杂化。

---

# 98. 第三阶段验收

RECC：

至少：

```text
Top1 Agreement ↑
Top2 Recall ↑
Router Regret ↓
Spearman ↑
```

且：

```text
MAE 不恶化
PCC 不恶化
```

---

# 99. 第四阶段验收

理想：

```text
MAE ↓
PCC ↑
rollout H2/H4/H8 改善
shuffled-HAMD 变差
competence matrix 对角优势增强
```

多 seed 一致。

---

# 100. 推荐最小实验顺序

严格按：

```text
1. tfm_film
2. tfm_moe
3. tfm_moe_diag
4. tfm_moe_recc
```

先跑。

如果：

```text
4 > 3 > 2
```

再：

```text
5. recc weight=0.01
6. recc weight=0.10
7. tau=0.5
8. tau=2.0
9. pair-recc
```

---

# 101. 不建议第一轮做的事情

不要：

```text
8 Experts
token-level routing
每层 Transformer MoE
CPM MoE
HAMD 多项式扩展
router attention pooling
expert graph network
expert-specific SC prior
expert-specific decoder
```

这些都会让变量失控。

---

# 102. 推荐最终结构图

```text
                    BOLD context
                         │
                         ▼
                 Context-only RevIN
                         │
                         ▼
               Soft Anatomical Prior
                         │
                       A_eff
                         │
                         ▼
                  Temporal Patching
                         │
                         ▼
          SC-guided Temporal–Variate Blocks
                         │
                         ▼
                Context tokens H
                  │              │
                  │              └────────────► CPM Head
                  │                               │
                  │                           HAMD-FiLM
                  │                               │
                  │                         t+1...t+H
                  │
                  ▼
             last token z_t
                  │
          A_eff @ z_t = z_g
                  │
        ┌─────────┴────────────┐
        │                      │
        ▼                      ▼
Shared Dynamics            Router
Transition              state + HAMD
        │                      │
   Δz_shared                  p_i
        │                      │
        │            ┌─────────┼───────────┐
        │            ▼         ▼           ▼
        │           E0        E1      ...  E3
        │            │         │           │
        │            └─────────┼───────────┘
        │                      │
        │                  Δz_path
        │                      │
        └──────────┬───────────┘
                   ▼
       z_(t+1)=z_t+Δz_shared+Δz_path
                   │
                   ▼
              Shared Decode
                   │
                   ▼
           x̂_(t+1)=x_t+Δx̂
```

训练期额外：

```text
E0 candidate ──► error_0 ─┐
E1 candidate ──► error_1  │
E2 candidate ──► error_2  ├─► competence q
E3 candidate ──► error_3 ─┘
                              │
                              ▼
                        KL(q || Router p)
```

---

# 103. 推荐默认参数

```text
tfm_patho_mode = moe

tfm_num_experts = 4
tfm_top_k = 2
tfm_expert_hidden_dim = 256
tfm_router_hidden_dim = 128
tfm_router_temperature = 1.0

tfm_moe_dense_warmup_epochs = 5

tfm_router_balance_weight = 0.01
tfm_router_z_weight = 0.001
tfm_router_entropy_weight = 0.0005

tfm_recc_mode = r2e
tfm_recc_weight = 0.05
tfm_competence_temperature = 1.0
tfm_competence_pcc_weight = 0.0

tfm_recc_start_epoch = 5
tfm_recc_ramp_epochs = 10
```

---

# 104. 最终执行优先级

## P0

```text
TFMPathologyExpert
TFMPathologyRouter
TFMPathologyMoE
One-Step integration
HC checkpoint compatibility
TFM-MoE baseline
```

## P1

```text
all-expert candidate
competence diagnosis
agreement / regret / oracle gap
```

## P2

```text
RECC-V1
gradient isolation
warmup schedule
```

## P3

```text
Pair-RECC
Expert-HAMD
Expert-network interpretability
```

---

# 105. 最终推荐

当前最值得实现的不是：

```text
把 ERC loss 直接贴到 TFM
```

而是先完成结构上的：

$$
\boxed{
Shared\ Healthy\ Dynamics
+
Pathology\ Residual\ Experts
}
$$

然后再引入：

$$
\boxed{
Ground\text{-}Truth\ Expert\ Competence
\rightarrow
Router\ Calibration
}
$$

这使 RECC 与 NeuroTwin-TFM 的核心任务天然一致：

> 在共享的健康脑动力学基础上，由不同病理专家学习状态依赖的 MDD 动力学偏离，并利用真实未来 BOLD 验证 Router 是否选择了真正更适合当前患者状态的 Expert。

相比将 MoE 扩散到整个 Transformer backbone，这种方案结构更干净、预训练兼容性更好、医学解释性更强，也更适合作为后续论文中的独立方法贡献。
