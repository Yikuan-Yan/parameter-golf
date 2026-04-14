# Parameter Golf 物理启发方法全景研究报告

## 写在前面：约束分析

在展开方法论之前，先精确量化比赛的核心约束：

| 约束 | 数值 | 物理类比 |
|------|------|----------|
| Artifact 大小 | ≤16,000,000 bytes | 系统总自由度 |
| 训练时间 | ≤10 min on 8×H100 | 能量注入上限 |
| 评价指标 | BPB on FineWeb | 自由能最小化 |
| 当前 SOTA | ~1.119 BPB | 当前基态能量 |
| Baseline | 1.224 BPB | 初始热力学态 |

**关键推论**：16MB 在 INT6 量化下约可容纳 ~21.3M 个参数（16e6 × 8/6），在 INT8 下约 16M 个参数。压缩后可能更多，但这给出了参数空间的量级——O(10⁷)。这与凝聚态物理中中等规模系统的配分函数计算量级相当。

---

## 第一部分：物理启发方法总览

我将方法按照其对应的物理原理进行分类，从最实用到最前沿排列。

---

### 方法 A：连续动力系统视角 —— State Space Models (SSM/Mamba)

#### 物理原理

SSM 的核心是**线性时变动力系统**的离散化：

$$h_t = A_t h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t$$

这与物理中的**受驱阻尼振子**、**控制论中的状态空间表示**、以及**量子力学中的含时演化**直接对应。Mamba 的创新在于让 $A_t, B_t, C_t$ 成为输入的函数（selective mechanism），这对应于**非自洽场论**中的自适应势能面。

**Mamba-3 的最新突破**（2026年3月，与 Parameter Golf 同期发布）引入了三个关键改进：
1. **指数-梯形离散化**（更精确的连续→离散映射）
2. **复数值状态更新**（等价于数据依赖的 RoPE，对应量子力学中的相位累积）
3. **MIMO 结构**（多输入多输出，提高硬件利用率）

#### 定量分析

| 指标 | Transformer (baseline) | Mamba-2 | Mamba-3 | 预期对比赛的影响 |
|------|----------------------|---------|---------|-----------------|
| 参数效率 | 1× | ~1.5× (同参数下更好) | ~2× | 相同 16MB 可获得更低 BPB |
| 训练速度 (seq len) | O(n²) | O(n) | O(n) | 允许更长序列/更多数据 |
| 推理内存 | O(n) | O(1) | O(1) | 不直接影响评估 |
| 状态压缩比 | N/A | state_size=64 | state_size=32（质量相当） | 更小状态 = 更多参数用于其他组件 |

**参数预算估算**：一个 Mamba-3 块约使用 $3 \times \text{expand} \times d^2$ 参数。若 d=384, expand=2：
- 每块参数量：$3 \times 2 \times 384^2 = 884,736 \approx 0.88M$
- 16MB (INT6) 可容纳约 21M 参数 → 约 23 层
- 对比 Transformer baseline 的 9 层 512-dim → 显著更深

#### 优势
- **线性时间复杂度**：10分钟内可处理更多训练 token
- **参数效率极高**：Mamba-3B 匹配 Transformer-6B 的下游性能
- **物理解释清晰**：直接对应连续时间动力系统的离散化
- **复数态允许相位信息**：对 state tracking 任务（如括号匹配、语法结构）至关重要

#### 劣势
- **In-context learning 弱于 Transformer**：ICL 能力约为同规模 Transformer 的 38-82%（Mamba-1/2），不过这对 BPB 评估影响不大
- **自定义 CUDA kernel 依赖**：需要 causal-conv1d 和 selective-scan kernel（比赛允许外部包）
- **训练不如 Transformer 稳定**：需要仔细调参
- **Retrieval 任务表现较差**：但 FineWeb 的 BPB 评估主要是语言建模，不是 retrieval

#### 推荐原因
**强烈推荐作为主架构的候选之一**。Mamba 的参数效率在 16MB 约束下是决定性优势。Mamba-3 的复数态更新对应于量子力学中的酉演化，你的物理背景可以直接利用。

#### 注意要点
1. 使用 Mamba-2 的 SSD formulation 而非 Mamba-1，因为 SSD 可以利用矩阵乘法加速训练
2. 考虑 Hybrid 架构：少量 attention 层 + 多数 Mamba 层（如 Mamba-3 论文建议）
3. 复数值 SSM 需要特殊的量化策略——实部和虚部分别量化

#### 预期效果
在相同 16MB 约束下，纯 Mamba 架构预计比纯 Transformer 获得 **0.02-0.05 BPB 的改善**，主要来自更深的网络和更高的参数效率。

---

### 方法 B：Neural ODE / 连续深度网络

#### 物理原理

将 Transformer 的离散层视为 ODE 的 Euler 步进：

$$h_{l+1} = h_l + f_\theta(h_l, l) \quad \xrightarrow{\Delta l \to 0} \quad \frac{dh}{dt} = f_\theta(h, t)$$

这对应于 **ResNet 是 ODE 的 Euler 离散化** 的经典观察。核心物理洞察：

1. **参数共享**：$f_\theta$ 在所有"时间"共享参数 → 一组参数决定无限深度
2. **O(1) 内存**：adjoint method 反向传播不存中间激活
3. **自适应深度**：ODE solver 根据输入复杂度自动选择步数

**Neural ODE Transformer** (2025) 进一步将所有 attention 和 FFN 权重参数化为连续层索引的函数，使用 hyper-network 生成权重。

#### 定量分析

| 配置 | 独立参数量 | 等效深度 | 预期 BPB |
|------|-----------|---------|---------|
| 11层独立 Transformer | 16M | 11 | ~1.12 (当前 SOTA) |
| Neural ODE + 4步 Euler | 4M (×4复用) | 等效16 | ~1.13-1.15 |
| Neural ODE + 自适应 | 4M | 自适应 | ~1.12-1.14 |
| Relaxed Recursive (LoRA) | 4M + 2M LoRA | 等效16 | ~1.11-1.13 |

**核心权衡**：Neural ODE 用**等效深度**换**参数量**。在 16MB 约束下：
- 独立参数: 每层 ~1.5M × 11层 = 16.5M → 刚好不够
- 共享参数: 核心 ~5M × 1 + LoRA ~0.3M × 16 = ~10M → 余 6M 给 embedding 等

#### 优势
- **极致参数效率**：一组动力学参数定义任意深度网络
- **物理优美**：对应 Hamilton 力学、Lagrange 力学的连续演化
- **Lyapunov 稳定性**：Mamba SSM 被证明是 Lyapunov 稳定的，适合混合精度训练
- **自适应计算**：简单 token 用少步，复杂 token 用多步（评估时可能节省时间）

#### 劣势
- **训练效率低**：ODE solver 的开销在 10 分钟时间约束下是致命的
- **梯度问题**：adjoint method 引入额外数值误差
- **量化困难**：ODE solver 对数值精度敏感，INT6 量化可能破坏动力学
- **实际性能不如预期**：在 NLP 任务上，Neural ODE 通常不如同参数的离散网络

#### 推荐原因
**不推荐作为主架构，但推荐作为设计哲学**。具体来说：用 **Relaxed Recursive Transformer**（离散化的 Neural ODE）代替连续 ODE solver。这保留了参数共享的好处，避免了 ODE solver 的开销。

#### 注意要点
1. 不要使用真正的 ODE solver——10分钟训练时间承受不起
2. 改用 **Universal Transformer + depth-wise LoRA**：共享主体参数，每层加小的 LoRA 适配器
3. 参数共享策略选择：CYCLE > SEQUENCE > UNIVERSAL（文献一致结论）

#### 预期效果
Relaxed Recursive Transformer 在 16MB 下预计比独立层 Transformer 改善 **0.01-0.02 BPB**，主要来自更大的等效深度。

---

### 方法 C：Kolmogorov-Arnold Networks (KAN)

#### 物理原理

基于 **Kolmogorov-Arnold 表示定理**：任何多元连续函数都可以表示为有限次一元函数的叠加与复合：

$$f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

与 MLP 的关键区别：MLP 在节点上使用固定激活函数、在边上使用可学习线性权重；KAN 在边上使用可学习的非线性函数（B-spline 参数化）。

物理类比：这对应于将多体相互作用**精确分解为两体相互作用的组合**——类似于 Born-Oppenheimer 近似或 Hartree-Fock 方法中的平均场分解。

#### 定量分析

| 模型 | 参数方式 | 参数量/边 | 压缩性 | 适用规模 |
|------|---------|----------|--------|---------|
| MLP | 1 weight + 1 bias | 2 | INT6 友好 | 任意 |
| KAN (B-spline, k=3, G=5) | ~8 控制点 | 8 | 浮点密集，量化损失大 | 小规模 |
| KAN (低阶多项式) | ~4 系数 | 4 | 尚可 | 小-中规模 |

**关键问题**：KAN 的参数密度远高于 MLP。对于 [n_in, n_out] 层：
- MLP 参数量：$n_{in} \times n_{out} + n_{out}$
- KAN 参数量：$n_{in} \times n_{out} \times G$（G 为 spline 阶数）

在 16MB 约束下，KAN 的参数效率**显著差于 MLP**。

#### KAN 在语言建模中的现状

截至 2026 年，**KAN 在大规模语言建模中没有成功案例**。KAN 的优势（可解释性、函数拟合精度）集中在小规模科学计算任务。对于语言建模所需的高维离散 token 空间，KAN 的 B-spline 基函数缺乏归纳偏置。

#### 优势
- 理论上更优的 neural scaling law（$\alpha \sim 1$ vs MLP 的 $\alpha \sim 1/d$）
- 可解释性强——可以直接观察学到的一元函数
- 小规模下精度优于 MLP

#### 劣势
- **参数效率极差**：每条边 8 个参数 vs MLP 的 1 个
- **训练极慢**：spline 计算开销大，10 分钟内训练量严重不足
- **量化困难**：spline 控制点需要高精度
- **无语言建模先例**：在 NLP 领域几乎没有成功应用
- **缺乏高效实现**：没有类似 FlashAttention 的优化

#### 推荐原因
**不推荐用于主架构**。但可以考虑一个折中方案：用 **KAN-inspired 激活函数**替换 MLP 中的固定激活函数。例如，用可学习的分段线性函数替代 ReLU²/SiLU：

$$\sigma(x) = \sum_{i} a_i \max(0, x - b_i)$$

这仅增加 O(k) 个参数/激活函数，但引入了 KAN 的可学习非线性。

#### 预期效果
纯 KAN 替换 MLP：性能**下降** 0.05-0.1 BPB（参数效率损失）。KAN-inspired 激活函数：可能改善 **0.005-0.01 BPB**。

---

### 方法 D：张量网络 (Tensor Networks / MPS)

#### 物理原理

张量网络是量子多体物理中表示高维波函数的核心工具。**Matrix Product State (MPS)** 将一个指数大的张量分解为局部张量的链式缩并：

$$\Psi(s_1, \ldots, s_N) = \sum_{\{i\}} A^{s_1}_{i_1} A^{s_2}_{i_1 i_2} \cdots A^{s_N}_{i_{N-1}}$$

对于语言建模：将句子看作"量子态"，每个词是一个局部自由度，MPS 通过有限的 bond dimension $\chi$ 捕获有限范围的关联。

**TensorGPT** 提出用 Tensor-Train Decomposition 压缩 token embedding 矩阵，将 $V \times d$ 的 embedding 表分解为一系列小张量，实现训练无关的压缩。

**Tensor Train Language Model (TTLM)** 直接用张量网络作为语言模型的骨架，在指数大的 tensor product 空间中工作，同时保持低秩近似的高效性。

#### 定量分析

Embedding 压缩效果（TensorGPT，vocab_size=32000, d=512）：

| 方法 | 参数量 | 压缩比 | PPL 变化 |
|------|--------|--------|---------|
| 原始 embedding | 16.4M | 1× | baseline |
| TT 分解 (rank=32) | 1.6M | ~10× | +5-10% |
| TT 分解 (rank=64) | 3.2M | ~5× | +2-5% |

**关键洞察**：Embedding 矩阵通常占 16MB 模型的 30-50%。如果用 TT 分解压缩 embedding，省出的空间可以加深网络。

MPS 作为语言模型（实验结果来自文献）：

| 模型 | 数据集 | BPC/PPL | 对比 |
|------|--------|---------|------|
| u-MPS (bond dim=128) | Penn Treebank | ~1.4 BPC | 弱于同参数 LSTM |
| u-MPS (bond dim=256) | 合成 CFG | 接近完美 | 强于 RNN |

#### 优势
- **精确的压缩理论**：MPS 的 bond dimension 直接控制可表达的纠缠量
- **Embedding 压缩立竿见影**：释放参数预算给更重要的组件
- **并行化友好**：MPS 的缩并可以 O(log n) 并行化
- **信息论保证**：截断误差由奇异值的衰减速率精确量化

#### 劣势
- **不适合做主架构**：u-MPS 在自然语言上性能远不如 Transformer/Mamba
- **Bond dimension 限制**：自然语言的"纠缠熵"远高于一维量子系统
- **训练不稳定**：需要特殊的正交化/规范化技巧
- **工程实现复杂**：缺乏成熟的 GPU 优化库

#### 推荐原因
**推荐用于 Embedding 压缩，不推荐用于主架构**。具体方案：
1. 将 embedding 矩阵 TT-分解，释放 ~5MB 参数预算
2. 用释放的预算加深主网络（增加 3-4 层）

#### 预期效果
TT-embedding 压缩 + 更深网络：预计改善 **0.01-0.02 BPB**（净效果，扣除 embedding 质量损失后）。

---

### 方法 E：重整化群 (Renormalization Group) 启发的多尺度架构

#### 物理原理

重整化群 (RG) 是统计物理和量子场论的核心框架，描述系统在不同尺度下的有效理论。核心思想：

1. **粗粒化**：将短程自由度积掉，得到有效的长程描述
2. **不动点**：RG flow 的不动点对应相变临界点
3. **标度律**：临界点附近的 universality class 由少数 relevant operator 决定

对语言建模的类比：
- 字符 → 子词 → 词 → 短语 → 句子 → 段落 → 文档
- 每一层对应一次"粗粒化"——底层处理局部模式，高层处理全局语义
- **关键洞察**：不同层应该有不同的"分辨率"

**具体实现**：Hourglass Transformer / Funnel Transformer 风格的架构，在中间层降采样序列长度（对应 RG 的粗粒化），在输出层恢复。

#### 定量分析

| 架构 | 序列处理方式 | 计算量 (vs 标准) | 质量 |
|------|------------|----------------|------|
| 标准 Transformer | 全长 attention 每层 | 1× | baseline |
| Funnel (2x下采样) | 中间层序列减半 | ~0.6× | -0.5% PPL |
| Hourglass (4x) | 中间层序列/4 | ~0.4× | -1-2% PPL |
| 分级 window attention | 底层窗口小，高层窗口大 | ~0.5× | ≈ baseline |

**在 Parameter Golf 中的价值**：计算量节省意味着 10 分钟内可以训练更多 step/data。

#### 优势
- **计算效率显著提升**：中间层的序列长度减半，attention 成本降到 1/4
- **物理动机清晰**：多尺度处理是自然语言的固有结构
- **与其他方法正交**：可以叠加在 Mamba/Transformer 上

#### 劣势
- **下采样信息损失**：粗粒化不可逆，一些细节信息会丢失
- **实现复杂**：需要仔细设计上采样/下采样策略
- **超参数多**：采样率、采样位置、上采样方式都需要调优

#### 推荐原因
**中等推荐**。主要价值在于节省计算而非改善质量。如果训练时间是瓶颈（很可能是），这可以让模型在 10 分钟内看到更多数据。

#### 预期效果
训练效率提升 ~1.5×，最终 BPB 改善 **0.005-0.015**（间接效果，通过更多训练步数）。

---

### 方法 F：统计力学视角 —— 量化即相变

#### 物理原理

模型量化可以用**统计力学的语言**来理解：

1. **连续权重** → **离散权重**：类似于 Ising 模型中连续自旋 → 离散自旋的映射
2. **量化误差**：类似于**热涨落**——在有限温度下偏离基态
3. **量化感知训练 (QAT)**：类似于**模拟退火**——在训练中逐渐降温，让系统找到离散配置空间中的最优态

**关键物理洞察**：存在一个**临界量化精度**，低于该精度模型性能会突然崩溃（类似相变）。对于 Parameter Golf：

- INT8: 安全区，几乎无损
- INT6: 次临界区，需要 QAT 保护
- INT4: 临界区，需要混合精度或特殊处理
- INT2/1-bit: 可能在超临界区——但有研究表明，通过足够的训练，1-bit 模型仍可工作

#### 当前 SOTA 量化策略分析

| 方法 | bits/param | 16MB 可容纳参数 | BPB 损失 | 训练开销 |
|------|-----------|---------------|---------|---------|
| FP32 | 32 | 4M | 0 | 1× |
| INT8 | 8 | 16M | +0.002 | 1.1× |
| INT6 | 6 | 21.3M | +0.005-0.01 | 1.2× |
| INT5 | 5 | 25.6M | +0.01-0.02 | 1.3× |
| INT4 | 4 | 32M | +0.02-0.05 | 1.5× |
| 1-bit (±1) | 1 | 128M | +0.05-0.15 | 2× |

**比赛中的最优策略**：当前 leaderboard 上几乎所有顶尖提交都使用 INT6 + QAT。1-bit 量化已有提交达到 1.1239 BPB（非限时赛道），证明极端量化是可行的。

#### 物理启发的量化改进

**1. 各向异性量化（Anisotropic Quantization）**

类比：晶体中不同方向的弹性模量不同。对应到模型：
- Attention 的 QK 投影对量化**高度敏感**（类似"硬"方向）
- FFN 的权重对量化**相对鲁棒**（类似"软"方向）
- 策略：对敏感层用更高精度（INT8），对鲁棒层用更低精度（INT4-5）

**2. 量化退火（Quantization Annealing）**

从高精度逐渐降到目标精度，类似模拟退火：
- 训练前 50%：FP16 正常训练
- 训练 50-80%：引入量化噪声，STE 梯度
- 训练 80-100%：固定到目标精度，fine-tune

**3. GPTQ-lite / 后量化微调**

训练结束后，用 Hessian 信息指导量化——优先量化 Hessian 对角元小的权重（"软模"），保护 Hessian 大的权重（"硬模"）。

#### 推荐原因
**强烈推荐**。量化是 Parameter Golf 中影响最大的单一优化。必须采用。

#### 预期效果
INT6 QAT vs INT8 naive：释放 ~33% 更多参数空间，换算为 **0.02-0.03 BPB 改善**。

---

### 方法 G：Test-Time Training (TTT) —— 非平衡统计力学

#### 物理原理

TTT 对应于**非平衡态统计力学**中的**弛豫过程**：模型在评估时，面对具体的测试数据，做一次快速的"弛豫"来适应当前数据的统计特性。

物理类比：
- 训练好的模型 = 平衡态系统
- 测试数据 = 外加微扰
- TTT = 系统在微扰下寻找新的局部平衡（类似 Onsager 弛豫）

**实现**：在评估时，用 LoRA 对已评估的 token 做梯度下降，动态调整模型参数。

#### 定量分析

| TTT 配置 | LoRA rank | 额外参数 | BPB 改善 | 评估时间增加 |
|---------|-----------|---------|---------|------------|
| 无 TTT | - | 0 | 0 | 1× |
| TTT (rank=4) | 4 | ~0.1M | -0.01 | ~3× |
| TTT (rank=8) | 8 | ~0.2M | -0.015 | ~5× |
| TTT (rank=16) | 16 | ~0.4M | -0.02 | ~8× |

当前 record holder 就使用了 TTT！它是从 1.1228 突破到 1.1194 的关键技术。

#### 优势
- **显著的 BPB 改善**：几乎免费获得 0.01-0.02 BPB
- **物理意义深刻**：在线自适应，匹配当前数据分布
- **与主架构正交**：任何架构都可以加 TTT
- **比赛规则允许**：只要只训练在已评估的 token 上

#### 劣势
- **评估时间大幅增加**：可能触及 10 分钟评估时限
- **LoRA 参数不计入 16MB（它们是运行时生成的）**，但 LoRA 的初始化代码需要在 artifact 中
- **不稳定**：在某些序列上 TTT 可能反而降低性能
- **超参数敏感**：学习率、更新步数、rank 都需要仔细调优

#### 推荐原因
**强烈推荐**。这是当前 SOTA 的关键成分。物理上对应非平衡弛豫，你可以利用 Onsager 倒易关系等物理直觉来设计更好的 TTT 策略。

#### 预期效果
改善 **0.01-0.02 BPB**，是性价比最高的单一优化之一。

---

### 方法 H：激活函数设计 —— 势能面工程

#### 物理原理

激活函数对应于系统的**势能面形状**。不同的势能面导致不同的动力学行为：

| 势能面 | 激活函数 | 物理行为 |
|--------|---------|---------|
| 谐振子 | Linear | 简谐振动，无非线性 |
| 双势阱 | tanh | 双稳态系统 |
| 单侧势垒 | ReLU | 单向传导 |
| 带陷阱的单侧 | LeakyReLU | 弱双向 |
| 光滑单侧 | SiLU/Swish | 量子隧穿效应 |
| **二次单侧** | **LeakyReLU²** | **非线性弹性** |

**LeakyReLU²** 是当前 record holder 使用的激活函数。其物理意义：$\sigma(x) = (\text{LeakyReLU}(x))^2$ 引入了**二次非线性**，类似于非线性弹性介质中的应力-应变关系。这提供了更强的特征交互，但保持了计算效率。

#### 定量分析

| 激活函数 | 每次计算成本 | 参数量 | 对 BPB 的影响 |
|---------|-----------|--------|-------------|
| ReLU | 1 op | 0 | baseline |
| SiLU | ~5 ops | 0 | -0.005 |
| LeakyReLU² | 2 ops | 0 | -0.008 |
| Learnable PWL (k=4) | ~8 ops | 4/激活 | -0.010? |

#### 推荐原因
**推荐使用 LeakyReLU²**。计算成本极低，但有实证效果。物理上对应非线性势能面，提供更丰富的梯度信息。

---

### 方法 I：Embedding 的物理设计 —— 信息论最优编码

#### 物理原理

Tokenizer + Embedding 的设计对应于**信息论中的信源编码**问题。Shannon 定理告诉我们，最优编码的平均码长等于信源熵。

**核心权衡**：
- 大词表 → embedding 表大，占用大量 16MB 预算 → 但每个 token 携带更多信息
- 小词表 → embedding 表小 → 但序列更长，需要更多计算

**最优词表大小估算**：

设 vocab_size = V, embedding_dim = d, 模型其余参数 = M_rest
- Embedding 参数量（tied input/output）: V × d
- 总约束: V × d × bits_per_param / 8 + M_rest_bytes ≤ 16MB

对于 INT6, d=384:
- V=1024: embedding = 1024×384 = 393K params → 294KB → 余 15.7MB
- V=4096: embedding = 4096×384 = 1.57M params → 1.18MB → 余 14.8MB  
- V=16384: embedding = 16384×384 = 6.29M params → 4.72MB → 余 11.3MB
- V=65536: embedding = 65536×384 = 25.2M params → 18.9MB → **超出！**

**Bigram Hash Embedding**（当前 leaderboard 技术）：

用 hash 函数将 bigram (前一个 token, 当前 byte) 映射到 embedding，不需要显式存储大词表。这对应于**特征哈希**——一种有损压缩编码。

#### 推荐原因
**推荐采用小词表 (1024-4096) + Bigram Hash**。释放参数预算给模型主体。

#### 预期效果
从 V=32000 减到 V=1024 + bigram hash：释放 ~5MB → 等效 **0.02-0.03 BPB 改善**。

---

## 第二部分：综合推荐架构

### 物理启发模型："弛豫动力系统" (Relaxation Dynamics Model)

基于上述分析，我推荐以下架构。名称取自非平衡统计力学中的弛豫过程——模型的每一层对应 token 表示在"势能面"上的一步弛豫。

```
┌──────────────────────────────────────────────────────────┐
│                    ARCHITECTURE OVERVIEW                   │
│                   "Relaxation Dynamics"                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Input: bytes → Bigram Hash Embedding (V=1024, d=384)    │
│                                                          │
│  ┌────────────────────────────────────────────┐          │
│  │  Shared Core Block (参数共享，循环 3 次)      │          │
│  │                                            │          │
│  │  Layer 1-4: Mamba-2 SSD Block              │          │
│  │    - d_model = 384                         │          │
│  │    - d_state = 32 (Mamba-3 style)          │          │
│  │    - d_conv = 4                            │          │
│  │    - expand = 2                            │          │
│  │    - 复数值状态 (RoPE trick)                 │          │
│  │    - LeakyReLU² 激活                        │          │
│  │    - RMSNorm (pre-norm)                    │          │
│  │                                            │          │
│  │  Layer 5: Cross-Sequence Attention (XSA)   │          │
│  │    - 4 heads, d_head = 96                  │          │
│  │    - GQA: 2 KV heads                       │          │
│  │    - Sliding window = 128                  │          │
│  │                                            │          │
│  │  + Depth-wise LoRA (rank=8) per iteration  │          │
│  └────────────────────────────────────────────┘          │
│                                                          │
│  共享块循环 3 次 → 等效 15 层深度                          │
│                                                          │
│  Output Head: Tied embedding + temperature scaling       │
│                                                          │
│  TTT: LoRA (rank=4) on attention layers during eval      │
│                                                          │
│  Quantization: INT6 QAT + GPTQ-lite post-training       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 参数预算详细分解

```
Component                          | Params (raw) | Bytes (INT6)
─────────────────────────────────────────────────────────────
Embedding (V=1024, d=384, tied)    |    393,216   |    294,912
                                   |              |
Shared Mamba Block ×4:             |              |
  Input proj (d→3×expand×d=2304)   |    884,736   |    663,552
  dt_proj, A, D per block          |     ~3,000   |      2,250
  Conv1d (d_conv=4)                |      1,536   |      1,152
  Out proj (2304→384)              |    884,736   |    663,552
  RMSNorm                          |        384   |        288
  Subtotal per Mamba layer         |    ~890,000  |    ~668,000
  × 4 layers                       |  3,560,000   |  2,670,000
                                   |              |
XSA Layer ×1:                      |              |
  Q,K,V proj (384→384)            |    442,368   |    331,776
  Output proj                      |    147,456   |    110,592
  RMSNorm                          |        384   |        288
  Subtotal                         |    ~590,000  |    ~443,000
                                   |              |
Depth-wise LoRA ×3 iterations:     |              |
  per Mamba layer: 2×(384×8) ×2    |     12,288   |      9,216
  × 4 Mamba + 1 XSA = 5 layers    |     61,440   |     46,080
  × 3 iterations                   |    184,320   |    138,240
                                   |              |
FFN (SwiGLU, 3× expansion):       |              |
  Up proj (384→1152) ×2 (gate)     |    884,736   |    663,552
  Down proj (1152→384)             |    442,368   |    331,776
  × 5 layers in shared block       |  ×1 (shared) |    995,328
                                   |              |
─────────────────────────────────────────────────────────────
Total raw parameters               | ~5,620,000   |            
Total bytes (INT6)                  |              | ~4,522,000
After zlib compression (~0.85×)     |              | ~3,844,000
─────────────────────────────────────────────────────────────
Remaining budget                    |              | ~12,156,000
→ 可用于增大模型或添加更多循环迭代
```

**预算大幅剩余！** 这意味着可以：
1. 增大 d_model 到 512 (参数量约 ×1.8)
2. 增加循环次数到 5 次 (等效 25 层)
3. 增加 LoRA rank
4. 或以上组合

### 修订后的推荐配置

```
─────────────────────────────────────────────────
FINAL RECOMMENDED CONFIGURATION
─────────────────────────────────────────────────

d_model          = 512
Vocab (BPE)      = 1024 + bigram hash (8192 buckets)
Embedding        = 1024×512 + 8192×64 (hash embed, 小维度)
                   → concat: 512 = 448 (main) + 64 (hash)

Shared Block (循环 4 次 = 等效 20 层):
  4× Mamba-2 SSD:
    d_state    = 32
    d_conv     = 4  
    expand     = 2
    activation = LeakyReLU²
  1× XSA (在最后一层, 最深3-4个循环中启用):
    heads      = 4
    KV heads   = 2
    d_head     = 128

  Depth LoRA: rank=12, 每循环独立

FFN: SwiGLU, expansion=3 (512→1536→512)
     共享于所有循环迭代

Quantization:
  Attention QKV: INT8 (敏感层)
  Mamba projections: INT6
  FFN: INT5 (最鲁棒)
  Embedding: INT8
  LoRA: INT8

Training:
  Optimizer: Muon (parallel variant)
  LR schedule: cosine with warmdown
  Batch size: maximize within 8×H100 memory
  Data: FineWeb train split, maximize tokens seen
  EMA: decay=0.999, 用 EMA 权重做最终评估

Evaluation:
  TTT: LoRA rank=4 on XSA layers
  Sliding window stride: 64
  Eval batch: 最大化 XSA 的跨序列效果

─────────────────────────────────────────────────
```

### 参数量估算 (修订后)

```
Embedding:  1024×512 = 524K + 8192×64 = 524K → 1.05M
Mamba ×4:   ~4×(3×2×512²) = 6.29M
XSA ×1:     ~4×512² = 1.05M  
FFN ×1:     ~3×3×512² = 2.36M (SwiGLU, shared)
LoRA:       5 layers × 4 iters × 2×(512×12) = 0.25M
Norms etc:  ~0.01M
─────────────────────
Total:      ~11.0M params

INT6 平均: 11.0M × 6/8 = 8.25MB
混合精度 (加权): ~8.8MB
zlib 压缩后: ~7.5MB

Remaining: ~8.5MB → 可以进一步增大模型
```

### 最终调整建议

由于还有 ~8MB 余量，有两个方向：

**方向 A（推荐）：增大到 d=640, 循环 5 次**
- 参数量 ~17M，压缩后约 13-14MB
- 等效 25 层深度
- 预期 BPB: ~1.10-1.11

**方向 B：保持 d=512，增加独立层**
- 5 个共享层 + 3 个独立层（首尾不共享）
- "Sandwich" 策略：首尾独立层捕获输入/输出特有模式
- 预期 BPB: ~1.10-1.12

---

## 第三部分：各方法对比总表

| 方法 | 物理原理 | 预期 BPB 改善 | 实现难度 | 风险 | 推荐度 |
|------|---------|-------------|---------|------|--------|
| A. Mamba SSM 主架构 | 线性时变动力系统 | 0.02-0.05 | 中 | 低 | ★★★★★ |
| B. 参数共享/Recursive | Neural ODE 离散化 | 0.01-0.02 | 低 | 低 | ★★★★★ |
| C. KAN | K-A 表示定理 | -0.05 (负!) | 高 | 高 | ★☆☆☆☆ |
| C'. KAN-inspired 激活 | 可学习势能面 | 0.005-0.01 | 低 | 低 | ★★★☆☆ |
| D. TT-Embedding | MPS/张量网络 | 0.01-0.02 | 中 | 中 | ★★★★☆ |
| E. 多尺度/RG | 重整化群 | 0.005-0.015 | 高 | 中 | ★★★☆☆ |
| F. INT6 QAT | 相变/退火 | 0.02-0.03 | 中 | 低 | ★★★★★ |
| G. TTT | 非平衡弛豫 | 0.01-0.02 | 中 | 中 | ★★★★★ |
| H. LeakyReLU² | 非线性势能面 | 0.005-0.008 | 极低 | 极低 | ★★★★☆ |
| I. 小词表+Hash | 信源编码 | 0.02-0.03 | 低 | 低 | ★★★★★ |

**累计预期改善（全部采用）**：~0.10-0.13 BPB（不可线性叠加，实际约 0.06-0.08）

**从 baseline 1.224 出发**：预期可达 **~1.14-1.16 BPB**

**从当前 SOTA 1.119 出发**：要超越需要极致优化或新的突破性方法。

---

## 第四部分：实施路线图

### Phase 1: 基础框架 (Day 1-3)
1. Fork `openai/parameter-golf` 仓库
2. 实现 Mamba-2 SSD 基础架构 (d=512)
3. 实现 Recursive (参数共享) 框架
4. 验证在 baseline 数据上能跑通

### Phase 2: 核心优化 (Day 4-10)
1. 集成 INT6 QAT
2. 实现 bigram hash embedding
3. 调优 Muon optimizer + cosine warmdown
4. 添加 EMA

### Phase 3: 高级技巧 (Day 11-18)
1. 实现 XSA (跨序列 attention)
2. 添加 depth-wise LoRA
3. 实现 TTT 评估流程
4. 混合精度量化 (各向异性)

### Phase 4: 调优与提交 (Day 19-24，deadline April 30)
1. 超参数搜索 (d_model, 循环次数, LoRA rank)
2. GPTQ-lite 后量化微调
3. 滑动窗口评估优化
4. 三种子统计显著性检验
5. 撰写 write-up，提交 PR

---

## 结语

这个架构的核心物理哲学是：**语言建模是一个弛豫过程**。Token 的表示从初始嵌入出发，在由共享参数定义的"势能面"上演化（Mamba SSM 的连续动力学），通过多次循环迭代逐步趋近"平衡态"（最终预测分布）。在评估时，TTT 提供了一次额外的非平衡弛豫，让模型适应当前数据的局部统计特性。整个过程中，量化对应于将连续态空间离散化（类似格点场论），而激活函数的选择对应于势能面的形状设计。

这不仅是一个工程方案，更是一个连贯的物理图像。
