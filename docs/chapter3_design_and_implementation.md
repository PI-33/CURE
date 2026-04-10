# Chapter 3: Design and Implementation

> **说明**：若你需要的是「奖励怎么算、尤其 case 奖励的公式与代码对应」，请以 **`reward_computation_reference.md`** 为准；下面内容是早期按论文章节搭的框架，可作提纲用，不必当作实现细节来源。

## 3.1 Motivation: Using Unit Tests for Inference

在代码生成任务中，单次采样（one-shot generation）往往无法稳定命中正确程序。一个常见提升策略是 Best-of-N：先生成多份候选代码，再利用外部信号进行排序与筛选。对于代码任务而言，最自然的外部信号是单元测试执行结果。  
然而，直接依赖 ground-truth (GT) 测试存在两个现实限制：

1. **标注成本高**：高质量 GT 测试集构建昂贵，且覆盖度有限。  
2. **训练可扩展性差**：在大规模 RL 迭代中，完全依赖 GT 信号会限制可用样本规模。  

基于此，CURE 的核心思想是将“代码生成器（coder）”与“测试生成器（unit tester）”放在同一个 RL 闭环中共同优化。即：模型不仅生成代码，也生成可用于区分代码质量的测试；测试再反过来为代码提供训练信号。这一机制将推理阶段的“测试驱动筛选”前移到训练阶段，使模型获得更强的自校验能力。

---

## 3.2 Ground-Truth-Free Reward Estimation

### 3.2.1 Problem Setup

对每个问题，系统采样得到：

- `k_code` 个代码候选
- `k_case` 个生成测试候选

执行模块输出布尔矩阵：

\[
M \in \{0,1\}^{k_{code} \times k_{case}}, \quad
M_{ij}=1 \iff \text{code}_i \text{ passes case}_j
\]

在 GT-Free 模式中，奖励估计仅依赖 \(M\)（对应实现中的 `all_case_table_i`），不使用 GT 列参与主奖励计算。

### 3.2.2 EM-Based Joint Estimation

我们联合估计三组潜变量：

- \(q_i \in [0,1]\)：第 \(i\) 个代码候选的质量分数（code quality）  
- \(w_j \ge 0\)：第 \(j\) 个测试候选的可靠性权重（test reliability）  
- \(e_j \in \{0,1\}\)：测试 \(j\) 的期望行为（expected behavior）

其中，\(e_j=1\) 表示高质量代码应通过该测试，\(e_j=0\) 表示高质量代码应不通过该测试。

#### Initialization

设列通过率 \(p_j=\frac{1}{k_{code}}\sum_i M_{ij}\)。  
仅保留可区分测试列（discriminative columns）：

\[
0 < p_j < 1
\]

初始化规则：

\[
e_j = \mathbb{I}(p_j>0.5), \quad
w_j = \max(p_j, 1-p_j) \text{ (for discriminative columns)}
\]

非可区分列权重置零。  
代码质量初值设为常数 0.5；若启用 anchor（公开样例弱监督），则采用：

\[
q_i^{(0)} = \alpha \cdot a_i + (1-\alpha)\cdot 0.5
\]

其中 \(a_i\) 为代码在公开样例上的通过率，\(\alpha\) 对应 `anchor_strength`。

#### Iterative Updates

每轮迭代包括：

1. **更新代码质量 \(q\)**（agreement estimation）：
\[
q_i'=\frac{\sum_j w_j \cdot \mathbb{I}(M_{ij}=e_j)}{\sum_j w_j}
\]
若使用 anchor：
\[
q_i \leftarrow \alpha a_i + (1-\alpha)q_i'
\]

2. **按中位数划分好/差代码组**：
\[
\text{good}=\{i\mid q_i>\mathrm{median}(q)\},\quad
\text{bad}=\{i\mid q_i\le \mathrm{median}(q)\}
\]

3. **更新测试权重与期望行为**：
\[
g_j=\mathbb{E}[M_{ij}\mid i\in \text{good}],\quad
b_j=\mathbb{E}[M_{ij}\mid i\in \text{bad}]
\]
\[
w_j=|g_j-b_j|,\quad
e_j=\mathrm{round}(g_j)
\]

该过程对应 `reward.py` 中 `em_estimate()` 的实现。

### 3.2.3 Filtering Strategy

为避免无信息样本污染训练，系统会跳过以下情形：

- `M` 为空（无代码或无测试）
- 不存在可区分测试列（所有列全 0 或全 1）
- 后续 reward 方差约等于 0（无法形成组内相对偏好）

这些过滤策略可显著减少噪声奖励带来的策略漂移。

---

## 3.3 Reward Construction and Joint Optimization with RL

### 3.3.1 Code Reward

EM 收敛后，代码候选的原始奖励定义为：

\[
r_i^{code}=\frac{\sum_j w_j \cdot \mathbb{I}(M_{ij}=e_j)}{\sum_j w_j}
\]

直观上，该分数衡量“代码行为与高置信测试期望的一致程度”。

### 3.3.2 Test Reward

当前实现中，测试奖励采用“测试列与代码质量分数的相关性”：

\[
r_j^{case} = \mathrm{corr}(q, M_{:j})
\]

其中 `corr` 为 Pearson 相关系数；全通/全不通列直接置 0。  
该定义鼓励“更易被高质量代码通过、被低质量代码拒绝”的测试。

### 3.3.3 Group Normalization and Length Regularization

在写入 RL 数据前，reward 先进行组内标准化：

\[
\hat{r}=\frac{r-\mu(r)}{\sigma(r)}
\]

当 `enable_efficient` 与 `sb_use_length_reg` 同时开启时，系统可叠加长度正则化（`length_regularize()`），用于约束长响应偏置。

此外，在训练器 `trainer.py` 中，reward 会再次进行组内中心化/标准化，以适配 GRPO 的相对优势优化目标（relative preference learning）。

### 3.3.4 Joint Optimization in RL


这实现了 coder 与 tester 的联合优化：代码质量提升会提升测试区分能力；测试区分能力提升又会反哺代码奖励质量，形成闭环。

---

## 3.4 Implementation Overview

### 3.4.1 End-to-End Pipeline

完整训练迭代在 `run.py` 中按固定顺序执行：

1. `sample.py`：生成代码/测试候选与文本响应  
2. `execute.py`：执行所有代码-测试组合，得到布尔通过矩阵  
3. `reward.py`：计算并写出 RL 训练样本 + 诊断信息  
4. `train.py`：加载 RL 样本执行策略优化  
5. （按间隔）`evaluation/eval.py`：阶段性评估并记录指标

### 3.4.2 Key Artifacts


### 3.4.3 Engineering Notes for Reproducibility

实现中包含若干工程稳定性设计：

- 在子进程调用前清理 NCCL/CUDA 相关环境变量，降低多阶段串行流程中的上下文污染风险
- 每步落盘 diagnostics，支持按 step 追踪 reward 分布与有效样本量
- 按实验目录隔离 temp/results/ckpt，保证多实验并行时的可复现性

---

## Appendix: Mapping to Existing GT-Based Reward

为保持与原版 CURE 对照，当前实现保留 GT-based 分支（`use_self_bootstrap=False`）。  
即便在 GT-Free 模式下，系统仍可额外计算 GT-based code reward 与自举 code reward 的 Spearman 相关系数（仅用于诊断，不参与训练更新），用于验证自举信号是否与 GT 信号同向。
