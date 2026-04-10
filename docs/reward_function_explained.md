# 奖励函数完全解析

本文档说明仓库中**两套奖励**（GT-based 与 Self-bootstrap）的数学含义与实现要点，对应 `use_self_bootstrap` 的两种取值。

---

## 0. 前置：数据结构

每个题目经采样与执行后，得到一个 **bool_table**（行 = 代码候选，列 = 测试用例）：

```
              GT测试₁ … GT测试ₜ | 生成测试₁ … 生成测试ₖ
code₁:          …                 |    …
code₂:          …                 |    …
  …                               |
codeₙ:          …                 |    …

              ← GT 子矩阵 →       | ← 生成测试子矩阵 M →
              ←────────── full_table (n × (t+k)) ──────────→
```

- `t`：GT 测试数量（数据集中私有标准测例）
- `k`：模型生成的测试条数
- 格子为真当且仅当：该代码在该测例上执行输出与期望一致

---

## 第一部分：GT-Based 奖励

> `use_self_bootstrap = False`

### 1.1 Code 奖励

**公式**

\[
\text{raw\_reward}_i = \frac{1}{t}\sum_{c=1}^{t} \mathbb{1}[\text{code}_i \text{ 通过 GT}_c]
\]

即第 \(i\) 条代码在 **全部 GT 列**上的平均通过率。

**后处理**：对同一题、同一 prompt 下所有代码候选的 `raw_reward` 做 **组内 z-score**（`normalize_reward`）。若 \(\sigma=0\)，整组丢弃。  
若 `enable_efficient` 为真，可再对标准化后的向量做 **长度正则**（`length_regularize`）。

### 1.2 Case 奖励

**Step 1 — 正确代码集合（依赖 GT）**

- `correct_codes`：在 **所有 GT 列**上均为通过的代码下标  
- `wrong_codes`：其余代码  

若无任何 `correct_codes` → **本题不产生 case 奖励**。

**Step 2 — 符号 `sign_j`**

在仅含正确代码的子表上按列：若该生成测试被**每一个**正确代码通过，则 `sign_j = +1`，否则 `sign_j = -1`。

**Step 3 — 缩放 `scale_j`**

当存在错误代码时：

- **好测试列**（`sign=+1`）：用错误代码在该列上的失败程度放大（`post_stage` 为假时常为失败率均值；为真时可变为「是否全体失败」的硬指标）。  
- **坏测试列**（`sign=-1`）：用错误代码在该列上的误通过程度放大。

**Step 4**

\[
\text{case\_reward}_j = \text{sign}_j \times \text{scale}_j
\]

再 `normalize_reward`，若 `enable_efficient` 可再 `length_regularize`。

### 1.3 说明

- Code 信号完全来自 GT；Case 信号依赖「至少一条代码全对 GT」，否则无 case 组。

---

## 第二部分：Self-Bootstrap 奖励

> `use_self_bootstrap = True`

**主奖励**只使用生成测试子矩阵 \(\mathbf{M}\)。GT 列 **不进入** code/case 的主奖励公式。若 \(t>0\)，可额外计算「GT code 分数」与「自举 code 分数」的 **Spearman 相关**作诊断，**不参与**训练标签。

### 2.1 本题是否进入 EM

- \(\mathbf{M}\) 行数或列数为 0 → 跳过。  
- 若不存在任一列满足 \(0 < p_j < 1\)（\(p_j\) 为该列通过率）→ **整题跳过**。

### 2.2 EM：估计 `q`, `w`, `e`

**输入**：\(\mathbf{M} \in \{0,1\}^{n \times k}\)。

**可区分列**：\(0 < p_j < 1\)。非可区分列在迭代中 **`w[j]` 恒为 0**，且不参与 M 步更新。

**初始化**

- \(e_j = \mathbb{1}(p_j > 0.5)\)  
- 对可区分列：\(w_j = \max(p_j, 1-p_j)\)  
- 若启用 anchor：`q_i = \alpha \cdot a_i + (1-\alpha)\cdot 0.5\)，其中 \(a_i\) 为第 \(i\) 份代码在 **前 `min(len(example_input), t)` 列**（GT 段）上的平均通过；否则 \(q_i = 0.5\)。

**每一轮迭代**

1. **更新 \(q\)**  
   \[
   q'_i = \frac{\sum_j w_j \,\mathbb{1}(M_{ij}=e_j)}{\sum_j w_j}
   \]  
   若有 anchor：\(q \leftarrow \alpha a + (1-\alpha) q'\)；否则 \(q \leftarrow q'\)。

2. **划分好 / 差代码**  
   \(\text{good} = \{i \mid q_i > \mathrm{median}(q)\}\)，\(\text{bad} = \{i \mid q_i \le \mathrm{median}(q)\}\)。  
   若任一侧为空 → **提前结束迭代**。

3. **更新 \(w\) 与 \(e\)（仅可区分列）**  
   记 \(g_j = \mathbb{E}[M_{ij}\mid i\in\text{good}]\)，\(b_j = \mathbb{E}[M_{ij}\mid i\in\text{bad}]\)。  
   \[
   w_j = |g_j - b_j|,\quad e_j = \mathrm{round}(g_j)
   \]  
   非可区分列保持 \(w_j=0\)。

### 2.3 Code 奖励（自举）

EM 结束后：

\[
r^{\text{code}}_i = \frac{\sum_j w_j \,\mathbb{1}(M_{ij}=e_j)}{\sum_j w_j}
\]

要求 \(\sum_j w_j > 0\)，且组内 \(r^{\text{code}}\) 的标准差 \(\ge 10^{-6}\)，否则本题 code 组不写入。

再 `normalize_reward`；仅当 `enable_efficient` **且** `sb_use_length_reg` 时做 `length_regularize`。

### 2.4 Case 奖励（自举）

对每条生成测试列 \(j\)，用 EM 得到的 **\(q\)**（与 \(w,e\) 同一次估计）与该列向量算 **Pearson 相关系数** \(\rho\)。

- 若 \(\mathrm{std}(q) < 10^{-6}\) → 本题 case 奖励无效。  
- 对第 \(j\) 列：若 \(p_j \in \{0,1\}\) → \(r^{\text{case}}_j = 0\)；若该列 0/1 向量标准差 \(< 10^{-8}\) → \(0\)；否则 \(r^{\text{case}}_j = \rho(q,\, M_{\cdot j})\)，`NaN` 记为 0。

再对 \(r^{\text{case}}\) 做 `normalize_reward`（组内 std \(\ge 10^{-6}\)）；长度正则条件同 2.3。

---

## 第三部分：两套方案对比

| 项目 | GT-based | Self-bootstrap |
|------|----------|----------------|
| Code 信号 | GT 列平均通过率 | \(\sum_j w_j \mathbb{1}(M_{ij}=e_j)/\sum_j w_j\)（仅用 \(\mathbf{M}\) 与 EM） |
| Case 信号 | sign × scale（依赖全对 GT 的代码） | Pearson(\(q\), 第 \(j\) 列) |
| 主奖励是否用 GT | 是 | 否（GT 可用于 anchor 或 Spearman 诊断） |
| Code 组丢弃 | z-score 方差为 0 | 无区分列；\(\sum w=0\)；或 raw code 方差 \(<10^{-6}\) |
| Case 组丢弃 | 无全对 GT 代码 | \(\mathrm{std}(q)\approx 0\)；或 raw case 方差 \(<10^{-6}\) |

---

## 第四部分：后处理与训练端

### 4.1 组内 z-score

\[
\hat r_k = \frac{r_k - \mu}{\sigma},\quad \sigma=0 \Rightarrow \text{丢弃该组}
\]

若 `enable_efficient` 且**所有**奖励均为 1，则 `normalize_reward` **原样返回**（不中心化）。

### 4.2 长度正则

- **GT 路径**：`enable_efficient` 时可对 code/case 做 `length_regularize`。  
- **自举路径**：`enable_efficient` **且** `sb_use_length_reg`。

### 4.3 训练中的再一次组内标准化

对每个 prompt 组内标量奖励，训练阶段通常再做一次减均值、除标准差，用于相对优势估计。

---

## 第五部分：Anchor（自举）

- `n_anchor = min(len(example_input), t)`；若 \(n_anchor>0\)，  
  \(a_i = \frac{1}{n_anchor}\sum_{c=1}^{n_anchor} B_{i,c}\)（矩阵前 \(n_anchor\) 列为 GT 段）。  
- 每轮 E 步后：\(q \leftarrow \alpha a + (1-\alpha) q'\)（`anchor_strength` = \(\alpha\)）。

---

## 第六部分：诊断与调参

| 现象 | 可能原因 | 方向 |
|------|----------|------|
| 自举 case 组很少 | \(\mathrm{std}(q)\approx 0\) 或列全 0/全 1 | 增大采样代码数、提高生成测试多样性 |
| 自举 code 与 GT 诊断相关偏低 | EM 不稳定 | 增加 `em_iterations`、`anchor_strength` |
| 长度行为异常 | 自举下长度正则开关 | 检查 `sb_use_length_reg` 是否应开启 |

---

## 附录：实现映射

| 逻辑 | 说明 |
|------|------|
| GT code | `compute_gt_code_reward` |
| GT case | `compute_gt_case_reward` |
| EM | `em_estimate` |
| 自举 code | `compute_sb_code_reward` |
| 自举 case | `compute_sb_case_reward`（Pearson） |
| 标准化 / 长度 | `normalize_reward`、`length_regularize` |
| 组格式 | `build_group_entry` |
| 训练端再标准化 | `make_experience` |
