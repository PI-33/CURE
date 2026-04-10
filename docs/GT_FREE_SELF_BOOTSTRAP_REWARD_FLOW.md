# GT-Free 自举奖励全过程（基于当前 `reward.py` 实现）

本文档对应当前仓库的 `optimization/reward.py`，解释在 `use_self_bootstrap=True` 时，奖励如何从执行结果一步步变成训练可用的 `rl_code_data.json` / `rl_case_data.json`。

## 1. 这套奖励要解决什么问题

原版 CURE 的 reward 依赖 GT 测试（`all_test_table`），而 GT-Free 自举奖励希望主要依赖模型自己生成的测试（`all_case_table`）来估计：

- 哪些代码更可靠（code quality）
- 哪些测试更可靠（test reliability）

实现方式是：对代码-测试的通过矩阵做 EM 迭代，得到 `q`（代码质量）、`w`（测试权重）、`e`（测试预期行为），再分别构造 code reward 和 case reward。

## 2. 输入数据长什么样

`reward.py` 读取文件：

- `./temp_data/outputs-rl-{pretrained_model}-{dataset}.json`

每个样本里最核心字段是：

- `all_case_bool_table`: 形状 `(k_code, t + k_case_gen)` 的布尔矩阵
  - 前 `t` 列是 GT 测试结果
  - 后 `k_case_gen` 列是生成测试结果
- `num_ground_truth_test = t`
- `code_response_length` / `case_response_length`
- `full_code_generation` / `full_case_generation`
- 对应 prompt 字段

在自举分支里，真正用于 EM 的矩阵是：

- `M = all_case_table_i = full_table[:, t:]`（只用生成测试列）

## 3. 每道题的主流程

对每道题，`reward.py` 做下面步骤：

1. 检查 `all_case_bool_table` 是否为空；空则跳过。
2. 切分 `full_table`：
   - `all_test_table_i = full_table[:, :t]`（GT 部分，仅用于诊断相关性）
   - `all_case_table_i = full_table[:, t:]`（自举主输入）
3. 进入 `use_self_bootstrap` 分支，计算自举奖励。

## 4. 自举前置过滤

在 EM 前会做两层过滤：

1. `M` 为空（行或列为 0） -> 跳过，状态 `skipped_empty`
2. `M` 没有可区分测试列 -> 跳过，状态 `skipped_no_disc`
   - 这里“可区分测试列”定义为：该列通过率 `pass_rate` 满足 `0 < pass_rate < 1`

直觉：如果某列所有代码都通过或都失败，那列没有任何排序信息。

## 5. Anchor（可选）

当 `use_anchor=True` 时，代码会用公开样例（`example_input`）作为弱锚定：

- `n_anchor = min(len(example_input), t)`
- 若 `n_anchor > 0`，则
  - `anchor_scores = mean(full_table[:, :n_anchor], axis=1)`
  - 即每条代码在公开样例上的通过率

Anchor 不直接作为最终 reward，而是参与 EM 中 `q` 的更新混合。

## 6. EM 迭代细节（`em_estimate`）

输入：`M`（`k_code x k_case`）

输出：

- `q[i]`：代码质量分数（0~1）
- `w[j]`：测试权重
- `e[j]`：该测试的“预期行为”（0 或 1）

### 6.1 初始化

1. `pass_rate[j] = mean(M[:, j])`
2. 可区分列掩码：`disc_mask = (pass_rate > 0) & (pass_rate < 1)`
3. `e[j] = 1 if pass_rate[j] > 0.5 else 0`
4. `w[j]` 初始化：
   - 非可区分列 `w[j] = 0`
   - 可区分列 `w[j] = max(pass_rate[j], 1 - pass_rate[j])`
5. `q` 初始化：
   - 有 anchor：`q = alpha * anchor_scores + (1-alpha) * 0.5`
   - 无 anchor：全 0.5

若可区分列数量为 0，直接返回初值。

### 6.2 迭代更新（循环 `n_iter` 次）

1. **更新 `q`（E-step 风格）**
   - `agreement_i = sum_j w[j] * 1(M[i,j] == e[j])`
   - `q_new[i] = agreement_i / sum(w)`（若 `sum(w)==0` 则保持旧值）
   - 有 anchor 时混合：`q = alpha * anchor_scores + (1-alpha) * q_new`

2. **按 `q` 二分代码**
   - `threshold = median(q)`
   - `good_mask = q > threshold`
   - `bad_mask = q <= threshold`
   - 若任一组为空，提前停止迭代

3. **更新每个测试的 `w` 与 `e`（M-step 风格）**
   - 非可区分列：`w[j]=0`
   - 可区分列：
     - `good_pass = mean(M[good_mask, j])`
     - `bad_pass = mean(M[bad_mask, j])`
     - `w[j] = abs(good_pass - bad_pass)`（区分力）
     - `e[j] = round(good_pass)`（好代码群体的多数行为）

## 7. Code reward 如何算

函数：`compute_sb_code_reward(M, q, w, e)`

原始奖励（未标准化）：

- `r_code[i] = sum_j w[j] * 1(M[i,j]==e[j]) / sum(w)`

过滤逻辑：

1. `sum(w)==0` -> 整组无效
2. `std(r_code) < 1e-6` -> 组内无区分度，跳过

后处理：

1. `normalize_reward`：组内 z-score
2. 若 `enable_efficient and sb_use_length_reg`：
   - 调用 `length_regularize`（先符号化再按长度做平衡标准化）
3. 通过 `build_group_entry` 打包写入 `code_data`

## 8. Case reward 如何算

函数：`compute_sb_case_reward(M, q)`

当前实现不是用 `w/e`，而是“每列通过结果与 `q` 的相关性”：

1. 若 `std(q) < 1e-6` -> 无法计算，返回空
2. 对每个测试列 `j`：
   - 若该列 `pass_rate` 为 0 或 1 -> reward 置 0
   - 否则计算 `corr_j = corrcoef(q, M[:,j])`
   - `NaN` 设为 0
3. 得到 `r_case[j] = corr_j`

过滤与后处理：

1. `std(r_case) < 1e-6` -> 跳过
2. z-score 标准化
3. 可选长度正则（同 code）
4. `build_group_entry` 打包写入 `case_data`

直觉：如果某个测试“更容易被高质量代码通过、被低质量代码拒绝”，它与 `q` 的相关性会更高，奖励更正向。

## 9. 结果如何落盘

遍历全部题目后：

- `separate_training=False`：写 `./temp_data/rl_data.json`
- `separate_training=True`（当前默认）：分别写
  - `./temp_data/rl_code_data.json`
  - `./temp_data/rl_case_data.json`

同时会输出诊断：

- `./temp_data/reward_diagnostics.json`
  - 包含每题状态（used/skipped）、`n_disc_tests`、`q_std`、`w_nonzero`、reward 统计等
- 追加文本日志到 `./results/results-{outputs_name}.txt`

## 10. 与 GT-based 分支的关系

即使在 `use_self_bootstrap=True`，代码仍会在有 GT 时额外计算：

- `gt_cr = compute_gt_code_reward(all_test_table_i)`
- 再与 `code_reward_raw` 计算 Spearman 相关（仅诊断，不参与训练）

这能帮助判断自举信号是否和 GT 信号同向。

## 11. 配置项与影响

关键参数在 `optimization/optimization_config.py`：

- `use_self_bootstrap`: 是否启用自举
- `em_iterations`: EM 迭代次数
- `use_anchor`: 是否启用公开样例锚定
- `anchor_strength`: Anchor 混合系数 `alpha`
- `sb_use_length_reg`: 自举模式是否做长度正则
- `enable_efficient`: 长度正则总开关（自举分支需与上项同时为 True 才生效）

## 12. 当前实现中容易误解的点

1. `min_weight_threshold` 虽然在参数中解析了，但当前 `reward.py` 主流程没有实际使用该阈值做过滤。
2. Case reward 当前实现是“`corr(q, M[:,j])`”，不是旧设计文档里常见的 `sign × discriminability` 版本。
3. 自举分支的核心输入是生成测试矩阵 `M`；GT 矩阵主要用于诊断相关性，不参与 reward 主计算。

## 13. 一张流程图（文字版）

1. 读取 `outputs-rl-*.json`
2. 对每题切 `full_table -> all_test_table + M`
3. 过滤空样本 / 无可区分列
4. （可选）构造 anchor
5. EM 得到 `q,w,e`
6. 算 `code_reward_raw` -> 标准化 ->（可选长度正则）-> 写 `code_data`
7. 算 `case_reward_raw` -> 标准化 ->（可选长度正则）-> 写 `case_data`
8. 汇总写入 `rl_code_data.json` / `rl_case_data.json`
9. 产出 `reward_diagnostics.json` 与文本日志
