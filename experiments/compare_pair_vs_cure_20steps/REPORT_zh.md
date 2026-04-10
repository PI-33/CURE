# Pair vs CURE 实验对比报告（优化诊断 step 1–20）

## 1. 数据与设定

| 项目 | 实验 1（pair） | 实验 2（cure，已更正） |
|------|----------------|-------------------------|
| 诊断目录 | `0409_Qwen2.5-7B-Instruct-2gpu_350step_pair/optimization/diagnostics` | `04031200_Qwen2.5-7B-Instruct_2gpu_350step_cure/optimization/diagnostics` |
| 对比步数 | step 1–20 | step 1–20 |
| 基座模型 | Qwen2.5-7B-Instruct | **同左** |
| `reward.reward_mode` | `pairwise_disagreement` | `gt_based` |

**说明：** 两侧为**同一基座模型**，执行类指标可比性比「跨模型」更好；`code_reward` / `case_reward` 的标量仍因 **reward 定义不同**（pairwise vs GT）而不能简单用绝对值论优劣，但可在各自语义下看曲线形态。

**`gt_correlation`：** pair 在 step 1–20 的 JSON 中**均有** `mean_spearman` 与 `n`。**本 CURE 实验**在 step 1–20 的 JSON 中**未写入** `gt_correlation` 字段（全为缺失），故**无法与 pair 对齐对比**该诊断量；图中 cure 曲线为空，仅 pair 有值。

**Reward 聚合规模：** CURE 侧 `code_reward.num_groups` / `case_reward.num_groups` 明显小于 pair（见下表），`case_reward.num_samples` 亦常低于 1600。解读 reward 的 mean/std 时需注意**聚合子集不同**，方差与均值波动可能更噪。

---

## 2. 生成的图表与数据

目录：`/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiments/compare_pair_vs_cure_20steps/`

| 文件 | 内容 |
|------|------|
| `fig1_sampling_execution.png` | 采样 token + `code_*` / `case_*` 准确率 |
| `fig2_p01_bon.png` | `p_01` / `p_00` + BoN（K=4 / K=16） |
| `fig3_rewards.png` | code/case reward 的 mean、std、num_groups |
| `fig4_samples_gt_corr.png` | `num_samples` + `gt_correlation.mean_spearman`（cure 无 gt 字段） |
| `fig5_gt_corr_n.png` | `gt_correlation.n`（仅 pair） |
| `series_steps_1_20.csv` | 逐步标量 |
| `plot_compare.py` | 复现：`python3 plot_compare.py` |

---

## 3. 各指标逐项对比（step 1–20）

**「20 步均值」** 为算术平均；**「第 20 步」** 为单点。箭头含义：仅表示数值高低，不隐含算法优劣。

### 3.1 `sampling`

| 指标 | pair 20 步均值 | cure 20 步均值 | 第 20 步 pair | 第 20 步 cure | 简评 |
|------|----------------|----------------|---------------|---------------|------|
| `mean_code_response_tokens` | 669.3 | 655.0 | 658.4 | 621.6 | 同量级，cure 略短或接近 |
| `mean_case_response_tokens` | 737.9 | 784.8 | 675.3 | 705.0 | cure 均值略高；第 20 步 pair 因 case 指标跳水而 context 不同，需结合执行曲线 |

### 3.2 `execution`

| 指标 | pair 均值 | cure 均值 | step 20 pair | step 20 cure | 简评 |
|------|-----------|-----------|--------------|--------------|------|
| `code_acc` | 0.214 | 0.207 | 0.264 | 0.214 | 20 步均值接近；第 20 步 pair 更高 |
| `code_accumulate_acc` | 0.301 | 0.291 | 0.336 | 0.320 | 同阶，pair 略高 |
| `case_acc` | 0.260 | 0.286 | **0.150** | **0.348** | 均值 cure 略高；**第 20 步 pair 骤降**（与此前报告一致），cure 相对稳定 |
| `case_accumulate_acc` | 0.337 | 0.355 | **0.193** | **0.361** | 同上 |
| `p_01_as_logged` | 0.299 | 0.293 | 0.229 | 0.319 | 均值接近；step 20 cure 更高 |
| `p_00` | 0.049 | 0.045 | 0.056 | 0.079 | 同量级 |
| BoN K=4 `acc` | 0.242 | 0.248 | 0.28 | 0.23 | 均值 cure 略高；step 20 pair 略高 |
| K=4 `accumulate_acc` | 0.338 | 0.345 | 0.349 | 0.371 | cure 略高 |
| K=16 `acc` | 0.254 | 0.275 | 0.27 | 0.25 | cure 均值略高 |
| K=16 `accumulate_acc` | 0.370 | 0.396 | 0.35 | 0.418 | cure 略高 |

### 3.3 `reward`

| 指标 | pair 均值 | cure 均值 | step 20 pair | step 20 cure | 简评 |
|------|-----------|-----------|--------------|--------------|------|
| `code_reward.mean` | 0.301 | 0.291 | 0.336 | 0.321 | 同量级小幅差异 |
| `code_reward.std` | 0.412 | 0.406 | 0.437 | 0.417 | 接近 |
| `code_reward.num_groups` | 76.2 | **9.6** | 77 | **6** | **CURE 聚合组数远小于 pair** |
| `code_reward.num_samples` | 1600 | 1600 | 1600 | 1600 | 一致 |
| `case_reward.mean` | 0.627 | **0.147** | 0.652 | **0.172** | pairwise vs GT，**标度不同**；数值上 pair 远高于 cure 为预期现象 |
| `case_reward.std` | 0.325 | 0.476 | 0.310 | 0.502 | cure 更分散（且子集更小） |
| `case_reward.num_groups` | 100 | **12.7** | 100 | **8** | **CURE 远小于 pair** |
| `case_reward.num_samples` | 1600 | **670** | 1600 | **704** | **CURE 长期低于 1600** |

### 3.4 `reward.problems`

| 指标 | pair | cure |
|------|------|------|
| `total` / `used` / `skipped` | 100 / 100 / 0 | 100 / 100 / 0 |

### 3.5 `reward.gt_correlation`

| 指标 | pair（step 1–20） | cure（step 1–20） |
|------|-------------------|-------------------|
| `mean_spearman` | 均值约 **-0.455**（各步约 -0.27～-0.57） | **JSON 无此字段，无法对比** |
| `n` | 均值约 **71.7** | **无** |

### 3.6 其他 JSON 字段

- `recorded_at`：仅时间戳。
- `_field_help_zh_file`：均为 `metrics_field_help_zh.json`。
- `rollout_debug_meta`：两侧均为 `enabled: false`。

---

## 4. 综合阅读建议

1. **同模型下的执行指标**：`code_*` 在 20 步均值上二者接近；`case_*` 均值 cure 略好，但 pair 在 **step 20** 出现 **case_acc / case_accumulate_acc 明显塌陷**，cure 在同一步更平稳，宜对照 `fig1` 看全程形态而非单看均值。
2. **BoN**：cure 在 K=16 的 accumulate_acc 上 20 步均值略占优，pair 在 step 20 的 K=4/K=16 acc 部分点更高，互有胜负。
3. **Reward 标量**：`case_reward.mean` 差异主要来自 **reward_mode**，不是「cure 更差」；更需关注 **num_groups / num_samples**：CURE 日志里 reward 聚合规模小得多，std 更大属合理现象。
4. **gt_correlation**：仅 pair 有记录；若要对 CURE 做同类诊断，需在训练代码/日志中开启或写入同名字段后再比。

---

*数值由 `series_steps_1_20.csv` 汇总，四舍五入至约 3–4 位有效数字。若需与旧版（误用 Qwen3-8B CURE）对照，请以本报告路径为准。*
