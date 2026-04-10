# GT-Free Self-Bootstrapped Reward 实验完整指南

## 👋 欢迎

本文档为你提供完整的实验流程和所有必要的工具。你只需要按步骤运行，然后提供结果，我会进行详细的分析和改进。

## 📋 快速导航

| 阶段 | 任务 | 预期耗时 | 是否需GPU |
|---|---|---|---|
| 1️⃣ 准备 | 环境检查和快速验证 | 5分钟 | ❌ 否 |
| 2️⃣ 对比实验 | 运行Baseline和新方案 | 4-6小时 | ✅ 是 |
| 3️⃣ 提交结果 | 运行分析脚本 | 2分钟 | ❌ 否 |
| 4️⃣ 分析改进 | AI分析和建议 | - | - |

---

## 🚀 第1步：快速验证（5分钟）

### 目的
验证新的奖励计算逻辑是否能正确运行，无需GPU。

### 命令

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE
bash quick_verify.sh
```

### 预期输出

日志会输出：
```
========================================================================
快速验证：Self-bootstrap 奖励计算
========================================================================

运行 Self-bootstrap 奖励计算...
----------------------------------------------------------------------
✓ Self-bootstrap 奖励计算完成

📊 关键输出文件：
  - rl_code_data.json（代码奖励）
  - rl_case_data.json（测试奖励）
  - results/results-*.txt（诊断日志）

📈 验证结果：

--- 最近的结果日志 ---
reward_mode: self_bootstrap
estimated_code_reward: mean=0.0234, std=0.8932, num_groups=8, num_samples=128
self_bootstrap_stats: total_problems=10, skipped=2, used=8
gt_correlation(code_reward): mean_spearman=0.4567, n=8
```

### ✅ 成功标准

- ✓ 无Python错误
- ✓ `gt_correlation` 在 [-1, 1] 范围内
- ✓ `used > 0`（至少使用了一些问题）
- ✓ `correlation > 0.2`（可选，更强的信号更好）

### ❌ 如果出错

常见问题和解决方案：

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `KeyError: 'num_ground_truth_test'` | 缺少输入数据 | 先运行 `python optimization/sample.py` 和 `python optimization/execute.py` |
| `ImportError: scipy` | 缺少依赖包 | `pip install scipy` |
| `correlation = nan` | 样本不足 | 忽略，在完整实验中会有足够样本 |

---

## 🧪 第2步：对比实验（4-6小时）

### 目的
在相同条件下对比GT-based Baseline和Self-bootstrap新方案的训练效果。

### 前置准备

确保两个配置文件都已创建：
```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/optimization

# 检查配置文件
ls -la config_baseline_gtbased.py config_selfbootstrap_new.py
```

### 启动实验

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE
bash run_compare_experiment.sh
```

这个脚本会：
1. 运行 Baseline 实验（2-3小时）
2. 等待30秒（清理GPU资源）
3. 运行 Self-bootstrap 实验（2-3小时）
4. 自动提取并对比结果

### 📊 预期输出

运行完毕后，你会看到：

```
========================================================================
实验完成！正在提取结果...
========================================================================

【baseline_gtbased】
  奖励模式: gt_based
  代码奖励: mean=0.0123, std=0.9456, ...
  测试奖励: mean=-0.0098, std=0.9123, ...
  问题统计: 总数=40, 使用=40, 跳过=0 (跳过率=0.0%)

【selfbootstrap_new】
  奖励模式: self_bootstrap
  代码奖励: mean=0.0145, std=0.8923, ...
  测试奖励: mean=-0.0112, std=0.9234, ...
  GT相关性: 0.4523
  问题统计: 总数=40, 使用=38, 跳过=2 (跳过率=5.0%)

========================================================================
关键发现
========================================================================

1. 奖励分布对比
   Baseline 代码奖励 std: 0.9456
   Self-bootstrap 代码奖励 std: 0.8923
   → 方差比: 0.94x

2. GT相关性对比
   Self-bootstrap GT相关性: 0.4523
   → 评估: ✓ 正相关，有效

3. 问题覆盖率对比
   Baseline 使用问题数: 40
   Self-bootstrap 使用问题数: 38
   Self-bootstrap 使用率: 95.0%

📁 完整结果已保存到: experiment_results
```

### 🔍 查看详细日志

```bash
# 查看最后100行Baseline日志
tail -100 /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiment_results/baseline_gtbased/train_*.log

# 查看最后100行Self-bootstrap日志
tail -100 /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiment_results/selfbootstrap_new/train_*.log
```

### 📈 监控实验进度

在另一个终端实时监控：

```bash
# Baseline 实验
tail -f /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/logs/train_*.log

# Self-bootstrap 实验（30秒后）
tail -f /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiment_results/selfbootstrap_new/train_*.log
```

---

## 📊 第3步：提交结果

### 生成分析报告

实验完成后，运行分析脚本：

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE
python scripts/analyze_results.py experiment_results
```

这会自动：
1. 提取所有关键指标
2. 生成对比报告
3. 提供诊断建议

### 💾 提交内容

请将以下内容提供给我进行分析：

**1. 命令行输出（复制并粘贴）**
```
python scripts/analyze_results.py experiment_results
```
的完整输出

**2. 实验配置信息**
```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE
echo "=== Baseline 配置 ===" && grep -E "n_sample_per_step|total_steps|use_self_bootstrap" optimization/config_baseline_gtbased.py
echo "=== Self-bootstrap 配置 ===" && grep -E "n_sample_per_step|total_steps|use_self_bootstrap|em_iterations|anchor" optimization/config_selfbootstrap_new.py
```

**3. 详细日志文件路径**
```
experiment_results/baseline_gtbased/train_*.log
experiment_results/selfbootstrap_new/train_*.log
experiment_results/analysis_report.txt
```

### 📝 结果报告模板

```markdown
# 实验结果报告

## 实验配置
- **数据集**: CodeContests_train
- **样本数/步**: 10
- **总步数**: 4
- **GPU配置**: 8 GPUs (2节点)
- **运行日期**: [日期]

## Baseline（GT-based）结果
[从analyze_results.py输出中复制]

## Self-bootstrap（新方案）结果
[从analyze_results.py输出中复制]

## 关键观察
[自由描述任何异常或有趣的现象]

## 附加信息
- 是否有任何错误或警告？
- 训练过程是否平稳？
- 是否有性能下降或异常？
```

---

## 🔧 进阶：手动操作

### 仅运行Baseline

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE
cp optimization/config_baseline_gtbased.py optimization/optimization_config.py
bash start.sh
```

### 仅运行Self-bootstrap

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE
cp optimization/config_selfbootstrap_new.py optimization/optimization_config.py
bash start.sh
```

### 清空并重新开始

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE
rm -rf optimization/temp_data/outputs-rl-*.json
rm -rf optimization/ckpt/*
rm -rf logs/*
rm -rf experiment_results/*
```

### 修改小规模参数

编辑配置文件中的这些参数，快速测试：

```python
n_sample_per_step = 5      # 更小的样本
total_steps = 2            # 更少的步数
eval_interval = 1          # 每步评估
```

---

## 📌 关键指标解释

### GT Correlation（GT相关性）

- **含义**: Self-bootstrap生成的奖励与GT-based奖励的Spearman相关系数
- **范围**: [-1, 1]
- **理想值**: > 0.3（表示两种方法的奖励方向一致）
- **解读**:
  - `> 0.4`: 很好，新方法学到了有意义的信号
  - `0.2 - 0.4`: 中等，有改进空间
  - `< 0.2`: 较弱，可能需要调参
  - `< 0`: 反向相关，可能有问题

### 代码/测试奖励标准差

- **含义**: 生成的奖励分布的分散程度
- **理想值**: 0.8 - 1.2（标准化后）
- **解读**:
  - `std < 0.5`: 奖励信号太弱，难以区分好坏
  - `std ≈ 1.0`: 理想状态
  - `std > 1.5`: 奖励波动较大，可能不稳定

### 问题使用率

- **含义**: (total_problems - skipped_problems) / total_problems
- **理想值**: > 80%
- **低使用率原因**: 
  - 生成的测试太uniform（无差异）
  - `min_weight_threshold` 设置过高
  - 生成的测试质量不均

---

## ❓ 常见问题

### Q: 实验需要多长时间？
**A**: 
- 快速验证：5分钟
- 完整对比实验：4-6小时（Baseline 2-3小时 + Self-bootstrap 2-3小时）

### Q: 如果实验中途中断怎么办？
**A**: 
- 对比实验可以分别重新运行
- 配置会自动选择
- 中断后重新运行相同命令即可继续

### Q: 新方案的性能比Baseline差怎么办？
**A**: 
这很正常，是我们要诊断的。提交完整结果后，我会：
1. 分析性能差异的根本原因
2. 提出具体的参数调优建议
3. 可能改进EM算法逻辑
4. 重新运行大规模对比

### Q: 可以修改配置参数吗？
**A**: 
可以，但建议保留两个模板配置供对比。如果想尝试不同参数，创建新文件：
```bash
cp optimization/config_selfbootstrap_new.py optimization/config_selfbootstrap_custom.py
# 编辑 config_selfbootstrap_custom.py
# 手动运行测试
```

### Q: 如何判断实验是否成功？
**A**: 
至少满足以下条件：
- ✓ 无崩溃或Python错误
- ✓ 两个方案都完成了所有训练步骤
- ✓ 生成了日志和结果文件
- ✓ 能够成功运行分析脚本

性能优劣不是成功的必要条件。

---

## 📞 需要帮助？

如果遇到问题：

1. 查看 `DEBUG_EXPERIENCE.md` 中的常见错误解决方案
2. 检查日志文件中的错误信息
3. 提供完整的错误日志给我

---

## ✨ 你现在可以开始了！

```bash
# 步骤1：快速验证（5分钟）
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE && bash quick_verify.sh

# 步骤2：对比实验（4-6小时）
# (申请到机器卡后运行)
bash run_compare_experiment.sh

# 步骤3：提交结果
python scripts/analyze_results.py experiment_results
```

祝实验顺利！🚀