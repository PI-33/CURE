# GT-Free Self-Bootstrapped Reward 实验前置准备

## 项目概述

本项目实现了 **执行一致性（Execution Consistency）** 驱动的自举奖励机制，用于CURE框架中的代码和测试协同进化。相比原GT-based方案，新方案无需依赖ground truth测试用例。

## 文件结构

```
vllm/CURE/
├── optimization/
│   ├── optimization_config.py          # ✨ 5个新参数控制Self-bootstrap
│   ├── reward.py                       # ✨ 完全重构，支持两种奖励计算
│   ├── sample.py, execute.py, train.py # 无变更
│   ├── run.py                          # 无变更
│   ├── ckpt/                           # 保存训练模型
│   └── temp_data/                      # 中间JSON文件
├── start.sh                            # 主启动脚本
├── run.py                              # 训练编排脚本
├── logs/                               # 训练日志输出
├── results/                            # 结果日志（metrics)
├── DESIGN_SELF_BOOTSTRAPPED_GRPO.md   # 算法设计文档
├── DEBUG_EXPERIENCE.md                 # 调试经验
└── EXPERIMENT_SETUP.md                 # 本文件
```

## 核心新参数

在 `optimization/optimization_config.py` 中：

```python
# ============= config for self-bootstrapped reward =================
use_self_bootstrap = True              # 启用新方案（False = GT-based baseline）
em_iterations = 3                      # EM算法迭代次数
use_anchor = True                      # 用公开例子作为弱监督
anchor_strength = 0.3                  # 锚的影响权重 [0, 1]
min_weight_threshold = 0.1             # 测试权重低于此阈值则跳过该问题
```

## 实验工作流

### 阶段1：快速验证（无GPU）

验证新的奖励计算逻辑是否正确运行，预期时间：**5分钟**

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/optimization

# 用现有 outputs-rl-*.json 文件验证
python reward.py \
  --pretrained_model "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-7B-Instruct" \
  --dataset "CodeContests_train" \
  --use_self_bootstrap true
```

**预期输出**：
- `temp_data/rl_code_data.json` 和 `rl_case_data.json` 生成成功
- `results/results-*.txt` 中有如下日志：
  ```
  reward_mode: self_bootstrap
  estimated_code_reward: mean=..., std=..., num_groups=..., num_samples=...
  self_bootstrap_stats: total_problems=..., skipped=..., used=...
  gt_correlation(code_reward): mean_spearman=..., n=...
  ```

**成功标准**：
- ✓ 无Python错误
- ✓ correlation值在 [-1, 1] 范围内
- ✓ used problems > 0（至少有一些问题未被跳过）

---

### 阶段2：小规模对比实验（需要GPU）

用小数据集对比 **GT-based（Baseline）** 和 **Self-bootstrap（新方案）** 的训练效果，预期时间：**2-3小时/方案**

#### 2.1 准备配置文件

使用提供的配置模板，创建两个配置版本：

**版本A：GT-based Baseline**
```bash
cp optimization/optimization_config.py optimization/config_baseline.py
# 编辑 config_baseline.py
# 修改以下参数：
n_sample_per_step = 10
total_steps = 4
eval_interval = 2
use_self_bootstrap = False
```

**版本B：Self-bootstrap 新方案**
```bash
cp optimization/optimization_config.py optimization/config_selfbootstrap.py
# 编辑 config_selfbootstrap.py
# 修改以下参数：
n_sample_per_step = 10
total_steps = 4
eval_interval = 2
use_self_bootstrap = True
em_iterations = 3
use_anchor = True
anchor_strength = 0.3
min_weight_threshold = 0.1
```

#### 2.2 运行基线实验（GT-based）

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE

# 备份原配置
cp optimization/optimization_config.py optimization/optimization_config.py.backup

# 使用基线配置
cp optimization/config_baseline.py optimization/optimization_config.py

# 设置日志文件名以区分实验
export EXP_TAG="baseline_gt_based"
LOG_FILE="logs/train_${EXP_TAG}_$(date +%Y%m%d_%H%M%S).log"
export CURE_LOG_FILE="$LOG_FILE"

# 运行实验
bash start.sh 2>&1 | tee "$LOG_FILE"
```

#### 2.3 运行新方案实验（Self-bootstrap）

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE

# 清空中间文件，确保新的运行不会混淆
rm -rf optimization/temp_data/outputs-rl-*.json
rm -rf optimization/ckpt/*

# 使用新方案配置
cp optimization/config_selfbootstrap.py optimization/optimization_config.py

# 设置日志文件名
export EXP_TAG="selfbootstrap_new"
LOG_FILE="logs/train_${EXP_TAG}_$(date +%Y%m%d_%H%M%S).log"
export CURE_LOG_FILE="$LOG_FILE"

# 运行实验
bash start.sh 2>&1 | tee "$LOG_FILE"
```

#### 2.4 提取结果

运行完成后，使用分析脚本自动提取关键指标：

```bash
cd /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE

# 生成对比报告
python scripts/extract_metrics.py \
  --baseline_log "logs/train_baseline_gt_based_*.log" \
  --selfbootstrap_log "logs/train_selfbootstrap_new_*.log" \
  --output "experiment_results.json"

# 生成可读的对比表格
python scripts/analyze_results.py --input "experiment_results.json"
```

---

## 关键指标解读

### 训练日志中的指标

#### 奖励阶段（reward.py 输出）

```
reward_mode: self_bootstrap                          # 使用的奖励模式
estimated_code_reward: mean=0.0234, std=0.8932, ... # 代码奖励的统计
estimated_case_reward: mean=-0.0156, std=0.9145, .. # 测试奖励的统计
self_bootstrap_stats: total_problems=10, skipped=2, used=8    # 问题统计
gt_correlation(code_reward): mean_spearman=0.456, n=8        # 与GT的相关性
```

**解读**：
- `mean ≈ 0, std ≈ 1`：奖励已正规化（正常）
- `gt_correlation > 0.3`：新旧奖励方向一致（好信号）
- `skipped / total < 20%`：大多数问题未被跳过（好信号）

#### 训练阶段（train.py 输出）

在 `logs/train_*.log` 中查找：
```
policy loss: ...
kl loss: ...
reward mean: ...
```

关键观察：
- Policy loss 应该逐步下降
- 前几步可能波动较大（正常）
- 最后2-3步应趋于稳定

### 评估指标

在 `results/results-*.txt` 中：
```
pass@1: X.XX%    # 单次采样成功率
pass@5: X.XX%    # 5次采样中至少有一次成功
BoN (16,16): X.XX%  # 最优的采样结果
```

**对比方法**：
1. 在相同的steps内，比较最终的 pass@1 和 pass@5
2. 新方案理想情况：与baseline相近或略优
3. 如果新方案明显劣化 → 需要调参或重新检查EM算法

---

## 实验命令速查表

| 任务 | 命令 |
|---|---|
| 快速验证 | `cd optimization && python reward.py --use_self_bootstrap true` |
| 运行Baseline | `cp config_baseline.py optimization_config.py && bash start.sh` |
| 运行新方案 | `cp config_selfbootstrap.py optimization_config.py && bash start.sh` |
| 查看最新日志 | `tail -f logs/train_*.log` |
| 提取指标 | `python scripts/extract_metrics.py --baseline_log ... --selfbootstrap_log ...` |
| 清空中间文件 | `rm -rf optimization/temp_data/*.json optimization/ckpt/*` |

---

## 故障排查

### 问题1：reward.py 运行失败

**错误**：`KeyError: 'num_ground_truth_test'`

**原因**：没有输入数据 `temp_data/outputs-rl-*.json`

**解决**：
```bash
# 先运行 sample.py 和 execute.py 生成中间文件
python optimization/sample.py
python optimization/execute.py
```

### 问题2：训练过程中CUDA错误

**错误**：`RuntimeError: CUDA out of memory` 或 `Error 304`

**原因**：GPU显存不足或CUDA上下文污染

**解决**：
```bash
# 减少batch大小或样本数
n_sample_per_step = 5
# 清空GPU缓存
rm -rf optimization/ckpt/*
```

### 问题3：新方案性能明显下降

**可能原因**：
1. EM算法参数不合适
2. 锚机制过强或过弱
3. 跳过阈值设置不当

**调试步骤**：
1. 降低 `anchor_strength` 到 0.1，重新运行
2. 降低 `min_weight_threshold` 到 0.05，确保更多问题被使用
3. 增加 `em_iterations` 到 5，让EM收敛更好

---

## 实验报告模板

运行完实验后，请按以下格式提供结果：

```markdown
# 实验结果报告

## 实验配置
- 数据集：CodeContests_train
- 样本数/步：10
- 总步数：4
- GPU数量：8（2节点）

## Baseline（GT-based）结果
- 最终 pass@1: X.XX%
- 最终 pass@5: X.XX%
- 平均每步耗时：Y 分钟
- 训练loss趋势：[可附图]

## Self-bootstrap（新方案）结果
- 最终 pass@1: X.XX%
- 最终 pass@5: X.XX%
- 平均每步耗时：Y 分钟
- GT correlation：Z.ZZ
- 跳过率：W%
- 训练loss趋势：[可附图]

## 对比分析
- 性能差异：...
- 关键观察：...
- 建议改进方向：...
```

---

## 下一步

实验完成后，我将根据你提供的详细结果：
1. 分析奖励函数是否有效工作
2. 诊断性能差异的根本原因
3. 提出具体的改进方向（调参、算法改进等）
4. 为进一步的大规模实验提供指导

