# CURE 项目调试经验总结

## 概述

本文档记录在复现和调试 CURE（Co-Evolving LLM Coder and Unit Tester）项目过程中遇到的主要问题及解决方案，供后续开发者参考。

---

## 一、日志分析能力建设

### 如何读懂训练日志

#### 1. 识别日志的整体结构

日志通常遵循这个流程：
```
Sampling (vLLM推理) → Execution (执行代码) → Reward (计算奖励) → Train (PPO训练)
```

在 `run.py` 的最后会打印步骤索引，清晰显示完成了多少个 step：
```
This is the 0-th step for sampling.
This is the 0-th step for execution.
This is the 0-th step for training.
This is the 1-th step for sampling.
...
```

#### 2. 清理日志中的冗余信息

原始日志通常包含大量噪音（进度条、vLLM worker 日志、DeepSpeed 配置转储等）。使用脚本去重能将 4900+ 行压缩到 500 行以下：

```python
import re

# 跳过规则示例
if re.search(r'Processed prompts:\s+\d+%', s) and '100%' not in s:
    continue  # 只保留 100% 的进度条
if 'PolicyRayActorBase' in s and 'DeepSpeedEngine configuration' not in s:
    continue  # 过滤 Ray actor 的冗余日志
```

#### 3. 采样阶段指标解读

```
Processed prompts: 100%|██████████| 160/160 [03:42<00:00, 1.39s/it, est. speed input: 559.85 toks/s, output: 4809.36 toks/s]
code response length: 8400.7125
code acc: 0.35, code accumulate acc: 0.3546875
case acc: 0.25, case accumulate acc: 0.3482142857142857
BoN setting (4, 4): acc: 0.45, accumulate acc: 0.46875
```

**关键指标**：
- `code acc`：当前生成的代码在测试用例上的通过率
- `case acc`：生成的测试用例的质量（能否正确判断代码对错）
- `BoN acc`：Best-of-N 策略后的准确率（从 N 份代码中选最好的）
- `response length`：平均响应长度（下降通常表示模型学到了更简洁的表达）

#### 4. 训练阶段指标解读

```
experiences size: 11
policy_loss=0.0198, kl_loss=0.0000, clip_ratio=0.0000, entropy=0.3182, lr=1.00e-06
```

**关键指标**：
- `experiences size`：有效训练样本数（经过 reward normalization 过滤后）
  - 太小（< 5）说明数据不足或 reward 方差为零
  - 为零说明所有样本要么全对要么全错，无法产生对比学习信号
- `policy_loss`：策略网络的 PPO 损失
- `kl_loss`：KL 散度约束项
- `clip_ratio`：PPO 剪辑的比例（0 表示没有样本被剪辑，可能说明策略变化不大）
- `entropy`：策略的熵（太低表示过于确定性，太高表示过于随机）

---

## 二、常见错误及修复

### 错误 1：`KeyError: 'test_input'` 在评估阶段

**症状**：
```
KeyError: 'test_input'
File "eval.py", line 719, in execute_scripts
    if len(data[i]["test_input"]) * len(data[i]["generated_code"]) == 0:
```

**原因**：评估数据集（如 MBPP.json）的字段格式不兼容，缺少 `test_input`/`test_output` 字段。

**解决方案**：
1. 检查数据集格式，确保有 `test_input` 和 `test_output` 字段
2. 或者从已验证的数据源创建小规模测试集：
   ```python
   # 从 CodeContests_train.json 提取前 10 条作为 small_eval.json
   import json
   with open('data/CodeContests_train.json') as f:
       data = json.load(f)
   with open('data/small_eval.json', 'w') as f:
       json.dump(data[6:16], f, indent=2)  # 6-15 条数据
   ```
3. 在 `optimization_config.py` 中指定正确的数据集

---

### 错误 2：`PermissionError: [Errno 13] Permission denied` 写文件

**症状**：
```
PermissionError: [Errno 13] Permission denied: './optimization/results/results-*.txt'
```

**原因**：`optimization/results/` 目录由 root 所有，当前用户无写权限。

**解决方案**：修改 `execute.py` 和 `reward.py`，改用 `./test_results/` 替代：

```python
output_dir = "./test_results"
os.makedirs(output_dir, exist_ok=True)
results_path = os.path.join(output_dir, "results-" + outputs_name + ".txt")

try:
    with open(results_path, "a") as f:
        f.write(...)
except PermissionError:
    logger.warning(f"Cannot write to {results_path}, skipping")
```

---

### 错误 3：`ImportError: cannot import name '_shutdown_backend'` 

**症状**：
```
ImportError: cannot import name '_shutdown_backend' from 'torch.distributed.distributed_c10d'
```

**原因**：PyTorch 版本不兼容。旧代码用的 `_shutdown_backend(pg)` 在 PyTorch 2.9+ 已改为 `pg.shutdown()`。

**解决方案**：在 `optimization/train_utils/exp_engine/parallels/orz_distributed_c10d.py` 中添加兼容性垫片：

```python
def _shutdown_backend(pg):
    """Compat shim: PyTorch >=2.9 replaced _shutdown_backend(pg) with pg.shutdown()."""
    if hasattr(pg, "shutdown"):
        pg.shutdown()
    elif hasattr(pg, "_shutdown"):
        pg._shutdown()
```

---

### 错误 4：`RuntimeError: Unexpected error from cudaGetDeviceCount(). Error 304`

**症状**：第一次训练成功，但第二个 step 的采样阶段崩溃：
```
ERROR 02-28 00:37:42 [multiproc_executor.py:435] RuntimeError: Unexpected error from cudaGetDeviceCount(). 
Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? 
Error 304: OS call failed or operation not supported on this OS
```

**根本原因**：
- Step 0 的训练使用 DeepSpeed + Ray，在 GPU 上初始化了 NCCL 通信组和 CUDA 上下文
- 训练完成后，这些 CUDA 资源**没有被完全清理**
- Step 1 的采样通过 `multiprocessing.Process` fork 出新的 vLLM workers
- 子进程**继承了被污染的 CUDA 上下文**，导致 `cudaGetDeviceCount()` 失败
- vLLM 的 `multiproc_executor` 再次 fork 时，CUDA 已经处于错误状态

**为什么 Step 0 的采样没问题**：因为那时还没运行过 DeepSpeed 训练，CUDA 上下文是干净的。

**解决方案**（选一个）：

1. **最优方案**：让 `sample.py` 在独立子进程运行（已在 `run.py` 中通过 `subprocess.run()` 实现）
   - 但需确保 `run.py` 主进程不初始化任何 CUDA
   - 在 `run.py` 开头添加：
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **备选方案**：在 `train.py` 训练完成后显式清理 CUDA：
   ```python
   import torch
   torch.cuda.empty_cache()
   torch.cuda.reset_peak_memory_stats()
   # 以及 NCCL 进程组的清理
   ```

3. **临时绕过**：在 Step 1 前重启 Python 进程（但治标不治本）

---

## 三、数据管理最佳实践

### 小规模测试数据集的构造

对于快速迭代开发，需要小规模但完整的数据集。以下是推荐的方式：

```python
import json

# 从 CodeContests_train.json 提取作为训练集
with open('data/CodeContests_train.json') as f:
    full_data = json.load(f)

# 创建 small_train.json（前 20 条）
with open('data/small_train.json', 'w') as f:
    json.dump(full_data[:20], f, indent=2)

# 创建 small_eval.json（任务 6-15）
with open('data/small_eval.json', 'w') as f:
    json.dump(full_data[6:16], f, indent=2)
```

**为什么需要这样的数据集**：
- 完整训练集太大，一个 step 需要数小时
- 小规模数据集让开发者能在 30 分钟内完成一个完整 step
- 验证 pipeline 流程和 bug 修复

### 必需字段检查

所有训练/评估数据必须包含：
- `problem`：问题描述
- `language`：编程语言（通常是 "python"）
- `test_input`：真实的测试输入（列表）
- `test_output`：真实的测试输出（列表）
- `test_input_example`：公开示例（用于 prompt）
- `test_output_example`：公开示例对应的输出

如果数据集缺少这些字段，脚本会在 execution 或 evaluation 阶段崩溃。

---

## 四、配置文件管理

### 主配置 vs. 备选配置

`optimization/optimization_config.py` 是主训练配置。对于不同实验，在 `optimization/configs/` 下保存变体：

```
optimization/configs/
├── optimization_config_debug.py          # 快速测试（3 steps，小数据）
├── optimization_config_对标论文.py        # 论文复现配置
```

在 `run.py` 中通过修改导入来切换：
```python
# 默认：from optimization import optimization_config
# 或者：from optimization.configs import optimization_config_debug as optimization_config
```

### 关键配置参数说明

| 参数 | 含义 | 对快速测试的推荐值 | 对完整训练的推荐值 |
|------|------|-------------------|-------------------|
| `total_steps` | 训练总步数 | 3 | 150 |
| `n_sample_per_step` | 每步采样的任务数 | 5 | 20 |
| `k_code` / `k_case` | 每题生成的代码/测试数 | 4 | 16 |
| `num_chunks` | 执行时的并行分片数 | 8 | 32 |
| `gpu_groups` | vLLM 引擎配置 | `[[0,1]]` | `[[0,1],[2,3],[4,5],[6,7]]` |
| `total_num_nodes` | 训练用 GPU 数 | 2 | 2 |

---

## 五、代码库结构和修改要点

### 重要的改进

1. **experiment 目录管理**（`run.py` 第 35-62 行）
   - 自动创建 `experiments/{YYYYMMDD}_{HHMM}_{model}_{dataset}/` 目录
   - 保存配置快照、训练日志、结果文件
   - 便于追踪不同实验

2. **TensorBoard 分离**（`train.py` 第 71-72 行）
   - 改用 `tb_logs/` 替代 `ckpt/`，避免 TB 文件混入模型 shard
   - 各 step 的 TB 日志分开存储

3. **Per-step 指标记录**（`train_utils/rl/trainer.py` 第 240-248 行）
   - 记录 policy_loss、kl_loss、clip_ratio、entropy、lr
   - 便于分析训练动态

4. **Reward 统计输出**（`reward.py` 第 200-219 行）
   - 输出 estimated_code_reward / estimated_case_reward 的均值和方差
   - 帮助诊断数据质量问题

---

## 六、性能指标预期

### 小规模测试（small_train.json，20 条数据，3 steps）

| Step | 时间 | code acc | case acc | BoN acc | experiences |
|------|------|----------|----------|---------|-------------|
| 0 | ~5 min | 0.35 | 0.25 | 0.45 | 11, 9 |
| 1 | ~5 min | 0.21 | 0.35 | 0.25 | 4, 6 |
| 2 | ~5 min | 0.51 | 0.41 | 0.55 | 2, 6 |

**观察**：
- Step 1 通常会出现性能下降（policy oscillation），这是 RL 训练的正常现象
- Step 2 应该恢复并超越基线
- 如果 experiences 持续很小（< 3），说明数据集太小或质量不好

### 完整训练（CodeContests_train，150 steps）

- 每个 step：~10 分钟（4×TP2 vLLM + 2 GPU DeepSpeed）
- 完整训练：~25 小时
- 评估间隔：每 25 步进行一次（Figure 2 曲线需要）

---

## 七、下一步改进建议

1. **立即可做**：
   - 实现 CUDA 上下文的显式清理（解决 Error 304）
   - 添加更详细的日志时间戳
   - 创建数据验证脚本（检查必需字段）

2. **中期改进**：
   - 支持断点续训（保存/加载优化器状态）
   - 实现多机训练（目前只支持单机多 GPU）
   - 添加在线评估（不必等到 eval_interval）

3. **长期方向**：
   - 与 Weights & Biases 集成以可视化训练曲线
   - 支持不同的 reward 函数（当前只有规则学习）
   - 实现分布式 RL（Ray 的多节点支持）

---

## 附录：常用命令速查表

```bash
# 查看日志
tail -f logs/train_*.log

# 清理日志中的噪音
python3 << 'EOF'
import re
with open('logs/train_XXX.log') as f:
    lines = f.readlines()
# ... 应用上述过滤规则
EOF

# 检查 GPU 内存
nvidia-smi

# 查看 TensorBoard
tensorboard --logdir=experiments/*/optimization/tb_logs

# 重新设置 git remote（用 token push）
git remote set-url origin https://USER:TOKEN@github.com/PI-33/CURE.git
git push origin main

# 恢复到 https URL（避免明文存储 token）
git remote set-url origin https://github.com/PI-33/CURE.git
```

---

**最后更新**：2026-02-28  
**贡献者**：CURE 项目调试工程师
