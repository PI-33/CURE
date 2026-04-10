# GRPO/PPO 训练参数说明文档

本文档基于 OpenRLHF 风格配置，解释 CURE 项目中使用的训练参数含义。

---

## 1. 奖励模型配置 (Reward Model)

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `reward_num_nodes` | 1 | Reward 模型使用的节点数量 |
| `reward_num_gpus_per_node` | 2 | 每个节点上 Reward 模型的 GPU 数量 |
| `reward_pretrain` | example_path | Reward 模型的预训练权重路径 |
| `reward_clip_range` | (-10, 10) | Reward 值的裁剪范围，防止极端奖励值 |
| `use_compute_reward_fn` | True | 是否使用自定义奖励计算函数 |

---

## 2. Actor (策略模型) 配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `actor_num_nodes` | 2 | Actor 模型使用的节点数量 |
| `actor_num_gpus_per_node` | 1 | 每个节点上 Actor 模型的 GPU 数量 |
| `actor_learning_rate` | 1e-06 | Actor 模型的初始学习率 |
| `pretrain` | /path/to/model | Actor 预训练模型路径 |
| `target_modules` | all-linear | 需要训练的模块名称 |
| `flash_attn` | True | 是否使用 Flash Attention 加速 |
| `gradient_checkpointing` | True | 是否使用梯度检查点节省显存 |
| `gradient_checkpointing_use_reentrant` | False | 梯度检查点的 reentrant 模式 |
| `max_norm` | 1.0 | 梯度裁剪的最大范数 (gradient clipping) |
| `freezing_actor_steps` | -1 | 冻结 Actor 的步数，-1 表示不冻结 |

---

## 3. Critic (价值模型) 配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `critic_num_nodes` | 2 | Critic 模型使用的节点数量 |
| `critic_num_gpus_per_node` | 1 | 每个节点上 Critic 模型的 GPU 数量 |
| `critic_learning_rate` | 9e-06 | Critic 模型的初始学习率 |
| `critic_pretrain` | example_path | Critic 模型的预训练权重路径 |
| `critic_update_steps` | 4 | Critic 模型每次更新的步数 |
| `value_head_prefix` | value_head | 价值头 (value head) 的模块前缀 |
| `value_clip` | 0.2 | Value 值的 PPO 裁剪范围 |

---

## 4. 参考模型 (Reference Model) 配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `ref_num_nodes` | 2 | Reference 模型使用的节点数量 |
| `ref_num_gpus_per_node` | 1 | 每个节点上 Reference 模型的 GPU 数量 |
| `ref_reward_offload` | False | 是否将 Reference 模型卸载到 CPU |

---

## 5. vLLM 推理配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `vllm_num_engines` | 2 | vLLM 推理引擎数量 |
| `vllm_tensor_parallel_size` | 1 | 张量并行大小 (每个引擎的 GPU 数) |
| `vllm_sync_backend` | nccl | 多 GPU 同步后端 |
| `gpu_memory_utilization` | 0.85 | vLLM 的 GPU 显存使用比例 (0-1) |
| `max_num_batched_tokens` | 2048 | 最大批处理的 token 数量 |
| `enable_prefix_caching` | True | 是否启用前缀缓存 |
| `enable_chunked_prefill` | False | 是否启用分块预填充 |
| `enforce_eager` | False | 是否强制使用 eager 模式 |
| `disable_trace_cache` | False | 是否禁用 Triton trace 缓存 |
| `prompt_max_len` | 2000 | 输入提示的最大长度 (token) |
| `generate_max_len` | 8000 | 生成内容的最大长度 (token) |

---

## 6. PPO/GRPO 训练超参数

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `eps_clip` | 0.2 | PPO 策略更新的裁剪范围 (epsilon) |
| `gamma` | 1.0 | 折扣因子 (discount factor)，奖励衰减率 |
| `lambd` | 1.0 | GAE (Generalized Advantage Estimation) 的 lambda 参数 |
| `num_episodes` | 1 | 每个 prompt 采样的 episode 数量 |
| `max_epochs` | 1 | 每次更新的最大 epoch 数 |
| `policy_update_steps` | 1 | 策略模型每次更新的步数 |
| `normalize_reward` | True | 是否对奖励进行归一化 |
| `advantage_normalize` | False | 是否对 advantage 进行归一化 |
| `top_p` | 1.0 | Nucleus sampling 的 top-p 值 |
| `temperature` | 1.0 | 生成温度，控制随机性 |
| `n_samples_per_prompt` | 1 | 每个 prompt 采样的样本数量 |

---

## 7. KL 正则化配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `init_kl_coef` | 0 | KL 惩罚系数的初始值 |
| `kl_target` | None | 目标 KL 值 |
| `use_kl_estimator_k3` | True | 是否使用 K3 KL 估计器 |
| `use_abs_kl` | False | 是否使用绝对 KL |
| `use_kl_loss` | True | 是否使用 KL loss |
| `kl_loss_coef` | 0.01 | KL loss 的系数 |

---

## 8. 优化器配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `adam_betas` | (0.9, 0.95) | Adam 优化器的 beta 参数 |
| `l2` | 0.0 | L2 正则化系数 |
| `adam_offload` | False | 是否将 Adam 优化器状态卸载到 CPU |
| `zpg` | 1 | ZeRO++ 专属参数 |
| `num_warmup_steps` | 0 | 学习率预热步数 |

---

## 9. DeepSpeed ZeRO 配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `zero_stage` | 3 | ZeRO 优化阶段 (0=disabled, 1=optimizer, 2=optimizer+gradient, 3=optimizer+gradient+param) |
| `bf16` | True | 是否使用 bfloat16 混合精度 |

---

## 10. 数据配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `rl_data` | temp_data/rl_data.json | RL 训练数据路径 |
| `rl_code_data` | temp_data/rl_code_data.json | 代码生成任务数据路径 |
| `rl_case_data` | temp_data/rl_case_data.json | 测试用例生成任务数据路径 |
| `separate_training` | True | 是否将 code 和 case 分开训练 |
| `packing_max_len` | 20000 | 打包序列的最大长度 |
| `load_checkpoint` | False | 是否加载检查点 |
| `ckpt_path` | ckpt | 检查点保存路径 |
| `save_path` | ckpt | 模型保存路径 |

---

## 11. 训练流程配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `train_batch_size` | 256 | 训练批大小 (总) |
| `micro_train_batch_size` | 1 | 微批次训练大小 |
| `rollout_batch_size` | 256 | Rollout 批大小 |
| `micro_rollout_batch_size` | 32 | 微 rollout 批次大小 |
| `micro_forward_batch_size` | 1 | 微 forward 批次大小 |
| `eval_steps` | -1 | 评估间隔步数，-1 表示不评估 |
| `save_steps` | -1 | 保存检查点间隔，-1 表示不保存 |
| `save_interval` | 100 | 保存间隔 |
| `global_step` | 0 | 当前全局步数 |
| `max_len` | None | 序列最大长度限制 |

---

## 12. 其他配置

| 参数 | 值示例 | 说明 |
|------|--------|------|
| `seed` | 42 | 随机种子 |
| `local_rank` | -1 | 本地进程排名，-1 表示非分布式 |
| `colocate_critic_reward` | True | 是否将 Critic 和 Reward 放在同一设备 |
| `colocate_actor_ref` | True | 是否将 Actor 和 Reference 放在同一设备 |
| `colocate_all` | True | 是否将所有模型放在同一设备 |
| `total_num_nodes` | 2 | 总节点数 |
| `disable_fast_tokenizer` | False | 是否禁用快速 tokenizer |
| `optimized_model_name` | optimized | 优化后的模型名称 |
| `update_ref_every_epoch` | False | 是否每个 epoch 更新 Reference 模型 |
| `enable_eval` | False | 是否启用评估模式 |

---

## 13. 训练指标说明

### 奖励指标

| 指标 | 说明 |
|------|------|
| `estimated_code_reward` | 代码生成任务的估计奖励 |
| `estimated_case_reward` | 测试用例生成任务的估计奖励 |
| `mean` | 奖励的均值 |
| `std` | 奖励的标准差 |
| `num_groups` | 分组数量 (用于 Group GRPO) |
| `num_samples` | 样本总数 |

### 训练指标

| 指标 | 说明 |
|------|------|
| `policy_loss` | 策略模型的损失 |
| `kl_loss` | KL 正则化损失 |
| `clip_ratio` | 被裁剪的策略变化比例 |
| `entropy` | 策略的熵 (表示探索程度) |
| `lr` | 当前学习率 |
| `avg_custom_rewards` | 平均自定义奖励 |
| `avg_response_length` | 平均响应长度 |
| `avg_group_raw_reward` | 分组原始奖励均值 |
| `avg_group_reward_std` | 分组奖励标准差 |

### 其他指标

| 指标 | 说明 |
|------|------|
| `code_acc` | 代码生成准确率 |
| `case_acc` | 测试用例生成准确率 |
| `accumulate_acc` | 累积准确率 |
| `BoN (Best of N)` | 从 N 个样本中选择最优的准确率 |
| `gt_correlation` | 与 ground truth 的相关性 (Spearman 相关系数) |

---

## 14. reward_mode 说明

本项目支持多种奖励模式：

| 模式 | 说明 |
|------|------|
| `self_bootstrap` | 自举模式：使用模型自身生成的结果作为奖励信号 |
| `ground_truth` | 使用 ground truth 作为奖励 |
| `combined` | 结合多种奖励信号 |

---

## 15. 典型配置示例

### 基础 GRPO 配置

```yaml
# 策略更新
eps_clip: 0.2
gamma: 1.0
lambd: 1.0
policy_update_steps: 1

# 学习率
actor_learning_rate: 1e-6

# KL 正则化
init_kl_coef: 0.01
use_kl_loss: True
kl_loss_coef: 0.01

# 生成
temperature: 1.0
top_p: 1.0
n_samples_per_prompt: 1
```

### DeepSpeed ZeRO-3 配置

```yaml
zero_stage: 3
bf16: True
gradient_checkpointing: True
gpu_memory_utilization: 0.85
```
