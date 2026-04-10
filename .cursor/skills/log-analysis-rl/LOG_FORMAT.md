# CURE Log Format Reference

## Log Source Locations

```
/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/logs/
├── train_YYYYMMDD_HHMMSS.log    # Main training logs
├── 0228/
│   ├── train_*.log              # Timestamped runs
│   └── train_*_cleaned.log      # Previously cleaned versions
└── fail/
    └── train_*.log              # Failed runs
```

## Log Sections and Markers

### Step Start
```
This is the 0-th step for sampling.
This is the 0-th step for execution.
This is the 0-th step for training.
```

### Sampling Phase (vLLM)
```
Loading model on GPUs [0, 1]...
INFO 02-24 00:53:51 [config.py:717] Automatically detected platform cuda.
INFO 02-24 00:53:51 [config.py:1770] Defaulting to use mp for distributed inference
Processed prompts: 100%|██████████| 40/40
generation job done!
code response length: 3918.8, case response length: 2978.45
```

### Execution Phase Results
```
code acc: 0.8, code accumulate acc: 0.975
case acc: 0.125, case accumulate acc: 0.125
p_01: 1
p_00: 0
BoN setting (4, 4):
acc: 0.8, accumulate acc: 0.975
BoN setting (16, 16):
acc: 0.75, accumulate acc: 0.85
```

### Self-bootstrap reward (after each sampling block, when enabled)
```
reward_mode: self_bootstrap
estimated_code_reward: mean=0.8542, std=0.3529, num_groups=6, num_samples=96
estimated_case_reward: mean=0.2858, std=0.6479, num_groups=6, num_samples=96
self_bootstrap_stats: total_problems=20, skipped=14, used=6
gt_correlation(code_reward): mean_spearman=0.6670, n=6
```

### PPO experience / update (`train_utils.rl.trainer`)
```
[step=0 tag=code] avg_custom_rewards=0.0130, avg_response_length=5381.3, avg_group_raw_reward=0.0130, avg_group_reward_std=1.0000
[step=0 tag=code] policy_loss=-0.0813, kl_loss=0.0000, clip_ratio=0.0000, entropy=0.4185, lr=1.00e-06
```

### Training Phase
```
Training with temp_data/rl_code_data.json
Calculate custom rewards, time cost: 0.00s
[INFO] | train_utils.rl.trainer:train:68 - Training with temp_data/rl_code_data.json
```

### Tensorboard Logging (if not cleaned up)
```
avg_raw_rewards: 0.5
avg_kl: 0.25
policy_entropy: 0.8
critic_loss: 0.123
```

## Noise Patterns the Cleaner Removes

### vLLM/PyTorch Logs
```
INFO 02-24 00:53:51 [config.py:717] This model supports multiple tasks...
WARNING 02-24 00:53:28 [multiproc_executor.py:786] WorkerProc was terminated
[1;36m(VllmWorker rank=0 pid=4716)[0;0m INFO 02-24 00:55:46 [custom_all_reduce.py:195]
```

### GPU Operations
```
Loading safetensors checkpoint shards: 100% Completed | 3/3
Graph capturing finished in 4 secs, took 0.70 GiB
Registering 7446 cuda graph addresses
numasched_setaffinity_v2_int() failed: Invalid argument
```

### Progress Bars (repetitive)
```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 6%|▌         Capturing CUDA graphs: 12%|█▏
Processed prompts: 2%|▎         | 1/40 [00:03<01:59,  3.06s/it]
```

### Process Messages
```
(PolicyRayActorBase pid=6443) Loading extension module fused_adam...
(Worker_TP0 pid=1465) /mnt/.../tvm_ffi/_optional_torch_c_dlpack.py:174: UserWarning
Process Process-1:
Traceback (most recent call last):
```

## Key Metrics in Logs

### Accuracy Metrics
- `code acc`: Accuracy of generated code on ground-truth tests
- `code accumulate acc`: Per-token accuracy across all code
- `case acc`: Accuracy of generated test cases
- `BoN acc`: Best-of-N accuracy (sampling multiple solutions)
- `p_01`: Probability that bad code passes good tests
- `p_00`: Probability that bad code passes bad tests

### Length Metrics
- `code response length`: Average tokens in generated code
- `case response length`: Average tokens in generated test case

### Training Diagnostics
- `avg_kl`: KL divergence between current and reference policy
- `policy_entropy`: Shannon entropy of policy (mode collapse indicator)
- `clip_ratio`: PPO clip frequency (learning stability)
- `critic_loss`: Value function prediction error

## Error Patterns

### Common Failures
```
IndexError: list index out of range
→ Empty training data (all rewards filtered out)

KeyError: 'test_input'
→ Dataset missing private test cases

ValueError: cannot reshape tensor
→ Tokenizer mismatch or empty responses

FileNotFoundError: outputs-rl-*.json
→ Sample or execute step failed
```

### Success Indicators
- `generation job done!` → Sampling completed
- `code acc: [0.6-0.9]` → Training learning
- `avg_kl: [0.1-1.0]` → Stable policy updates
- `No errors after step 5+` → Converging training

## Timeline Order

Healthy run timeline:
1. This is the 0-th step for sampling.
2. Processed prompts: 100%
3. generation job done!
4. [metrics printed]
5. This is the 0-th step for execution.
6. code acc: X, case acc: Y
7. This is the 0-th step for training.
8. Training with temp_data/rl_code_data.json
9. [training logs]
10. This is the 1-th step for sampling. (next iteration)

Failed run indicators:
- Sampling stops before "generation job done!"
- Crash right after "execution" starts
- Training phase never reached
- Error appears without step marker
