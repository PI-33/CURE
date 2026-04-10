# EM 实验步进指标汇总（自动生成）
- **diagnostics 源目录**: `/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiments/04031200_Qwen2.5-7B-Instruct_2gpu_350step_em/optimization/diagnostics`
- **实验目录**: `/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiments/04031200_Qwen2.5-7B-Instruct_2gpu_350step_em`
- **总步数**: 350
- **输出目录（图表与 CSV）**: `/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/analysis/em_summary_04031200_Qwen2.5-7B-Instruct_2gpu_350step_em`

## 文件说明
- `em_step_metrics_wide.csv`：每行一步，每列一指标（宽表，可用 Excel 打开）。
- `em_step_metrics_preview.md`：前 20 步表格预览。
- `em_trend__*.png`：各数值指标随 step 的折线图。

## 核心观察（自动统计）
- `execution_code_acc`：首步 0.140625，末步 0.21875，全程均值 0.21813，最小 0.133125，最大 0.33875
- `execution_code_accumulate_acc`：首步 0.230937，末步 0.302969，全程均值 0.302586，最小 0.1925，最大 0.456062
- `execution_bon_16x16_acc`：首步 0.19，末步 0.23，全程均值 0.221571，最小 0.13，最大 0.35
- `execution_bon_16x16_accumulate_acc`：首步 0.31875，末步 0.34125，全程均值 0.307978，最小 0.18125，最大 0.454887
- `sampling_mean_code_response_tokens`：首步 677.821，末步 641.837，全程均值 641.731，最小 588.356，最大 713.409
- `sampling_mean_case_response_tokens`：首步 769.803，末步 658.755，全程均值 673.101，最小 548.599，最大 933.251
- **`reward_problems_used`**：末步 0；等于 0 的步数 220 / 350；≥50 的最后一步 step = 15
- **`reward_gt_correlation_mean_spearman`**：非 NaN 步数 106；均值 0.03145

### 说明
若 `reward_problems_used` 在后期长期接近 0，表示自举 reward 过滤后几乎没有题目进入 RL，训练信号会变弱，需结合 `min_weight_threshold`、采样规模等排查。
