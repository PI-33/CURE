# 步进指标汇总（自动生成）
- **diagnostics 源目录**: `/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiments/04031200_Qwen2.5-7B-Instruct_2gpu_350step_cure/optimization/diagnostics`
- **实验目录**: `/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiments/04031200_Qwen2.5-7B-Instruct_2gpu_350step_cure`
- **总步数**: 350
- **输出目录（图表与 CSV）**: `/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/analysis/em_summary_04031200_Qwen2.5-7B-Instruct_2gpu_350step_cure`

## 文件说明
- `em_step_metrics_wide.csv`：每行一步，每列一指标（宽表，可用 Excel 打开）。
- `em_step_metrics_preview.md`：前 20 步表格预览。
- `em_trend__*.png`：各数值指标随 step 的折线图。

## 核心观察（自动统计）
- `execution_code_acc`：首步 0.213125，末步 0.174375，全程均值 0.208362，最小 0.13625，最大 0.319375
- `execution_code_accumulate_acc`：首步 0.299766，末步 0.266352，全程均值 0.296608，最小 0.208516，最大 0.405703
- `execution_bon_16x16_acc`：首步 0.25，末步 0.26，全程均值 0.278429，最小 0.16，最大 0.43
- `execution_bon_16x16_accumulate_acc`：首步 0.35625，末步 0.4，全程均值 0.398092，最小 0.29875，最大 0.525
- `sampling_mean_code_response_tokens`：首步 662.842，末步 541.074，全程均值 510.205，最小 368.693，最大 698.263
- `sampling_mean_case_response_tokens`：首步 802.451，末步 501.106，全程均值 581.766，最小 458.167，最大 858.303
- **`reward_problems_used`**：末步 100；等于 0 的步数 0 / 350；≥50 的最后一步 step = 349

### 说明
若 `reward_problems_used` 在后期长期接近 0，表示自举 reward 过滤后几乎没有题目进入 RL，训练信号会变弱，需结合 `min_weight_threshold`、采样规模等排查。
