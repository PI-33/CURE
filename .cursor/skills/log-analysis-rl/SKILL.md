---
name: log-analysis-rl
description: Analyze CURE RL training logs by automatically cleaning noise and extracting metrics, errors, and timelines. Use when analyzing training logs from /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/logs/, debugging failed runs, comparing training performance, or investigating errors.
---

# CURE RL Training Log Analysis

Analyze CURE training logs efficiently by removing noise and extracting structured data. This skill automates log cleaning via a pre-built Python script, then provides clean context for AI analysis.

## Quick Start

### Step 1: Clean the log file

Run the cleaning script on your log file:

```bash
python .cursor/skills/log-analysis-rl/scripts/clean_log.py /path/to/train_*.log
```

If the log directory is not writable, write elsewhere:

```bash
python .cursor/skills/log-analysis-rl/scripts/clean_log.py /path/to/train.log -o /path/to/train_cleaned.md
python .cursor/skills/log-analysis-rl/scripts/clean_log.py /path/to/train.log --output json -o /path/to/train_cleaned.json
```

The script removes vLLM noise, GPU messages, and progress bar clutter. Default output: `<log_stem>_cleaned.md` in the **current working directory** (not necessarily next to the log).

### Step 2: Read the cleaned log

View the cleaned markdown file to see structured sections: errors, timeline, and metrics.

### Step 3: Analyze the cleaned output

Share the cleaned markdown with me for analysis. The cleaner output provides:
- **Summary**: Line counts, error counts
- **Errors**: Extracted exceptions and tracebacks
- **Timeline**: Markdown **table of all** retained events (no standalone `timestamp_*` noise); optional `time` column when the source line contains a parseable time
- **Metrics**: Each series is a **full table** (`idx`, `value`, optional `step`, truncated log line) plus a one-line summary (count / min / max / latest)

## What the Cleaner Removes

The cleaning script filters out:
- vLLM INFO/WARNING log lines (loading model, CUDA operations)
- GPU initialization messages
- Repetitive progress bars (keeps start/end only)
- Process ID noise
- ANSI color codes
- PyTorch warnings

**Result**: ~70% of raw log becomes analyzable context.

## What the Cleaner Extracts

### Metrics Extracted

**Self-bootstrap & reward assignment** (one sample per sampling summary block; aligns with `optimization/diagnostics/step_*.json` when both exist):

| JSON key | Source log line |
|----------|-----------------|
| `estimated_code_reward_mean` | `estimated_code_reward: mean=...` |
| `estimated_case_reward_mean` | `estimated_case_reward: mean=...` |
| `estimated_code_reward_std` | same line, `std=...` |
| `estimated_case_reward_std` | same line, `std=...` |
| `gt_correlation_code_spearman` | `gt_correlation(code_reward): mean_spearman=...` (skipped if `no valid correlations`) |
| `self_bootstrap_used` / `self_bootstrap_skipped` | `self_bootstrap_stats: ... used=... skipped=...` |

**PPO / trainer** (per optimization step; entries include `step` in JSON when present):

| JSON key | Meaning |
|----------|---------|
| `ppo_avg_custom_reward_code` / `_case` | Batch mean after normalization — `make_experience` line `avg_custom_rewards=` |
| `ppo_policy_loss_*`, `ppo_kl_loss_*`, `ppo_clip_ratio_*`, `ppo_entropy_*` | `train:` line for `tag=code` / `tag=case` |

**BoN** (`acc` is on the line *after* `BoN setting (N, M):`):

- `bon4_acc`, `bon4_accumulate_acc`, `bon16_acc`, `bon16_accumulate_acc`

Note: `bon16_*` counts may exceed training steps if the log also contains **evaluation** BoN blocks (e.g. every 5 steps).

**Sampling accuracy & lengths** (as before):

- `code_acc`, `code_accumulate_acc`, `case_acc`, `case_accumulate_acc`
- `code_response_length`, `case_response_length`, `p_01`, `p_00`
- `clip_ratio`, `policy_entropy`, `critic_loss`, `avg_kl` — when logged in tensorboard-style lines

### Timeline Events
- Step transitions, sampling summaries (`code acc`, self-bootstrap, PPO `avg_custom_rewards`), training dataset switches, etc.
- **Excluded**: lines that matched only a bare datetime (removed to reduce clutter)
- Rendered as a **complete** markdown table (not truncated to the first N rows)

### Errors
- Tracebacks and exceptions (up to first 5)
- Full stack traces preserved
- Line context maintained

## Usage Workflow

### Debugging a Failed Run

```bash
# 1. Clean the log
python .cursor/skills/log-analysis-rl/scripts/clean_log.py logs/train_20260227_190857.log

# 2. Share the cleaned version
# → Use the generated train_20260227_190857_cleaned.md

# 3. Ask for analysis
# "Analyze this cleaned log - why did training fail?"
```

### Comparing Training Runs

```bash
# Clean multiple runs
python .cursor/skills/log-analysis-rl/scripts/clean_log.py logs/run1.log
python .cursor/skills/log-analysis-rl/scripts/clean_log.py logs/run2.log

# Share both cleaned files for comparison
```

### Extracting Metrics for Plotting

```bash
# Use JSON output for programmatic access
python .cursor/skills/log-analysis-rl/scripts/clean_log.py logs/train.log --output json

# → Generates train_cleaned.json with structured metrics
```

## JSON Output Format

Use `--output json` to get data in structured format:

```bash
python .cursor/skills/log-analysis-rl/scripts/clean_log.py train.log --output json
```

Output file `train_cleaned.json` contains:
```json
{
  "summary": {
    "total_lines": 8234,
    "errors_found": 2,
    "metrics_tracked": {"code_acc": 5, "case_acc": 5},
    "timeline_events": 23
  },
  "errors": ["Traceback..."],
  "timeline": [{"event": "step_0_sampling", "line": "...", "timestamp": "02-27 19:08:57"}],
  "metrics": {
    "code_acc": [{"value": 0.8, "line": "code acc: 0.8"}],
    "estimated_code_reward_mean": [{"value": 0.85, "line": "estimated_code_reward: mean=0.8542, ..."}],
    "ppo_avg_custom_reward_code": [{"value": 0.013, "line": "...", "step": 0}]
  }
}
```

## Key Metrics to Monitor

When analyzing cleaned logs, focus on:

| Metric | Healthy Range | Warning Signs |
|--------|---------------|----------------|
| `code_acc` | 0.6-0.9+ | Sudden drops, stuck at 0.5 |
| `estimated_code_reward_mean` | stable 0.7–0.95 typical | Collapse toward 0.5 with `used=1` → almost no valid groups |
| `gt_correlation_code_spearman` | positive preferred | Strongly negative or missing → misaligned self-bootstrap signal |
| `ppo_avg_custom_reward_*` | small magnitude (normalized) | Drifting far negative on one head only → check case/code balance |
| `avg_kl` | 0.1-1.0 | >5.0 = policy drift, <0.01 = frozen |
| `ppo_entropy_*` / `policy_entropy` | >0.5 | <0.1 = mode collapse |
| `ppo_clip_ratio_*` / `clip_ratio` | <0.2 | >0.3 = learning rate too high |
| `critic_loss` | Trending down | Stuck high = learning failure |

## Troubleshooting

### Script requires dependencies
```bash
# Ensure Python 3.7+ and no special dependencies needed
# Script uses only stdlib: re, sys, pathlib, datetime, collections, json
```

### Cleaned file is too large
Markdown lists **all** timeline rows and **every** sample per metric (tables can be long). Errors are still capped at 5 in the MD report. For filtering or plotting, prefer `--output json` and slice in Python.

### Metrics not found
Ensure log file contains the step summaries. Incomplete logs may not have all metrics recorded.

Self-bootstrap keys appear only when `reward_mode: self_bootstrap` (or equivalent) is used. PPO keys require `train_utils.rl.trainer` log lines with `[step=N tag=code|case]`.

## Advanced: Custom Analysis

After cleaning, the structured JSON can be used for custom analysis:

```python
import json

data = json.load(open('train_cleaned.json'))

# Extract metric trend
code_accs = [m['value'] for m in data['metrics']['code_acc']]
print(f"Code accuracy trend: {code_accs}")
ec = data['metrics'].get('estimated_code_reward_mean', [])
if ec:
    print(f"Self-bootstrap code reward mean: {[m['value'] for m in ec]}")
ppo = data['metrics'].get('ppo_avg_custom_reward_code', [])
if ppo:
    ppo.sort(key=lambda x: x.get('step', 0))
    print(f"PPO custom reward (code) by step: {[(x.get('step'), x['value']) for x in ppo]}")

# Find first error
if data['errors']:
    print(f"First error: {data['errors'][0][:200]}")
```

## References

- Script location: `.cursor/skills/log-analysis-rl/scripts/clean_log.py`
- For detailed log structure, see [LOG_FORMAT.md](LOG_FORMAT.md)
- For analysis patterns, see [ANALYSIS_PATTERNS.md](ANALYSIS_PATTERNS.md)
