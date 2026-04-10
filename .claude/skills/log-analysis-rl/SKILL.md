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

The script removes vLLM noise, GPU messages, and progress bar clutter. Output: `train_*_cleaned.md`

### Step 2: Read the cleaned log

View the cleaned markdown file to see structured sections: errors, timeline, and metrics.

### Step 3: Analyze the cleaned output

Share the cleaned markdown with me for analysis. The cleaner output provides:
- **Summary**: Line counts, error counts
- **Errors**: Extracted exceptions and tracebacks
- **Timeline**: Step-by-step execution flow
- **Metrics**: Code accuracy, case accuracy, BoN scores, loss values

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
- `code_acc`, `code_accumulate_acc` — code generation accuracy
- `case_acc`, `case_accumulate_acc` — test case generation accuracy
- `bon_acc` — Best-of-N accuracy at different scales
- `code_response_length`, `case_response_length` — generation lengths
- `clip_ratio`, `policy_entropy`, `critic_loss`, `avg_kl` — training diagnostics

### Timeline Events
- Step transitions (sampling → execution → reward → training)
- Timestamps for each event
- Critical transitions like dataset changes

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
    "code_acc": [{"value": 0.8, "line": "code acc: 0.8"}]
  }
}
```

## Key Metrics to Monitor

When analyzing cleaned logs, focus on:

| Metric | Healthy Range | Warning Signs |
|--------|---------------|----------------|
| `code_acc` | 0.6-0.9+ | Sudden drops, stuck at 0.5 |
| `avg_kl` | 0.1-1.0 | >5.0 = policy drift, <0.01 = frozen |
| `policy_entropy` | >0.5 | <0.1 = mode collapse |
| `clip_ratio` | <0.2 | >0.3 = learning rate too high |
| `critic_loss` | Trending down | Stuck high = learning failure |

## Troubleshooting

### Script requires dependencies
```bash
# Ensure Python 3.7+ and no special dependencies needed
# Script uses only stdlib: re, sys, pathlib, datetime, collections, json
```

### Cleaned file is too large
The markdown output is truncated to first 20 timeline events and 5 errors. For full data, use `--output json` and process programmatically.

### Metrics not found
Ensure log file contains the step summaries. Incomplete logs may not have all metrics recorded.

## Advanced: Custom Analysis

After cleaning, the structured JSON can be used for custom analysis:

```python
import json

data = json.load(open('train_cleaned.json'))

# Extract metric trend
code_accs = [m['value'] for m in data['metrics']['code_acc']]
print(f"Code accuracy trend: {code_accs}")

# Find first error
if data['errors']:
    print(f"First error: {data['errors'][0][:200]}")
```

## References

- Script location: `.cursor/skills/log-analysis-rl/scripts/clean_log.py`
- For detailed log structure, see [LOG_FORMAT.md](LOG_FORMAT.md)
- For analysis patterns, see [ANALYSIS_PATTERNS.md](ANALYSIS_PATTERNS.md)
