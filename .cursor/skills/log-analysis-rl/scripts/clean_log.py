#!/usr/bin/env python3
"""Clean CURE training logs by removing noise and extracting key information."""

import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class LogCleaner:
    """Clean and structure CURE training logs."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.lines = self.log_path.read_text().split('\n')
        self.metrics = defaultdict(list)
        self.errors = []
        self.timeline = []
        self.sections = defaultdict(list)

    def remove_noise(self):
        """Filter out vLLM loading logs, GPU messages, and repetitive progress bars."""
        noise_patterns = [
            r'INFO.*\[.*\.py:\d+\]',  # vLLM INFO lines
            r'WARNING.*\[.*\.py:\d+\]',  # vLLM warnings
            r'\[1;36m\(.*\[0;0m',  # ANSI colored process outputs
            r'Loading model on GPUs',
            r'Loading safetensors checkpoint',
            r'Capturing CUDA graphs',
            r'Registering.*cuda graph',
            r'Graph capturing finished',
            r'numasched_setaffinity',
            r'Process.*pid=',
            r'Time to load',
            r'Config mesh_device',
            r'FutureWarning.*pynvml',
        ]

        filtered = []
        for line in self.lines:
            if any(re.search(pattern, line) for pattern in noise_patterns):
                continue
            # Skip repetitive progress bar lines (keep only start/end)
            if 'Processed prompts:' in line or 'Capturing CUDA' in line:
                if '%' in line and not line.endswith('100%'):
                    continue
            filtered.append(line)

        self.lines = filtered
        return len(self.lines)

    def extract_timeline(self):
        """Extract step-by-step execution timeline with timestamps."""
        patterns = {
            r'This is the (\d+)-th step for (\w+)': lambda m: f"step_{m.group(1)}_{m.group(2)}",
            # Intentionally no bare-timestamp-only events (too noisy in markdown output).
            r'code acc: ([\d.]+)': lambda m: f"code_acc_{m.group(1)}",
            r'case acc: ([\d.]+)': lambda m: f"case_acc_{m.group(1)}",
            r'estimated_code_reward: mean=([\d.]+)': lambda m: f"est_code_reward_{m.group(1)}",
            r'estimated_case_reward: mean=([\d.eE\-+]+)': lambda m: f"est_case_reward_{m.group(1)}",
            r'gt_correlation\(code_reward\): mean_spearman=([\d.\-]+)': lambda m: f"gt_spearman_{m.group(1)}",
            r'\[step=(\d+) tag=(code|case)\] .*avg_custom_rewards=([\d.eE\-+]+)': lambda m: f"ppo_custom_r_step{m.group(1)}_{m.group(2)}_{m.group(3)}",
            r'Training with (.+?)$': lambda m: f"training_dataset_{m.group(1).strip()}",
            r'avg_raw_rewards': lambda m: 'training_started',
        }

        for line in self.lines:
            for pattern, format_fn in patterns.items():
                match = re.search(pattern, line)
                if match:
                    self.timeline.append({
                        'event': format_fn(match),
                        'line': line.strip(),
                        'timestamp': self._extract_timestamp(line)
                    })

    def _append_metric(self, name, value, line, **meta):
        """Append one metric sample; optional step/tag in meta."""
        entry = {'value': float(value), 'line': line.strip()}
        entry.update({k: v for k, v in meta.items() if v is not None})
        self.metrics[name].append(entry)

    def extract_metrics(self):
        """Extract performance metrics from logs."""
        metric_patterns = {
            'code_acc': r'code acc: ([\d.]+)',
            'code_accumulate_acc': r'code accumulate acc: ([\d.]+)',
            'case_acc': r'case acc: ([\d.]+)',
            'case_accumulate_acc': r'case accumulate acc: ([\d.]+)',
            'p_01': r'p_01: ([\d.]+)',
            'p_00': r'p_00: ([\d.]+)',
            'code_response_length': r'code response length: ([\d.]+)',
            'case_response_length': r'case response length: ([\d.]+)',
            'clip_ratio': r'ppo_clip_count.*?clip_ratio.*?([\d.]+)',
            'policy_entropy': r'policy_entropy.*?([\d.e\-]+)',
            'critic_loss': r'critic_loss.*?([\d.e\-]+)',
            'avg_kl': r'avg_kl.*?([\d.e\-]+)',
            # Self-bootstrap / reward assignment (one row per sampling round)
            'estimated_code_reward_mean': r'estimated_code_reward: mean=([\d.]+)',
            'estimated_case_reward_mean': r'estimated_case_reward: mean=([\d.eE\-+]+)',
            'estimated_code_reward_std': r'estimated_code_reward: mean=[\d.]+, std=([\d.]+)',
            'estimated_case_reward_std': r'estimated_case_reward: mean=[\d.eE\-+]+, std=([\d.eE\-+]+)',
            'gt_correlation_code_spearman': r'gt_correlation\(code_reward\): mean_spearman=([\d.\-]+)',
            'self_bootstrap_used': r'self_bootstrap_stats: total_problems=\d+, skipped=\d+, used=(\d+)',
            'self_bootstrap_skipped': r'self_bootstrap_stats: total_problems=\d+, skipped=(\d+), used=\d+',
        }

        ppo_make_code = re.compile(
            r'\[step=(\d+) tag=code\] .*?avg_custom_rewards=([\d.eE\-+]+)'
        )
        ppo_make_case = re.compile(
            r'\[step=(\d+) tag=case\] .*?avg_custom_rewards=([\d.eE\-+]+)'
        )
        ppo_train = re.compile(
            r'\[step=(\d+) tag=(code|case)\] policy_loss=([\d.eE\-+]+), kl_loss=([\d.eE\-+]+), '
            r'clip_ratio=([\d.eE\-+]+), entropy=([\d.eE\-+]+)'
        )

        for line in self.lines:
            for metric_name, pattern in metric_patterns.items():
                if metric_name == 'gt_correlation_code_spearman' and 'no valid' in line:
                    continue
                match = re.search(pattern, line)
                if match:
                    try:
                        self._append_metric(metric_name, match.group(1), line)
                    except (ValueError, IndexError):
                        pass

            mc = ppo_make_code.search(line)
            if mc:
                try:
                    self._append_metric(
                        'ppo_avg_custom_reward_code',
                        mc.group(2),
                        line,
                        step=int(mc.group(1)),
                    )
                except (ValueError, IndexError):
                    pass
            ms = ppo_make_case.search(line)
            if ms:
                try:
                    self._append_metric(
                        'ppo_avg_custom_reward_case',
                        ms.group(2),
                        line,
                        step=int(ms.group(1)),
                    )
                except (ValueError, IndexError):
                    pass
            pt = ppo_train.search(line)
            if pt:
                try:
                    st, tag = int(pt.group(1)), pt.group(2)
                    self._append_metric(
                        f'ppo_policy_loss_{tag}',
                        pt.group(3),
                        line,
                        step=st,
                    )
                    self._append_metric(
                        f'ppo_kl_loss_{tag}',
                        pt.group(4),
                        line,
                        step=st,
                    )
                    self._append_metric(
                        f'ppo_clip_ratio_{tag}',
                        pt.group(5),
                        line,
                        step=st,
                    )
                    self._append_metric(
                        f'ppo_entropy_{tag}',
                        pt.group(6),
                        line,
                        step=st,
                    )
                except (ValueError, IndexError):
                    pass

        self._extract_bon_lines()

    def _extract_bon_lines(self):
        """BoN metrics sit on the line after 'BoN setting (N, M):'."""
        acc_line = re.compile(
            r'acc:\s*([\d.]+)(?:,\s*accumulate acc:\s*([\d.]+))?'
        )
        for i, line in enumerate(self.lines):
            if i + 1 >= len(self.lines):
                break
            nxt = self.lines[i + 1]
            if 'BoN setting (4, 4)' in line:
                m = acc_line.search(nxt)
                if m:
                    try:
                        self._append_metric('bon4_acc', m.group(1), nxt)
                        if m.group(2):
                            self._append_metric('bon4_accumulate_acc', m.group(2), nxt)
                    except (ValueError, IndexError):
                        pass
            if 'BoN setting (16, 16)' in line:
                m = acc_line.search(nxt)
                if m:
                    try:
                        self._append_metric('bon16_acc', m.group(1), nxt)
                        if m.group(2):
                            self._append_metric('bon16_accumulate_acc', m.group(2), nxt)
                    except (ValueError, IndexError):
                        pass

    def extract_errors(self):
        """Extract error messages, tracebacks, and exceptions."""
        error_keywords = ['Traceback', 'Error', 'Exception', 'Failed', 'FAILED', 'IndexError', 'ValueError']
        current_error = []

        for line in self.lines:
            if any(kw in line for kw in error_keywords):
                if current_error and 'Traceback' in line:
                    # Save previous error
                    self.errors.append('\n'.join(current_error))
                    current_error = [line]
                else:
                    current_error.append(line)
            elif current_error and (line.startswith('  ') or line.startswith('\t')):
                current_error.append(line)
            elif current_error and line.strip() == '':
                self.errors.append('\n'.join(current_error))
                current_error = []

        if current_error:
            self.errors.append('\n'.join(current_error))

    def _extract_timestamp(self, line):
        """Extract timestamp from log line."""
        match = re.search(r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if match:
            return match.group(1)
        return 'unknown'

    def get_summary(self):
        """Generate summary of log analysis."""
        summary = {
            'total_lines': len(self.lines),
            'errors_found': len(self.errors),
            'metrics_tracked': {k: len(v) for k, v in self.metrics.items() if v},
            'timeline_events': len(self.timeline),
        }
        return summary

    @staticmethod
    def _md_cell(text, max_len=120):
        """Escape pipes/newlines for markdown table cells."""
        if text is None:
            return ""
        s = str(text).replace("\n", " ").replace("|", "\\|")
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s

    def to_markdown(self):
        """Output cleaned log as structured markdown."""
        output = []
        output.append("# CURE Training Log Analysis\n")

        # Summary
        summary = self.get_summary()
        output.append("## Summary\n")
        output.append(f"- **Total lines processed**: {summary['total_lines']}\n")
        output.append(f"- **Errors found**: {summary['errors_found']}\n")
        output.append(f"- **Timeline events**: {summary['timeline_events']}\n")
        output.append("\n")

        # Errors
        if self.errors:
            output.append("## Errors & Exceptions\n")
            for i, error in enumerate(self.errors[:5], 1):  # Show first 5 errors
                output.append(f"### Error {i}\n")
                output.append("```\n")
                output.append(error[:500])  # Truncate long errors
                output.append("\n```\n")
                output.append("\n")

        # Timeline (no standalone timestamp rows; list all retained events)
        if self.timeline:
            output.append("## Execution Timeline\n")
            output.append(
                "| idx | event | time | log (truncated) |\n"
                "|-----|-------|------|------------------|\n"
            )
            for i, event in enumerate(self.timeline):
                ts = event.get("timestamp") or ""
                if ts == "unknown":
                    ts = ""
                output.append(
                    f"| {i} | {self._md_cell(event.get('event', ''), 80)} | "
                    f"{self._md_cell(ts, 24)} | {self._md_cell(event.get('line', ''), 100)} |\n"
                )
            output.append("\n")

        # Metrics: rewards / PPO first, then the rest
        if self.metrics:
            reward_first = (
                'estimated_code_reward_mean', 'estimated_case_reward_mean',
                'estimated_code_reward_std', 'estimated_case_reward_std',
                'gt_correlation_code_spearman', 'self_bootstrap_used',
                'self_bootstrap_skipped',
                'ppo_avg_custom_reward_code', 'ppo_avg_custom_reward_case',
                'ppo_policy_loss_code', 'ppo_policy_loss_case',
                'ppo_kl_loss_code', 'ppo_kl_loss_case',
                'ppo_clip_ratio_code', 'ppo_clip_ratio_case',
                'ppo_entropy_code', 'ppo_entropy_case',
                'bon4_acc', 'bon4_accumulate_acc', 'bon16_acc', 'bon16_accumulate_acc',
            )
            ordered = [k for k in reward_first if k in self.metrics and self.metrics[k]]
            ordered += [k for k in sorted(self.metrics.keys()) if k not in ordered and self.metrics[k]]

            output.append("## Performance metrics\n")
            output.append(
                "Order: **rewards & PPO first** — self-bootstrap `estimated_*_reward_*`, "
                "`gt_correlation_code_spearman`, `self_bootstrap_used/skipped`; "
                "BoN `bon4_*` / `bon16_*`; PPO `ppo_avg_custom_reward_*`, "
                "`ppo_entropy_*`, `ppo_policy_loss_*`, `ppo_clip_ratio_*`. "
                "Then code/case acc, lengths, etc.\n\n"
            )

            for metric_name in ordered:
                values = self.metrics[metric_name]
                if not values:
                    continue
                nums = [v['value'] for v in values]
                has_step = any('step' in v for v in values)
                output.append(f"### {metric_name}\n")
                output.append(
                    f"- **Summary**: count={len(nums)}, latest={nums[-1]:.6g}"
                )
                if len(nums) > 1:
                    output.append(f", min={min(nums):.6g}, max={max(nums):.6g}")
                output.append("\n\n")
                if has_step:
                    output.append(
                        "| idx | value | step | log (truncated) |\n"
                        "|-----|-------|------|------------------|\n"
                    )
                    for i, v in enumerate(values):
                        st = v.get("step", "")
                        output.append(
                            f"| {i} | {v['value']:.6g} | {st} | "
                            f"{self._md_cell(v.get('line', ''), 100)} |\n"
                        )
                else:
                    output.append(
                        "| idx | value | log (truncated) |\n"
                        "|-----|-------|------------------|\n"
                    )
                    for i, v in enumerate(values):
                        output.append(
                            f"| {i} | {v['value']:.6g} | "
                            f"{self._md_cell(v.get('line', ''), 120)} |\n"
                        )
                output.append("\n")

        return "".join(output)

    def save_json(self, output_path, metrics_max_per_key=2000):
        """Save cleaned data as JSON for programmatic access."""
        import json
        slim_metrics = {}
        for k, v in self.metrics.items():
            slim_metrics[k] = v if len(v) <= metrics_max_per_key else v[:metrics_max_per_key]
        data = {
            'summary': self.get_summary(),
            'errors': self.errors[:10],
            'timeline': self.timeline[:200],
            'metrics': slim_metrics,
        }
        Path(output_path).write_text(json.dumps(data, indent=2))


def _parse_cli():
    argv = sys.argv[1:]
    output_format = 'markdown'
    out_file = None
    positional = []
    i = 0
    while i < len(argv):
        if argv[i] == '--output' and i + 1 < len(argv):
            output_format = argv[i + 1]
            i += 2
        elif argv[i] in ('-o', '--output-file') and i + 1 < len(argv):
            out_file = argv[i + 1]
            i += 2
        elif argv[i] in ('-h', '--help'):
            return None, None, None
        else:
            positional.append(argv[i])
            i += 1
    if not positional:
        return None, None, None
    return positional[0], output_format, out_file


def main():
    parsed = _parse_cli()
    if parsed[0] is None:
        print("Usage: python clean_log.py <log_file> [--output markdown|json] [-o OUT_PATH]")
        print("  -o, --output-file  Write cleaned result here (use when cwd is not writable).")
        print("Formats: markdown (default), json")
        sys.exit(1)

    log_path, output_format, out_file = parsed

    cleaner = LogCleaner(log_path)
    print(f"[*] Removing noise...", file=sys.stderr)
    removed = len(cleaner.lines) - cleaner.remove_noise()
    print(f"[+] Removed {removed} noise lines", file=sys.stderr)

    print(f"[*] Extracting timeline...", file=sys.stderr)
    cleaner.extract_timeline()

    print(f"[*] Extracting metrics...", file=sys.stderr)
    cleaner.extract_metrics()

    print(f"[*] Extracting errors...", file=sys.stderr)
    cleaner.extract_errors()

    stem = Path(log_path).stem
    if output_format == 'json':
        output_path = out_file or f'{stem}_cleaned.json'
        cleaner.save_json(output_path)
        print(f"[+] Saved to {output_path}", file=sys.stderr)
        print(output_path)
    else:
        output_path = out_file or f'{stem}_cleaned.md'
        Path(output_path).write_text(cleaner.to_markdown())
        print(f"[+] Saved to {output_path}", file=sys.stderr)
        print(output_path)


if __name__ == '__main__':
    main()
