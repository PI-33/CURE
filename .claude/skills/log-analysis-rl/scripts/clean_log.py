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
            r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})': lambda m: f"timestamp_{m.group(1)}",
            r'code acc: ([\d.]+)': lambda m: f"code_acc_{m.group(1)}",
            r'case acc: ([\d.]+)': lambda m: f"case_acc_{m.group(1)}",
            r'BoN setting.*acc: ([\d.]+)': lambda m: f"bon_acc_{m.group(1)}",
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

    def extract_metrics(self):
        """Extract performance metrics from logs."""
        metric_patterns = {
            'code_acc': r'code acc: ([\d.]+)',
            'code_accumulate_acc': r'code accumulate acc: ([\d.]+)',
            'case_acc': r'case acc: ([\d.]+)',
            'case_accumulate_acc': r'case accumulate acc: ([\d.]+)',
            'bon_acc': r'BoN setting.*acc: ([\d.]+)',
            'bon_accumulate_acc': r'BoN setting.*accumulate acc: ([\d.]+)',
            'p_01': r'p_01: ([\d.]+)',
            'p_00': r'p_00: ([\d.]+)',
            'code_response_length': r'code response length: ([\d.]+)',
            'case_response_length': r'case response length: ([\d.]+)',
            'clip_ratio': r'ppo_clip_count.*?clip_ratio.*?([\d.]+)',
            'policy_entropy': r'policy_entropy.*?([\d.e\-]+)',
            'critic_loss': r'critic_loss.*?([\d.e\-]+)',
            'avg_kl': r'avg_kl.*?([\d.e\-]+)',
        }

        for line in self.lines:
            for metric_name, pattern in metric_patterns.items():
                match = re.search(pattern, line)
                if match:
                    try:
                        value = float(match.group(1))
                        self.metrics[metric_name].append({
                            'value': value,
                            'line': line.strip(),
                        })
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

        # Timeline
        if self.timeline:
            output.append("## Execution Timeline\n")
            for event in self.timeline[:20]:  # Show first 20 events
                output.append(f"- [{event['timestamp']}] {event['event']}\n")
            output.append("\n")

        # Metrics
        if self.metrics:
            output.append("## Performance Metrics\n")
            for metric_name, values in sorted(self.metrics.items()):
                if values:
                    nums = [v['value'] for v in values]
                    output.append(f"### {metric_name}\n")
                    output.append(f"- Count: {len(nums)}\n")
                    output.append(f"- Latest: {nums[-1]:.4f}\n")
                    if len(nums) > 1:
                        output.append(f"- Min: {min(nums):.4f}, Max: {max(nums):.4f}\n")
                    output.append(f"- Trend: {' → '.join(f'{v:.3f}' for v in nums[-5:])}\n")
                    output.append("\n")

        return "".join(output)

    def save_json(self, output_path):
        """Save cleaned data as JSON for programmatic access."""
        import json
        data = {
            'summary': self.get_summary(),
            'errors': self.errors[:10],  # First 10 errors
            'timeline': self.timeline[:50],  # First 50 events
            'metrics': {k: v[:50] for k, v in self.metrics.items()},  # First 50 points per metric
        }
        Path(output_path).write_text(json.dumps(data, indent=2))


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_log.py <log_file> [--output <format>]")
        print("Formats: markdown (default), json")
        sys.exit(1)

    log_path = sys.argv[1]
    output_format = 'markdown'

    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_format = sys.argv[idx + 1]

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

    if output_format == 'json':
        output_path = Path(log_path).stem + '_cleaned.json'
        cleaner.save_json(output_path)
        print(f"[+] Saved to {output_path}", file=sys.stderr)
        print(output_path)
    else:
        output_path = Path(log_path).stem + '_cleaned.md'
        Path(output_path).write_text(cleaner.to_markdown())
        print(f"[+] Saved to {output_path}", file=sys.stderr)
        print(output_path)


if __name__ == '__main__':
    main()
