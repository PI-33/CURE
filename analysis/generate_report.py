#!/usr/bin/env python3
"""
CURE Training Results Report Generator

Parses training logs, evaluation results, and reward statistics from a
completed CURE experiment, then produces:
  - train_curve.csv   (per-step training metrics)
  - eval_summary.csv  (periodic evaluation on CodeContests eval set)
  - comparison_table.txt (baseline vs trained plain-text table)
  - figures/A-G       (publication-ready matplotlib plots)

Usage:
    python generate_report.py [--exp_dir <path>]

If --exp_dir is not given, it auto-detects the latest experiment under
  <project_root>/experiments/
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── path helpers ───────────────────────────

def find_experiment_dir(project_root: str) -> str:
    exp_base = os.path.join(project_root, "experiments")
    candidates = sorted(glob.glob(os.path.join(exp_base, "*")))
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        sys.exit(f"No experiment directories found under {exp_base}")
    return candidates[-1]


def find_file(directory: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        return ""
    return matches[0]


# ─────────────────────────── parsers ────────────────────────────────

def parse_rl_blocks(filepath: str) -> list[dict]:
    """Parse results-rl-*.txt: 9 lines per step block."""
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    blocks = []
    i = 0
    while i + 8 <= len(lines):
        block = {}
        # line 0: code response length: X, case response length: Y
        m = re.search(r"code response length:\s*([\d.]+).*case response length:\s*([\d.]+)", lines[i])
        if not m:
            i += 1
            continue
        block["mean_code_len"] = float(m.group(1))
        block["mean_case_len"] = float(m.group(2))

        # line 1: code acc: X, code accumulate acc: Y
        m = re.search(r"code acc:\s*([\d.]+).*code accumulate acc:\s*([\d.]+)", lines[i+1])
        block["code_acc"] = float(m.group(1))
        block["code_acc_acc"] = float(m.group(2))

        # line 2: case acc: X, case accumulate acc: Y
        m = re.search(r"case acc:\s*([\d.]+).*case accumulate acc:\s*([\d.]+)", lines[i+2])
        block["case_acc"] = float(m.group(1))
        block["case_acc_acc"] = float(m.group(2))

        # line 3: p_01: X
        block["p_01"] = float(lines[i+3].split(":")[1].strip())

        # line 4: p_00: X
        block["p_00"] = float(lines[i+4].split(":")[1].strip())

        # line 5-6: BoN setting (4, 4): \n acc: X, accumulate acc: Y
        m = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i+6])
        block["bon_4_4_acc"] = float(m.group(1))
        block["bon_4_4_acc_acc"] = float(m.group(2))

        # line 7-8: BoN setting (16, 16): \n acc: X, accumulate acc: Y
        m = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i+8])
        block["bon_16_16_acc"] = float(m.group(1))
        block["bon_16_16_acc_acc"] = float(m.group(2))

        blocks.append(block)
        i += 9

    return blocks


def parse_reward_blocks(filepath: str) -> list[dict]:
    """Parse results-*.txt (non-rl): 2 lines per step block."""
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    blocks = []
    i = 0
    while i + 1 < len(lines):
        block = {}
        m = re.search(r"mean=([\d.-]+),\s*std=([\d.-]+)", lines[i])
        if m:
            block["code_reward_mean"] = float(m.group(1))
            block["code_reward_std"] = float(m.group(2))
        m = re.search(r"mean=([\d.-]+),\s*std=([\d.-]+)", lines[i+1])
        if m:
            block["case_reward_mean"] = float(m.group(1))
            block["case_reward_std"] = float(m.group(2))
        blocks.append(block)
        i += 2

    return blocks


def parse_eval_blocks(filepath: str) -> list[dict]:
    """Parse results-eval-*.txt: 11 lines per eval block."""
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    blocks = []
    i = 0
    while i + 10 <= len(lines):
        block = {}
        # line 0: code acc (...): X
        m = re.search(r":\s*([\d.]+)\s*$", lines[i])
        if not m:
            i += 1
            continue
        block["code_acc"] = float(m.group(1))

        # line 1: code accumulate acc
        m = re.search(r":\s*([\d.]+)\s*$", lines[i+1])
        block["code_acc_acc"] = float(m.group(1))

        # line 2: estimated unit test acc
        m = re.search(r":\s*([\d.]+)\s*$", lines[i+2])
        block["case_acc"] = float(m.group(1))

        # line 3: estimated unit test accumulate acc
        m = re.search(r":\s*([\d.]+)\s*$", lines[i+3])
        block["case_acc_acc"] = float(m.group(1))

        # line 4: estimated p_01
        block["p_01"] = float(lines[i+4].split(":")[1].strip())

        # line 5: estimated p_00
        block["p_00"] = float(lines[i+5].split(":")[1].strip())

        # line 6-7: BoN (4,4)
        m = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i+7])
        block["bon_4_4_acc"] = float(m.group(1))
        block["bon_4_4_acc_acc"] = float(m.group(2))

        # line 8-9: BoN (16,16)
        m = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i+9])
        block["bon_16_16_acc"] = float(m.group(1))
        block["bon_16_16_acc_acc"] = float(m.group(2))

        # line 10: response lengths
        m = re.search(r"code average response length:\s*([\d.]+).*unit test average response length:\s*([\d.]+)", lines[i+10])
        block["mean_code_len"] = float(m.group(1))
        block["mean_case_len"] = float(m.group(2))

        blocks.append(block)
        i += 11

    return blocks


# ────────────────────── build train_curve.csv ───────────────────────

TRAIN_COLUMNS = [
    "step",
    "code_acc", "code_acc_acc",
    "case_acc", "case_acc_acc",
    "p_01", "p_00",
    "bon_4_4_acc", "bon_4_4_acc_acc",
    "bon_16_16_acc", "bon_16_16_acc_acc",
    "mean_code_len", "mean_case_len",
    "code_reward_mean", "code_reward_std",
    "case_reward_mean", "case_reward_std",
]


def build_train_curve(exp_dir: str) -> list[dict]:
    opt_results = os.path.join(exp_dir, "optimization", "results")

    # find files
    pretrain_rl = find_file(opt_results, "results-rl-*ai4good2*")
    optimized_rl = find_file(opt_results, "results-rl-*optimized*")
    pretrain_rw = find_file(opt_results, "results-*ai4good2*")
    pretrain_rw = pretrain_rw if pretrain_rw and "results-rl" not in pretrain_rw else ""
    optimized_rw = find_file(opt_results, "results-*optimized*")
    optimized_rw = optimized_rw if optimized_rw and "results-rl" not in optimized_rw else ""

    # handle the non-rl files (reward) - need to filter out rl files
    rw_candidates = glob.glob(os.path.join(opt_results, "results-.*ai4good2*"))
    for f in glob.glob(os.path.join(opt_results, "*")):
        bn = os.path.basename(f)
        if bn.startswith("results-") and "results-rl-" not in bn and "results-eval-" not in bn:
            if "ai4good2" in bn:
                pretrain_rw = f
            elif "optimized" in bn:
                optimized_rw = f

    rl_pretrain = parse_rl_blocks(pretrain_rl)
    rl_optimized = parse_rl_blocks(optimized_rl)
    rw_pretrain = parse_reward_blocks(pretrain_rw)
    rw_optimized = parse_reward_blocks(optimized_rw)

    n_opt_steps = len(rl_optimized)
    total_steps = 1 + n_opt_steps  # step 0 + steps 1..N

    # align reward blocks: take last n_opt_steps from optimized reward
    if len(rw_optimized) > n_opt_steps:
        rw_optimized = rw_optimized[len(rw_optimized) - n_opt_steps:]

    rows = []
    for step in range(total_steps):
        row = {"step": step}

        if step == 0:
            if rl_pretrain:
                row.update(rl_pretrain[-1])  # last block = this run's step 0
            if rw_pretrain:
                row.update(rw_pretrain[-1])
        else:
            idx = step - 1
            if idx < len(rl_optimized):
                row.update(rl_optimized[idx])
            if idx < len(rw_optimized):
                row.update(rw_optimized[idx])

        rows.append(row)

    return rows


# ────────────────────── build eval_summary.csv ──────────────────────

EVAL_COLUMNS = [
    "model", "checkpoint", "dataset", "eval_size",
    "n_code", "m_test", "temperature", "top_p", "step",
    "code_acc", "code_acc_acc",
    "case_acc", "case_acc_acc",
    "p_01", "p_00",
    "bon_4_4_acc", "bon_4_4_acc_acc",
    "bon_16_16_acc", "bon_16_16_acc_acc",
    "mean_code_len", "mean_case_len",
]


def build_eval_summary(exp_dir: str, eval_interval: int = 25) -> list[dict]:
    eval_results_dir = os.path.join(exp_dir, "evaluation", "results")
    eval_file = find_file(eval_results_dir, "results-eval-*CodeContests*")
    if not eval_file or "small_eval" in eval_file or "MBPP" in eval_file:
        for f in glob.glob(os.path.join(eval_results_dir, "results-eval-*")):
            if "CodeContests" in f and "small_eval" not in f and "MBPP" not in f:
                eval_file = f
                break

    blocks = parse_eval_blocks(eval_file)

    rows = []
    # baseline placeholder
    rows.append({
        "model": "baseline (Qwen2.5-7B-Instruct)",
        "checkpoint": "pretrained",
        "dataset": "CodeContests",
        "eval_size": 239,
        "n_code": 16, "m_test": 16,
        "temperature": 1.0, "top_p": 1.0,
        "step": 0,
    })

    for i, blk in enumerate(blocks):
        step = (i + 1) * eval_interval
        row = {
            "model": f"trained_step{step}",
            "checkpoint": f"iter{step}",
            "dataset": "CodeContests",
            "eval_size": 239,
            "n_code": 16, "m_test": 16,
            "temperature": 1.0, "top_p": 1.0,
            "step": step,
        }
        row.update(blk)
        rows.append(row)

    return rows


# ────────────────────── comparison table ────────────────────────────

def build_comparison_table(train_rows: list[dict], eval_rows: list[dict]) -> str:
    step0 = train_rows[0] if train_rows else {}
    best_eval = {}
    for r in eval_rows[1:]:
        if r.get("code_acc") and (not best_eval or r["code_acc"] > best_eval.get("code_acc", 0)):
            best_eval = r

    last_train = train_rows[-1] if train_rows else {}

    sep = "=" * 80
    header = f"{sep}\n  CURE 复现实验: Baseline vs Trained 对比表\n{sep}\n\n"

    def fmt(v, pct=True):
        if v is None or v == "":
            return "N/A"
        if pct:
            return f"{float(v):.4f}"
        return f"{float(v):.1f}"

    metrics = [
        ("Code Accuracy",       "code_acc"),
        ("Code Accumulate Acc", "code_acc_acc"),
        ("UT Accuracy",         "case_acc"),
        ("UT Accumulate Acc",   "case_acc_acc"),
        ("p_01",                "p_01"),
        ("p_00",                "p_00"),
        ("BoN(4,4) Acc",        "bon_4_4_acc"),
        ("BoN(16,16) Acc",      "bon_16_16_acc"),
        ("Code Resp Length",    "mean_code_len"),
        ("Case Resp Length",    "mean_case_len"),
    ]

    lines = [header]
    lines.append(f"{'Metric':<25s} {'Baseline (Step 0)':>18s} {'Best Eval':>18s} {'Eval Step':>10s}\n")
    lines.append("-" * 75 + "\n")

    for label, key in metrics:
        b_val = fmt(step0.get(key), pct=(key not in ("mean_code_len", "mean_case_len")))
        e_val = fmt(best_eval.get(key), pct=(key not in ("mean_code_len", "mean_case_len")))
        e_step = str(best_eval.get("step", "N/A"))
        lines.append(f"{label:<25s} {b_val:>18s} {e_val:>18s} {e_step:>10s}\n")

    lines.append("\n" + "-" * 75 + "\n")
    lines.append("Note: Baseline uses Step 0 training-set metrics (not eval-set).\n")
    lines.append("      Best Eval is the checkpoint with highest code_acc on CodeContests eval.\n")
    lines.append("      To get a true baseline, run eval.py on the pretrained model.\n")

    return "".join(lines)


# ────────────────────── plotting ────────────────────────────────────

def rolling_avg(values, window=10):
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    out = np.full_like(arr, np.nan)
    cumsum = np.zeros(len(arr))
    count = np.zeros(len(arr))
    for i in range(len(arr)):
        if mask[i]:
            start = max(0, i - window + 1)
            segment = arr[start:i+1]
            valid = segment[~np.isnan(segment)]
            if len(valid):
                out[i] = np.mean(valid)
    return out


def setup_plot_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_single_metric(steps, values, title, ylabel, save_path,
                       window=10, eval_steps=None, eval_values=None,
                       extra_series=None):
    setup_plot_style()
    fig, ax = plt.subplots()

    arr = np.array(values, dtype=float)
    ax.plot(steps, arr, alpha=0.3, color="C0", linewidth=0.8)
    smoothed = rolling_avg(values, window)
    ax.plot(steps, smoothed, color="C0", linewidth=2.0,
            label=f"{ylabel} (smoothed, w={window})")

    if extra_series:
        for label, vals, color in extra_series:
            arr2 = np.array(vals, dtype=float)
            ax.plot(steps, arr2, alpha=0.3, color=color, linewidth=0.8)
            sm2 = rolling_avg(vals, window)
            ax.plot(steps, sm2, color=color, linewidth=2.0,
                    label=f"{label} (smoothed)")

    if eval_steps and eval_values:
        ax.scatter(eval_steps, eval_values, color="red", zorder=5, s=60,
                   marker="D", label="Eval on CodeContests")

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_all(train_rows, eval_rows, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    setup_plot_style()

    steps = [r["step"] for r in train_rows]

    def col(key):
        return [r.get(key, float("nan")) for r in train_rows]

    eval_steps_all = [r["step"] for r in eval_rows[1:] if r.get("code_acc")]
    eval_code_acc = [r["code_acc"] for r in eval_rows[1:] if r.get("code_acc")]
    eval_case_acc_vals = [r.get("case_acc") for r in eval_rows[1:] if r.get("code_acc")]
    eval_bon16 = [r.get("bon_16_16_acc") for r in eval_rows[1:] if r.get("code_acc")]

    # A: Code Accuracy vs Steps
    plot_single_metric(
        steps, col("code_acc"),
        "Code Accuracy vs Training Steps",
        "Code Accuracy",
        os.path.join(fig_dir, "A_code_acc_vs_steps.png"),
        eval_steps=eval_steps_all, eval_values=eval_code_acc,
    )

    # B: Case (UT) Accuracy vs Steps
    plot_single_metric(
        steps, col("case_acc"),
        "Unit Test Accuracy vs Training Steps",
        "UT Accuracy",
        os.path.join(fig_dir, "B_case_acc_vs_steps.png"),
        eval_steps=[s for s, v in zip(eval_steps_all, eval_case_acc_vals) if v],
        eval_values=[v for v in eval_case_acc_vals if v],
    )

    # C: BoN Accuracy vs Steps (both settings)
    fig, ax = plt.subplots(figsize=(10, 6))
    bon44 = col("bon_4_4_acc")
    bon1616 = col("bon_16_16_acc")
    ax.plot(steps, np.array(bon44, dtype=float), alpha=0.25, color="C0", linewidth=0.8)
    ax.plot(steps, rolling_avg(bon44), color="C0", linewidth=2.0, label="BoN(4,4) smoothed")
    ax.plot(steps, np.array(bon1616, dtype=float), alpha=0.25, color="C1", linewidth=0.8)
    ax.plot(steps, rolling_avg(bon1616), color="C1", linewidth=2.0, label="BoN(16,16) smoothed")
    valid_eval_bon = [(s, v) for s, v in zip(eval_steps_all, eval_bon16) if v]
    if valid_eval_bon:
        ax.scatter([s for s, v in valid_eval_bon], [v for s, v in valid_eval_bon],
                   color="red", zorder=5, s=60, marker="D", label="Eval BoN(16,16)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("BoN Accuracy")
    ax.set_title("Best-of-N Accuracy vs Training Steps")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "C_bon_acc_vs_steps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # D: Response Length vs Steps
    plot_single_metric(
        steps, col("mean_code_len"),
        "Response Length vs Training Steps",
        "Token Length",
        os.path.join(fig_dir, "D_response_length_vs_steps.png"),
        extra_series=[("Case Response Length", col("mean_case_len"), "C1")],
    )

    # E: p01 / p00 vs Steps
    fig, ax = plt.subplots(figsize=(10, 6))
    p01 = col("p_01")
    p00 = col("p_00")
    ax.plot(steps, np.array(p01, dtype=float), alpha=0.25, color="C0", linewidth=0.8)
    ax.plot(steps, rolling_avg(p01), color="C0", linewidth=2.0, label="p_01 (smoothed)")
    ax.plot(steps, np.array(p00, dtype=float), alpha=0.25, color="C3", linewidth=0.8)
    ax.plot(steps, rolling_avg(p00), color="C3", linewidth=2.0, label="p_00 (smoothed)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Proportion")
    ax.set_title("p_01 / p_00 vs Training Steps")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "E_p01_p00_vs_steps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # F: Reward Mean vs Steps
    plot_single_metric(
        steps, col("code_reward_mean"),
        "Reward Mean vs Training Steps",
        "Reward Mean",
        os.path.join(fig_dir, "F_reward_vs_steps.png"),
        extra_series=[("Case Reward Mean", col("case_reward_mean"), "C1")],
    )

    # G: Baseline vs Trained bar chart
    step0 = train_rows[0] if train_rows else {}
    best_eval = {}
    for r in eval_rows[1:]:
        if r.get("code_acc") and (not best_eval or r["code_acc"] > best_eval.get("code_acc", 0)):
            best_eval = r

    if step0 and best_eval:
        labels = ["Code Acc", "UT Acc", "BoN(4,4)", "BoN(16,16)"]
        keys = ["code_acc", "case_acc", "bon_4_4_acc", "bon_16_16_acc"]
        baseline_vals = [step0.get(k, 0) or 0 for k in keys]
        trained_vals = [best_eval.get(k, 0) or 0 for k in keys]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (Step 0)", color="C7")
        bars2 = ax.bar(x + width/2, trained_vals, width,
                       label=f"Trained (Step {best_eval.get('step', '?')})", color="C0")

        ax.set_ylabel("Accuracy")
        ax.set_title("Baseline vs Trained: Key Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, max(max(baseline_vals), max(trained_vals)) * 1.3)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(fig_dir, "G_baseline_vs_trained.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ────────────────────── CSV writers ─────────────────────────────────

def write_train_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  Saved: {path}  ({len(rows)} rows)")


def write_eval_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  Saved: {path}  ({len(rows)} rows)")


# ────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CURE Training Report Generator")
    parser.add_argument("--exp_dir", type=str, default="",
                        help="Path to experiment directory")
    parser.add_argument("--project_root", type=str,
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="CURE project root")
    args = parser.parse_args()

    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        exp_dir = find_experiment_dir(args.project_root)

    print(f"Experiment directory: {exp_dir}")
    out_dir = os.path.join(args.project_root, "analysis")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Build train_curve
    print("\n[1/5] Parsing training metrics...")
    train_rows = build_train_curve(exp_dir)
    train_csv = os.path.join(out_dir, "train_curve.csv")
    write_train_csv(train_rows, train_csv)

    # 2. Build eval_summary
    print("\n[2/5] Parsing evaluation results...")
    eval_rows = build_eval_summary(exp_dir)
    eval_csv = os.path.join(out_dir, "eval_summary.csv")
    write_eval_csv(eval_rows, eval_csv)

    # 3. Build comparison table
    print("\n[3/5] Building comparison table...")
    table = build_comparison_table(train_rows, eval_rows)
    table_path = os.path.join(out_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"  Saved: {table_path}")
    print(table)

    # 4. Generate figures
    print("\n[4/5] Generating figures...")
    plot_all(train_rows, eval_rows, fig_dir)

    # 5. Summary
    print("\n[5/5] Report generation complete!")
    print(f"\nAll outputs saved to: {out_dir}")
    print(f"  - train_curve.csv      ({len(train_rows)} steps)")
    print(f"  - eval_summary.csv     ({len(eval_rows)} entries)")
    print(f"  - comparison_table.txt")
    print(f"  - figures/             ({len(os.listdir(fig_dir))} plots)")


if __name__ == "__main__":
    main()
