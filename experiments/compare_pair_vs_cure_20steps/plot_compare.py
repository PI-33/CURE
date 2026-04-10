#!/usr/bin/env python3
"""Load diagnostics step_1..step_20 for pair vs cure experiments and emit plots + CSV summary."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAIR_DIR = Path(
    "/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiments/"
    "0409_Qwen2.5-7B-Instruct-2gpu_350step_pair/optimization/diagnostics"
)
CURE_DIR = Path(
    "/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/experiments/"
    "04031200_Qwen2.5-7B-Instruct_2gpu_350step_cure/optimization/diagnostics"
)
OUT = Path(__file__).resolve().parent


def load_steps(diag_dir: Path, max_step: int = 20):
    rows = []
    for s in range(1, max_step + 1):
        p = diag_dir / f"step_{s}.json"
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def bon_item(bon_list, num_code: int):
    for b in bon_list or []:
        if b.get("num_code") == num_code:
            return b
    return {}


def flatten_row(row: dict) -> dict:
    ex = row["execution"]
    rw = row["reward"]
    bon4 = bon_item(ex.get("bon_by_scale"), 4)
    bon16 = bon_item(ex.get("bon_by_scale"), 16)
    gc = rw.get("gt_correlation") or {}
    out = {
        "step": row["step"],
        "mean_code_response_tokens": row["sampling"]["mean_code_response_tokens"],
        "mean_case_response_tokens": row["sampling"]["mean_case_response_tokens"],
        "code_acc": ex["code_acc"],
        "code_accumulate_acc": ex["code_accumulate_acc"],
        "case_acc": ex["case_acc"],
        "case_accumulate_acc": ex["case_accumulate_acc"],
        "p_01_as_logged": ex["p_01_as_logged"],
        "p_00": ex["p_00"],
        "bon_k4_acc": bon4.get("acc"),
        "bon_k4_accumulate_acc": bon4.get("accumulate_acc"),
        "bon_k16_acc": bon16.get("acc"),
        "bon_k16_accumulate_acc": bon16.get("accumulate_acc"),
        "reward_mode": rw.get("reward_mode"),
        "code_reward_mean": rw["code_reward"]["mean"],
        "code_reward_std": rw["code_reward"]["std"],
        "code_reward_num_groups": rw["code_reward"]["num_groups"],
        "code_reward_num_samples": rw["code_reward"]["num_samples"],
        "case_reward_mean": rw["case_reward"]["mean"],
        "case_reward_std": rw["case_reward"]["std"],
        "case_reward_num_groups": rw["case_reward"]["num_groups"],
        "case_reward_num_samples": rw["case_reward"]["num_samples"],
        "problems_total": rw["problems"]["total"],
        "problems_used": rw["problems"]["used"],
        "problems_skipped": rw["problems"]["skipped"],
        "gt_correlation_mean_spearman": gc.get("mean_spearman"),
        "gt_correlation_n": gc.get("n"),
    }
    return out


def plot_panel(ax, steps, y_pair, y_cure, title, ylab, pair_label="pair", cure_label="cure"):
    ax.plot(steps, y_pair, "o-", label=pair_label, markersize=3, linewidth=1.2)
    ax.plot(steps, y_cure, "s--", label=cure_label, markersize=3, linewidth=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel(ylab, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def main():
    pair_rows = [flatten_row(r) for r in load_steps(PAIR_DIR, 20)]
    cure_rows = [flatten_row(r) for r in load_steps(CURE_DIR, 20)]
    steps = np.array([r["step"] for r in pair_rows])

    def col(name, which="pair"):
        data = pair_rows if which == "pair" else cure_rows
        return np.array([r[name] for r in data], dtype=float)

    # Figure 1: sampling + main execution accuracies
    fig1, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    fig1.suptitle(
        "Qwen2.5-7B-Instruct: pair vs CURE (gt_based) — steps 1–20: sampling & execution",
        fontsize=11,
    )
    metrics_f1 = [
        (0, 0, "mean_code_response_tokens", "mean code response tokens"),
        (0, 1, "mean_case_response_tokens", "mean case response tokens"),
        (0, 2, "code_acc", "code_acc"),
        (1, 0, "code_accumulate_acc", "code_accumulate_acc"),
        (1, 1, "case_acc", "case_acc"),
        (1, 2, "case_accumulate_acc", "case_accumulate_acc"),
    ]
    for i, j, key, title in metrics_f1:
        plot_panel(
            axes[i, j],
            steps,
            col(key, "pair"),
            col(key, "cure"),
            title,
            title,
        )
    fig1.savefig(OUT / "fig1_sampling_execution.png", dpi=150)
    plt.close(fig1)

    # Figure 2: p_01, p_00, BoN
    fig2, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    fig2.suptitle(
        "Qwen2.5-7B-Instruct: pair vs CURE — steps 1–20: p_01 / p_00 / BoN",
        fontsize=11,
    )
    m2 = [
        (0, 0, "p_01_as_logged", "p_01_as_logged"),
        (0, 1, "p_00", "p_00"),
        (0, 2, "bon_k4_acc", "BoN acc (K=4)"),
        (1, 0, "bon_k4_accumulate_acc", "BoN accumulate_acc (K=4)"),
        (1, 1, "bon_k16_acc", "BoN acc (K=16)"),
        (1, 2, "bon_k16_accumulate_acc", "BoN accumulate_acc (K=16)"),
    ]
    for i, j, key, title in m2:
        plot_panel(axes[i, j], steps, col(key, "pair"), col(key, "cure"), title, title)
    fig2.savefig(OUT / "fig2_p01_bon.png", dpi=150)
    plt.close(fig2)

    # Figure 3: rewards + meta
    fig3, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    fig3.suptitle(
        "Qwen2.5-7B-Instruct: pair vs CURE — steps 1–20: reward statistics",
        fontsize=11,
    )
    m3 = [
        (0, 0, "code_reward_mean", "code_reward mean"),
        (0, 1, "code_reward_std", "code_reward std"),
        (0, 2, "case_reward_mean", "case_reward mean"),
        (1, 0, "case_reward_std", "case_reward std"),
        (1, 1, "code_reward_num_groups", "code_reward num_groups"),
        (1, 2, "case_reward_num_groups", "case_reward num_groups"),
    ]
    for i, j, key, title in m3:
        plot_panel(axes[i, j], steps, col(key, "pair"), col(key, "cure"), title, title)
    fig3.savefig(OUT / "fig3_rewards.png", dpi=150)
    plt.close(fig3)

    # Figure 4: sample counts + gt_correlation (NaN where missing)
    fig4, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    fig4.suptitle(
        "Qwen2.5-7B-Instruct: pair vs CURE — steps 1–20: samples & GT correlation",
        fontsize=11,
    )
    plot_panel(
        axes[0],
        steps,
        col("code_reward_num_samples", "pair"),
        col("code_reward_num_samples", "cure"),
        "code_reward num_samples",
        "count",
    )
    plot_panel(
        axes[1],
        steps,
        col("case_reward_num_samples", "pair"),
        col("case_reward_num_samples", "cure"),
        "case_reward num_samples",
        "count",
    )
    sp_pair = col("gt_correlation_mean_spearman", "pair")
    sp_cure = col("gt_correlation_mean_spearman", "cure")
    plot_panel(
        axes[2],
        steps,
        sp_pair,
        sp_cure,
        "gt_correlation mean_spearman",
        "rho",
    )
    fig4.savefig(OUT / "fig4_samples_gt_corr.png", dpi=150)
    plt.close(fig4)

    # Optional: n for gt correlation
    fig5, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(
        steps,
        col("gt_correlation_n", "pair"),
        "o-",
        label="pair n",
        markersize=3,
    )
    ax.plot(
        steps,
        col("gt_correlation_n", "cure"),
        "s--",
        label="cure n",
        markersize=3,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("gt_correlation n")
    ax.set_title("gt_correlation sample count n")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig5.savefig(OUT / "fig5_gt_corr_n.png", dpi=150)
    plt.close(fig5)

    # CSV for report tables
    keys = [
        k
        for k in pair_rows[0].keys()
        if k not in ("step", "reward_mode")
    ]
    lines = ["step,exp," + ",".join(keys)]
    for r in pair_rows:
        lines.append(
            f"{r['step']},pair," + ",".join(str(r[k]) for k in keys)
        )
    for r in cure_rows:
        lines.append(
            f"{r['step']},cure," + ",".join(str(r[k]) for k in keys)
        )
    (OUT / "series_steps_1_20.csv").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote figures and series_steps_1_20.csv to", OUT)


if __name__ == "__main__":
    main()
