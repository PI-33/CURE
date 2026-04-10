#!/usr/bin/env python3
"""
将 experiments/.../optimization/diagnostics/step_*.json 汇总为宽表 CSV，
并为每个数值指标绘制 step 趋势图（PNG）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def flatten_step(obj: dict) -> dict:
    """展平单步 JSON；bon_by_scale 展开为 execution_bon_{N}x{M}_acc 等。"""
    out: dict = {}

    def flatten_dict(d: dict, prefix: str) -> None:
        for k, v in d.items():
            if str(k).startswith("_"):
                continue
            key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            elif isinstance(v, bool):
                out[key] = int(v)
            elif isinstance(v, (int, float, str)) or v is None:
                out[key] = v
            else:
                out[key] = json.dumps(v, ensure_ascii=False)

    if "step" in obj:
        out["step"] = obj["step"]
    if "recorded_at" in obj:
        out["recorded_at"] = obj["recorded_at"]

    samp = obj.get("sampling")
    if isinstance(samp, dict):
        flatten_dict(samp, "sampling")

    ex = obj.get("execution")
    if isinstance(ex, dict):
        for k, v in ex.items():
            if k == "bon_by_scale" and isinstance(v, list):
                for item in v:
                    if not isinstance(item, dict):
                        continue
                    nc = item.get("num_code")
                    ng = item.get("num_gen_tests")
                    lab = f"execution_bon_{nc}x{ng}"
                    out[f"{lab}_acc"] = item.get("acc")
                    out[f"{lab}_accumulate_acc"] = item.get("accumulate_acc")
            elif isinstance(v, dict):
                flatten_dict(v, f"execution_{k}")
            elif isinstance(v, bool):
                out[f"execution_{k}"] = int(v)
            else:
                out[f"execution_{k}"] = v

    rw = obj.get("reward")
    if isinstance(rw, dict):
        flatten_dict(rw, "reward")

    return out


def safe_filename(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name)
    return s[:max_len].strip("_") or "metric"


def resolve_charts_dir(
    explicit: Path | None, exp_root: Path, diag: Path
) -> Path:
    if explicit is not None:
        p = explicit.resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    script_root = Path(__file__).resolve().parent.parent
    candidates = [
        exp_root / "em_summary_charts",
        diag / "em_summary_charts",
        script_root / "analysis" / f"em_summary_{exp_root.name}",
    ]
    last_err: OSError | None = None
    for p in candidates:
        try:
            p = p.resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError as e:
            last_err = e
    raise SystemExit(
        f"无法在实验目录或 diagnostics 下创建输出文件夹（最后错误: {last_err}）。"
        f"请使用 --charts-dir 指定可写路径。"
    )


def write_zh_report(df: pd.DataFrame, charts_dir: Path, diag: Path, exp_root: Path) -> None:
    """基于宽表写简短中文汇总（执行 / 采样 / reward 诊断）。"""
    lines = [
        "# 步进指标汇总（自动生成）\n",
        f"- **diagnostics 源目录**: `{diag}`\n",
        f"- **实验目录**: `{exp_root}`\n",
        f"- **总步数**: {len(df)}\n",
        f"- **输出目录（图表与 CSV）**: `{charts_dir}`\n",
        "\n## 文件说明\n",
        "- `em_step_metrics_wide.csv`：每行一步，每列一指标（宽表，可用 Excel 打开）。\n",
        "- `em_step_metrics_preview.md`：前 20 步表格预览。\n",
        "- `em_trend__*.png`：各数值指标随 step 的折线图。\n",
        "\n## 核心观察（自动统计）\n",
    ]

    def stat_col(name: str) -> str:
        if name not in df.columns:
            return f"- `{name}`：无此列\n"
        s = pd.to_numeric(df[name], errors="coerce")
        def fmt(x):
            if pd.isna(x):
                return "NaN"
            return f"{float(x):.6g}"

        return (
            f"- `{name}`：首步 {fmt(s.iloc[0])}，末步 {fmt(s.iloc[-1])}，"
            f"全程均值 {fmt(s.mean())}，最小 {fmt(s.min())}，最大 {fmt(s.max())}\n"
        )

    for c in (
        "execution_code_acc",
        "execution_code_accumulate_acc",
        "execution_bon_16x16_acc",
        "execution_bon_16x16_accumulate_acc",
        "sampling_mean_code_response_tokens",
        "sampling_mean_case_response_tokens",
    ):
        lines.append(stat_col(c))

    if "reward_problems_used" in df.columns:
        u = df["reward_problems_used"]
        lines.append(
            f"- **`reward_problems_used`**：末步 {u.iloc[-1]}；"
            f"等于 0 的步数 {(u == 0).sum()} / {len(df)}；"
            f"≥50 的最后一步 step = {df.loc[u >= 50, 'step'].max() if (u >= 50).any() else '无'}\n"
        )
    if "reward_gt_correlation_mean_spearman" in df.columns:
        g = pd.to_numeric(df["reward_gt_correlation_mean_spearman"], errors="coerce")
        lines.append(
            f"- **`reward_gt_correlation_mean_spearman`**：非 NaN 步数 {g.notna().sum()}；"
            f"均值 {g.mean():.6g}\n"
        )

    lines.append(
        "\n### 说明\n"
        "若 `reward_problems_used` 在后期长期接近 0，表示自举 reward 过滤后几乎没有题目进入 RL，"
        "训练信号会变弱，需结合 `min_weight_threshold`、采样规模等排查。\n"
    )

    out = charts_dir / "em_analysis_report_zh.md"
    out.write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "diagnostics_dir",
        type=Path,
        help="含 step_*.json 的目录，如 .../optimization/diagnostics",
    )
    ap.add_argument(
        "--charts-dir",
        type=Path,
        default=None,
        help=(
            "图表与 CSV 输出目录；默认依次尝试 <实验根>/em_summary_charts、"
            "diagnostics/em_summary_charts、CURE/analysis/em_summary_<实验名>"
        ),
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=120,
    )
    args = ap.parse_args()
    diag: Path = args.diagnostics_dir.resolve()
    if not diag.is_dir():
        raise SystemExit(f"Not a directory: {diag}")

    exp_root = diag.parent.parent  # diagnostics -> optimization -> 实验根目录
    charts_dir = resolve_charts_dir(args.charts_dir, exp_root, diag)

    step_files = sorted(
        diag.glob("step_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not step_files:
        raise SystemExit(f"No step_*.json under {diag}")

    rows = []
    for p in step_files:
        with open(p, encoding="utf-8") as f:
            rows.append(flatten_step(json.load(f)))

    df = pd.DataFrame(rows)
    if "step" not in df.columns:
        raise SystemExit("Missing 'step' column after flatten")

    df = df.sort_values("step").reset_index(drop=True)

    # 列顺序：step 在前，其余按名字排序
    cols = ["step"] + sorted(c for c in df.columns if c != "step")
    df = df.reindex(columns=[c for c in cols if c in df.columns])

    csv_path = charts_dir / "em_step_metrics_wide.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 可选：紧凑 Markdown（前 20 行预览）
    md_preview = charts_dir / "em_step_metrics_preview.md"
    with open(md_preview, "w", encoding="utf-8") as mf:
        mf.write(f"# EM 步进指标预览（前 20 行）\n\n")
        mf.write(f"- 总步数: {len(df)}\n")
        mf.write(f"- 宽表 CSV: `{csv_path.name}`\n\n")
        mf.write(df.head(20).to_markdown(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numeric_cols = []
    for c in df.columns:
        if c == "step":
            continue
        if df[c].dtype in ("float64", "int64", "float32", "int32"):
            numeric_cols.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() >= max(3, len(df) // 10):
                df[c] = coerced
                numeric_cols.append(c)

    x = df["step"].values
    for col in numeric_cols:
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 4), dpi=args.dpi)
        ax.plot(x, y, linewidth=1.0, color="#2563eb", alpha=0.85)
        ax.set_xlabel("step")
        ax.set_ylabel(col)
        ax.set_title(col, fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fn = safe_filename(f"em_trend__{col}") + ".png"
        fig.savefig(charts_dir / fn)
        plt.close(fig)

    summary_path = charts_dir / "em_run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(f"diagnostics_dir={diag}\n")
        sf.write(f"exp_root={exp_root}\n")
        sf.write(f"num_steps={len(df)}\n")
        sf.write(f"num_columns={len(df.columns)}\n")
        sf.write(f"num_plots={len(numeric_cols)}\n")
        sf.write(f"csv={csv_path}\n")
        sf.write(f"charts_dir={charts_dir}\n")

    write_zh_report(df, charts_dir, diag, exp_root)
    report_zh = charts_dir / "em_analysis_report_zh.md"

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_preview}")
    print(f"Wrote {len(numeric_cols)} plots under {charts_dir}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_zh}")


if __name__ == "__main__":
    main()
