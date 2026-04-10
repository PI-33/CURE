"""Paths for the current training run.

When ``run.py`` sets ``CURE_EXPERIMENT_DIR`` to ``experiments/<exp_name>`` (absolute),
all temp data, optimization text results, and per-run ckpt live under that tree so
concurrent jobs with different ``exp_name`` do not clobber each other.

If the env var is unset (e.g. running ``sample.py`` alone), fall back to
``<this_dir>/temp_data`` and ``<this_dir>/results`` / ``<this_dir>/ckpt``.
"""

import os

_OPT_ROOT = os.path.dirname(os.path.abspath(__file__))


def experiment_dir():
    d = os.environ.get("CURE_EXPERIMENT_DIR")
    return os.path.abspath(d) if d else None


def temp_data_dir():
    e = experiment_dir()
    d = os.path.join(e, "temp_data") if e else os.path.join(_OPT_ROOT, "temp_data")
    os.makedirs(d, exist_ok=True)
    return d


def optimization_results_dir():
    e = experiment_dir()
    d = (
        os.path.join(e, "optimization", "results")
        if e
        else os.path.join(_OPT_ROOT, "results")
    )
    os.makedirs(d, exist_ok=True)
    return d


def experiment_ckpt_root():
    e = experiment_dir()
    d = os.path.join(e, "ckpt") if e else os.path.join(_OPT_ROOT, "ckpt")
    os.makedirs(d, exist_ok=True)
    return d


def experiment_ckpt_for_optimized(optimized_model_name: str) -> str:
    """仅用于「已是完整模型目录」的路径拼接；勿用作 train 的 save_path（见 actors.save_model 会再拼一层 name）。"""
    p = os.path.join(experiment_ckpt_root(), optimized_model_name)
    os.makedirs(p, exist_ok=True)
    return p


def resolve_path_after_parent_rename(primary: str) -> str:
    """项目根目录从 .../zpy3/grpo/ 迁到 .../zpy3/vllm/ 后，sanitize 文件名会从 zpy3.grpo 变为 zpy3.vllm。

    若无法重命名磁盘上旧产物（只读/权限），仍可能保留带 zpy3.grpo 的文件名；读文件时回退到旧名。
    """
    if os.path.isfile(primary):
        return primary
    if "zpy3.vllm." not in primary:
        return primary
    legacy = primary.replace("zpy3.vllm.", "zpy3.grpo.")
    return legacy if os.path.isfile(legacy) else primary
