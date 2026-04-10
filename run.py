import json
import os
import sys
import shutil
import subprocess
import copy
from datetime import datetime
from termcolor import cprint

from optimization import optimization_config


# 写入 experiments/.../optimization/diagnostics/step_*.json 时的字段说明（与训练数据无关）
STEP_METRICS_HELP_ZH = {
    "sampling": {
        "mean_code_response_tokens": "本步 vLLM 生成代码的平均 token 数（与日志 code response length 一致）。",
        "mean_case_response_tokens": "本步生成单测相关文本的平均 token 数（与日志 case response length 一致）。",
    },
    "execution": {
        "code_acc": "题目维度：至少有一份生成代码在全部 GT 私有测试上通过的题目占比。",
        "code_accumulate_acc": "所有「生成代码 × GT 测试」对上，执行输出与期望一致的比例。",
        "case_acc": "仅在存在「全对 GT 的代码」的题目上统计：生成测试能区分/匹配好代码的题目级比例。",
        "case_accumulate_acc": "在好代码与生成测试的交叉上，执行是否一致的比例。",
        "p_01_as_logged": "日志中 p_01 行打印的值，等于 1−(内部 p_01)：错误代码在「好测试」上仍通过的比例相关量。",
        "p_00": "错误代码与错误测试组合上，执行仍判为通过的比例。",
        "bon_by_scale": "Best-of-N：从前 num_code 份代码、num_gen_tests 份生成测试中按生成测试得分选最优代码，再在 GT 上算的 acc / accumulate_acc。",
    },
    "reward": {
        "reward_mode": "gt_based | self_bootstrap | pairwise_disagreement（见 optimization_config.reward_mode）。",
        "code_reward / case_reward": "本步写入 RL json 的原始 reward 在组内归一化前的统计（mean/std/组数/样本数）。",
        "problems": "本步参与 reward 的题目总数、实际使用数、跳过数（自举时低信心等会 skip）。",
        "gt_correlation": "自举得到的 code reward 与 GT-based code reward 的 Spearman 相关（仅诊断，不参与训练）。",
        "rollout_debug_meta": "是否在 reward_diagnostics 里嵌入大段 rollout（默认关；开则 temp 文件会很大）。",
    },
}


def _clean_env():
    """Return a copy of os.environ with stale CUDA/NCCL IPC state removed.

    After a DeepSpeed/Ray training subprocess exits, shared-memory handles
    and NCCL rendezvous files may linger.  Passing a clean environment to
    the next subprocess prevents vLLM workers from inheriting a polluted
    CUDA context (Error 304 / cudaGetDeviceCount failure).
    """
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("NCCL_", "MASTER_ADDR", "MASTER_PORT",
                           "RANK", "LOCAL_RANK", "WORLD_SIZE",
                           "GROUP_RANK", "LOCAL_WORLD_SIZE")):
            del env[key]
    return env


# if you are the first time to train the model, set this to be True.
# if you have stopped the process and want to keep training, simply set this to be False and run this script.
start_from_scratch = True


eval_interval = optimization_config.eval_interval
save_interval = optimization_config.save_interval
total_steps = optimization_config.total_steps
pretrain_model = optimization_config.pretrained_model
eval_dataset = optimization_config.eval_dataset
train_dataset = optimization_config.train_dataset
gpu_groups = optimization_config.gpu_groups
eval_k_code = optimization_config.eval_k_code
eval_k_case = optimization_config.eval_k_case
eval_scale_tuple_list = optimization_config.eval_scale_tuple_list
eval_num_chunks = optimization_config.eval_num_chunks
eval_no_example = optimization_config.eval_no_example
eval_max_test = optimization_config.eval_max_test


# ===================== Experiment Directory Setup =====================

def _build_exp_name():
    raw = optimization_config.exp_name
    if raw:
        return raw
    model_short = os.path.basename(pretrain_model.rstrip("/"))
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{ts}_{model_short}_{train_dataset}"

exp_name = _build_exp_name()
exp_dir = os.path.join("experiments", exp_name)
exp_dir_abs = os.path.abspath(exp_dir)

# 本实验独占：temp、optimization 文本结果、评测、可训练 ckpt 均写在该目录下，避免多任务共用 optimization/temp_data
os.environ["CURE_EXPERIMENT_DIR"] = exp_dir_abs

_optimized_name = optimization_config.optimized_model_name
os.makedirs(os.path.join(exp_dir_abs, "temp_data"), exist_ok=True)
os.makedirs(os.path.join(exp_dir_abs, "optimization", "results"), exist_ok=True)
os.makedirs(os.path.join(exp_dir_abs, "optimization", "tb_logs"), exist_ok=True)
os.makedirs(os.path.join(exp_dir_abs, "optimization", "ckpt"), exist_ok=True)
os.makedirs(os.path.join(exp_dir_abs, "optimization", "diagnostics"), exist_ok=True)
os.makedirs(os.path.join(exp_dir_abs, "evaluation", "results"), exist_ok=True)
os.makedirs(os.path.join(exp_dir_abs, "evaluation", "temp_data"), exist_ok=True)
# 权重目录为 <exp>/ckpt/<optimized_model_name>/，由 train 的 save_model(join(save_path, name)) 创建，勿提前建错层级
os.makedirs(os.path.join(exp_dir_abs, "ckpt"), exist_ok=True)

model = os.path.join(exp_dir_abs, "ckpt", _optimized_name)
if start_from_scratch is False:
    pretrain_model = model

exp_tb_dir = os.path.abspath(os.path.join(exp_dir, "optimization", "tb_logs"))

shutil.copy2("optimization/optimization_config.py",
             os.path.join(exp_dir, "config_snapshot.py"))

cprint(f"{'='*60}", color="yellow")
cprint(f"  Experiment: {exp_name}", color="yellow")
cprint(f"  Directory:  {os.path.abspath(exp_dir)}", color="yellow")
cprint(f"{'='*60}", color="yellow")


def _sanitize_model_path_for_results(m: str) -> str:
    """与 sample.py / execute.py / reward.py 中 outputs_name 一致：路径分隔符换成 ."""
    return m.replace("/", ".")


def _sanitize_variants_for_results(m: str):
    """目录 grpo→vllm 后 sanitize 串从 zpy3.grpo 变为 zpy3.vllm；保留旧串以免误删仍带旧文件名的结果。"""
    s = _sanitize_model_path_for_results(m)
    out = {s}
    if "zpy3.vllm." in s:
        out.add(s.replace("zpy3.vllm.", "zpy3.grpo."))
    if "zpy3.grpo." in s:
        out.add(s.replace("zpy3.grpo.", "zpy3.vllm."))
    return out


def _expected_optimization_result_basenames():
    """本实验会读写的 optimization/results 下 txt 基名（基座与 ckpt 各一对，可能重合）。"""
    basenames = set()
    for m in (pretrain_model, model):
        if not m:
            continue
        for s in _sanitize_variants_for_results(m):
            basenames.add(f"results-rl-{s}-{train_dataset}.txt")
            basenames.add(f"results-{s}-{train_dataset}.txt")
    return basenames


def _expected_evaluation_result_basenames():
    """训练过程中 eval.py（is_final_eval=False）写入的 results 文件名。"""
    basenames = set()
    for s in _sanitize_variants_for_results(model):
        basenames.add(f"results-eval-{s}-{eval_dataset}.txt")
    return basenames


def archive_optimization_results():
    """子进程已直接写入 experiments/<exp>/optimization/results/，此处仅清理非本实验命名的残留 txt。"""
    allowed = _expected_optimization_result_basenames()
    dest_dir = os.path.join(exp_dir_abs, "optimization", "results")
    os.makedirs(dest_dir, exist_ok=True)
    for name in list(os.listdir(dest_dir)):
        if not name.endswith(".txt"):
            continue
        if name not in allowed:
            try:
                os.remove(os.path.join(dest_dir, name))
            except OSError:
                pass


def archive_eval_results():
    allowed = _expected_evaluation_result_basenames()
    dest_dir = os.path.join(exp_dir_abs, "evaluation", "results")
    os.makedirs(dest_dir, exist_ok=True)
    for name in list(os.listdir(dest_dir)):
        if not name.endswith(".txt"):
            continue
        if name not in allowed:
            try:
                os.remove(os.path.join(dest_dir, name))
            except OSError:
                pass

def archive_step_summary(step):
    """将本步与日志对齐的关键指标写入 experiments/.../diagnostics/step_<step>.json（不含 per_problem 等大字段）。"""
    diag_dir = os.path.join(exp_dir_abs, "optimization", "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    help_path = os.path.join(diag_dir, "metrics_field_help_zh.json")
    if not os.path.isfile(help_path):
        with open(help_path, "w", encoding="utf-8") as hf:
            json.dump(STEP_METRICS_HELP_ZH, hf, indent=2, ensure_ascii=False)

    merged = {
        "step": int(step),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "_field_help_zh_file": "metrics_field_help_zh.json",
        "sampling": None,
        "execution": None,
        "reward": None,
    }

    sample_path = os.path.join(exp_dir_abs, "temp_data", "last_step_sample_metrics.json")
    if os.path.isfile(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            merged["sampling"] = json.load(f)

    exec_path = os.path.join(exp_dir_abs, "temp_data", "last_step_execute_metrics.json")
    if os.path.isfile(exec_path):
        with open(exec_path, "r", encoding="utf-8") as f:
            merged["execution"] = json.load(f)

    reward_path = os.path.join(exp_dir_abs, "temp_data", "reward_diagnostics.json")
    if os.path.isfile(reward_path):
        with open(reward_path, "r", encoding="utf-8") as f:
            reward_slim = json.load(f)
        reward_slim = copy.deepcopy(reward_slim)
        reward_slim.pop("per_problem", None)
        gc = reward_slim.get("gt_correlation")
        if isinstance(gc, dict) and "all_values" in gc:
            gc = dict(gc)
            gc.pop("all_values", None)
            reward_slim["gt_correlation"] = gc
        merged["reward"] = reward_slim

    out_path = os.path.join(diag_dir, f"step_{step}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

def archive_log_symlink():
    log_file = os.environ.get("CURE_LOG_FILE")
    if log_file:
        abs_log = os.path.abspath(log_file)
        link_path = os.path.join(exp_dir_abs, "train.log")
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(abs_log, link_path)


# ===================== Pipeline Steps =====================

def begin_with(file_name):
    with open(file_name, "w") as f:
        f.write("")

if start_from_scratch:
    _ores = os.path.join(exp_dir_abs, "optimization", "results")
    _eres = os.path.join(exp_dir_abs, "evaluation", "results")
    for bn in _expected_optimization_result_basenames():
        begin_with(os.path.join(_ores, bn))
    for bn in _expected_evaluation_result_basenames():
        begin_with(os.path.join(_eres, bn))

def evaluation(model, eval_dataset, gpu_groups):
    cprint(f"This is the {i}-th step for evaluation.", color = "green")
    subprocess.run(
        f'python eval.py '
        f'--pretrained_model {model} '
        f'--dataset {eval_dataset} '
        '--use_api False '
        '--exe_verbose False '
        '--is_final_eval False '
        '--single_eval False '
        f'--k_code {eval_k_code} '
        f'--k_case {eval_k_case} '
        f'--scale_tuple_list "{repr(eval_scale_tuple_list)}" '
        f'--num_chunks {eval_num_chunks} '
        f'--no_example {eval_no_example} '
        f'--max_test {eval_max_test} '
        f'--gpu_groups "{repr(gpu_groups)}" ',
        shell=True,
        cwd='evaluation',
        check=True,
        env=_clean_env(),
    )
    archive_eval_results()

def sample(model):
    cprint(f"This is the {i}-th step for sampling.", color = "green")
    subprocess.run(
        f'python sample.py '
        f'--pretrained_model {model} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=_clean_env(),
    )

def execute(model):
    cprint(f"This is the {i}-th step for execution.", color = "green")
    subprocess.run(
        f'python execute.py '
        f'--pretrained_model {model} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=_clean_env(),
    )

def assign_reward(model):
    # 显式传入，避免「以为开了 EM 但实际未传参、仅靠 default」难以核对；与 optimization_config 一致
    sb = optimization_config.use_self_bootstrap
    rm = getattr(optimization_config, "reward_mode", "gt_based")
    pd_pen = getattr(optimization_config, "pairwise_disagreement_zero_penalty", 0.0)
    pd_lr = getattr(optimization_config, "pairwise_disagreement_use_length_reg", False)
    subprocess.run(
        f'python reward.py '
        f'--pretrained_model {model} '
        f'--use_self_bootstrap {"true" if sb else "false"} '
        f'--reward_mode {rm} '
        f'--pairwise_disagreement_zero_penalty {pd_pen} '
        f'--pairwise_disagreement_use_length_reg {"true" if pd_lr else "false"} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=_clean_env(),
    )
    archive_optimization_results()

def train(model):
    cprint(f"This is the {i}-th step for training.", color = "green")
    subprocess.run(
        f'python -m train '
        f'--pretrain {model} '
        f'--step {i} '
        f'--tb_dir {exp_tb_dir} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=_clean_env(),
    )

def save(model_from, model_to):
    os.makedirs(model_to, exist_ok=True)
    subprocess.run(f"rm -rf {model_to}/*", shell=True, check=True)
    subprocess.run(f"cp -r {model_from}/* {model_to}/", shell=True, check=True)

# the first step if train from scratch
i = 0
#evaluation(pretrain_model, eval_dataset, gpu_groups)
sample(pretrain_model)
execute(pretrain_model)
assign_reward(pretrain_model)
archive_step_summary(i)
train(pretrain_model)
i += 1

# start the iterative optimization
while i <= total_steps:

    if i % eval_interval == 0:
        evaluation(model, eval_dataset, gpu_groups)
    if i % save_interval == 0:
        exp_ckpt = os.path.join(exp_dir_abs, "optimization", "ckpt", f"iter{i}")
        save(model, exp_ckpt)

    if i == total_steps:
        break

    sample(model)
    execute(model)
    assign_reward(model)
    archive_step_summary(i)
    train(model)

    i += 1

# archive final model and log
exp_final_ckpt = os.path.join(exp_dir_abs, "optimization", "ckpt", "final")
save(model, exp_final_ckpt)
archive_optimization_results()
archive_eval_results()
archive_log_symlink()

cprint(f"{'='*60}", color="yellow")
cprint(f"  Experiment finished: {exp_name}", color="yellow")
cprint(f"  All artifacts saved to: {os.path.abspath(exp_dir)}", color="yellow")
cprint(f"{'='*60}", color="yellow")






