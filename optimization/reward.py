import os
import ast
import json
import random
import argparse
import numpy as np
from scipy.stats import spearmanr
from termcolor import cprint

import optimization_config
from runtime_paths import (
    temp_data_dir,
    optimization_results_dir,
    resolve_path_after_parent_rename,
)


# ======================== Argument Parsing ========================

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--max_generation_len", type=int, default=optimization_config.max_generation_token)
    parser.add_argument("--max_len_threshold", type=int, default=optimization_config.max_len_threshold)
    parser.add_argument("--min_len_threshold", type=int, default=optimization_config.min_len_threshold)
    parser.add_argument("--separate_training", type=str2bool, default=optimization_config.separate_training)
    parser.add_argument("--enable_efficient", type=str2bool, default=optimization_config.enable_efficient)
    parser.add_argument("--post_stage", type=str2bool, default=optimization_config.post_stage)
    parser.add_argument("--use_self_bootstrap", type=str2bool, default=optimization_config.use_self_bootstrap)
    parser.add_argument("--em_iterations", type=int, default=optimization_config.em_iterations)
    parser.add_argument("--use_anchor", type=str2bool, default=optimization_config.use_anchor)
    parser.add_argument("--anchor_strength", type=float, default=optimization_config.anchor_strength)
    parser.add_argument("--min_weight_threshold", type=float, default=optimization_config.min_weight_threshold)
    parser.add_argument("--sb_use_length_reg", type=str2bool, default=optimization_config.sb_use_length_reg)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default=getattr(optimization_config, "reward_mode", "gt_based"),
    )
    parser.add_argument(
        "--pairwise_disagreement_zero_penalty",
        type=float,
        default=getattr(optimization_config, "pairwise_disagreement_zero_penalty", 0.0),
    )
    parser.add_argument(
        "--pairwise_disagreement_use_length_reg",
        type=str2bool,
        default=getattr(optimization_config, "pairwise_disagreement_use_length_reg", False),
    )
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))

reward_diagnostics_include_rollouts = getattr(
    optimization_config, "reward_diagnostics_include_rollouts", False
)
reward_diagnostics_max_rollout_problems = getattr(
    optimization_config, "reward_diagnostics_max_rollout_problems", None
)
reward_diagnostics_truncate_chars = getattr(
    optimization_config, "reward_diagnostics_truncate_chars", None
)


# ======================== Data Loading ========================

outputs_name = pretrained_model.replace("/", ".") + "-" + dataset

_td = temp_data_dir()
_rl_json = os.path.join(_td, "outputs-rl-" + outputs_name + ".json")
_rl_json = resolve_path_after_parent_rename(_rl_json)
os.makedirs(os.path.dirname(_rl_json), exist_ok=True)
with open(_rl_json, "r") as f:
    data = json.load(f)


# ======================== Utility Functions ========================

def normalize_reward(reward_arr):
    if np.all(reward_arr == 1) and enable_efficient:
        return reward_arr
    mean = np.mean(reward_arr)
    std = np.std(reward_arr)
    if std.item() == 0:
        return None
    return (reward_arr - mean) / std


def normalize_reward_allow_zero_std(reward_arr):
    """Z-score; if std==0 return zeros (keeps learning signal when all raw rewards are equal)."""
    reward_arr = np.asarray(reward_arr, dtype=float)
    std = np.std(reward_arr)
    if std.item() == 0:
        return np.zeros_like(reward_arr, dtype=float)
    return (reward_arr - np.mean(reward_arr)) / std

def normalize_balance_std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0
    sum_pos = x[pos_mask].sum()
    sum_neg_abs = abs(x[neg_mask].sum())
    if sum_pos * sum_neg_abs == 0:
        return None
    scale_factor = sum_neg_abs / sum_pos
    x[pos_mask] *= scale_factor
    return x / x.std()

def length_regularize(reward_arr, response_length_list):
    reward_arr = np.sign(reward_arr)
    pos_list = np.where(reward_arr == 1)[0].tolist()
    neg_list = np.where(reward_arr == -1)[0].tolist()
    pos_response_length = np.array([response_length_list[j] for j in pos_list])
    threshold = np.median(pos_response_length).item()
    if np.sum((pos_response_length - threshold)**2) == 0:
        return normalize_balance_std(np.sign(reward_arr))
    threshold = max(min(threshold, max_len_threshold), min_len_threshold)
    length_reg_reward = np.zeros(len(reward_arr), float)
    length_reg_reward[pos_list] = - pos_response_length + threshold
    length_reg_reward[neg_list] = np.min(length_reg_reward).copy()
    length_reg_reward = normalize_balance_std(length_reg_reward)
    return length_reg_reward

def _truncate_text(s, max_chars):
    if s is None:
        return ""
    s = str(s)
    if max_chars is None or max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 40] + f"\n... [truncated, total_chars={len(s)}]"


def build_rollout_debug(
    item,
    full_table,
    all_test_table_i,
    all_case_table_i,
    t,
    *,
    M=None,
    pass_rate_generated=None,
    q=None,
    w=None,
    e=None,
    code_reward_raw=None,
    case_reward_raw=None,
):
    """Full prompt + generations + execution matrix for debugging reward design."""
    codes = item.get("full_code_generation") or []
    tests = item.get("full_case_generation") or []
    tc = reward_diagnostics_truncate_chars
    code_lens = item.get("code_response_length") or []
    test_lens = item.get("case_response_length") or []
    blob = {
        "question": _truncate_text(item.get("question", ""), tc),
        "code_generation_prompt": _truncate_text(item.get("code_generation_prompt", ""), tc),
        "case_generation_prompt": _truncate_text(item.get("case_generation_prompt", ""), tc),
        "codes": [_truncate_text(c, tc) for c in codes],
        "generated_tests": [_truncate_text(x, tc) for x in tests],
        "code_response_length": [int(x) for x in code_lens],
        "case_response_length": [int(x) for x in test_lens],
        "matrix_layout": (
            "matrix_all_tests[code_i][col]: cols [0..num_gt_test-1] = hidden GT tests, "
            "then generated tests in generation order"
        ),
        "num_gt_test": int(t),
        "matrix_shape": [int(full_table.shape[0]), int(full_table.shape[1])],
        "matrix_all_tests": full_table.astype(int).tolist(),
    }
    if M is not None and M.size > 0:
        blob["matrix_generated_tests_only"] = M.astype(int).tolist()
    if pass_rate_generated is not None:
        blob["pass_rate_generated_tests"] = [
            round(float(x), 6) for x in np.asarray(pass_rate_generated).tolist()
        ]
    if q is not None:
        blob["em_q"] = [round(float(x), 6) for x in np.asarray(q).tolist()]
    if w is not None:
        blob["em_w"] = [round(float(x), 6) for x in np.asarray(w).tolist()]
    if e is not None:
        blob["em_e"] = [round(float(x), 6) for x in np.asarray(e).tolist()]
    if code_reward_raw is not None:
        blob["code_reward_raw"] = [
            round(float(x), 6) for x in np.asarray(code_reward_raw).tolist()
        ]
    if case_reward_raw is not None:
        blob["case_reward_raw"] = [
            round(float(x), 6) for x in np.asarray(case_reward_raw).tolist()
        ]
    return blob


def build_group_entry(prompt, responses_raw, response_lengths, rewards, max_gen_len):
    """Pack responses and rewards into the grouped GRPO format."""
    group_entry = {"prompt": prompt, "responses": [], "rewards": []}
    for j, reward_j in enumerate(rewards.tolist()):
        if response_lengths[j] < max_gen_len:
            resp = responses_raw[j] + "<|im_end|>"
        else:
            resp = responses_raw[j]
        group_entry["responses"].append(resp)
        group_entry["rewards"].append(reward_j)
    if len(group_entry["responses"]) > 0:
        return group_entry
    return None


# ======================== EM Self-Bootstrap Functions ========================

def em_estimate(M, anchor_scores=None, n_iter=3, alpha=0.3):
    """EM iteration to jointly estimate code quality and test reliability.

    Key improvement: only use discriminative tests (0 < pass_rate < 1) for EM,
    as uniform tests provide zero information about code quality differences.

    Args:
        M: (k_code, k_case) bool matrix — generated tests only
        anchor_scores: (k_code,) optional, public-example pass rates
        n_iter: EM iterations
        alpha: anchor blending strength

    Returns:
        q: (k_code,) code quality scores in [0, 1]
        w: (k_case,) test reliability weights (0 for uniform tests)
        e: (k_case,) expected test behavior (0 or 1)
    """
    k_code, k_case = M.shape
    M_f = M.astype(float)
    pass_rate = M_f.mean(axis=0)

    disc_mask = (pass_rate > 0.0) & (pass_rate < 1.0)
    n_disc = disc_mask.sum()

    e = (pass_rate > 0.5).astype(float)
    w = np.zeros(k_case)
    if n_disc > 0:
        w[disc_mask] = np.maximum(pass_rate[disc_mask], 1.0 - pass_rate[disc_mask])

    if anchor_scores is not None:
        q = alpha * anchor_scores + (1.0 - alpha) * 0.5
    else:
        q = np.full(k_code, 0.5)

    if n_disc == 0:
        return q, w, e

    for _ in range(n_iter):
        agreement = np.array([np.sum(w * (M_f[i, :] == e)) for i in range(k_code)])
        w_sum = np.sum(w)
        if w_sum > 0:
            q_new = agreement / w_sum
        else:
            q_new = q
        if anchor_scores is not None:
            q = alpha * anchor_scores + (1.0 - alpha) * q_new
        else:
            q = q_new

        threshold = np.median(q)
        good_mask = q > threshold
        bad_mask = q <= threshold

        if good_mask.sum() == 0 or bad_mask.sum() == 0:
            break

        for j in range(k_case):
            if not disc_mask[j]:
                w[j] = 0.0
                continue
            good_pass = M_f[good_mask, j].mean()
            bad_pass = M_f[bad_mask, j].mean()
            w[j] = abs(good_pass - bad_pass)
            e[j] = round(good_pass)

    return q, w, e


def compute_sb_code_reward(M, q, w, e):
    """Self-bootstrapped code reward: weighted agreement with expected test behavior.

    Returns raw (un-normalized) reward to let the caller decide normalization.
    """
    k_code = M.shape[0]
    M_f = M.astype(float)
    w_sum = np.sum(w)
    if w_sum == 0:
        return None
    code_reward = np.array([np.sum(w * (M_f[i, :] == e)) / w_sum for i in range(k_code)])
    return code_reward


def compute_sb_case_reward(M, q):
    """Self-bootstrapped test reward based on discriminability.

    Uses code quality scores directly instead of requiring a strict good/bad
    partition. For each test j, compute Pearson correlation between q and M[:,j].
    A good test should positively correlate with code quality.
    """
    k_code, k_case = M.shape
    M_f = M.astype(float)
    pass_rate = M_f.mean(axis=0)

    if q.std() < 1e-6:
        return None

    case_reward = np.zeros(k_case)
    for j in range(k_case):
        if pass_rate[j] == 0.0 or pass_rate[j] == 1.0:
            case_reward[j] = 0.0
            continue
        col = M_f[:, j]
        col_std = col.std()
        if col_std < 1e-8:
            case_reward[j] = 0.0
            continue
        corr = np.corrcoef(q, col)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        case_reward[j] = corr

    return case_reward


# ======================== GT-Based Reward (Fallback) ========================

def compute_gt_code_reward(all_test_table):
    """Original CURE code reward: mean GT test pass rate per code."""
    return np.mean(all_test_table, axis=1)

def compute_gt_case_reward(all_test_table, all_case_table, post_stage_flag):
    """Original CURE case reward with discriminability scaling."""
    correct_code_list = np.where(all_test_table.all(axis=1))[0].tolist()
    if len(correct_code_list) == 0:
        return None

    correct_code_table = all_case_table[correct_code_list, :].copy()
    good_case_indices = np.where(np.all(correct_code_table, axis=0))[0].tolist()
    reward_sign = -np.ones(correct_code_table.shape[1], dtype=float)
    reward_sign[good_case_indices] = 1
    case_reward = reward_sign.copy()

    wrong_code_list = [j for j in range(all_case_table.shape[0]) if j not in correct_code_list]
    if len(wrong_code_list) > 0:
        reward_scale = np.ones(correct_code_table.shape[1], dtype=float)
        correct_case_list = np.where(correct_code_table.all(axis=0))[0].tolist()
        wrong_case_list = [j for j in range(all_case_table.shape[1]) if j not in correct_case_list]
        if len(correct_case_list):
            wc_cc = all_case_table[wrong_code_list, :][:, correct_case_list].copy()
            if not post_stage_flag:
                mean_p01 = np.mean(~wc_cc, 0)
            else:
                mean_p01 = (~np.any(wc_cc, axis=0)).astype(float)
            reward_scale[correct_case_list] *= mean_p01
        if len(wrong_case_list):
            wc_wc = all_case_table[wrong_code_list, :][:, wrong_case_list].copy()
            if not post_stage_flag:
                mean_p00 = np.mean(wc_wc, 0)
            else:
                mean_p00 = np.any(wc_wc, axis=0).astype(float)
            reward_scale[wrong_case_list] *= mean_p00
        case_reward = case_reward * reward_scale

    return case_reward


# ======================== Pairwise Disagreement (no EM, no GT for case) ========================

def normalize_execution_output_for_disagreement(raw):
    """Align with execute.test_if_eq whitespace handling; keep execution errors distinct."""
    if raw is None:
        return "<NONE>"
    s = str(raw)
    if s.startswith("Execution Error:"):
        return s.strip()
    return " ".join(s.split())


def column_pairwise_disagreement(norm_outputs):
    """Fraction of code pairs that disagree on this test (doc: 2/(n(n-1)) * sum 1[out_i != out_j])."""
    n = len(norm_outputs)
    if n < 2:
        return 0.0
    disagree = 0
    pairs = 0
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            pairs += 1
            if norm_outputs[i1] != norm_outputs[i2]:
                disagree += 1
    return disagree / pairs


def build_normalized_exe_matrix_generated(exe_grid, t, n_code, k_gen):
    """exe_grid[row][col] strings for full test list; return (n_code, k_gen) normalized strings."""
    mat = []
    for i in range(n_code):
        row = []
        for j in range(k_gen):
            col_idx = t + j
            raw = exe_grid[i][col_idx] if col_idx < len(exe_grid[i]) else None
            row.append(normalize_execution_output_for_disagreement(raw))
        mat.append(row)
    return np.array(mat, dtype=object)


def compute_pd_case_rewards(norm_matrix, zero_penalty=0.0):
    """Per generated test column: pairwise output disagreement in [0, 1]."""
    n_code, k_gen = norm_matrix.shape
    if k_gen == 0 or n_code < 2:
        return None
    rewards = np.zeros(k_gen, dtype=float)
    for j in range(k_gen):
        col = [norm_matrix[i, j] for i in range(n_code)]
        rj = column_pairwise_disagreement(col)
        if zero_penalty > 0.0 and rj == 0.0:
            rj -= zero_penalty
        rewards[j] = rj
    return rewards


def compute_pd_code_rewards(norm_matrix):
    """Per code: mean over tests of fraction of other codes with different output (doc auxiliary code signal)."""
    n_code, k_gen = norm_matrix.shape
    if n_code < 2 or k_gen == 0:
        return None
    out = np.zeros(n_code, dtype=float)
    for i in range(n_code):
        acc = 0.0
        for j in range(k_gen):
            col = norm_matrix[:, j]
            diff = sum(1 for i2 in range(n_code) if i2 != i and col[i] != col[i2])
            acc += diff / (n_code - 1)
        out[i] = acc / k_gen
    return out


# ======================== Main Reward Computation ========================

code_data = []
case_data = []
raw_code_rewards_all = []
raw_case_rewards_all = []

total_problems = 0
skipped_problems = 0
gt_correlations = []
per_problem_diag = []

_rollout_remaining = [0]
if reward_diagnostics_include_rollouts:
    _rollout_remaining[0] = (
        10**9
        if reward_diagnostics_max_rollout_problems is None
        else max(0, int(reward_diagnostics_max_rollout_problems))
    )


def _try_attach_rollout_debug(prob_diag, item, full_table, att, act, t, **extra):
    if _rollout_remaining[0] <= 0:
        return
    blob = build_rollout_debug(item, full_table, att, act, t, **extra)
    _metric_keys = (
        "status",
        "avg_code_len",
        "avg_test_len",
        "n_disc_tests",
        "n_total_gen_tests",
        "q_mean",
        "q_std",
        "w_max",
        "w_nonzero",
        "code_reward_valid",
        "case_reward_valid",
        "code_reward_raw_std",
        "case_reward_raw_std",
        "gt_corr",
        "code_reward_skipped",
        "gt_code_acc",
        "gt_code_reward_mean",
        "pd_case_mean_raw",
        "pd_code_source",
    )
    blob["per_problem_metrics"] = {k: prob_diag[k] for k in _metric_keys if k in prob_diag}
    prob_diag["rollout_debug"] = blob
    _rollout_remaining[0] -= 1


for i in range(len(data)):
    if data[i]["all_case_bool_table"] is None:
        continue

    total_problems += 1
    t = data[i]["num_ground_truth_test"]
    full_table = np.array(data[i]["all_case_bool_table"])
    all_test_table_i = full_table[:, :t].copy()
    all_case_table_i = full_table[:, t:].copy()

    prob_diag = {
        "idx": i,
        "question_preview": data[i].get("question", "")[:80],
        "num_gt_test": t,
        "k_code": full_table.shape[0],
        "k_gen_test": all_case_table_i.shape[1],
        "gt_code_acc": float(all_test_table_i.all(axis=1).mean()) if t > 0 else None,
        "avg_code_len": round(float(np.mean(data[i].get("code_response_length", [0]))), 1),
        "avg_test_len": round(float(np.mean(data[i].get("case_response_length", [0]))), 1),
    }

    if reward_mode == "pairwise_disagreement":
        n_code = full_table.shape[0]
        k_gen = all_case_table_i.shape[1]
        exe_grid = data[i].get("all_case_exe_results")

        def _pd_valid_exe():
            if not exe_grid or len(exe_grid) != n_code:
                return False
            need_cols = t + k_gen
            for row in exe_grid:
                if not row or len(row) < need_cols:
                    return False
            return True

        if k_gen == 0 or not _pd_valid_exe():
            skipped_problems += 1
            prob_diag["status"] = "skipped_pd_missing_exe_or_empty_gen"
            _try_attach_rollout_debug(
                prob_diag,
                data[i],
                full_table,
                all_test_table_i,
                all_case_table_i,
                t,
                M=all_case_table_i if all_case_table_i.size else None,
            )
            per_problem_diag.append(prob_diag)
            continue

        if n_code < 2:
            skipped_problems += 1
            prob_diag["status"] = "skipped_pd_n_code_lt_2"
            _try_attach_rollout_debug(
                prob_diag,
                data[i],
                full_table,
                all_test_table_i,
                all_case_table_i,
                t,
                M=all_case_table_i if all_case_table_i.size else None,
            )
            per_problem_diag.append(prob_diag)
            continue

        norm_matrix = build_normalized_exe_matrix_generated(exe_grid, t, n_code, k_gen)

        case_reward_raw = None
        if n_code >= 2:
            case_reward_raw = compute_pd_case_rewards(
                norm_matrix, zero_penalty=pairwise_disagreement_zero_penalty
            )
        prob_diag["case_reward_valid"] = case_reward_raw is not None
        if case_reward_raw is not None:
            prob_diag["pd_case_mean_raw"] = round(float(np.mean(case_reward_raw)), 4)
            prob_diag["case_reward_raw_std"] = round(float(np.std(case_reward_raw)), 4)

        if t > 0:
            code_reward_raw = compute_gt_code_reward(all_test_table_i)
            prob_diag["pd_code_source"] = "gt_mean_pass"
        else:
            code_reward_raw = compute_pd_code_rewards(norm_matrix) if n_code >= 2 else None
            prob_diag["pd_code_source"] = "pairwise_disagreement"

        prob_diag["code_reward_valid"] = code_reward_raw is not None

        if code_reward_raw is not None:
            cr_std = float(np.std(code_reward_raw))
            prob_diag["code_reward_raw_std"] = round(cr_std, 4)
            raw_code_rewards_all.extend(np.asarray(code_reward_raw, dtype=float).tolist())

            if t > 0:
                code_reward_norm = normalize_reward(code_reward_raw)
            else:
                code_reward_norm = normalize_reward_allow_zero_std(code_reward_raw)

            if code_reward_norm is not None:
                if enable_efficient and pairwise_disagreement_use_length_reg:
                    code_reward_final = length_regularize(
                        code_reward_norm, data[i]["code_response_length"]
                    )
                else:
                    code_reward_final = code_reward_norm

                if code_reward_final is not None:
                    entry = build_group_entry(
                        data[i]["code_generation_prompt"],
                        data[i]["full_code_generation"],
                        data[i]["code_response_length"],
                        code_reward_final,
                        max_generation_len,
                    )
                    if entry:
                        code_data.append(entry)

            if t > 0:
                pd_cr = compute_pd_code_rewards(norm_matrix)
                gt_cr = np.asarray(code_reward_raw, dtype=float)
                if (
                    pd_cr is not None
                    and gt_cr.std() > 1e-6
                    and pd_cr.std() > 1e-6
                ):
                    corr, _ = spearmanr(gt_cr, pd_cr)
                    if not np.isnan(corr):
                        gt_correlations.append(corr)
                        prob_diag["gt_corr"] = round(float(corr), 4)

        if case_reward_raw is not None:
            raw_case_rewards_all.extend(case_reward_raw.tolist())
            case_reward_norm = normalize_reward_allow_zero_std(case_reward_raw)
            if enable_efficient and pairwise_disagreement_use_length_reg:
                case_reward_final = length_regularize(
                    case_reward_norm, data[i]["case_response_length"]
                )
            else:
                case_reward_final = case_reward_norm

            if case_reward_final is not None:
                entry = build_group_entry(
                    data[i]["case_generation_prompt"],
                    data[i]["full_case_generation"],
                    data[i]["case_response_length"],
                    case_reward_final,
                    max_generation_len,
                )
                if entry:
                    case_data.append(entry)

        prob_diag["status"] = "used"
        pr_gen = (
            all_case_table_i.astype(float).mean(axis=0)
            if all_case_table_i.size
            else None
        )
        _try_attach_rollout_debug(
            prob_diag,
            data[i],
            full_table,
            all_test_table_i,
            all_case_table_i,
            t,
            M=all_case_table_i if all_case_table_i.size else None,
            pass_rate_generated=pr_gen,
            code_reward_raw=code_reward_raw,
            case_reward_raw=case_reward_raw,
        )
        per_problem_diag.append(prob_diag)

    elif use_self_bootstrap:
        M = all_case_table_i

        if M.shape[0] == 0 or M.shape[1] == 0:
            skipped_problems += 1
            prob_diag["status"] = "skipped_empty"
            _try_attach_rollout_debug(
                prob_diag,
                data[i],
                full_table,
                all_test_table_i,
                all_case_table_i,
                t,
                M=M if M.size else None,
                pass_rate_generated=None,
            )
            per_problem_diag.append(prob_diag)
            continue

        pass_rate = M.astype(float).mean(axis=0)
        n_disc = int(np.sum((pass_rate > 0.0) & (pass_rate < 1.0)))
        prob_diag["n_disc_tests"] = n_disc
        prob_diag["n_total_gen_tests"] = int(M.shape[1])

        if n_disc == 0:
            skipped_problems += 1
            prob_diag["status"] = "skipped_no_disc"
            _try_attach_rollout_debug(
                prob_diag,
                data[i],
                full_table,
                all_test_table_i,
                all_case_table_i,
                t,
                M=M,
                pass_rate_generated=pass_rate,
            )
            per_problem_diag.append(prob_diag)
            continue

        anchor_scores = None
        if use_anchor:
            n_example = len(data[i].get("example_input", []))
            n_anchor = min(n_example, t)
            if n_anchor > 0:
                anchor_scores = full_table[:, :n_anchor].astype(float).mean(axis=1)

        q, w, e = em_estimate(M, anchor_scores, n_iter=em_iterations, alpha=anchor_strength)

        prob_diag["q_mean"] = round(float(q.mean()), 4)
        prob_diag["q_std"] = round(float(q.std()), 4)
        prob_diag["w_max"] = round(float(np.max(w)), 4)
        prob_diag["w_nonzero"] = int(np.sum(w > 1e-4))

        # --- Code reward ---
        code_reward_raw = compute_sb_code_reward(M, q, w, e)
        prob_diag["code_reward_valid"] = code_reward_raw is not None

        if code_reward_raw is not None:
            cr_std = code_reward_raw.std()
            prob_diag["code_reward_raw_std"] = round(float(cr_std), 4)
            raw_code_rewards_all.extend(code_reward_raw.tolist())

            if cr_std < 1e-6:
                prob_diag["code_reward_skipped"] = "std_zero"
            else:
                code_reward_norm = normalize_reward(code_reward_raw)
                if code_reward_norm is not None:
                    if enable_efficient and sb_use_length_reg:
                        code_reward_final = length_regularize(code_reward_norm, data[i]["code_response_length"])
                    else:
                        code_reward_final = code_reward_norm

                    if code_reward_final is not None:
                        entry = build_group_entry(
                            data[i]["code_generation_prompt"],
                            data[i]["full_code_generation"],
                            data[i]["code_response_length"],
                            code_reward_final, max_generation_len,
                        )
                        if entry:
                            code_data.append(entry)

            if t > 0:
                gt_cr = compute_gt_code_reward(all_test_table_i)
                if code_reward_raw is not None and gt_cr.std() > 1e-6 and cr_std > 1e-6:
                    corr, _ = spearmanr(gt_cr, code_reward_raw)
                    if not np.isnan(corr):
                        gt_correlations.append(corr)
                        prob_diag["gt_corr"] = round(float(corr), 4)

        # --- Case reward ---
        case_reward_raw = compute_sb_case_reward(M, q)
        prob_diag["case_reward_valid"] = case_reward_raw is not None

        if case_reward_raw is not None:
            cs_std = case_reward_raw.std()
            prob_diag["case_reward_raw_std"] = round(float(cs_std), 4)

            if cs_std < 1e-6:
                prob_diag["case_reward_skipped"] = "std_zero"
            else:
                raw_case_rewards_all.extend(case_reward_raw.tolist())
                case_reward_norm = normalize_reward(case_reward_raw)
                if case_reward_norm is not None:
                    if enable_efficient and sb_use_length_reg:
                        case_reward_final = length_regularize(case_reward_norm, data[i]["case_response_length"])
                    else:
                        case_reward_final = case_reward_norm

                    if case_reward_final is not None:
                        entry = build_group_entry(
                            data[i]["case_generation_prompt"],
                            data[i]["full_case_generation"],
                            data[i]["case_response_length"],
                            case_reward_final, max_generation_len,
                        )
                        if entry:
                            case_data.append(entry)

        prob_diag["status"] = "used"
        _try_attach_rollout_debug(
            prob_diag,
            data[i],
            full_table,
            all_test_table_i,
            all_case_table_i,
            t,
            M=M,
            pass_rate_generated=pass_rate,
            q=q,
            w=w,
            e=e,
            code_reward_raw=code_reward_raw,
            case_reward_raw=case_reward_raw,
        )
        per_problem_diag.append(prob_diag)

    else:
        code_reward = compute_gt_code_reward(all_test_table_i)
        raw_code_rewards_all.extend(code_reward.tolist())
        prob_diag["gt_code_reward_mean"] = round(float(code_reward.mean()), 4)
        code_reward = normalize_reward(code_reward)
        if code_reward is not None:
            if enable_efficient:
                code_reward = length_regularize(code_reward, data[i]["code_response_length"])
            if code_reward is not None:
                entry = build_group_entry(
                    data[i]["code_generation_prompt"],
                    data[i]["full_code_generation"],
                    data[i]["code_response_length"],
                    code_reward, max_generation_len,
                )
                if entry:
                    code_data.append(entry)

        case_reward = compute_gt_case_reward(all_test_table_i, all_case_table_i, post_stage)
        if case_reward is not None:
            raw_case_rewards_all.extend(case_reward.tolist())
            case_reward = normalize_reward(case_reward)
            if case_reward is not None:
                if enable_efficient:
                    case_reward = length_regularize(case_reward, data[i]["case_response_length"])
                if case_reward is not None:
                    entry = build_group_entry(
                        data[i]["case_generation_prompt"],
                        data[i]["full_case_generation"],
                        data[i]["case_response_length"],
                        case_reward, max_generation_len,
                    )
                    if entry:
                        case_data.append(entry)

        prob_diag["status"] = "used"
        pr_gen = (
            all_case_table_i.astype(float).mean(axis=0)
            if all_case_table_i.size
            else None
        )
        _try_attach_rollout_debug(
            prob_diag,
            data[i],
            full_table,
            all_test_table_i,
            all_case_table_i,
            t,
            M=all_case_table_i if all_case_table_i.size else None,
            pass_rate_generated=pr_gen,
        )
        per_problem_diag.append(prob_diag)


# ======================== Output Training Data ========================

final_data = code_data + case_data
random.shuffle(final_data)

if separate_training == False:
    with open(os.path.join(_td, "rl_data.json"), "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
else:
    with open(os.path.join(_td, "rl_code_data.json"), "w", encoding="utf-8") as f:
        json.dump(code_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(_td, "rl_case_data.json"), "w", encoding="utf-8") as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)


# ======================== Diagnostic Logging ========================

raw_code_arr = np.array(raw_code_rewards_all) if raw_code_rewards_all else np.array([0.0])
raw_case_arr = np.array(raw_case_rewards_all) if raw_case_rewards_all else np.array([0.0])

if reward_mode == "pairwise_disagreement":
    _reward_mode_logged = "pairwise_disagreement"
elif use_self_bootstrap:
    _reward_mode_logged = "self_bootstrap"
else:
    _reward_mode_logged = "gt_based"

step_summary = {
    "reward_mode": _reward_mode_logged,
    "rollout_debug_meta": {
        "enabled": bool(reward_diagnostics_include_rollouts),
        "max_problems_per_step": reward_diagnostics_max_rollout_problems,
        "truncate_chars": reward_diagnostics_truncate_chars,
        "slots_remaining_after_step": _rollout_remaining[0],
    },
    "code_reward": {
        "mean": round(float(raw_code_arr.mean()), 4),
        "std": round(float(raw_code_arr.std()), 4),
        "num_groups": len(code_data),
        "num_samples": len(raw_code_rewards_all),
    },
    "case_reward": {
        "mean": round(float(raw_case_arr.mean()), 4),
        "std": round(float(raw_case_arr.std()), 4),
        "num_groups": len(case_data),
        "num_samples": len(raw_case_rewards_all),
    },
    "problems": {
        "total": total_problems,
        "used": total_problems - skipped_problems,
        "skipped": skipped_problems,
    },
    "per_problem": per_problem_diag,
}

if gt_correlations:
    step_summary["gt_correlation"] = {
        "mean_spearman": round(float(np.mean(gt_correlations)), 4),
        "all_values": [round(c, 4) for c in gt_correlations],
        "n": len(gt_correlations),
    }

with open(os.path.join(_td, "reward_diagnostics.json"), "w") as f:
    json.dump(step_summary, f, indent=2)

# text log (appended, same format as before for log parsing compatibility)
_res_dir = optimization_results_dir()
_reward_txt = os.path.join(_res_dir, "results-" + outputs_name + ".txt")
os.makedirs(os.path.dirname(_reward_txt), exist_ok=True)
with open(_reward_txt, "a") as f:
    def save_and_print(text):
        cprint(text, color="cyan")
        f.write(text + "\n")

    save_and_print(f"reward_mode: {step_summary['reward_mode']}")
    save_and_print(
        f"estimated_code_reward: mean={raw_code_arr.mean():.4f}, std={raw_code_arr.std():.4f}, "
        f"num_groups={len(code_data)}, num_samples={len(raw_code_rewards_all)}"
    )
    save_and_print(
        f"estimated_case_reward: mean={raw_case_arr.mean():.4f}, std={raw_case_arr.std():.4f}, "
        f"num_groups={len(case_data)}, num_samples={len(raw_case_rewards_all)}"
    )
    # 与 self_bootstrap 实验共用同一行格式，便于 results-*.txt 解析（clean_log 等）
    if use_self_bootstrap or reward_mode == "pairwise_disagreement":
        save_and_print(
            f"self_bootstrap_stats: total_problems={total_problems}, "
            f"skipped={skipped_problems}, used={total_problems - skipped_problems}"
        )
        if gt_correlations:
            save_and_print(
                f"gt_correlation(code_reward): mean_spearman={np.mean(gt_correlations):.4f}, n={len(gt_correlations)}"
            )
        else:
            save_and_print("gt_correlation(code_reward): no valid correlations computed")
