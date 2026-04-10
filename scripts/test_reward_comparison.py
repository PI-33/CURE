#!/usr/bin/env python3
"""
Compare old vs new reward.py on existing data WITHOUT running training.
Shows side-by-side: how many groups each method produces, GT correlation, etc.
"""

import json
import sys
import os
import numpy as np
from scipy.stats import spearmanr

DATA_FILES = [
    "temp_data/outputs-rl-.mnt.shared-storage-user.ai4good2-share.models.Qwen.Qwen2.5-7B-Instruct-small_train.json",
    "temp_data/outputs-rl-.mnt.shared-storage-user.zhupengyu1.zpy3.vllm.CURE.optimization.ckpt.optimized-small_train.json",
]


def em_old(M, anchor_scores=None, n_iter=3, alpha=0.3):
    k_code, k_case = M.shape
    M_f = M.astype(float)
    pass_rate = M_f.mean(axis=0)
    e = (pass_rate > 0.5).astype(float)
    w = np.maximum(pass_rate, 1.0 - pass_rate)
    q = alpha * anchor_scores + (1.0 - alpha) * 0.5 if anchor_scores is not None else np.full(k_code, 0.5)
    for _ in range(n_iter):
        agreement = np.array([np.sum(w * (M_f[i, :] == e)) for i in range(k_code)])
        w_sum = np.sum(w)
        q_new = agreement / w_sum if w_sum > 0 else q
        q = alpha * anchor_scores + (1.0 - alpha) * q_new if anchor_scores is not None else q_new
        threshold = np.median(q)
        good_mask = q > threshold
        bad_mask = q <= threshold
        if good_mask.sum() == 0 or bad_mask.sum() == 0:
            break
        for j in range(k_case):
            good_pass = M_f[good_mask, j].mean()
            bad_pass = M_f[bad_mask, j].mean()
            w[j] = abs(good_pass - bad_pass) * max(pass_rate[j], 1.0 - pass_rate[j])
            e[j] = round(good_pass)
    return q, w, e


def em_new(M, anchor_scores=None, n_iter=3, alpha=0.3):
    k_code, k_case = M.shape
    M_f = M.astype(float)
    pass_rate = M_f.mean(axis=0)
    disc_mask = (pass_rate > 0.0) & (pass_rate < 1.0)
    e = (pass_rate > 0.5).astype(float)
    w = np.zeros(k_case)
    if disc_mask.sum() > 0:
        w[disc_mask] = np.maximum(pass_rate[disc_mask], 1.0 - pass_rate[disc_mask])
    q = alpha * anchor_scores + (1.0 - alpha) * 0.5 if anchor_scores is not None else np.full(k_code, 0.5)
    if disc_mask.sum() == 0:
        return q, w, e
    for _ in range(n_iter):
        agreement = np.array([np.sum(w * (M_f[i, :] == e)) for i in range(k_code)])
        w_sum = np.sum(w)
        q_new = agreement / w_sum if w_sum > 0 else q
        q = alpha * anchor_scores + (1.0 - alpha) * q_new if anchor_scores is not None else q_new
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


def code_reward(M, w, e):
    M_f = M.astype(float)
    w_sum = np.sum(w)
    if w_sum == 0:
        return None
    return np.array([np.sum(w * (M_f[i, :] == e)) / w_sum for i in range(M.shape[0])])


def case_reward_old(M, q, w, e):
    k_code, k_case = M.shape
    M_f = M.astype(float)
    threshold = np.median(q)
    good_mask = q > threshold
    bad_mask = q <= threshold
    if good_mask.sum() == 0 or bad_mask.sum() == 0:
        return None
    case_r = np.zeros(k_case)
    for j in range(k_case):
        gp = M_f[good_mask, j].mean()
        bp = M_f[bad_mask, j].mean()
        disc = abs(gp - bp)
        direction = (gp - bp) if e[j] == 1 else ((1 - gp) - (1 - bp))
        case_r[j] = disc * np.sign(direction) if direction != 0 else 0.0
    return case_r


def case_reward_new(M, q):
    k_code, k_case = M.shape
    M_f = M.astype(float)
    if q.std() < 1e-6:
        return None
    case_r = np.zeros(k_case)
    for j in range(k_case):
        col = M_f[:, j]
        if col.std() < 1e-8:
            case_r[j] = 0.0
            continue
        corr = np.corrcoef(q, col)[0, 1]
        case_r[j] = corr if not np.isnan(corr) else 0.0
    return case_r


def compare(data_path):
    try:
        with open(data_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as ex:
        print(f"  Skip {data_path}: {ex}")
        return

    print(f"\n{'='*80}")
    print(f"数据文件: {os.path.basename(data_path)}")
    print(f"题目数: {len(data)}")
    print(f"{'='*80}")

    old_code_groups = 0
    old_case_groups = 0
    new_code_groups = 0
    new_case_groups = 0
    old_corrs = []
    new_corrs = []

    for d in data:
        if d.get("all_case_bool_table") is None:
            continue
        t = d["num_ground_truth_test"]
        full = np.array(d["all_case_bool_table"])
        gt_table = full[:, :t]
        gen_table = full[:, t:]
        M = gen_table
        if M.shape[0] == 0 or M.shape[1] == 0:
            continue

        gt_cr = gt_table.astype(float).mean(axis=1)
        anchor = full[:, :min(1, t)].astype(float).mean(axis=1) if t > 0 else None

        # OLD
        pass_rate = M.astype(float).mean(axis=0)
        if not np.all((pass_rate == 0) | (pass_rate == 1)):
            q_o, w_o, e_o = em_old(M, anchor, 3, 0.3)
            if np.max(w_o) >= 0.1:
                cr_o = code_reward(M, w_o, e_o)
                if cr_o is not None and cr_o.std() > 1e-6:
                    old_code_groups += 1
                    if gt_cr.std() > 1e-6:
                        c, _ = spearmanr(gt_cr, cr_o)
                        if not np.isnan(c):
                            old_corrs.append(c)

                cs_o = case_reward_old(M, q_o, w_o, e_o)
                if cs_o is not None and cs_o.std() > 1e-6:
                    old_case_groups += 1

        # NEW
        n_disc = np.sum((pass_rate > 0) & (pass_rate < 1))
        if n_disc > 0:
            q_n, w_n, e_n = em_new(M, anchor, 3, 0.3)
            cr_n = code_reward(M, w_n, e_n)
            if cr_n is not None and cr_n.std() > 1e-6:
                new_code_groups += 1
                if gt_cr.std() > 1e-6:
                    c, _ = spearmanr(gt_cr, cr_n)
                    if not np.isnan(c):
                        new_corrs.append(c)

            cs_n = case_reward_new(M, q_n)
            if cs_n is not None and cs_n.std() > 1e-6:
                new_case_groups += 1

    print(f"\n{'指标':<25} {'旧版(OLD)':<15} {'新版(NEW)':<15}")
    print("-" * 55)
    print(f"{'Code groups':<25} {old_code_groups:<15} {new_code_groups:<15}")
    print(f"{'Case groups':<25} {old_case_groups:<15} {new_case_groups:<15}")
    
    if old_corrs:
        print(f"{'GT corr (mean)':<25} {np.mean(old_corrs):<15.4f} ", end="")
    else:
        print(f"{'GT corr (mean)':<25} {'N/A':<15} ", end="")
    
    if new_corrs:
        print(f"{np.mean(new_corrs):<15.4f}")
    else:
        print(f"{'N/A':<15}")

    if old_corrs:
        print(f"{'GT corr (median)':<25} {np.median(old_corrs):<15.4f} ", end="")
    else:
        print(f"{'GT corr (median)':<25} {'N/A':<15} ", end="")
    
    if new_corrs:
        print(f"{np.median(new_corrs):<15.4f}")
    else:
        print(f"{'N/A':<15}")

    gain_code = new_code_groups - old_code_groups
    gain_case = new_case_groups - old_case_groups
    print(f"\n  Code group 增加: {'+' if gain_code >= 0 else ''}{gain_code}")
    print(f"  Case group 增加: {'+' if gain_case >= 0 else ''}{gain_case}")

    if old_corrs and new_corrs:
        delta = np.mean(new_corrs) - np.mean(old_corrs)
        print(f"  GT correlation 变化: {'+' if delta >= 0 else ''}{delta:.4f}")
    print()


if __name__ == "__main__":
    os.chdir("/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/optimization")
    for path in DATA_FILES:
        if os.path.exists(path):
            compare(path)
        else:
            print(f"Skip: {path} not found")
