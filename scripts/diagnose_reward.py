#!/usr/bin/env python3
"""
诊断脚本：分析 outputs-rl-*.json 中的奖励计算过程
在每步训练之前运行，输出详细的诊断报告

用法:
  cd optimization
  python ../scripts/diagnose_reward.py --data temp_data/outputs-rl-*.json
  
或指定具体文件:
  python ../scripts/diagnose_reward.py --data temp_data/outputs-rl-XXXX.json
"""

import json
import glob
import argparse
import numpy as np
from scipy.stats import spearmanr


def diagnose(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print(f"诊断报告: {data_path}")
    print(f"总题目数: {len(data)}")
    print(f"{'='*80}\n")

    valid_problems = 0
    sb_code_groups = 0
    sb_case_groups = 0
    gt_code_groups = 0
    gt_case_groups = 0
    skip_reasons = {"no_bool_table": 0, "empty_matrix": 0, "uniform_tests": 0,
                    "low_weight": 0, "code_std_zero": 0, "case_no_split": 0}
    correlations = []
    
    per_problem = []

    for i, d in enumerate(data):
        info = {"idx": i, "name": d.get("problem_name", f"problem_{i}")[:60]}

        if d.get("all_case_bool_table") is None:
            skip_reasons["no_bool_table"] += 1
            info["status"] = "SKIP: no bool_table"
            per_problem.append(info)
            continue

        valid_problems += 1
        t = d["num_ground_truth_test"]
        full_table = np.array(d["all_case_bool_table"])
        gt_table = full_table[:, :t].copy()
        gen_table = full_table[:, t:].copy()
        k_code, k_case = gen_table.shape

        info["k_code"] = k_code
        info["k_case"] = k_case
        info["num_gt_test"] = t
        info["table_shape"] = f"{full_table.shape[0]}x{full_table.shape[1]} (gt={t}, gen={k_case})"

        # GT-based code reward
        gt_code_reward = np.mean(gt_table, axis=1)
        gt_code_std = gt_code_reward.std()
        info["gt_code_mean"] = f"{gt_code_reward.mean():.3f}"
        info["gt_code_std"] = f"{gt_code_std:.3f}"
        if gt_code_std > 0:
            gt_code_groups += 1

        # GT-based case reward
        correct_codes = np.where(gt_table.all(axis=1))[0]
        info["num_correct_codes"] = len(correct_codes)
        if len(correct_codes) > 0:
            gt_case_groups += 1

        # Self-bootstrap analysis
        M = gen_table
        if M.shape[0] == 0 or M.shape[1] == 0:
            skip_reasons["empty_matrix"] += 1
            info["status"] = "SKIP: empty matrix"
            per_problem.append(info)
            continue

        pass_rate = M.astype(float).mean(axis=0)
        info["pass_rate_range"] = f"[{pass_rate.min():.2f}, {pass_rate.max():.2f}]"
        
        n_uniform = np.sum((pass_rate == 0.0) | (pass_rate == 1.0))
        info["n_uniform_tests"] = int(n_uniform)
        info["n_discriminative_tests"] = int(k_case - n_uniform)

        if np.all((pass_rate == 0.0) | (pass_rate == 1.0)):
            skip_reasons["uniform_tests"] += 1
            info["status"] = "SKIP: all tests uniform"
            per_problem.append(info)
            continue

        # EM estimation
        M_f = M.astype(float)
        e = (pass_rate > 0.5).astype(float)
        w = np.maximum(pass_rate, 1.0 - pass_rate)
        q = np.full(k_code, 0.5)

        for iteration in range(3):
            agreement = np.array([np.sum(w * (M_f[ii, :] == e)) for ii in range(k_code)])
            w_sum = np.sum(w)
            if w_sum > 0:
                q = agreement / w_sum

            threshold = np.median(q)
            good_mask = q > threshold
            bad_mask = q <= threshold

            if good_mask.sum() == 0 or bad_mask.sum() == 0:
                break

            for j in range(k_case):
                good_pass = M_f[good_mask, j].mean()
                bad_pass = M_f[bad_mask, j].mean()
                discriminability = abs(good_pass - bad_pass)
                consensus = max(pass_rate[j], 1.0 - pass_rate[j])
                w[j] = discriminability * consensus
                e[j] = round(good_pass)

        info["em_q_range"] = f"[{q.min():.3f}, {q.max():.3f}]"
        info["em_q_std"] = f"{q.std():.4f}"
        info["em_w_max"] = f"{w.max():.4f}"
        info["em_w_mean"] = f"{w.mean():.4f}"
        info["em_w_nonzero"] = int(np.sum(w > 0.01))

        if np.max(w) < 0.1:
            skip_reasons["low_weight"] += 1
            info["status"] = "SKIP: max(w) < 0.1"
            per_problem.append(info)
            continue

        # SB code reward
        w_sum = np.sum(w)
        if w_sum > 0:
            sb_code_reward = np.array([np.sum(w * (M_f[ii, :] == e)) / w_sum for ii in range(k_code)])
        else:
            sb_code_reward = np.zeros(k_code)
        
        sb_code_std = sb_code_reward.std()
        info["sb_code_std"] = f"{sb_code_std:.4f}"

        if sb_code_std < 1e-6:
            skip_reasons["code_std_zero"] += 1
            info["status"] = "SKIP: SB code reward std=0"
            per_problem.append(info)
            continue

        sb_code_groups += 1
        sb_code_norm = (sb_code_reward - sb_code_reward.mean()) / sb_code_std

        # SB case reward
        threshold = np.median(q)
        good_mask = q > threshold
        bad_mask = q <= threshold
        if good_mask.sum() > 0 and bad_mask.sum() > 0:
            case_reward = np.zeros(k_case)
            for j in range(k_case):
                good_pass = M_f[good_mask, j].mean()
                bad_pass = M_f[bad_mask, j].mean()
                disc = abs(good_pass - bad_pass)
                if e[j] == 1:
                    direction = good_pass - bad_pass
                else:
                    direction = (1.0 - good_pass) - (1.0 - bad_pass)
                case_reward[j] = disc * np.sign(direction) if direction != 0 else 0.0
            
            case_std = case_reward.std()
            info["sb_case_std"] = f"{case_std:.4f}"
            if case_std >= 1e-6:
                sb_case_groups += 1
                info["sb_case_reward_range"] = f"[{case_reward.min():.3f}, {case_reward.max():.3f}]"
            else:
                info["sb_case_status"] = "std=0, no case group"
        else:
            skip_reasons["case_no_split"] += 1
            info["sb_case_status"] = "cannot split good/bad"

        # GT correlation
        if gt_code_std > 1e-6 and sb_code_std > 1e-6:
            gt_norm = (gt_code_reward - gt_code_reward.mean()) / gt_code_std
            corr, _ = spearmanr(gt_code_reward, sb_code_reward)
            if not np.isnan(corr):
                correlations.append(corr)
                info["gt_correlation"] = f"{corr:.4f}"

        info["status"] = "OK"
        per_problem.append(info)

    # ==================== 打印汇总 ====================
    
    print("="*80)
    print("一、汇总统计")
    print("="*80)
    print(f"  有效题目数:        {valid_problems}/{len(data)}")
    print(f"  SB code groups:    {sb_code_groups}")
    print(f"  SB case groups:    {sb_case_groups}")
    print(f"  GT code groups:    {gt_code_groups} (仅供对比)")
    print(f"  GT case groups:    {gt_case_groups} (仅供对比)")
    print()

    print("跳过原因分布:")
    for reason, count in skip_reasons.items():
        if count > 0:
            print(f"  {reason}: {count}")
    print()

    if correlations:
        print(f"GT Correlation: mean={np.mean(correlations):.4f}, "
              f"median={np.median(correlations):.4f}, "
              f"min={np.min(correlations):.4f}, max={np.max(correlations):.4f}, "
              f"n={len(correlations)}")
    print()

    # ==================== 逐题详情 ====================
    
    print("="*80)
    print("二、逐题详情")
    print("="*80)
    
    for info in per_problem:
        status = info.get("status", "?")
        marker = "✓" if status == "OK" else "✗"
        print(f"\n  [{marker}] 题目 {info['idx']}: {info.get('name', '?')}")
        print(f"      状态: {status}")
        
        if "table_shape" in info:
            print(f"      矩阵: {info['table_shape']}")
        if "num_correct_codes" in info:
            print(f"      GT正确代码数: {info['num_correct_codes']}/{info.get('k_code', '?')}")
        if "gt_code_mean" in info:
            print(f"      GT code reward: mean={info['gt_code_mean']}, std={info['gt_code_std']}")
        if "n_discriminative_tests" in info:
            print(f"      有区分度的测试: {info['n_discriminative_tests']}/{info.get('k_case', '?')}")
        if "em_q_range" in info:
            print(f"      EM q: range={info['em_q_range']}, std={info['em_q_std']}")
            print(f"      EM w: max={info['em_w_max']}, mean={info['em_w_mean']}, nonzero={info['em_w_nonzero']}")
        if "sb_code_std" in info:
            print(f"      SB code reward std: {info['sb_code_std']}")
        if "sb_case_std" in info:
            print(f"      SB case reward std: {info['sb_case_std']}")
        if "gt_correlation" in info:
            print(f"      GT correlation: {info['gt_correlation']}")

    # ==================== 核心问题和建议 ====================

    print(f"\n{'='*80}")
    print("三、核心问题诊断")
    print("="*80)

    issues = []

    if sb_code_groups < valid_problems * 0.3:
        issues.append(
            f"Code group 产出率过低: {sb_code_groups}/{valid_problems} = "
            f"{sb_code_groups/max(valid_problems,1)*100:.0f}%\n"
            f"    → 大量数据被浪费，训练信号极度稀疏"
        )
    
    if sb_case_groups < sb_code_groups * 0.5:
        issues.append(
            f"Case group 产出率过低: {sb_case_groups} vs code的{sb_code_groups}\n"
            f"    → 测试生成器几乎得不到训练信号"
        )

    if correlations and np.mean(correlations) < 0.3:
        issues.append(
            f"GT correlation 偏低: mean={np.mean(correlations):.3f}\n"
            f"    → 自举奖励方向可能不正确，有误导风险"
        )

    if skip_reasons["uniform_tests"] > valid_problems * 0.2:
        issues.append(
            f"大量题目因'所有测试uniform'被跳过: {skip_reasons['uniform_tests']}\n"
            f"    → 生成的测试缺乏多样性"
        )
    
    if skip_reasons["code_std_zero"] > valid_problems * 0.2:
        issues.append(
            f"大量题目因'SB code std=0'被跳过: {skip_reasons['code_std_zero']}\n"
            f"    → EM未能区分代码质量，所有code得分相同"
        )

    # 检查空response问题
    n_empty_response_problems = 0
    for info in per_problem:
        if info.get("status") == "OK":
            idx = info["idx"]
            responses = data[idx].get("full_code_generation", [])
            n_empty = sum(1 for r in responses if len(r.strip()) < 10)
            if n_empty > len(responses) * 0.5:
                n_empty_response_problems += 1
    
    if n_empty_response_problems > 0:
        issues.append(
            f"有 {n_empty_response_problems} 个题目超过半数code回复为空/极短\n"
            f"    → 模型可能已经开始 collapse（生成空回答）"
        )

    if not issues:
        print("\n  ✓ 未检测到明显问题")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"\n  问题{i}: {issue}")

    print(f"\n{'='*80}\n")

    # 保存诊断结果为JSON
    report = {
        "data_path": data_path,
        "total_problems": len(data),
        "valid_problems": valid_problems,
        "sb_code_groups": sb_code_groups,
        "sb_case_groups": sb_case_groups,
        "gt_code_groups": gt_code_groups,
        "gt_case_groups": gt_case_groups,
        "skip_reasons": skip_reasons,
        "gt_correlation_mean": float(np.mean(correlations)) if correlations else None,
        "gt_correlation_all": [float(c) for c in correlations],
        "issues": issues,
    }
    
    report_path = data_path.replace(".json", "_diagnosis.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"诊断报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="outputs-rl-*.json 文件路径")
    args = parser.parse_args()

    if args.data:
        files = glob.glob(args.data)
    else:
        files = glob.glob("temp_data/outputs-rl-*.json")

    if not files:
        print("未找到数据文件")
    else:
        for f in sorted(files):
            diagnose(f)
