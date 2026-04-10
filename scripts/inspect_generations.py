#!/usr/bin/env python3
"""
查看当前 temp_data 中的生成代码和测试用例。

用法:
  python scripts/inspect_generations.py                  # 显示所有问题的摘要
  python scripts/inspect_generations.py --problem 3      # 详细展示第3个问题
  python scripts/inspect_generations.py --problem 3 --code 0   # 展示第3题的第0个代码
  python scripts/inspect_generations.py --problem 3 --test 2   # 展示第3题的第2个测试
"""

import json
import sys
import argparse
import os
import glob
import numpy as np


def find_data_file():
    patterns = [
        "optimization/temp_data/outputs-rl-*.json",
        "temp_data/outputs-rl-*.json",
    ]
    for p in patterns:
        files = glob.glob(p)
        if files:
            return files[0]
    return None


def truncate(s, n=120):
    s = s.replace("\n", "\\n")
    return s[:n] + "..." if len(s) > n else s


def show_summary(data):
    print(f"\n{'='*90}")
    print(f"  共 {len(data)} 个问题")
    print(f"{'='*90}\n")

    for i, d in enumerate(data):
        if d.get("all_case_bool_table") is None:
            status = "⊘ 无执行结果"
        else:
            tbl = np.array(d["all_case_bool_table"])
            t = d["num_ground_truth_test"]
            gt_pass = tbl[:, :t].all(axis=1).sum()
            status = f"GT通过: {gt_pass}/{tbl.shape[0]}"

        n_code = len(d.get("full_code_generation", []))
        n_test = len(d.get("full_case_generation", []))
        question = d.get("question", "")[:60]
        code_lens = d.get("code_response_length", [])
        test_lens = d.get("case_response_length", [])

        avg_code_len = f"{np.mean(code_lens):.0f}" if code_lens else "-"
        avg_test_len = f"{np.mean(test_lens):.0f}" if test_lens else "-"

        print(f"[{i:>3}] {question}...")
        print(f"      代码×{n_code} (avg {avg_code_len} tok) | 测试×{n_test} (avg {avg_test_len} tok) | {status}")
        print()


def show_problem(data, idx):
    if idx < 0 or idx >= len(data):
        print(f"错误: 索引 {idx} 超范围 (0-{len(data)-1})")
        return

    d = data[idx]
    print(f"\n{'='*90}")
    print(f"  问题 [{idx}]")
    print(f"{'='*90}")

    print(f"\n题目:")
    question = d.get("question", "无")
    for line in question.split("\n")[:15]:
        print(f"  {line}")
    if question.count("\n") > 15:
        print(f"  ... (共 {question.count(chr(10))+1} 行)")

    codes = d.get("full_code_generation", [])
    tests = d.get("full_case_generation", [])
    code_lens = d.get("code_response_length", [])
    test_lens = d.get("case_response_length", [])
    gen_codes = d.get("generated_code", [])

    if d.get("all_case_bool_table") is not None:
        tbl = np.array(d["all_case_bool_table"])
        t = d["num_ground_truth_test"]
        print(f"\n执行矩阵: {tbl.shape[0]} codes × {tbl.shape[1]} tests (前{t}列是GT)")
        gt_pass_count = tbl[:, :t].all(axis=1).sum()
        gen_pass_rate = tbl[:, t:].mean()
        print(f"  GT全通过的代码: {gt_pass_count}/{tbl.shape[0]}")
        print(f"  生成测试平均通过率: {gen_pass_rate:.3f}")

    print(f"\n--- 代码 ({len(codes)}个) ---")
    for j, code in enumerate(codes):
        length = code_lens[j] if j < len(code_lens) else "?"
        extracted = gen_codes[j] if j < len(gen_codes) else ""
        gt_pass = ""
        if d.get("all_case_bool_table") is not None:
            tbl = np.array(d["all_case_bool_table"])
            t = d["num_ground_truth_test"]
            if t > 0:
                passed = tbl[j, :t].all()
                gt_pass = " ✓GT" if passed else " ✗GT"
        print(f"  [{j:>2}] {length:>5} tok{gt_pass} | {truncate(extracted, 100)}")

    print(f"\n--- 测试 ({len(tests)}个) ---")
    case_inputs = d.get("case_input", [])
    case_outputs = d.get("case_output", [])
    for j, test in enumerate(tests):
        length = test_lens[j] if j < len(test_lens) else "?"
        inp = case_inputs[j] if j < len(case_inputs) else ""
        out = case_outputs[j] if j < len(case_outputs) else ""
        print(f"  [{j:>2}] {length:>5} tok | in: {truncate(inp, 40)} → out: {truncate(out, 40)}")

    print()


def show_code_detail(data, prob_idx, code_idx):
    d = data[prob_idx]
    codes = d.get("full_code_generation", [])
    gen_codes = d.get("generated_code", [])

    if code_idx >= len(codes):
        print(f"错误: 代码索引 {code_idx} 超范围 (0-{len(codes)-1})")
        return

    print(f"\n{'='*90}")
    print(f"  问题[{prob_idx}] 代码[{code_idx}]")
    print(f"{'='*90}")

    if d.get("all_case_bool_table") is not None:
        tbl = np.array(d["all_case_bool_table"])
        t = d["num_ground_truth_test"]
        row = tbl[code_idx]
        gt_results = row[:t].tolist()
        gen_results = row[t:].tolist()
        gt_pass = all(gt_results)
        print(f"\nGT测试: {'✓ 全通过' if gt_pass else '✗ 未全通过'} {gt_results}")
        print(f"生成测试通过率: {sum(gen_results)}/{len(gen_results)}")

    print(f"\n--- 提取的代码 ---")
    if code_idx < len(gen_codes):
        print(gen_codes[code_idx])
    else:
        print("(无提取结果)")

    print(f"\n--- 完整LLM输出 ---")
    print(codes[code_idx])


def show_test_detail(data, prob_idx, test_idx):
    d = data[prob_idx]
    tests = d.get("full_case_generation", [])
    case_inputs = d.get("case_input", [])
    case_outputs = d.get("case_output", [])

    if test_idx >= len(tests):
        print(f"错误: 测试索引 {test_idx} 超范围 (0-{len(tests)-1})")
        return

    print(f"\n{'='*90}")
    print(f"  问题[{prob_idx}] 测试[{test_idx}]")
    print(f"{'='*90}")

    if d.get("all_case_bool_table") is not None:
        tbl = np.array(d["all_case_bool_table"])
        t = d["num_ground_truth_test"]
        col_idx = t + test_idx
        if col_idx < tbl.shape[1]:
            col = tbl[:, col_idx].tolist()
            pass_count = sum(col)
            print(f"\n通过情况: {pass_count}/{len(col)} 个代码通过此测试")

    if test_idx < len(case_inputs):
        print(f"\n--- 测试输入 ---")
        print(case_inputs[test_idx])
    if test_idx < len(case_outputs):
        print(f"\n--- 测试输出 ---")
        print(case_outputs[test_idx])

    print(f"\n--- 完整LLM输出 ---")
    print(tests[test_idx])


def main():
    parser = argparse.ArgumentParser(description="查看生成的代码和测试用例")
    parser.add_argument("--file", type=str, help="指定 outputs-rl-*.json 路径")
    parser.add_argument("--problem", "-p", type=int, help="查看指定问题的详情")
    parser.add_argument("--code", "-c", type=int, help="查看指定代码的完整内容")
    parser.add_argument("--test", "-t", type=int, help="查看指定测试的完整内容")
    args = parser.parse_args()

    path = args.file or find_data_file()
    if not path or not os.path.exists(path):
        print("找不到数据文件。请在项目根目录运行，或用 --file 指定路径。")
        sys.exit(1)

    print(f"读取: {path}")
    with open(path) as f:
        data = json.load(f)

    if args.problem is not None:
        if args.code is not None:
            show_code_detail(data, args.problem, args.code)
        elif args.test is not None:
            show_test_detail(data, args.problem, args.test)
        else:
            show_problem(data, args.problem)
    else:
        show_summary(data)


if __name__ == "__main__":
    main()
