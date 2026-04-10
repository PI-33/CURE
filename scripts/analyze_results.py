#!/usr/bin/env python3

"""
实验结果分析工具
自动从日志文件中提取关键指标，生成对比报告和诊断建议
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RewardMetrics:
    """奖励阶段的指标"""
    mode: str  # "gt_based" or "self_bootstrap"
    code_mean: Optional[float] = None
    code_std: Optional[float] = None
    case_mean: Optional[float] = None
    case_std: Optional[float] = None
    gt_correlation: Optional[float] = None
    total_problems: Optional[int] = None
    used_problems: Optional[int] = None
    skipped_problems: Optional[int] = None
    
    def __repr__(self):
        return (
            f"RewardMetrics(mode={self.mode}, code_std={self.code_std:.4f}, "
            f"case_std={self.case_std:.4f}, correlation={self.gt_correlation:.4f})"
        )


@dataclass
class TrainingMetrics:
    """训练阶段的指标"""
    total_steps: int = 0
    final_policy_loss: Optional[float] = None
    final_kl_loss: Optional[float] = None
    final_reward_mean: Optional[float] = None
    training_time_minutes: Optional[float] = None


@dataclass
class EvalMetrics:
    """评估指标"""
    pass_at_1: Optional[float] = None
    pass_at_5: Optional[float] = None
    bon_4_4: Optional[float] = None
    bon_16_16: Optional[float] = None


@dataclass
class ExperimentResult:
    """单个实验的完整结果"""
    name: str
    reward_metrics: Optional[RewardMetrics] = None
    training_metrics: Optional[TrainingMetrics] = None
    eval_metrics: Optional[EvalMetrics] = None
    log_files: List[str] = None
    result_files: List[str] = None
    
    def __post_init__(self):
        if self.log_files is None:
            self.log_files = []
        if self.result_files is None:
            self.result_files = []


def extract_reward_metrics(content: str) -> Optional[RewardMetrics]:
    """从内容中提取奖励相关指标"""
    metrics = RewardMetrics(mode="unknown")
    
    # 检测模式
    if "reward_mode: self_bootstrap" in content:
        metrics.mode = "self_bootstrap"
    elif "reward_mode: gt_based" in content:
        metrics.mode = "gt_based"
    else:
        return None
    
    # 提取代码奖励
    match = re.search(r"estimated_code_reward: mean=([\d.-]+), std=([\d.-]+)", content)
    if match:
        metrics.code_mean = float(match.group(1))
        metrics.code_std = float(match.group(2))
    
    # 提取测试奖励
    match = re.search(r"estimated_case_reward: mean=([\d.-]+), std=([\d.-]+)", content)
    if match:
        metrics.case_mean = float(match.group(1))
        metrics.case_std = float(match.group(2))
    
    # 提取GT相关性
    match = re.search(r"gt_correlation\(code_reward\): mean_spearman=([\d.-]+)", content)
    if match:
        metrics.gt_correlation = float(match.group(1))
    
    # 提取问题统计
    match = re.search(r"self_bootstrap_stats: total_problems=(\d+), skipped=(\d+), used=(\d+)", content)
    if match:
        metrics.total_problems = int(match.group(1))
        metrics.skipped_problems = int(match.group(2))
        metrics.used_problems = int(match.group(3))
    
    return metrics


def extract_training_metrics(content: str) -> TrainingMetrics:
    """从训练日志中提取训练阶段指标"""
    metrics = TrainingMetrics()
    
    # 提取总步数（查找最后的step信息）
    matches = re.findall(r"step (\d+)", content, re.IGNORECASE)
    if matches:
        metrics.total_steps = max(int(m) for m in matches)
    
    # 提取最后的loss值
    match = re.search(r"policy loss.*?(\d+\.\d+)", content[-5000:], re.IGNORECASE | re.DOTALL)
    if match:
        metrics.final_policy_loss = float(match.group(1))
    
    return metrics


def extract_eval_metrics(content: str) -> EvalMetrics:
    """从评估日志中提取评估指标"""
    metrics = EvalMetrics()
    
    # pass@1, pass@5 等
    match = re.search(r"pass@1[:\s]+([\d.]+)", content, re.IGNORECASE)
    if match:
        metrics.pass_at_1 = float(match.group(1))
    
    match = re.search(r"pass@5[:\s]+([\d.]+)", content, re.IGNORECASE)
    if match:
        metrics.pass_at_5 = float(match.group(1))
    
    return metrics


def load_experiment(exp_dir: Path) -> ExperimentResult:
    """从实验目录加载完整的实验结果"""
    result = ExperimentResult(name=exp_dir.name)
    
    # 查找日志文件
    log_files = sorted(exp_dir.glob("train_*.log"))
    if not log_files:
        log_files = sorted(exp_dir.glob("*.log"))
    result.log_files = [str(f) for f in log_files]
    
    # 查找结果文件
    result_files = sorted(exp_dir.glob("results-*.txt"))
    if not result_files:
        result_files = sorted(exp_dir.glob("*.txt"))
    result.result_files = [str(f) for f in result_files]
    
    # 合并所有文件内容
    all_content = ""
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_content += f.read() + "\n"
        except:
            pass
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_content += f.read() + "\n"
        except:
            pass
    
    # 提取各类指标
    if all_content:
        result.reward_metrics = extract_reward_metrics(all_content)
        result.training_metrics = extract_training_metrics(all_content)
        result.eval_metrics = extract_eval_metrics(all_content)
    
    return result


def compare_experiments(baseline: ExperimentResult, selfbootstrap: ExperimentResult) -> str:
    """生成对比分析报告"""
    report = []
    report.append("\n" + "="*90)
    report.append("实验对比分析报告".center(90))
    report.append("="*90)
    
    # 1. 奖励函数对比
    report.append("\n📊 【1】奖励函数对比")
    report.append("-"*90)
    
    if baseline.reward_metrics and selfbootstrap.reward_metrics:
        b_rm = baseline.reward_metrics
        s_rm = selfbootstrap.reward_metrics
        
        report.append(f"\n代码奖励标准差:")
        report.append(f"  Baseline (GT-based):       {b_rm.code_std:.4f}")
        report.append(f"  Self-bootstrap (新方案):   {s_rm.code_std:.4f}")
        if b_rm.code_std > 0:
            ratio = s_rm.code_std / b_rm.code_std
            report.append(f"  → 方差比: {ratio:.2f}x {'📈' if ratio > 1.2 else '📉' if ratio < 0.8 else '≈'}")
        
        report.append(f"\n测试奖励标准差:")
        report.append(f"  Baseline (GT-based):       {b_rm.case_std:.4f}")
        report.append(f"  Self-bootstrap (新方案):   {s_rm.case_std:.4f}")
        if b_rm.case_std > 0:
            ratio = s_rm.case_std / b_rm.case_std
            report.append(f"  → 方差比: {ratio:.2f}x {'📈' if ratio > 1.2 else '📉' if ratio < 0.8 else '≈'}")
        
        if s_rm.gt_correlation is not None:
            report.append(f"\nSelf-bootstrap GT相关性: {s_rm.gt_correlation:.4f}")
            if s_rm.gt_correlation > 0.4:
                report.append("  → ✓ 良好（新旧奖励方向一致）")
            elif s_rm.gt_correlation > 0.2:
                report.append("  → △ 中等（有一定相关性，但可能需要优化）")
            else:
                report.append("  → ✗ 弱（需要检查EM算法或参数设置）")
        
        if s_rm.total_problems and s_rm.used_problems:
            skip_rate = (s_rm.skipped_problems / s_rm.total_problems) * 100
            report.append(f"\nSelf-bootstrap 问题覆盖率:")
            report.append(f"  总问题数: {s_rm.total_problems}")
            report.append(f"  使用问题: {s_rm.used_problems}")
            report.append(f"  跳过率: {skip_rate:.1f}% {'✓' if skip_rate < 20 else '✗'}")
    
    # 2. 诊断和建议
    report.append("\n\n🔧 【2】诊断建议")
    report.append("-"*90)
    
    if baseline.reward_metrics and selfbootstrap.reward_metrics:
        s_rm = selfbootstrap.reward_metrics
        
        issues = []
        
        # 检查1：相关性
        if s_rm.gt_correlation is not None:
            if s_rm.gt_correlation < 0.2:
                issues.append(
                    "⚠️  相关性过低 (< 0.2)\n"
                    "   原因可能：EM算法未能正确学习代码/测试质量\n"
                    "   建议：① 增加 em_iterations 到 5\n"
                    "        ② 降低 anchor_strength 到 0.1\n"
                    "        ③ 检查生成测试的多样性"
                )
        
        # 检查2：跳过率
        if s_rm.total_problems and s_rm.used_problems:
            skip_rate = (s_rm.skipped_problems / s_rm.total_problems) * 100
            if skip_rate > 30:
                issues.append(
                    "⚠️  问题跳过率过高 (> 30%)\n"
                    "   原因可能：min_weight_threshold 过严，或生成测试质量不均\n"
                    "   建议：① 降低 min_weight_threshold 到 0.05\n"
                    "        ② 检查生成测试是否存在系统性偏差"
                )
        
        # 检查3：奖励方差
        if s_rm.code_std is not None:
            if s_rm.code_std < 0.1:
                issues.append(
                    "⚠️  代码奖励方差过小 (< 0.1)\n"
                    "   原因可能：奖励信号太弱，难以区分好坏代码\n"
                    "   建议：① 检查EM的M步是否正确更新w和e\n"
                    "        ② 增加生成测试数量（k_case）"
                )
        
        if not issues:
            report.append("\n✓ 未检测到明显问题。建议进行完整规模的训练对比。")
        else:
            report.append("\n检测到的潜在问题：\n")
            for i, issue in enumerate(issues, 1):
                report.append(f"{i}. {issue}\n")
    
    # 3. 后续实验建议
    report.append("\n\n📋 【3】后续建议")
    report.append("-"*90)
    report.append("\n基于快速验证结果：\n")
    
    if selfbootstrap.reward_metrics and selfbootstrap.reward_metrics.gt_correlation:
        if selfbootstrap.reward_metrics.gt_correlation > 0.3:
            report.append("✓ 新方案显示良好的信号（correlation > 0.3）")
            report.append("  → 建议进行完整规模的训练对比实验")
        else:
            report.append("△ 新方案信号较弱（correlation ≤ 0.3）")
            report.append("  → 建议先进行参数调优，然后再做大规模对比")
    
    report.append("\n" + "="*90 + "\n")
    
    return "\n".join(report)


def main():
    """主函数"""
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("experiment_results")
    
    if not results_dir.exists():
        print(f"错误：目录不存在 {results_dir}")
        sys.exit(1)
    
    # 查找所有实验目录
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    
    if not exp_dirs:
        print(f"未找到实验目录 (在 {results_dir} 中)")
        sys.exit(1)
    
    # 加载所有实验
    experiments = {}
    for exp_dir in exp_dirs:
        print(f"📂 加载实验: {exp_dir.name}")
        exp = load_experiment(exp_dir)
        if exp.reward_metrics:
            experiments[exp_dir.name] = exp
            print(f"   ✓ 已加载")
        else:
            print(f"   ✗ 未找到有效的指标")
    
    if not experiments:
        print("未找到任何有效的实验结果")
        sys.exit(1)
    
    # 生成报告
    print("\n" + "="*90)
    print("生成分析报告...".center(90))
    print("="*90)
    
    # 按名称查找baseline和新方案
    baseline = None
    selfbootstrap = None
    
    for name, exp in experiments.items():
        if "baseline" in name.lower() or "gtbased" in name.lower() or "gt_based" in name.lower():
            baseline = exp
        if "selfbootstrap" in name.lower() or "self_bootstrap" in name.lower():
            selfbootstrap = exp
    
    # 输出所有实验的详细信息
    print("\n" + "="*90)
    print("所有实验详细指标".center(90))
    print("="*90)
    
    for name, exp in experiments.items():
        print(f"\n【{name}】")
        if exp.reward_metrics:
            print(f"  奖励模式: {exp.reward_metrics.mode}")
            if exp.reward_metrics.code_std is not None:
                print(f"  代码奖励 std: {exp.reward_metrics.code_std:.4f}")
            if exp.reward_metrics.case_std is not None:
                print(f"  测试奖励 std: {exp.reward_metrics.case_std:.4f}")
            if exp.reward_metrics.gt_correlation is not None:
                print(f"  GT 相关性: {exp.reward_metrics.gt_correlation:.4f}")
            if exp.reward_metrics.total_problems:
                skip_rate = (exp.reward_metrics.skipped_problems / exp.reward_metrics.total_problems) * 100
                print(f"  问题统计: {exp.reward_metrics.used_problems}/{exp.reward_metrics.total_problems} "
                      f"(跳过率 {skip_rate:.1f}%)")
    
    # 输出对比分析
    if baseline and selfbootstrap:
        report = compare_experiments(baseline, selfbootstrap)
        print(report)
        
        # 保存报告
        report_file = results_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ 报告已保存: {report_file}")
    else:
        print("\n⚠️  未找到对比所需的实验配置")
        if not baseline:
            print("   缺少: Baseline (GT-based)")
        if not selfbootstrap:
            print("   缺少: Self-bootstrap")


if __name__ == "__main__":
    main()
