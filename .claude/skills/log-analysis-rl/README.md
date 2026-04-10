# 日志分析技能使用指南

## 快速开始（3步）

### 1️⃣ 清洗日志
```bash
python .cursor/skills/log-analysis-rl/scripts/clean_log.py /path/to/train_YYYYMMDD_HHMMSS.log
```

脚本输出路径：`train_YYYYMMDD_HHMMSS_cleaned.md`

### 2️⃣ 查看清洗后的文件
- 浏览 `*_cleaned.md` 查看结构化摘要
- 注意错误、时间线和性能指标

### 3️⃣ 分享给 AI 分析
复制整个清洁日志内容，让 AI 进行深度分析

## 实际例子

```bash
# 针对最新的训练运行
python .cursor/skills/log-analysis-rl/scripts/clean_log.py \
  /mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/logs/train_20260227_190857.log

# 输出自动生成为：
# → train_20260227_190857_cleaned.md
# → 可选：train_20260227_190857_cleaned.json (使用 --output json)
```

## 获取 JSON 格式（编程用）

```bash
python .cursor/skills/log-analysis-rl/scripts/clean_log.py train.log --output json
# 输出：train_cleaned.json（可用 Python 脚本处理）
```

## 分析焦点

清洁后的日志包含这些关键部分：

| 部分 | 用途 |
|------|------|
| **Summary** | 日志规模和错误计数 |
| **Errors** | 故障调试（查找 ValueError、IndexError 等） |
| **Timeline** | 执行流程和性能检查点 |
| **Metrics** | 训练趋势（acc, loss, entropy 等） |

## 技能文件位置

```
.cursor/skills/log-analysis-rl/
├── SKILL.md                 # 主要使用说明（这里就是）
├── ANALYSIS_PATTERNS.md     # 诊断模式和常见问题
├── LOG_FORMAT.md            # 日志格式和字段参考
└── scripts/
    └── clean_log.py         # 数据清洁脚本（需要调用）
```

## 问题排查

**脚本找不到文件？**
- 使用绝对路径：`/mnt/shared-storage-user/zhupengyu1/zpy3/vllm/CURE/logs/train_*.log`
- 相对路径也可以，但必须从正确的目录运行

**清洁后仍然很大？**
- 使用 `--output json` 获取紧凑的结构化格式
- JSON 文件更易于编程处理和可视化

**提取的指标不完整？**
- 检查日志是否完整（中途未中断）
- 部分指标可能在特定训练步骤才出现
