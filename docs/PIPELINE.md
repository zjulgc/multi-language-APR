# xCodeEval Multilingual APR Pipeline

基座：Qwen2.5-Coder-7B-Instruct · 方法：Shared APR Expert + Routing Experts + Adaptive Gate

详细目录说明见 [STRUCTURE.md](STRUCTURE.md)。

## 目录一览

| 目录 | 内容 | Git |
|------|------|-----|
| `moe_apr/` | MoE-LoRA 核心库 | 跟踪 |
| `train_*.py` / `generate_*.py` / `eval_*.py` / `prepare_*.py` | 训练 / 生成 / 评测 / 数据准备入口 | 跟踪 |
| `scripts/` | Shell 自动化（含 `scripts/launch/` 训练启动） | 跟踪 |
| `analysis/` | 离线统计、路由诊断、CI 脚本 | 跟踪 |
| `docs/` | 论文计划、实验记录、调研 | 跟踪 |
| `data/` | 主训练数据 + `variants/` 数据变体 | 忽略 |
| `checkpoints/` | 训练 checkpoint | 忽略 |
| `eval_outputs/` | 生成补丁 | 忽略 |
| `results/` | pass@k、路由诊断 JSON | 忽略 |
| `logs/` | 历史训练 / 监控日志 | 忽略 |

旧短路径不再保留符号链接；脚本和文档统一使用真实路径。

## 快速开始

```bash
cd /path/to/multi-language-APR
export PYTHONPATH=.

# 数据准备
python prepare_xcodeeval_by_family.py ...

# 训练（推荐新路径）
bash scripts/launch/_launch_e11_shared_top1.sh

# Smoke 评测
bash scripts/run_xcodeeval_pass1_smoke.sh
```

SwanLab 凭证：`.swanlab.env`（已 gitignore，勿提交）。
