# 目录结构说明

本 pipeline 按职责分为 **代码**、**脚本**、**数据**、**实验产物**、**文档** 五类。大文件与本地实验状态默认不进入 git（见仓库根 `.gitignore`）。

```text
multi-language-APR/
├── moe_apr/                 # 核心库：MoE 层、trainer、数据工具、单测
├── train_*.py               # 训练入口
├── generate_*.py            # 生成入口
├── eval_*.py                # 评测入口
├── prepare_*.py             # 数据准备入口
├── quick_eval.py            # 快速评测
├── smoke_test_moe.py        # MoE smoke
│
├── scripts/                 # Shell 自动化
│   ├── launch/              # 训练启动（原 _launch_*.sh）
│   └── …                    # eval / watchdog / 数据准备 runner
├── analysis/                # 离线统计与路由诊断（原 _*.py / analyze_*.py）
├── docs/                    # 论文计划、实验记录、调研笔记（*.md）
│
├── data/                    # 主训练数据（by_family / all）
│   └── variants/            # 数据变体实体目录
│
├── checkpoints/             # [gitignore] 训练 checkpoint / moe_state.pt
├── eval_outputs/            # [gitignore] 模型生成补丁
├── results/                 # [gitignore] pass@k JSON、路由诊断等
└── logs/                    # [gitignore] 历史训练 / 监控日志
```

## 路径规范

仓库不保留旧短路径符号链接。请直接使用真实路径：

- 数据变体：`data/variants/...`
- 训练启动：`scripts/launch/...`
- 离线分析：`analysis/...`
