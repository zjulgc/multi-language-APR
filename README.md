# MultiRepair：共享–路由 MoE-LoRA 多语言程序自动修复

多语言 APR（Automated Program Repair）研究代码。核心方法 **MultiRepair** 在冻结基座的每个
Transformer 块的 7 类线性投影中并联注入 MoE-LoRA 模块：3 条常驻共享专家（rank-4）+ 11 条
候选路由专家（rank-4，逐 token top-1 选择），由联合 softmax 门控加权，训练时附带负载均衡损失。

- **基座**：Qwen2.5-Coder-7B-Instruct（主）；Meta-Llama-3-8B-Instruct（跨基座验证）
- **训练数据**：xCodeEval APR，11 种语言，每语言 3k（`perlang3k`）
- **评测**：xCodeEval APR（分布内，1100 题分层子集 + 17,699 题全集，ExecEval 执行式打分）、
  HumanEvalFix（分布外，6 种语言）
- **对照方法**：Base / Few-shot ICL / Zero-shot CoT（免训练）；Joint-LoRA r64 / LoRA r128 /
  Full FT（联合训练）；PerLang-LoRA + TIES / DARE 合并（先分语言后合并）；消融 w/o-Shared、w/o-Shared-Top4

命名对照：论文 **MultiRepair** = `s3-11e-top1`，磁盘目录 `*_e11r4_s3r4`；论文 **Joint-LoRA** =
`vanilla_lora_r64_perlang3k`；论文 **PerLang-LoRA** = `mono_vanilla_<Lang>`。

## 仓库布局

```text
multi-language-APR/
├── moe_apr/                    # MoE-LoRA 核心实现（层/模型补丁/快速推理路径/单测）
├── train_moe_apr.py            # MoE-LoRA 训练入口（支持 torchrun DDP）
├── train_lora_xcodeeval.py     # LoRA 基线训练（Joint / PerLang / r128）
├── train_full_xcodeeval.py     # 全参微调
├── generate_moe_apr.py         # MoE 模型生成
├── generate_apr_local.py       # 基座/PEFT 模型生成（含 --prompt_style plain|icl|cot）
├── eval_xcodeeval_execeval.py  # xCodeEval ExecEval 打分
├── eval_humanevalfix.py        # HumanEvalFix 生成+本地执行评测（含批量快速路径）
├── quick_eval.py               # 训练中快速代理评测（LM loss + 贪心匹配）
├── prompt_utils.py             # 对话模板 / ICL / CoT 提示构造
├── icl_retrieval.py            # Few-shot ICL 同语言示例检索（Jaccard）
├── prepare_xcodeeval_sft.py    # 原始 xCodeEval → SFT 格式
├── prepare_xcodeeval_by_family.py  # 数据准备（含 all/ 混合切分）
├── smoke_test_moe.py           # MoE 注入冒烟测试
├── scripts/                    # 流水线脚本 + launch/（各方法训练超参档案）
├── docs/                       # PIPELINE.md / STRUCTURE.md
└── data/ checkpoints/ eval_outputs/ results/ logs/ instruction_dataset/
    analysis/ figures/          # [本地，不提交：数据/产物/论文分析与图]
```

## 典型流程

```bash
export PYTHONPATH=.

# 1. 数据准备（原始 xCodeEval → perlang3k 训练变体）
python prepare_xcodeeval_sft.py ...            # 见 docs/PIPELINE.md
python scripts/prep_perlang_data.py ...
python scripts/make_xcodeeval_subset.py ...    # 1100 题评测子集（含题面 join）

# 2. 训练 MultiRepair（超参见 launcher 内部；支持 torchrun 多卡）
CUDA_VISIBLE_DEVICES=0 bash scripts/launch/_launch_perlang3k_e11r4_s3r4.sh

# 3. 生成 + 打分（xCodeEval 子集；生成必须用含题面的 join 后 jsonl）
python generate_moe_apr.py --moe_state checkpoints/<run>/moe_state.pt \
    --subset_jsonl data/eval/xcodeeval_test_100perlang.jsonl ...
python eval_xcodeeval_execeval.py --gen_dir eval_outputs/<run> --max_workers 8 ...

# 4. HumanEvalFix（分布外；--gen_batch_size >1 走批量快速路径）
python eval_humanevalfix.py --mode moe --languages python js java cpp rust go \
    --gen_batch_size 8 --test_workers 4 ...
```

各方法训练命令与超参以 `scripts/launch/` 下对应脚本为准；完整流水线见
[docs/PIPELINE.md](docs/PIPELINE.md)，目录明细见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

## 实现要点

- **快速推理路径**：推理时把逐专家 Python 循环改写为拼接权重的两次矩阵乘，输出逐字节一致，
  长序列生成约 3.8× 提速（训练仍走原路径）；见 `moe_apr/`。
- **ExecEval 打分**：编译型语言对并发敏感，`--max_workers` 建议 ≤8，否则 JVM/Go 会因编译超时静默掉分。
- **DDP 训练**：`torchrun` 数据并行；47GB 卡上 7B 需 reentrant checkpointing + static graph。

## 不进入 Git 的内容

数据集、checkpoint、生成与评测产物、日志（`data/`、`instruction_dataset/`、`checkpoints/`、
`eval_outputs/`、`results/`、`logs/`）以及实验过程文档均在 `.gitignore` 中。
