# Shared Expert based MoE-LoRA for Multilingual APR (Engineering)

This module implements the MoE-LoRA + Shared APR Expert + Adaptive Gate
architecture from the ICSE 2026 proposal. Code lives entirely under
`moe_apr/`; we do **not** depend on any third-party MoE
library (no MoE-PEFT, no MixLoRA), only HuggingFace `transformers` + `peft`.

## Architecture in one picture

```
                ┌─────────────────────────────────────────┐
                │         frozen base nn.Linear            │
                └─────────────────────────────────────────┘
                                    +
                ┌─────────────────────────────────────────┐
                │   Shared APR Expert  (LoRA, fp16/bf16)   │  always active
                │             × adaptive_gate (fp32, scalar)│  joint softmax
                └─────────────────────────────────────────┘
                                    +
                ┌─────────────────────────────────────────┐
                │  Routing Experts (one per family, top-k) │
                │  c_family / jvm_family / dynamic_typed / │  top-2 each token
                │  systems     × routing_weight (fp32)      │
                └─────────────────────────────────────────┘
```

`adaptive_gate` and the four `routing_weight`s are **jointly softmax
normalized** so they sum to 1 across the union {shared, top-2 routing}.
This is the ASE-style adaptive gate; it empirically prevents the shared
expert from collapsing.

## Module layout


| file                   | responsibility                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| `lora_expert.py`       | Single LoRA expert (A B with scaling). Zero-init B → forward initially equals base.           |
| `moe_layer.py`         | `MoELoRALinear` (full layer) + `AdaptiveGate`. Wraps & freezes base linear.                   |
| `load_balance.py`      | Switch-style auxiliary loss; only over routing experts (shared excluded).                     |
| `model_patcher.py`     | `patch_model_with_moe_lora(model, cfg)` replaces every target nn.Linear with `MoELoRALinear`. |
| `data_utils.py`        | `BalancedFamilySampler`, `load_family_dataset`, `FamilyDataPaths`.                            |
| `trainer.py`           | `MoETrainer` (HF `Trainer` subclass) with aux loss, route supervision, and sampler hook.      |
| `tests/`               | 18 unit tests; CPU-only; <2s.                                                                 |


## Data

Run once (≈11 min, 55 GB output):

```bash
python prepare_xcodeeval_by_family.py \
  --apr_dir instruction_dataset/xCodeEval/apr \
  --problem_desc instruction_dataset/xCodeEval/problem_descriptions.jsonl \
  --output_dir data
```

Layout produced:

```
data/
├─ all/
│  ├─ train_sft.jsonl            (4,672,070 mixed-language)
│  ├─ validation_sft.jsonl       (5,068, balanced across 11 langs)
│  └─ test_infer.jsonl           (17,699 with per-language balance)
├─ by_family/
│  ├─ c_family/{train,validation,test}.jsonl   (3,581,566 / 2,113 / 5,985)
│  ├─ jvm_family/...                            ( 590,786 / 1,029 / 4,010)
│  ├─ dynamic_typed/...                         ( 478,965 / 1,427 / 5,372)
│  └─ systems/...                               (  20,753 /   499 / 2,332)
└─ distribution_report.json
```

> ⚠️ **systems family is 170× smaller than c_family**. Training should use
> `BalancedFamilySampler`; otherwise systems is effectively unseen.

## Training

Single-stage training updates the shared APR expert(s), routing experts, router,
and adaptive gate together on the per-language BALANCED set produced by
`scripts/prep_perlang_data.py` (11 languages x N rows).

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train_moe_apr.py \
  --base_model /data/share/qwen/Qwen2.5-Coder-7B-Instruct \
  --data_root  data/variants/perlang3k \
  --languages "C,C++,C#,Java,Kotlin,Python,Javascript,Ruby,PHP,Rust,Go" \
  --val_file   data/all/intrain_validation_sft.jsonl \
  --output_dir checkpoints/single_stage_perlang3k_e11r4_s1r16 \
  --num_epochs 1 \
  --max_seq_len 1024 \
  --train_batch_size 2 --grad_accum 8 \
  --num_routing_experts 11 --route_by language --top_k 1 \
  --lora_r 4 --lora_alpha 8 \
  --num_shared_experts 1 --shared_lora_r 16 --shared_lora_alpha 32 \
  --shared_expert_gate_mode adaptive \
  --max_samples_per_language 3000 \
  --router_aux_loss_coef 0.01
```

By default, `train_moe_apr.py` loads `<data_root>/by_language/*/train_sft.jsonl`
and uses balanced per-language sampling. Use `--languages` to restrict the set.
Ready-made launch scripts: `scripts/launch/_launch_perlang3k_*.sh`.

## Inference & evaluation

Generate APR candidates in `eval_apr.py`-compatible format:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python generate_moe_apr.py \
  --base_model /data/share/qwen/Qwen2.5-Coder-7B-Instruct \
  --moe_state  checkpoints/single_stage_adaptive_shared/moe_state.pt \
  --local_apr_dir instruction_dataset/xCodeEval/apr \
  --split test \
  --output_dir dumped/oai/apr_n_sample_20 \
  --num_samples 1 --temperature 0.2 --max_items 17699
```

Then run the existing xCodeEval evaluator:

```bash
export DUMP_FOLDER=$(pwd)/dumped
python instruction_dataset/xCodeEval_repo/evaluation/apr/eval_apr.py
python instruction_dataset/xCodeEval_repo/evaluation/apr/get_result.py
```

## Smoke test (correctness, ~25s on A6000)

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python smoke_test_moe.py \
  --base_model /data/share/qwen/Qwen2.5-Coder-7B-Instruct \
  --data_root  data \
  --output_dir checkpoints/_smoke
```

Validates: model load → patch (112 modules) → tokenize → forward → backward
→ loss decrease → save MoE state → reload → forward exact-match.

## Unit tests (CPU)

```bash
PYTHONPATH=. python -m unittest moe_apr.tests.test_moe_layer -v
```

Runs the CPU test suite in ~15 s.

## Frozen design decisions (proposal §7)


| #   | decision           | value                                               |
| --- | ------------------ | --------------------------------------------------- |
| 1   | Language families  | 4 — c_family / jvm_family / dynamic_typed / systems |
| 2   | Routing top-k      | 2                                                   |
| 3   | Training schedule  | Single-stage shared + routing MoE training          |
| 4   | Shared Expert rank | same as routing experts (16)                        |
| 5   | Adaptive Gate      | joint softmax over {shared, top-k routing}          |
| 6   | Eval metrics       | pass@1 + pass@10                                    |
