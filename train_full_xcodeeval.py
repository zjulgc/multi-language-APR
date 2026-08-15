import argparse
import os
from typing import Dict

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if os.environ.get("LGC_GPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LGC_GPU"]

print(f"[GPU env] LGC_GPU={os.environ.get('LGC_GPU')}, "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
      f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}", flush=True)

import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"[torch] device_count={torch.cuda.device_count()}, name={p.name}, "
          f"total_mem={p.total_memory/1024**3:.1f}GB", flush=True)
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from prompt_utils import tokenize_example


def _maybe_load_swanlab_env() -> None:
    """Load .swanlab.env if SWANLAB_API_KEY is unset."""
    if os.environ.get("SWANLAB_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".swanlab.env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def build_swanlab_callbacks(args):
    """Return HF Trainer callbacks for SwanLab (matches train_lora_xcodeeval.py)."""
    _maybe_load_swanlab_env()
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback

    api_key = os.environ.get("SWANLAB_API_KEY")
    if api_key:
        swanlab.login(api_key=api_key, save=True)
    project = args.swanlab_project or "multi-language-APR"
    workspace = os.environ.get("SWANLAB_WORKSPACE") or None
    experiment_name = args.swanlab_experiment or os.path.basename(args.output_dir.rstrip("/"))
    cb_kwargs = {
        "project": project,
        "experiment_name": experiment_name,
        "description": "Full-parameter fine-tune (FSDP full_shard)",
    }
    if workspace:
        cb_kwargs["workspace"] = workspace
    print(f"[swanlab] project={project} experiment={experiment_name}", flush=True)
    cb = SwanLabCallback(**cb_kwargs)
    cb.update_config(vars(args))
    return [cb]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--train_on_inputs", action="store_true")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="")
    parser.add_argument("--swanlab_experiment", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # FSDP: load the full bf16 model on CPU per rank; the Trainer's accelerate
    # integration shards it across ranks at prepare() time. No device_map here —
    # that is the LoRA/DDP path and conflicts with FSDP wrapping.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    raw = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    train_ds = raw["train"]
    val_ds = raw["validation"]
    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    train_ds = train_ds.map(
        lambda r: tokenize_example(tokenizer, args.max_seq_len, args.train_on_inputs, r),
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda r: tokenize_example(tokenizer, args.max_seq_len, args.train_on_inputs, r),
        remove_columns=val_ds.column_names,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type=args.lr_scheduler_type,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            # Optimizer/scheduler state under FSDP is ~4x model size and slow to
            # gather; crash recovery restarts from weights, so skip it.
            save_only_model=True,
            logging_steps=20,
            logging_first_step=True,
            bf16=True,
            fp16=False,
            report_to="none",
            dataloader_num_workers=4,
            group_by_length=False,
            max_grad_norm=args.max_grad_norm,
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
                "use_orig_params": True,
                "activation_checkpointing": True,
                "state_dict_type": "FULL_STATE_DICT",
                "limit_all_gathers": True,
            },
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=build_swanlab_callbacks(args) if args.swanlab else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
