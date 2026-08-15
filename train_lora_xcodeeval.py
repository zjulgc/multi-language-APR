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
from peft import LoraConfig, get_peft_model
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
    """Return HF Trainer callbacks for SwanLab (matches train_moe_apr.py)."""
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
        "description": f"Vanilla dense LoRA r={args.lora_r}",
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
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--train_on_inputs", action="store_true")
    parser.add_argument("--neftune_noise_alpha", type=float, default=0.0,
                        help="NEFTune embedding noise (0 = off; MORepair repro uses 5.0)")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--swanlab", action="store_true", help="Log to SwanLab (uses SWANLAB_API_KEY or .swanlab.env)")
    parser.add_argument("--swanlab_project", type=str, default="", help="SwanLab project (default multi-language-APR)")
    parser.add_argument("--swanlab_experiment", type=str, default="", help="SwanLab run name (default basename of output_dir)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # DDP (torchrun) awareness: each rank places its own replica on its own GPU.
    # Single-GPU runs (LOCAL_RANK unset) keep the original {"":0} placement
    # byte-for-byte, so nothing changes off torchrun.
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    _is_ddp = _local_rank >= 0

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": _local_rank} if _is_ddp else {"": 0},
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Gradient checkpointing: trade ~25% compute for a large activation-memory
    # saving so seq_len 2048+ fits on one 48GB GPU. enable_input_require_grads is
    # required for PEFT/LoRA (frozen base) so the checkpointed graph reaches the
    # adapters. use_cache is set False below (incompatible with checkpointing).
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

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
            neftune_noise_alpha=(args.neftune_noise_alpha or None),
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            logging_steps=50,
            logging_first_step=args.swanlab,
            bf16=True,
            fp16=False,
            report_to="none",
            dataloader_num_workers=4,
            group_by_length=False,
            max_grad_norm=args.max_grad_norm,
            # Plain LoRA activates every adapter parameter on every step, so
            # there are no unused params to search for.
            ddp_find_unused_parameters=False,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=build_swanlab_callbacks(args) if args.swanlab else None,
    )

    if _is_ddp:
        # static_graph makes reentrant activation-checkpointing safe under DDP
        # (otherwise "marked ready twice"). Valid here because a plain LoRA's
        # autograd graph is identical on every step. accelerate reads the handler
        # at prepare() time inside train(), so setting it now takes effect.
        _h = getattr(trainer.accelerator, "ddp_handler", None)
        if _h is not None:
            _h.static_graph = True
        else:
            from accelerate import DistributedDataParallelKwargs
            trainer.accelerator.ddp_handler = DistributedDataParallelKwargs(
                static_graph=True, find_unused_parameters=False
            )
        print("[ddp] static_graph=True", flush=True)

    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
