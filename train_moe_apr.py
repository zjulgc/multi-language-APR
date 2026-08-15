"""Train MoE-LoRA + Shared APR Expert on xCodeEval.

This is a single-stage trainer: shared APR experts, routing experts, router,
and adaptive gate are trained together on balanced per-language xCodeEval data.

The output checkpoint contains a state_dict of all MoELoRALinear submodules
(NOT the frozen base weights) so it can be re-loaded by re-patching the
base model and calling ``load_moe_state_dict``.

Example commands:

    python train_moe_apr.py \\
      --base_model /mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct \\
      --data_root data \\
      --output_dir checkpoints/single_stage_adaptive_shared \\
      --num_epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)

from moe_apr.data_utils import (
    DEFAULT_LANGUAGES,
    LANG_TO_EXPERT,
    BUGTYPE_TO_EXPERT,
    LanguageDataPaths,
    load_language_dataset,
    make_balanced_sampler,
)
from moe_apr.model_patcher import (
    MoEPatchConfig,
    patch_model_with_moe_lora,
    report_trainable_parameters,
)
from moe_apr.trainer import MoETrainer
from prompt_utils import tokenize_example


class RouteCollator:
    """Wrap a base collator, stripping the str 'language' column and stacking the
    int 'route_id' column into a (B,) tensor (consumed by the metrics, not the model)."""

    def __init__(self, base_collator) -> None:
        self.base = base_collator

    def __call__(self, features):
        route_ids = None
        if features and "route_id" in features[0]:
            route_ids = [int(f.pop("route_id")) for f in features]
        for f in features:
            f.pop("language", None)
        batch = self.base(features)
        if route_ids is not None:
            batch["route_id"] = torch.tensor(route_ids, dtype=torch.long)
        return batch


def _maybe_load_swanlab_env() -> None:
    """Load .swanlab.env if SWANLAB_API_KEY is unset."""
    if os.environ.get("SWANLAB_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(__file__), ".swanlab.env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def build_swanlab_callbacks(args: argparse.Namespace) -> List:
    """Return HF Trainer callbacks for SwanLab (transformers<4.50 needs SwanLabCallback)."""
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
        "description": f"MoE-APR single-stage experts={args.num_routing_experts}",
    }
    if workspace:
        cb_kwargs["workspace"] = workspace
    print(f"[swanlab] project={project} experiment={experiment_name}", flush=True)
    callback = SwanLabCallback(**cb_kwargs)
    # SwanLabCallback.__init__ ignores unknown kwargs; hyperparams must use update_config.
    callback.update_config(vars(args))
    return [callback]


def _parse_int_list(value: str) -> Optional[list[int]]:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# -------------------------------------------------------------------------- #
#                       Save / load MoE state-dict only                        #
# -------------------------------------------------------------------------- #


def save_moe_state(model: torch.nn.Module, path: str) -> None:
    """Save only the trainable MoE weights (LoRA experts + router + gate)."""
    sd = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            sd[n] = p.detach().cpu()
        else:
            if any(tag in n for tag in ("routing_experts", "shared_expert", "router", "adaptive_gate")):
                sd[n] = p.detach().cpu()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(sd, path)
    print(f"[save] MoE state -> {path} ({len(sd)} tensors)", flush=True)


def load_moe_state(model: torch.nn.Module, path: str, strict: bool = False) -> None:
    sd = torch.load(path, map_location="cpu")
    # Backward-compat: older checkpoints saved the single shared expert under
    # ``*shared_expert.lora_{A,B}.weight``. New code uses ``shared_experts: ModuleList``
    # so the same weights live at ``*shared_experts.0.lora_{A,B}.weight``.
    # Use a regex so the prefix may or may not have a leading dot (root vs nested).
    _legacy_re = re.compile(r"(^|\.)shared_expert\.")
    sd = {_legacy_re.sub(r"\1shared_experts.0.", k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    moe_missing = [k for k in missing if any(t in k for t in ("routing_experts", "shared_expert", "router", "adaptive_gate"))]

    # A tied parameter (HydraLoRA's shared lora_A) is saved once, because
    # named_parameters() de-duplicates -- so its other names are legitimately
    # "missing" and already have the right values. Any OTHER missing MoE key
    # means part of the adapter kept its random init while the model still
    # loads and generates, which is silent corruption. Name the difference.
    seen, tied_aliases = {}, set()
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            tied_aliases.add(name)
        else:
            seen[id(param)] = name
    unexplained = [k for k in moe_missing if k not in tied_aliases]
    if unexplained:
        print(f"[load] WARNING: {len(unexplained)} MoE key(s) missing and NOT explained by "
              f"parameter tying -- those weights keep their random init: {unexplained[:5]}", flush=True)
    if strict and (unexplained or unexpected):
        raise RuntimeError(f"[load] strict=True but missing={unexplained} unexpected={unexpected}")
    print(
        f"[load] {path}: {len(sd)} tensors loaded; "
        f"missing-MoE-keys={len(moe_missing)} (tied-alias={len(moe_missing) - len(unexplained)}, "
        f"unexplained={len(unexplained)}) unexpected={len(unexpected)}",
        flush=True,
    )


# -------------------------------------------------------------------------- #
#                                  Main                                        #
# -------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True, help="Root with by_language/<lang>/ splits (scripts/prep_perlang_data.py)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Optional HuggingFace Trainer checkpoint directory to resume optimizer/scheduler/global step.",
    )

    p.add_argument("--num_routing_experts", type=int, default=4)
    p.add_argument("--num_shared_experts", type=int, default=1,
                   help="Number of always-active shared APR experts (DeepSeekMoE Fine-grained Segmentation).")
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--share_routing_A", action="store_true",
                   help="Tie one LoRA A across all routing experts (HydraLoRA baseline: "
                        "shared A + N B heads; combine with --top_k == --num_routing_experts "
                        "for HydraLoRA's dense softmax router and --router_aux_loss_coef 0)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument(
        "--routing_ranks",
        type=str,
        default="",
        help="Optional comma-separated per-routing-expert LoRA ranks, e.g. 13,13,13,13,13",
    )
    p.add_argument(
        "--routing_alphas",
        type=str,
        default="",
        help="Optional comma-separated per-routing-expert LoRA alphas, e.g. 26,26,26,26,26",
    )
    p.add_argument("--shared_lora_r", type=int, default=-1, help="Optional shared expert LoRA rank")
    p.add_argument("--shared_lora_alpha", type=int, default=-1, help="Optional shared expert LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--shared_expert_gate_mode", type=str, default="adaptive", choices=["adaptive", "naive", "none"]
    )

    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--router_aux_loss_coef", type=float, default=0.01)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_strategy", type=str, default="steps", choices=["steps", "no"])
    p.add_argument("--disable_gradient_checkpointing", action="store_true")

    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_val_samples", type=int, default=-1)
    p.add_argument(
        "--val_file",
        type=str,
        default="",
        help=(
            "Path to validation jsonl with non-empty `output` field. "
            "Defaults to <data_root>/all/intrain_validation_sft.jsonl. "
            "DO NOT use the official validation_sft.jsonl: its outputs are empty "
            "(oracle hidden behind ExecEval) which produces meaningless eval_loss."
        ),
    )
    p.add_argument(
        "--max_samples_per_language",
        type=int,
        default=-1,
        help="When loading by_language/ data, cap each language at this many samples.",
    )
    p.add_argument("--language_weights_json", type=str, default="", help="Optional JSON dict for sampler weights")
    p.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated language names under <data_root>/by_language. Default is all 11 languages.",
    )
    p.add_argument("--train_on_inputs", action="store_true")
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument(
        "--route_by",
        choices=["language", "bug_type"],
        default="language",
        help="Routing-diagnostic target axis for the moe/route_purity + route_nmi metrics. "
        "'language' = 11 individual languages; 'bug_type' = 5 cross-lingual bug_exec_outcome classes.",
    )
    p.add_argument(
        "--model_parallel",
        action="store_true",
        help="Shard the base model across all visible GPUs (device_map=auto) and disable gradient "
        "checkpointing.",
    )
    p.add_argument(
        "--swanlab",
        action="store_true",
        help="Log train/eval metrics to SwanLab (uses SWANLAB_API_KEY or .swanlab.env).",
    )
    p.add_argument(
        "--swanlab_project",
        type=str,
        default="",
        help="SwanLab project name (default: env SWANLAB_PROJECT or 'multi-language-APR').",
    )
    p.add_argument(
        "--swanlab_experiment",
        type=str,
        default="",
        help="SwanLab experiment/run name (default: basename of --output_dir).",
    )
    p.add_argument(
        "--log_moe_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log MoE routing stats (expert utilization, entropy, route purity/NMI, shared gate) "
        "to Trainer/SwanLab under the moe/* prefix.",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False), flush=True)

    # ----------------------- Tokenizer + model ---------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # DDP (torchrun) awareness: each process gets its own LOCAL_RANK and must place
    # its full model replica on its own GPU. Single-GPU runs (LOCAL_RANK unset) keep
    # the original {"":0} placement byte-for-byte -> no behaviour change off torchrun.
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    _is_ddp = _local_rank >= 0
    _rank = int(os.environ.get("RANK", 0))

    print("Loading base model (this can take 1-2 min) ...", flush=True)
    # Prefer FlashAttention-2 for the seq-1024 attention (big share of step time);
    # fall back to PyTorch SDPA, then eager, if it is not installed.
    _load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.model_parallel else ({"": _local_rank} if _is_ddp else {"": 0}),
        trust_remote_code=True,
    )
    for _attn in ("flash_attention_2", "sdpa", None):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                **({**_load_kwargs, "attn_implementation": _attn} if _attn else _load_kwargs),
            )
            print(f"[speed] attn_implementation = {_attn or 'default'}", flush=True)
            break
        except (ImportError, ValueError) as e:
            print(f"[speed] attn_implementation={_attn} unavailable ({type(e).__name__}); falling back", flush=True)
    if args.model_parallel:
        # Naive model parallelism: let HF Trainer know the model spans devices so
        # it does not wrap it in DataParallel and places inputs on the right device.
        model.is_parallelizable = True
        model.model_parallel = True
        print(f"[model-parallel] device map spans: {sorted(set(model.hf_device_map.values()))}", flush=True)
    model.config.use_cache = False
    if (not args.disable_gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        # Reentrant checkpointing (default) is ~10GB lighter than non-reentrant, which
        # matters on the 47GB cards under DDP. Reentrant is unsafe under vanilla DDP
        # ("marked ready twice"), so the DDP path enables static_graph on the accelerator
        # below -- valid here because dense activates all experts every step (static graph).
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    elif args.disable_gradient_checkpointing:
        print("[speed] gradient checkpointing disabled", flush=True)

    # ----------------------- MoE patch ------------------------------------ #
    use_shared_expert = args.shared_expert_gate_mode != "none"
    routing_ranks = _parse_int_list(args.routing_ranks)
    routing_alphas = _parse_int_list(args.routing_alphas)
    shared_rank = args.shared_lora_r if args.shared_lora_r > 0 else None
    shared_alpha = args.shared_lora_alpha if args.shared_lora_alpha > 0 else None
    patch_cfg = MoEPatchConfig(
        target_modules=args.target_modules.split(","),
        num_routing_experts=args.num_routing_experts,
        top_k=args.top_k,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        use_shared_expert=use_shared_expert,
        shared_expert_gate_mode=args.shared_expert_gate_mode,
        routing_ranks=routing_ranks,
        routing_alphas=routing_alphas,
        shared_rank=shared_rank,
        shared_alpha=shared_alpha,
        num_shared_experts=args.num_shared_experts,
        share_routing_A=args.share_routing_A,
    )
    replaced = patch_model_with_moe_lora(model, patch_cfg)
    print(f"Patched {len(replaced)} target Linear modules into MoELoRALinear", flush=True)

    train_count, total_count, pct = report_trainable_parameters(model)
    print(f"Trainable params: {train_count:,} / {total_count:,} ({pct:.4f}%)", flush=True)

    # ----------------------- Data ----------------------------------------- #
    val_file = args.val_file or os.path.join(args.data_root, "all", "intrain_validation_sft.jsonl")
    print(f"[data] val_file = {val_file}", flush=True)
    val_ds = load_dataset("json", data_files=val_file, split="train")

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    train_paths = LanguageDataPaths(args.data_root, languages=languages, split="train")
    train_ds = load_language_dataset(train_paths)

    if args.max_samples_per_language > 0:
        keep_idx = []
        counts: Dict[str, int] = {}
        for i, lang in enumerate(train_ds["language"]):
            if counts.get(lang, 0) < args.max_samples_per_language:
                keep_idx.append(i)
                counts[lang] = counts.get(lang, 0) + 1
        train_ds = train_ds.select(keep_idx)
        print(f"After max_samples_per_language={args.max_samples_per_language}: {counts}", flush=True)

    weights = None
    if args.language_weights_json:
        weights = json.loads(args.language_weights_json)
    train_sampler = make_balanced_sampler(train_ds, language_weights=weights, num_samples=len(train_ds))
    print(f"Sampler language distribution: {train_sampler.language_distribution()}", flush=True)
    print(f"Sampler weights: {train_sampler.language_weights}", flush=True)

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))

    print(f"train_ds size = {len(train_ds):,}, val_ds size = {len(val_ds):,}", flush=True)

    # MoE metric logging (route_purity / route_nmi) needs a per-sample routing-
    # target expert id ("route_id"), derived from the individual language
    # (--route_by language) or the bug type (--route_by bug_type).
    use_route_id = args.log_moe_metrics
    if use_route_id:
        if args.route_by == "language" and "lang_cluster" not in train_ds.column_names:
            raise ValueError("--route_by language requires a 'lang_cluster' column in the data")
        if args.route_by == "bug_type" and "bug_exec_outcome" not in train_ds.column_names:
            raise ValueError("--route_by bug_type requires a 'bug_exec_outcome' column in the data")

        def _route_id(r):
            if args.route_by == "bug_type":
                eid = BUGTYPE_TO_EXPERT.get(r.get("bug_exec_outcome", ""))
                if eid is None:
                    raise ValueError(f"unknown bug_exec_outcome: {r.get('bug_exec_outcome')}")
                return {"route_id": eid}
            eid = LANG_TO_EXPERT.get(r.get("lang_cluster", ""))
            if eid is None:
                raise ValueError(f"unknown lang_cluster: {r.get('lang_cluster')}")
            return {"route_id": eid}

        def _val_has_route_field():
            if args.route_by == "bug_type":
                return "bug_exec_outcome" in val_ds.column_names
            return "lang_cluster" in val_ds.column_names

        train_ds = train_ds.map(_route_id)
        if _val_has_route_field():
            val_ds = val_ds.map(_route_id)

    keep_cols = {"language", "route_id"}
    train_ds = train_ds.map(
        lambda r: tokenize_example(tokenizer, args.max_seq_len, args.train_on_inputs, r),
        remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    )
    val_ds = val_ds.map(
        lambda r: tokenize_example(tokenizer, args.max_seq_len, args.train_on_inputs, r),
        remove_columns=[c for c in val_ds.column_names if c not in keep_cols],
    )

    # 'language' (str) and 'route_id' (int) are stripped from model inputs by
    # RouteCollator; route_id is forwarded to the MoE routing metrics.

    # ----------------------- Trainer args --------------------------------- #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        # HydraLoRA ties one lora_A across the routing experts, and safetensors
        # refuses to serialize shared tensors. Fall back to torch.save for the
        # Trainer's intermediate checkpoints in that case; the artifact we
        # actually load from (moe_state.pt via save_moe_state) is torch.save
        # either way, so nothing else changes.
        save_safetensors=not args.share_routing_A,
        load_best_model_at_end=args.eval_strategy != "no",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_first_step=args.swanlab,
        bf16=True,
        fp16=False,
        report_to="none",
        dataloader_num_workers=4,
        group_by_length=False,  # sampler controls order
        # When threading route_id to the metrics we must keep the column past HF's
        # unused-column filter; our RouteCollator strips str/int extras itself.
        remove_unused_columns=not use_route_id,
        # dense/A2 experts are all active every forward -> no unused params; False is
        # correct and faster. Ignored when world_size==1 (single-GPU unaffected).
        ddp_find_unused_parameters=False,
    )

    base_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    data_collator = RouteCollator(base_collator) if use_route_id else base_collator

    callbacks: List = []
    if args.swanlab:
        callbacks.extend(build_swanlab_callbacks(args))

    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=callbacks,
        router_aux_loss_coef=args.router_aux_loss_coef,
        train_sampler=train_sampler,
        log_moe_metrics=args.log_moe_metrics,
    )

    if _is_ddp:
        # Enable DDP static_graph so reentrant activation-checkpointing is safe (avoids
        # "marked ready twice") and comm is more efficient. accelerate reads the handler
        # at prepare() time (inside train()), so flipping it here takes effect. Valid
        # because dense's autograd graph is identical every step (all experts active).
        _h = getattr(trainer.accelerator, "ddp_handler", None)
        if _h is not None:
            _h.static_graph = True
            print("[ddp] static_graph=True on accelerator.ddp_handler", flush=True)
        else:
            from accelerate import DistributedDataParallelKwargs
            trainer.accelerator.ddp_handler = DistributedDataParallelKwargs(
                static_graph=True, find_unused_parameters=False
            )
            print("[ddp] created ddp_handler with static_graph=True", flush=True)

    train_count, total_count, pct = report_trainable_parameters(model)
    print(f"Trainable after Trainer init: {train_count:,} ({pct:.4f}%)", flush=True)

    # ----------------------- Train ---------------------------------------- #
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    # Under DDP all ranks hold identical (all-reduced) weights; only rank 0 persists
    # artifacts to avoid a 4-way write race. Non-zero ranks are done -> exit cleanly.
    # No collective ops run after train(), so returning here is safe.
    if _is_ddp and _rank != 0:
        return

    # Save MoE state (not the frozen base).
    save_moe_state(model, os.path.join(args.output_dir, "moe_state.pt"))
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "patch_config.json"), "w") as f:
        json.dump(
            {
                "target_modules": list(patch_cfg.target_modules),
                "num_routing_experts": patch_cfg.num_routing_experts,
                "top_k": patch_cfg.top_k,
                "rank": patch_cfg.rank,
                "alpha": patch_cfg.alpha,
                "routing_ranks": patch_cfg.routing_ranks,
                "routing_alphas": patch_cfg.routing_alphas,
                "shared_rank": patch_cfg.shared_rank,
                "shared_alpha": patch_cfg.shared_alpha,
                "num_shared_experts": patch_cfg.num_shared_experts,
                "share_routing_A": patch_cfg.share_routing_A,
                "dropout": patch_cfg.dropout,
                "use_shared_expert": patch_cfg.use_shared_expert,
                "shared_expert_gate_mode": patch_cfg.shared_expert_gate_mode,
                "training_mode": "single_stage",
                "route_by": args.route_by,
            },
            f,
            indent=2,
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
