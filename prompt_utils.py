"""Shared prompt construction using the base model's NATIVE chat template.

Qwen2.5-Coder-7B-Instruct is aligned to ChatML (<|im_start|>...<|im_end|>); both
trainers (train_moe_apr.py, train_lora_xcodeeval.py) and the generator
(generate_moe_apr.py) build prompts through ``tokenizer.apply_chat_template`` here
so training and inference use an identical format.

Message layout:
    system    : coding-assistant preamble
    user      : "{instruction}\n\nBuggy code:\n{buggy_code}"
    assistant : {fixed_code}          # only present at training time
"""

from typing import Dict, List

SYSTEM_PROMPT = (
    "You are an exceptionally intelligent coding assistant that consistently "
    "delivers accurate and reliable responses to user instructions."
)


def build_chat_messages(instruction: str, buggy_code: str) -> List[Dict[str, str]]:
    """System + user turns (no assistant turn -> for the generation prompt)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\nBuggy code:\n{buggy_code}"},
    ]


def render_chat_prompt(tokenizer, instruction: str, buggy_code: str) -> str:
    """Prompt STRING ending with the assistant generation header (for inference)."""
    return tokenizer.apply_chat_template(
        build_chat_messages(instruction, buggy_code),
        tokenize=False,
        add_generation_prompt=True,
    )


def build_icl_messages(
    instruction: str, buggy_code: str, exemplars: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Few-shot ICL layout (RING-style): system + k (user, assistant) exemplar
    turns + the real query turn.

    Exemplar user turns use a compact instruction (no problem description) so a
    2-shot prompt stays within budget; assistant turns are the raw fixed code,
    matching the SFT output format the finetuned rows were scored under.
    Each exemplar: {"lang": ..., "buggy": ..., "fixed": ...}.
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in exemplars:
        msgs.append({
            "role": "user",
            "content": f"Fix the buggy {ex['lang']} code.\n\nBuggy code:\n{ex['buggy']}",
        })
        msgs.append({"role": "assistant", "content": ex["fixed"]})
    msgs.append({"role": "user", "content": f"{instruction}\n\nBuggy code:\n{buggy_code}"})
    return msgs


def render_icl_prompt(
    tokenizer, instruction: str, buggy_code: str, exemplars: List[Dict[str, str]]
) -> str:
    return tokenizer.apply_chat_template(
        build_icl_messages(instruction, buggy_code, exemplars),
        tokenize=False,
        add_generation_prompt=True,
    )


COT_DIRECTIVE = (
    "First, briefly analyze step by step what is wrong with the code. "
    "Then output the complete fixed code in a single fenced code block (```)."
)


def cot_instruction(instruction: str) -> str:
    """Zero-shot CoT variant: swap the 'code only' directive for analyze-then-fix."""
    base = instruction.replace(
        "Return only the fixed code without extra explanation.", ""
    ).rstrip()
    return f"{base}\n\n{COT_DIRECTIVE}"


def tokenize_example(tokenizer, max_len: int, train_on_inputs: bool, row: Dict):
    """SFT tokenization: full conversation, loss only on the assistant response.

    Uses the native chat template. The prompt (system+user+assistant-header) is a
    strict token prefix of the full render, so masking the first ``len(prompt)``
    tokens leaves loss on ``{output}<|im_end|>`` exactly.
    """
    msgs = build_chat_messages(row["instruction"], row["input"])
    prompt_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True
    )
    output = row.get("output", "") or ""
    full_ids = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": output}],
        tokenize=True,
        add_generation_prompt=False,
    )
    full_ids = full_ids[:max_len]
    labels = list(full_ids)

    if not train_on_inputs:
        p = min(len(prompt_ids), len(full_ids))
        if p >= len(full_ids):
            # Response fully truncated away: keep loss on the last token only so
            # CE stays defined (such samples contribute ~nothing but don't NaN).
            p = max(0, len(full_ids) - 1)
        for i in range(p):
            labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
