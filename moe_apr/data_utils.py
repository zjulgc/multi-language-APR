"""Data utilities for per-language MoE-LoRA training.

Two main jobs:

1. ``load_language_dataset`` - load the per-language JSONL files into a single
   HuggingFace ``Dataset``, with an extra ``language`` column.
2. ``BalancedLanguageSampler`` - PyTorch sampler that draws indices from each
   language with a configurable ratio (default uniform across languages). This
   is the core balanced sampling mechanism for training.

The sampler is plug-compatible with HuggingFace ``Trainer`` via
``Trainer._get_train_sampler`` override (see ``trainer.py``).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler

try:  # optional dependency
    from datasets import Dataset, concatenate_datasets, load_dataset
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore

# Canonical 11 xCodeEval languages -> per-language routing-expert index (0..10).
# MUST stay fixed so the routing diagnostic (purity / NMI) and any per-language
# analysis agree on which expert "owns" which language.
LANGS_CANONICAL = (
    "C", "C++", "C#", "Java", "Kotlin",
    "Python", "Javascript", "Ruby", "PHP", "Rust", "Go",
)
LANG_TO_EXPERT: Dict[str, int] = {lang: i for i, lang in enumerate(LANGS_CANONICAL)}
LANG_TO_EXPERT["JavaScript"] = LANG_TO_EXPERT["Javascript"]  # spelling alias

DEFAULT_LANGUAGES = LANGS_CANONICAL


def lang_to_expert(lang_cluster: str) -> Optional[int]:
    """Return the canonical per-language expert index (0..10), or None if unknown."""
    return LANG_TO_EXPERT.get(lang_cluster)


# Canonical bug-type (xCodeEval bug_exec_outcome) -> expert index (5 classes -> 0..4).
# The alternative cross-lingual "repair axis": label experts by the failure mode of
# the buggy program rather than by its programming language. Used only as a routing
# diagnostic target (--route_by bug_type).
BUGTYPES_CANONICAL = (
    "COMPILATION_ERROR",
    "RUNTIME_ERROR",
    "WRONG_ANSWER",
    "TIME_LIMIT_EXCEEDED",
    "MEMORY_LIMIT_EXCEEDED",
)
BUGTYPE_TO_EXPERT: Dict[str, int] = {b: i for i, b in enumerate(BUGTYPES_CANONICAL)}


def bugtype_to_expert(bug_exec_outcome: str) -> Optional[int]:
    """Return the canonical expert index (0..4) for a bug_exec_outcome, or None if unknown."""
    return BUGTYPE_TO_EXPERT.get(bug_exec_outcome)


@dataclass
class LanguageDataPaths:
    """Paths to per-language SFT JSONL produced by scripts/prep_perlang_data.py."""

    base_dir: str
    languages: Sequence[str] = DEFAULT_LANGUAGES
    split: str = "train"  # "train" | "validation"

    def __post_init__(self) -> None:
        self.files: Dict[str, str] = {
            l: os.path.join(self.base_dir, "by_language", l, f"{self.split}_sft.jsonl")
            for l in self.languages
        }
        for l, path in self.files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing per-language file for {l}: {path}")


def load_language_dataset(paths: LanguageDataPaths):  # -> Dataset
    """Load per-language JSONLs into a single Dataset with an extra `language` column.

    Returned dataset has columns: instruction, input, output, lang_cluster,
    src_uid, ..., language.
    """
    if Dataset is None:
        raise ImportError("datasets library is required: pip install datasets")

    parts = []
    for language, path in paths.files.items():
        ds = load_dataset("json", data_files=path, split="train")
        ds = ds.add_column("language", [language] * len(ds))
        parts.append(ds)

    return concatenate_datasets(parts)


class BalancedLanguageSampler(Sampler[int]):
    """Weighted sampler that draws from each language with a configurable ratio.

    Args:
        language_per_index: ``language[i]`` for each index i in the (concatenated) dataset.
        language_weights:   target sampling probability per language (renormalized).
                            Default: uniform over the distinct languages present.
        num_samples:        total number of indices yielded per epoch. Default: len(dataset).
        seed:               random seed for reproducibility.
        replacement:        sample with replacement (default True).

    With 11 languages and uniform weights, every batch is expected to be ~1/11
    from each language in expectation, regardless of the raw data imbalance.
    """

    def __init__(
        self,
        language_per_index: Sequence[str],
        language_weights: Optional[Dict[str, float]] = None,
        num_samples: Optional[int] = None,
        seed: int = 0,
        replacement: bool = True,
    ) -> None:
        self.language_per_index = list(language_per_index)
        self.num_samples = num_samples or len(self.language_per_index)
        self.seed = seed
        self.replacement = replacement

        # Build language -> [indices] map.
        self._lang_to_indices: Dict[str, List[int]] = {}
        for i, l in enumerate(self.language_per_index):
            self._lang_to_indices.setdefault(l, []).append(i)

        languages = sorted(self._lang_to_indices.keys())
        if language_weights is None:
            w = {l: 1.0 for l in languages}
        else:
            w = {l: float(language_weights.get(l, 0.0)) for l in languages}
        total_w = sum(w.values())
        if total_w <= 0:
            raise ValueError("All language weights are zero")
        self.language_weights = {l: w[l] / total_w for l in languages}
        self.languages = languages

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        weights = [self.language_weights[l] for l in self.languages]

        for _ in range(self.num_samples):
            language = rng.choices(self.languages, weights=weights, k=1)[0]
            pool = self._lang_to_indices[language]
            yield pool[rng.randrange(len(pool))]

    def language_distribution(self) -> Dict[str, int]:
        """Return raw language -> count of indices in the underlying data."""
        return {l: len(idxs) for l, idxs in self._lang_to_indices.items()}


def make_balanced_sampler(
    dataset,  # HF Dataset with a 'language' column
    language_weights: Optional[Dict[str, float]] = None,
    num_samples: Optional[int] = None,
    seed: int = 0,
) -> BalancedLanguageSampler:
    if "language" not in dataset.column_names:
        raise KeyError("Dataset must have a 'language' column. Use load_language_dataset().")
    return BalancedLanguageSampler(
        language_per_index=dataset["language"],
        language_weights=language_weights,
        num_samples=num_samples,
        seed=seed,
    )
