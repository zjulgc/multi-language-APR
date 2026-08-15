"""Same-language exemplar retrieval for the few-shot ICL baseline (RING-style).

Bank rows: {"lang": <xcek lang_cluster>, "buggy": ..., "fixed": ...}, built from
the perlang3k training split only (no eval leakage). Retrieval is identifier-set
Jaccard between the query's buggy code and each bank entry's buggy code —
deterministic and cheap, no model in the loop.
"""
import json
import re
from typing import Dict, List

_IDENT = re.compile(r"[A-Za-z_]\w+")
# buggy+fixed combined; keeps a 2-shot prompt comfortably under ~1.5k tokens
MAX_EXEMPLAR_CHARS = 2400


def _toks(code: str) -> set:
    return set(_IDENT.findall(code or ""))


def load_bank(path: str) -> Dict[str, list]:
    bank: Dict[str, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r["buggy"]) + len(r["fixed"]) > MAX_EXEMPLAR_CHARS:
                continue
            bank.setdefault(r["lang"], []).append((_toks(r["buggy"]), r["buggy"], r["fixed"]))
    return bank


def retrieve(bank: Dict[str, list], lang: str, query_code: str, k: int = 2) -> List[Dict[str, str]]:
    cands = bank.get(lang, [])
    q = _toks(query_code)
    scored = []
    for toks, buggy, fixed in cands:
        union = len(q | toks)
        scored.append(((len(q & toks) / union) if union else 0.0, buggy, fixed))
    scored.sort(key=lambda t: -t[0])
    return [{"lang": lang, "buggy": b, "fixed": f} for _, b, f in scored[:k]]
