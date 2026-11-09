"""检索文本：title×3 权重 + content + tags（与 Rust doc_tokens 同序）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from .store import MemRecord
from .tokenize import tokenize

R = TypeVar("R")


@dataclass
class Scored(Generic[R]):
    score: float
    record: R


def doc_tokens(r: MemRecord) -> list[str]:
    toks = tokenize(r.content)
    toks.extend(tokenize(" ".join(r.tags)))
    for _ in range(3):
        toks.extend(tokenize(r.title))
    return toks


def bm25(records: list[R], query: str, limit: int, project: str) -> list[Scored[R]]:
    """BM25 打分（k1=1.2, b=0.75），降序返回；同分 ts 新者先。project 非空时过滤。"""
    q = tokenize(query)
    if not q:
        return []
    qtf: dict[str, int] = {}
    for t in q:
        qtf[t] = qtf.get(t, 0) + 1
    pool = list(records) if project == "" else [r for r in records if r.project == project]
    if not pool:
        return []
    docs: list[tuple[dict[str, int], int]] = []
    for r in pool:
        toks = doc_tokens(r)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        docs.append((tf, len(toks)))
    df: dict[str, int] = {}
    for tf, _ in docs:
        for t in tf:
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    avg_dl = sum(dl for _, dl in docs) / n
    k1 = 1.2
    b = 0.75
    scored: list[Scored[R]] = []
    for i, (tf, dl) in enumerate(docs):
        s = 0.0
        for term, qf in qtf.items():
            tfc = tf.get(term)
            if tfc is None:
                continue
            d = df.get(term, 1)
            idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
            s += qf * idf * ((tfc * (k1 + 1)) / (tfc + k1 * (1 - b + (b * dl) / max(avg_dl, 1e-9))))
        if s > 0:
            scored.append(Scored(s, pool[i]))
    scored.sort(key=lambda sc: (-sc.score, -sc.record.ts))
    return scored[:limit]


def default_title(content: str) -> str:
    """标题铁律：缺省截正文首句 ≤40 字。"""
    head = content.split("\n")[0] if content else ""
    cut = len(head)
    for sep in ("。", "！", "？", ". ", "! ", "? "):
        p = head.find(sep)
        if p >= 0 and p < cut:
            cut = p
    return head[:cut][:40]
