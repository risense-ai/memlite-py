"""余弦：零向量/长度不符返回 0（与 Rust cosine 同语义）。"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeVar

from .bm25 import Scored

R = TypeVar("R")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == 0 or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    na = na**0.5
    nb = nb**0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic(
    records: Iterable[R],
    query_emb: Sequence[float],
    limit: int,
    project: str,
    vec_getter: Callable[[str], Sequence[float] | None],
) -> list[Scored[R]]:
    """余弦语义：vec_getter 从侧车取向量，只排有向量的记录（与 Rust semantic 同语义）。"""
    if len(query_emb) == 0:
        return []
    scored: list[Scored[R]] = []
    for r in records:
        if project != "" and r.project != project:
            continue
        v = vec_getter(r.id)
        if v is None:
            continue
        s = cosine(query_emb, v)
        if s > 0:
            scored.append(Scored(s, r))
    scored.sort(key=lambda sc: (-sc.score, -sc.record.ts))
    return scored[:limit]
