"""RRF 融合（k=60）：各候选列表按位次累加 1/(k+位次+1)，只看位次不吃量纲——
语义余弦与 BM25 分数可直接合。同 id 合并，同分 ts 新者先（与单档排序同口径）。"""

from __future__ import annotations

from typing import Iterable

from .bm25 import Scored

K = 60


def rrf_fuse(lists: Iterable[Iterable[Scored]], limit: int) -> list[Scored]:
    fused: dict[str, Scored] = {}
    for rlist in lists:
        for rank, item in enumerate(rlist):
            rec = item.record
            slot = fused.get(rec.id)
            if slot is not None:
                slot.score += 1 / (K + rank + 1)
            else:
                fused[rec.id] = Scored(1 / (K + rank + 1), rec)
    out = list(fused.values())
    out.sort(key=lambda s: (-s.score, -s.record.ts))
    return out[:limit]
