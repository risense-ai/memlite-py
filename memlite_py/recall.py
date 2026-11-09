"""recall 主流程（含项目级→用户级回退）——Rust main.rs / memlite-js recall.ts 的移植。

BM25 永远算——既是融合臂也是兜底；语义可用且有命中才进 RRF；
嵌入失败永不抛异常——回退 BM25。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .bm25 import Scored, bm25
from .cosine import semantic
from .rrf import rrf_fuse
from .store import MemRecord, Store
from .vectors import VectorIndex

RecallBackend = str  # 'bm25' | 'rrf' | 'semantic'


class Logger(Protocol):
    def warn(self, msg: str) -> None: ...


@dataclass
class SearchOnceOpts:
    limit: int
    project: str
    """强制词面档"""
    bm25_only: bool = False


@dataclass
class SearchOnceResult:
    backend: RecallBackend
    hits: list[Scored[MemRecord]]


def _embedder_dim(embedder: "EmbedderLike") -> int:
    """dim 未定按 1（可试语义）；dim=0 = 纯词面档跳过语义臂。"""
    dim = getattr(embedder, "dim", None)
    return 1 if dim is None else dim


class EmbedderLike(Protocol):
    async def embed(self, text: str, *, timeout_ms: int | None = None) -> list[float]: ...

    dim: int | None


async def search_once(
    records: list[MemRecord],
    vectors: VectorIndex | None,
    embedder: EmbedderLike | None,
    query: str,
    opts: SearchOnceOpts,
    logger: Logger | None = None,
) -> SearchOnceResult:
    """单库检索一次：BM25 必算 → 嵌入 → RRF。vectors 为 None 时跳过语义臂。

    嵌入失败回退 BM25（logger.warn 提示），永不阻塞。"""
    pool = max(opts.limit * 10, 50)
    bm = bm25(records, query, pool, opts.project)
    if opts.bm25_only:
        return SearchOnceResult(backend="bm25", hits=bm[: opts.limit])
    if embedder is not None and _embedder_dim(embedder) != 0 and vectors is not None:
        try:
            qv = await embedder.embed(query)
            if len(qv) > 0:
                sem = semantic(records, qv, pool, opts.project, vectors.get)
                if len(sem) > 0:
                    return SearchOnceResult(backend="rrf", hits=rrf_fuse([sem, bm], opts.limit))
        except Exception as e:
            if logger is not None:
                logger.warn(f"嵌入失败，回退 BM25: {e}")
    return SearchOnceResult(backend="bm25", hits=bm[: opts.limit])


@dataclass
class RecallSearchOpts(SearchOnceOpts):
    embedder: EmbedderLike | None = None
    logger: Logger | None = None


async def recall_search(dir: str, query: str, opts: RecallSearchOpts) -> SearchOnceResult:
    """目录级检索：每次开新 Store（与 Rust 短进程语义一致；常驻场景用 Memlite.recall）。"""
    s = Store(dir)
    vectors = VectorIndex(dir) if opts.embedder is not None else None
    return await search_once(
        s.all(), vectors, opts.embedder, query, opts, opts.logger
    )


def perf_ms_since(t0: float) -> int:
    """自 t0=time.perf_counter() 起的毫秒数（四舍五入）。"""
    return round((time.perf_counter() - t0) * 1000)
