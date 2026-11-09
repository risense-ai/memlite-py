"""检索主流程：BM25 兜底、语义臂、RRF、项目级回退。"""

import asyncio

from memlite_py.recall import RecallSearchOpts, recall_search, search_once
from memlite_py.store import Store
from memlite_py.vectors import VectorIndex

from conftest import rec


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]] | None = None, fail: bool = False, dim: int | None = 2):
        self.vectors = vectors or {}
        self.fail = fail
        self.dim = dim
        self.calls = 0

    async def embed(self, text, opts=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embed down")
        return self.vectors.get(text, [0.1, 0.9])


def test_search_once_bm25_fallback_when_no_embedder():
    rs = asyncio.run(search_once([rec("1", 1.0, "rust", "rust 正文")], None, None, "rust", RecallSearchOpts(limit=3, project="")))
    assert rs.backend == "bm25" and len(rs.hits) == 1


def test_search_once_bm25_only():
    emb = FakeEmbedder({"rust": [1.0, 0.0]})
    rs = asyncio.run(search_once([rec("1", 1.0, "rust", "rust 正文")], VectorIndexStoreEmpty(), emb, "rust", RecallSearchOpts(limit=3, project="", bm25_only=True)))
    assert rs.backend == "bm25" and emb.calls == 0


def test_search_once_rrf_when_semantic_hits():
    records = [rec("1", 1.0, "rust 编译", "rust 正文"), rec("2", 2.0, "菜谱", "番茄炒蛋")]
    vectors = VectorIndexVectors({"1": [0.9, 0.1]})
    emb = FakeEmbedder({"rust": [0.9, 0.1]})
    rs = asyncio.run(search_once(records, vectors, emb, "rust", RecallSearchOpts(limit=3, project="")))
    assert rs.backend == "rrf"
    assert rs.hits[0].record.id == "1"


def test_search_once_embed_failure_falls_back():
    records = [rec("1", 1.0, "rust 编译", "rust 正文")]
    emb = FakeEmbedder(fail=True)
    warns = []

    class Log:
        def warn(self, msg):
            warns.append(msg)

    rs = asyncio.run(search_once(records, VectorIndexVectors({"1": [1.0, 0.0]}), emb, "rust", RecallSearchOpts(limit=3, project=""), Log()))
    assert rs.backend == "bm25" and warns


def test_search_once_dim0_skips_semantic():
    emb = FakeEmbedder(dim=0)  # none 档
    rs = asyncio.run(search_once([rec("1", 1.0, "rust", "x")], VectorIndexVectors({}), emb, "rust", RecallSearchOpts(limit=3, project="")))
    assert rs.backend == "bm25" and emb.calls == 0


def test_recall_search_directory_level(tmp_dir):
    s = Store(tmp_dir)
    s.put("Redis 大 key 排查正文", "redis 排查", [], "")
    rs = asyncio.run(recall_search(tmp_dir, "redis", RecallSearchOpts(limit=5, project="")))
    assert len(rs.hits) == 1 and rs.hits[0].record.title == "redis 排查"


# ---- 极简向量侧车桩 -------------------------------------------------------

class VectorIndexVectors:
    def __init__(self, m):
        self._m = m
        self.dim = 2

    def get(self, rid):
        return self._m.get(rid)


class VectorIndexStoreEmpty(VectorIndexVectors):
    def __init__(self):
        super().__init__({})
