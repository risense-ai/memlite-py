"""余弦语义与 RRF 融合。"""

from memlite_py.cosine import cosine, semantic
from memlite_py.rrf import rrf_fuse

from conftest import rec


def test_cosine_basics():
    assert cosine([1, 0], [0, 1]) == 0
    assert abs(cosine([1, 0], [1, 0]) - 1) < 1e-9
    assert cosine([], [1]) == 0
    assert cosine([1], [1, 2]) == 0  # 长度不符
    assert cosine([0, 0], [1, 1]) == 0  # 零向量


def test_semantic_filters_project_and_missing_vectors():
    r1 = rec("1", 1.0, "a", "a")
    r2 = rec("2", 2.0, "b", "b", project="p")
    vecs = {"1": [1.0, 0.0]}
    hits = semantic([r1, r2], [1.0, 0.0], 5, "", vecs.get)
    assert [h.record.id for h in hits] == ["1"]
    hits_p = semantic([r1, r2], [1.0, 0.0], 5, "p", lambda _id: [1.0, 0.0])
    assert [h.record.id for h in hits_p] == ["2"]


def test_rrf_fuse_merges_and_ranks():
    a = rec("a", 3.0, "a", "a")
    b = rec("b", 2.0, "b", "b")
    c = rec("c", 1.0, "c", "c")
    list1 = [type("S", (), {"score": 0.9, "record": a})(), type("S", (), {"score": 0.5, "record": b})()]
    list2 = [type("S", (), {"score": 4.2, "record": b})(), type("S", (), {"score": 0.1, "record": c})()]
    fused = rrf_fuse([list1, list2], 3)
    # b 在两列表都在 → 累加最高；a 与 c 各只在一个列表
    assert fused[0].record.id == "b"
    ids = [f.record.id for f in fused]
    assert set(ids) == {"a", "b", "c"}


def test_rrf_limit():
    recs = [rec(str(i), float(i), "t", "共同词") for i in range(5)]
    lst = [type("S", (), {"score": 1.0, "record": r})() for r in recs]
    assert len(rrf_fuse([lst], 2)) == 2
