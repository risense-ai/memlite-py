"""BM25 打分、title×3 权重、project 过滤、标题铁律。"""

from memlite_py.bm25 import bm25, default_title, doc_tokens

from conftest import rec


def test_bm25_finds_and_ranks():
    r1 = rec("1", 1.0, "rust 编译", "cargo build 慢用 sccache")
    r2 = rec("2", 2.0, "菜谱", "番茄炒蛋的做法")
    r3 = rec("3", 3.0, "", "rust 编译缓存 sccache 配置")
    hits = bm25([r1, r2, r3], "rust 编译", 3, "")
    assert len(hits) > 0
    assert hits[0].record.id in ("1", "3")
    assert all(h.score > 0 for h in hits)


def test_bm25_title_weight():
    r1 = rec("1", 1.0, "kubernetes 部署", "无关内容")
    r2 = rec("2", 2.0, "别的", "提到一次 kubernetes")
    hits = bm25([r1, r2], "kubernetes", 3, "")
    assert hits[0].record.id == "1"


def test_bm25_project_filter_and_fallback_empty():
    r1 = rec("1", 1.0, "rust", "rust 内容", project="proj-a")
    r2 = rec("2", 2.0, "rust", "rust 内容二", project="proj-b")
    hits = bm25([r1, r2], "rust", 3, "proj-b")
    assert len(hits) == 1 and hits[0].record.id == "2"
    # 未命中 project 的库为空 → 空表
    assert bm25([r1], "rust", 3, "proj-zzz") == []


def test_bm25_tie_breaks_by_ts_desc():
    a = rec("a", 1.0, "t", "共同词")
    b = rec("b", 2.0, "t", "共同词")
    hits = bm25([a, b], "共同词", 3, "")
    assert [h.record.id for h in hits] == ["b", "a"]


def test_bm25_tags_counted():
    r = rec("1", 1.0, "标题", "正文没提", tags=["redis"])
    hits = bm25([r], "redis", 3, "")
    assert len(hits) == 1


def test_doc_tokens_order():
    r = rec("1", 1.0, "标题", "正文", tags=["tag"])
    toks = doc_tokens(r)
    # title×3 + content + tags
    assert toks.count("标题") >= 3
    assert "正文" in toks
    assert "tag" in toks


def test_default_title_first_sentence():
    assert default_title("第一句。第二句继续") == "第一句"
    assert default_title("no sep here\nsecond line") == "no sep here"
    assert len(default_title("x" * 100)) == 40
    assert default_title("") == ""
