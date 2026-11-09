"""Memlite 主类端到端：remember/recall/show/因果操作/更新/状态/日志/压实。"""

import asyncio
import json
import os
import time

import pytest

from memlite_py import Memlite, MemliteConfig, MemliteDeps, RememberOpts, SensitiveError
from memlite_py.config import ConfigCtx


def make(data_dir, root):
    """hermetic：ctx 指向测试临时家目录，项目级无命中绝不回退到真实用户库。"""
    ctx = ConfigCtx(env={"HOME": root, "USERPROFILE": root}, cwd=root, home=root)
    return Memlite(MemliteConfig(data_dir=data_dir), MemliteDeps(ctx=ctx))


def test_remember_and_recall_roundtrip(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    rid = asyncio.run(mem.remember("甲关键词 正文内容", RememberOpts(title="甲题")))
    assert len(rid) == 12
    res = asyncio.run(mem.recall("甲关键词"))
    assert res.backend == "bm25"
    assert res.fallback == ""
    assert res.hits and res.hits[0].id == rid and res.hits[0].title == "甲题"
    assert res.ms >= 0
    # 默认标题：截首句
    rid2 = asyncio.run(mem.remember("第一句。第二句"))
    assert mem.show(rid2).record.title == "第一句"


def test_remember_with_parent_chain(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    a = asyncio.run(mem.remember("因正文", RememberOpts(title="因")))
    b = asyncio.run(mem.remember("果正文", RememberOpts(title="果", parent=a)))
    hit = asyncio.run(mem.recall("果正文")).hits[0]
    assert [x.id for x in hit.ancestors] == [a]
    assert [x.id for x in hit.children] == []
    hit_a = asyncio.run(mem.recall("因正文")).hits[0]
    assert [x.id for x in hit_a.children] == [b]
    with pytest.raises(ValueError, match="目标因"):
        asyncio.run(mem.remember("正文", RememberOpts(parent="nonexistent")))


def test_sensitive_gate(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    with pytest.raises(SensitiveError) as ei:
        asyncio.run(mem.remember("口令 password=x1x2x3x4", RememberOpts(title="t")))
    assert "拒绝入库" in str(ei.value)
    assert mem.store.count() == 0
    # 关闸放行
    mem2 = Memlite(MemliteConfig(data_dir=tmp_dir, sensitive_gate=False))
    rid = asyncio.run(mem2.remember("password=x1x2x3x4"))
    assert mem2.store.get(rid)


def test_show_and_chain(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    a = asyncio.run(mem.remember("因" * 100, RememberOpts(title="因条")))
    b = asyncio.run(mem.remember("果正文", RememberOpts(title="果条", parent=a)))
    view = mem.show(b)
    assert view.record.id == b
    assert view.ancestors[0].id == a
    assert any("因链" in ln for ln in view.causal_lines)
    cv = mem.chain(b, depth=2)
    assert cv.entry.id == b
    assert cv.ancestors[0].id == a
    assert cv.descendants == []
    cv_a = mem.chain(a, depth=2)
    assert [d[1].id for d in cv_a.descendants] == [b]
    assert cv_a.descendants[0][0] == 1
    assert mem.show("zzzzzzzz") is None
    # 前缀解析
    assert mem.show(b[:8]).record.id == b


def test_attach_promote_demote(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    a = asyncio.run(mem.remember("甲", RememberOpts(title="甲")))
    b = asyncio.run(mem.remember("乙", RememberOpts(title="乙")))
    r = mem.attach(b, a)
    assert r == {"child": b, "parent": a}
    with pytest.raises(ValueError, match="防环|不能挂自己"):
        mem.attach(a, b)
    p = mem.promote(b)  # 乙上浮到根（甲无父）
    assert p == {"id": b, "new_parent": ""}
    assert p["new_parent"] == ""


def test_forget_and_list(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    a = asyncio.run(mem.remember("甲正文", RememberOpts(title="甲")))
    time.sleep(0.02)  # Windows 计时器精度：确保 ts 严格可分
    b = asyncio.run(mem.remember("乙正文", RememberOpts(title="乙")))
    assert [r.id for r in mem.list(10)] == [b, a]  # ts 降序
    assert mem.forget(a[:8]) == a
    with pytest.raises(ValueError, match="记忆不存在"):
        mem.forget("nonexistent")


def test_update_keeps_parent_and_rejects_empty(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    a = asyncio.run(mem.remember("甲", RememberOpts(title="甲")))
    b = asyncio.run(mem.remember("乙", RememberOpts(title="乙", parent=a, tags=["old"])))
    upd = asyncio.run(mem.update(b, {"title": "乙新", "tags": ["new"], "extra": {"kind": "Task"}}))
    assert upd.id == b and upd.title == "乙新" and upd.embedded is False
    got = mem.store.get(b)
    assert got.tags == ["new"]
    assert got.extra.get("parent_id") == a  # parent 不随 patch 变
    assert got.extra.get("kind") == "Task"
    with pytest.raises(ValueError, match="content 不能为空"):
        asyncio.run(mem.update(b, {"content": "  "}))
    with pytest.raises(ValueError, match="记忆不存在"):
        asyncio.run(mem.update("zzzzzzzz", {"content": "x"}))


def test_tree_modes(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    a = asyncio.run(mem.remember("甲", RememberOpts(title="根条")))
    b = asyncio.run(mem.remember("乙", RememberOpts(title="叶条", parent=a)))
    rendered = mem.tree()
    assert any("🌳" in ln for ln in rendered) and any("叶条" in ln for ln in rendered)
    out = mem.tree(mode="outline")
    assert out["count"] == 2 and out["lines"][0].endswith("根条")
    j = mem.tree(mode="json", from_=b)
    assert j[0].id == b and j[0].leaf and j[0].children == []
    with pytest.raises(ValueError):
        mem.tree(from_="zzzzzzzz")


def test_querylog_written_and_exported(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    asyncio.run(mem.remember("甲关键词 正文", RememberOpts(title="甲题")))
    asyncio.run(mem.recall("甲关键词"))
    asyncio.run(mem.recall("零命中查询"))
    stats = mem.querylog_stats(5)
    assert stats["queries_window"] == 2
    assert stats["backend"]["bm25"] == 2
    assert stats["zero_hit"] == 1
    recent = mem.querylog_recent(10)
    assert len(recent) == 2 and recent[0]["query"] == "零命中查询"
    out = str(tmp_path / "train.jsonl")
    n = mem.querylog_export(out)
    assert n == 2
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert rows[0]["positive"] and rows[0]["query"]


def test_status_and_compact(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    asyncio.run(mem.remember("甲正文", RememberOpts(title="甲")))
    st = asyncio.run(mem.status())
    assert st["records"] == 1
    assert st["data_dir"] == tmp_dir
    assert st["embedding"]["provider"] == "none"
    assert st["vectors"] == 0
    assert st["querylog"] == "memory.querylog.jsonl"
    assert st["sync"]["repo"] is None or isinstance(st["sync"], dict)
    assert mem.compact() >= 1
    assert mem.store.count() == 1


def test_open_close_and_store_lazy(tmp_dir, tmp_path):
    mem = make(tmp_dir, str(tmp_path))
    assert not os.path.exists(tmp_dir)  # 构造不落盘（Store 装载惰性，与 JS 一致）
    rid = asyncio.run(mem.remember("首写正文", RememberOpts(title="t")))
    assert os.path.exists(tmp_dir)  # 首写创建目录
    mem.close()
    mem.open()  # 幂等
    assert mem.store.count() == 1 and mem.store.get(rid) is not None


def test_reembed_with_stub_embedder(tmp_dir, tmp_path):
    root = str(tmp_path)
    class StubEmb:
        dim = 3

        async def embed(self, text, opts=None):
            return [1.0, 0.0, 0.5]

    mem = Memlite(MemliteConfig(data_dir=tmp_dir), MemliteDeps(embedder=StubEmb(), ctx=ConfigCtx(env={"HOME": root, "USERPROFILE": root}, cwd=root, home=root)))
    a = asyncio.run(mem.remember("甲", RememberOpts(title="甲")))
    b = asyncio.run(mem.remember("乙", RememberOpts(title="乙")))
    # remember 侧车已经嵌入两条
    assert mem.vectors.count() == 2
    res = asyncio.run(mem.reembed())
    assert res.embedded == 0 and res.failed == 0 and res.total == 2
    assert mem.vectors.get(a) == [1.0, 0.0, 0.5]
    # 清掉整个侧车（两文件都删——ids 在册即算已有向量）后回填
    os.remove(os.path.join(tmp_dir, "emb.f32"))
    os.remove(os.path.join(tmp_dir, "emb-ids.txt"))
    mem.close()
    mem = Memlite(MemliteConfig(data_dir=tmp_dir), MemliteDeps(embedder=StubEmb()))
    res2 = asyncio.run(mem.reembed())
    assert res2.embedded == 2
    assert mem.vectors.has(a) and mem.vectors.has(b)


def test_host_embedder_fallback_to_bm25_on_error(tmp_dir):
    class BadEmb:
        dim = 2

        async def embed(self, text, opts=None):
            raise RuntimeError("host embed unavailable")

    warns = []

    class Log:
        def warn(self, msg):
            warns.append(msg)

    mem = Memlite(MemliteConfig(data_dir=tmp_dir))
    mem.deps.embedder = BadEmb()
    mem.deps.logger = Log()
    asyncio.run(mem.remember("甲关键词 正文", RememberOpts(title="甲题")))
    res = asyncio.run(mem.recall("甲关键词"))
    assert res.backend == "bm25" and res.hits  # 语义臂失败不拦词面命中
    assert warns  # 告警可见
