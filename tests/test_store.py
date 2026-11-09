"""存储层：put/delete/LWW/resolve/因果结构/轮转/压实。"""

import json
import os

import pytest

from memlite_py.store import DATA_FILE, Store, parent_id, parse_record, serialize_record


def test_put_and_roundtrip(tmp_dir):
    s = Store(tmp_dir)
    rid = s.put("正文内容", "标题", ["t1", " t2 ", ""], "proj")
    rec = s.get(rid)
    assert rec is not None
    assert rec.title == "标题"
    assert rec.tags == ["t1", "t2"]  # 清洗空白与空标签
    assert rec.project == "proj"
    assert parent_id(rec) == ""
    assert len(rid) == 12
    # 重新装载一致
    s2 = Store(tmp_dir)
    assert s2.count() == 1 and s2.get(rid).content == "正文内容"


def test_put_empty_content_rejected(tmp_dir):
    s = Store(tmp_dir)
    with pytest.raises(ValueError, match="content 为空"):
        s.put("   ", "t", [], "")


def test_put_child_requires_parent(tmp_dir):
    s = Store(tmp_dir)
    with pytest.raises(ValueError, match="目标因"):
        s.put_child("正文", "t", [], "", "nonexistent")
    p = s.put("父正文", "父", [], "")
    c = s.put_child("子正文", "子", [], "", p)
    assert parent_id(s.get(c)) == p


def test_delete_tombstone_and_revive(tmp_dir):
    s = Store(tmp_dir)
    rid = s.put("正文", "t", [], "")
    s.delete(rid)
    assert s.count() == 0
    with pytest.raises(ValueError):
        s.delete(rid)
    # del 后 put 复活（LWW）
    s.put_full(content="复活正文", title="t2", id=rid)
    assert s.get(rid).content == "复活正文"


def test_lww_put_with_id_keeps_extra(tmp_dir):
    s = Store(tmp_dir)
    rid = s.put_full(content="原始", title="t", tags=[], project="", extra={"kind": "Context", "importance": "important"})
    s.put_with_id(rid, "改后", "t", [], "")
    got = s.get(rid)
    assert got.content == "改后"
    assert got.extra.get("kind") == "Context"  # extra 全量保留
    assert got.extra.get("parent_id") == ""
    # parent_id 不能被 patch 偷改
    s.put_with_id(rid, "再改", "t", [], "", extra_patch={"parent_id": "evil"})
    assert parent_id(s.get(rid)) == ""


def test_put_full_rejects_existing_id(tmp_dir):
    s = Store(tmp_dir)
    rid = s.put("a", "t", [], "")
    with pytest.raises(ValueError, match="id 已存在"):
        s.put_full(content="b", title="t", id=rid)


def test_resolve_three_levels(tmp_dir):
    s = Store(tmp_dir)
    rid = s.put("正文", "唯一纲名", [], "")
    assert s.resolve(rid).id == rid
    assert s.resolve(rid[:8]).id == rid
    assert s.resolve("#唯一纲名").id == rid  # 纲名直挂（剥 #）
    assert s.resolve("zzzz") is None
    # 歧义前缀 → None（首字符相同才可断言）
    other = s.put("正文二", "t2", [], "")
    if rid[:1] == other[:1]:
        assert s.resolve(rid[:1]) is None


def test_reparent_checked_cycle(tmp_dir):
    s = Store(tmp_dir)
    a = s.put("甲", "甲", [], "")
    b = s.put_child("乙", "乙", [], "", a)
    c = s.put_child("丙", "丙", [], "", b)
    with pytest.raises(ValueError, match="不能挂自己"):
        s.reparent_checked(a, a)
    with pytest.raises(ValueError, match="防环"):
        s.reparent_checked(a, c)  # c 在 a 的后代链上
    with pytest.raises(ValueError, match="防环"):
        s.reparent_checked(a, b)  # b 的因链含 a，挂过去也成环
    # 合法改挂：丙升根
    ts_before = s.get(c).ts
    s.reparent_checked(c, "")
    assert parent_id(s.get(c)) == ""
    assert s.get(c).ts == ts_before  # reparent 保留原 ts


def test_promote_grandparent(tmp_dir):
    s = Store(tmp_dir)
    a = s.put("甲", "甲", [], "")
    b = s.put_child("乙", "乙", [], "", a)
    c = s.put_child("丙", "丙", [], "", b)
    s.reparent_checked(b, "")  # b 升根
    assert parent_id(s.get(c)) == b and parent_id(s.get(b)) == ""


def test_ancestor_chain_root_first(tmp_dir):
    s = Store(tmp_dir)
    a = s.put("甲", "甲", [], "")
    b = s.put_child("乙", "乙", [], "", a)
    c = s.put_child("丙", "丙", [], "", b)
    assert s.ancestor_chain(c) == [a, b]
    assert s.ancestor_chain(a) == []


def test_children_and_roots_sorted_by_ts(tmp_dir):
    s = Store(tmp_dir)
    r = s.put("根", "根", [], "")
    ids = [s.put_child(f"子{i}", f"子{i}", [], "", r) for i in range(3)]
    assert [c.id for c in s.children(r)] == ids  # 存入序
    assert [x.id for x in s.roots()] == [r]


def test_rotation_and_reload(tmp_dir):
    s = Store(tmp_dir, chunk_bytes=512)  # 缩小阈值测轮转
    for i in range(20):
        s.put("正文" * 20, f"标题{i}", [], "")
    files = os.listdir(tmp_dir)
    assert any(f.startswith("memory-0001") for f in files)
    s2 = Store(tmp_dir)
    assert s2.count() == 20


def test_versions_and_compact(tmp_dir):
    s = Store(tmp_dir)
    rid = s.put("第一版", "t", [], "")
    s.put_with_id(rid, "第二版", "t", [], "")
    vers = s.versions(rid)
    assert [v.content for v in vers] == ["第一版", "第二版"]
    rid2 = s.put("另一条", "t2", [], "")
    s.delete(rid2)
    before = s.line_count()
    removed = s.compact()
    assert removed == before
    s2 = Store(tmp_dir)
    assert s2.count() == 1
    assert os.listdir(tmp_dir) == [DATA_FILE]  # 归档已删


def test_bad_lines_counted(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    rid_line = json.dumps({"op": "put", "id": "aaaaaaaaaaaa", "ts": 1.0, "title": "", "content": "x", "tags": [], "project": ""}, ensure_ascii=False)
    with open(os.path.join(tmp_dir, DATA_FILE), "w", encoding="utf-8") as f:
        f.write("{broken json\n")
        f.write(rid_line + "\n")
        f.write('{"op":"nope"}\n')
    s = Store(tmp_dir)
    assert s.bad_lines == 2
    assert s.count() == 1


def test_tag_project_counts(tmp_dir):
    s = Store(tmp_dir)
    s.put("a", "a", ["x", "y"], "p1")
    s.put("b", "b", ["x"], "p1")
    s.put("c", "c", [], "")
    assert s.tag_counts() == [("x", 2), ("y", 1)]
    assert s.project_counts()[0] == ("p1", 2)
    assert ("（无）", 1) in s.project_counts()


def test_parse_and_serialize_shape():
    line = '{"op":"put","id":"abc123def456","ts":1792078629.79,"title":"t","content":"正文","tags":[],"project":"","parent_id":"","importance":"important"}'
    r = parse_record(line)
    assert r is not None
    assert r.extra == {"parent_id": "", "importance": "important"}
    out = json.loads(serialize_record(r))
    assert out["importance"] == "important" and out["parent_id"] == ""
    assert parse_record("not json") is None
    assert parse_record('{"op":"??"}') is None
