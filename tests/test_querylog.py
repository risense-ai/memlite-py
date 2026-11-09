"""检索日志：聚合、尾窗、导出训练集、Rust 形状行解析。"""

import json
import os

from memlite_py.querylog import Entry, Hit, LOG_FILE, Log, export_log


def entry(q, backend="bm25", hits=None, ts=1.0, fallback="", source="cli"):
    return Entry(ts=ts, query=q, backend=backend, limit=5, fallback=fallback, source=source, ms=3, hits=hits or [])


def test_append_and_reload(tmp_dir):
    log = Log(tmp_dir)
    log.append(tmp_dir, entry("甲", hits=[Hit(id="id1", score=0.5, title="t")], ts=100.0))
    log.append(tmp_dir, entry("乙", backend="rrf", hits=[Hit(id="id1", score=2.0, title="t"), Hit(id="id2", score=1.0, title="u")], ts=200.0))
    assert log.total_count() == 2
    assert log.count_of("id1") == 2 and log.count_of("id2") == 1
    log2 = Log(tmp_dir)  # 重载聚合一致
    assert log2.total_count() == 2 and log2.count_of("id1") == 2


def test_recent_newest_first(tmp_dir):
    log = Log(tmp_dir)
    for i in range(5):
        log.append(tmp_dir, entry(f"q{i}", ts=float(i)))
    assert [e.query for e in log.recent(3)] == ["q4", "q3", "q2"]


def test_stats_shape(tmp_dir):
    log = Log(tmp_dir)
    log.append(tmp_dir, entry("甲", backend="rrf", hits=[Hit(id="a", score=1.0, title="T甲")], ts=10.0))
    log.append(tmp_dir, entry("乙", hits=[], ts=20.0, fallback="user", source="web"))
    s = log.stats(3)
    assert s["queries_window"] == 2 and s["queries_total"] == 2
    assert s["backend"] == {"semantic": 0, "rrf": 1, "bm25": 1}
    assert s["fallback_user"] == 1
    assert s["source"] == {"cli": 1, "web": 1}
    assert s["zero_hit"] == 1
    assert s["hits_avg"] == 0.5
    assert s["since"] == 10.0 and s["until"] == 20.0
    assert s["top"][0] == {"id": "a", "count": 1, "title": "T甲"}
    assert s["distinct_recalled"] == 1


def test_recent_json_rounds_scores(tmp_dir):
    log = Log(tmp_dir)
    log.append(tmp_dir, entry("甲", hits=[Hit(id="a", score=0.123456789, title="t")]))
    r = log.recent_json(1)[0]
    assert r["hits"][0]["score"] == round(0.123456789 * 10000) / 10000


def test_export_training_set(tmp_dir, tmp_path):
    log = Log(tmp_dir)
    log.append(tmp_dir, entry("甲", hits=[Hit(id="a", score=0.9, title="t")], ts=50.0))
    log.append(tmp_dir, entry("乙", hits=[], ts=100.0))
    out = str(tmp_path / "sub" / "train.jsonl")
    assert export_log(tmp_dir, out, since=0) == 2
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert rows[0]["positive"] == ["a"]
    assert rows[0]["hits"][0]["score"] == 0.9
    assert "limit" in rows[0] and "fallback" in rows[0]
    # since 过滤
    assert export_log(tmp_dir, out, since=99) == 1
    with pytest.raises(ValueError, match="读日志失败"):
        export_log(os.path.join(tmp_dir, "none"), out, 0)


import pytest  # noqa: E402


def test_parses_rust_shape_lines(tmp_dir):
    """Rust 写的日志行（含未知来源）直接可读；坏行计数。"""
    line = '{"ts":1792078629.79,"query":"nginx","backend":"semantic","limit":10,"fallback":"","source":"dsh","ms":17,"hits":[{"id":"abc123def456","score":0.657,"title":"t"}]}'
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, LOG_FILE), "w", encoding="utf-8") as f:
        f.write(line + "\n" + "bad line\n")
    log = Log(tmp_dir)
    assert log.total_count() == 1 and log.bad_lines == 1
    assert log.count_of("abc123def456") == 1
