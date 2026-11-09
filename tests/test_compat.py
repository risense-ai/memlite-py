"""互通契约测试（compat.md）：手写 Rust/JS 形状的数据文件，验证 Python 端直接可读可写。"""

import asyncio
import json
import os
import struct

from memlite_py import Store, VectorIndex
from memlite_py.bm25 import bm25
from memlite_py.querylog import Log


RUST_PUT = (
    '{"op":"put","id":"abc123def456","ts":1792078629.79,'
    '"title":"rust 写 nginx","content":"Rust 写入的记忆：nginx 504 超时排查",'
    '"tags":["ops"],"project":"","parent_id":"","importance":"important","kind":"Context"}'
)
RUST_DEL = '{"op":"del","id":"dead00000000","ts":1792078700.00,"title":"","content":"","tags":[],"project":""}'
RUST_PUT_DEAD = '{"op":"put","id":"dead00000000","ts":1792078600.00,"title":"已删","content":"旧正文","tags":[],"project":""}'


def test_reads_rust_written_store(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "memory.jsonl"), "w", encoding="utf-8") as f:
        f.write(RUST_PUT_DEAD + "\n" + RUST_DEL + "\n" + RUST_PUT + "\n")
    s = Store(tmp_dir)
    assert s.bad_lines == 0
    assert s.count() == 1  # 墓碑生效
    r = s.get("abc123def456")
    assert r.title == "rust 写 nginx"
    assert r.extra == {"parent_id": "", "importance": "important", "kind": "Context"}
    assert r.ts == 1792078629.79
    hits = bm25(s.all(), "nginx 超时", 5, "")
    assert hits and hits[0].record.id == "abc123def456"
    # Python 写回的行仍是 flatten 形状，Rust serde 可解析
    all_lines = open(os.path.join(tmp_dir, "memory.jsonl"), encoding="utf-8").read().splitlines()
    put_line = json.loads([l for l in all_lines if '"abc123def456"' in l and '"put"' in l][0])
    assert put_line["importance"] == "important"
    rid = s.put("py 新条", "t", ["x"], "p")
    newline = [l for l in open(os.path.join(tmp_dir, "memory.jsonl"), encoding="utf-8").read().splitlines() if f'"{rid}"' in l][0]
    obj = json.loads(newline)
    assert obj["op"] == "put" and obj["parent_id"] == "" and obj["project"] == "p"


def test_python_rust_vector_sidecar_interop(tmp_dir):
    """Python 写 → 按 Rust 语义读回；Rust 字节 → Python 读。"""
    os.makedirs(tmp_dir, exist_ok=True)
    v = VectorIndex(tmp_dir)
    v.put("vecid1", [0.5, -0.25, 2.0])
    raw = open(os.path.join(tmp_dir, "emb.f32"), "rb").read()
    assert struct.unpack("<3f", raw) == (0.5, -0.25, 2.0)
    ids = open(os.path.join(tmp_dir, "emb-ids.txt"), encoding="utf-8").read().split("\n")
    assert ids == ["vecid1", ""]  # 尾换行
    # Rust/JS 端追加一条（模拟）
    with open(os.path.join(tmp_dir, "emb.f32"), "ab") as f:
        f.write(struct.pack("<3f", 1.0, 1.0, 1.0))
    with open(os.path.join(tmp_dir, "emb-ids.txt"), "a", encoding="utf-8") as f:
        f.write("vecid2\n")
    v2 = VectorIndex(tmp_dir)
    assert v2.dim == 3 and v2.count() == 2
    assert v2.get("vecid2") == [1.0, 1.0, 1.0]


def test_querylog_interop(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    line = '{"ts":1792078629.79,"query":"redis","backend":"rrf","limit":10,"fallback":"user","source":"cli","ms":42,"hits":[{"id":"abc123def456","score":0.6571,"title":"rust 写 nginx"}]}'
    with open(os.path.join(tmp_dir, "memory.querylog.jsonl"), "w", encoding="utf-8") as f:
        f.write(line + "\n")
    log = Log(tmp_dir)
    assert log.total_count() == 1
    assert log.count_of("abc123def456") == 1
    s = log.stats(3)
    assert s["backend"]["rrf"] == 1 and s["fallback_user"] == 1
    py_line = json.dumps({
        "ts": 1792078630.0, "query": "py 查询", "backend": "bm25", "limit": 5,
        "fallback": "", "source": "library", "ms": 7,
        "hits": [{"id": "abc123def456", "score": 1.2345, "title": "t"}],
    }, ensure_ascii=False)
    obj = json.loads(py_line)
    assert set(obj.keys()) == {"ts", "query", "backend", "limit", "fallback", "source", "ms", "hits"}


def test_config_json_rust_shape(tmp_dir, monkeypatch):
    """Rust 写的 config.json（embedding 三件套）Python 端 memlite_from_files 可解析。"""
    home = os.path.join(tmp_dir, "home")
    os.makedirs(os.path.join(home, ".memlite"))
    with open(os.path.join(home, ".memlite", "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data_dir": os.path.join(tmp_dir, "lib"),
            "embedding": {"api_base": "https://api.example.com/v1", "api_key": "sk-test-000", "model": "BAAI/bge-m3"},
        }, f, ensure_ascii=False)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("MEMLITE_DATA", raising=False)
    from memlite_py import memlite_from_files

    mem = memlite_from_files()
    assert mem.data_dir == os.path.join(tmp_dir, "lib")
    assert mem.config.embedding["provider"] == "openai-compatible"
    assert mem.config.embedding["api_base"] == "https://api.example.com/v1"
    # 密钥经 resolver 回落
    key = asyncio.run(mem.deps.credential_resolver("ref"))
    assert key == "sk-test-000"
