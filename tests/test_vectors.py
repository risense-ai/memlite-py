"""向量侧车：f32 格式、dim 自适应、覆盖、维度不符。"""

import os
import struct

import pytest

from memlite_py.vectors import F32_FILE, IDS_FILE, VectorIndex


def test_put_get_roundtrip(tmp_dir):
    v = VectorIndex(tmp_dir)
    assert v.dim is None and v.count() == 0
    v.put("aa", [1.0, 2.0, 3.0])
    v.put("bb", [4.0, 5.0, 6.0])
    assert v.dim == 3 and v.count() == 2 and v.has("aa")
    v2 = VectorIndex(tmp_dir)  # 重载推导 dim
    assert v2.dim == 3
    assert v2.get("bb") == [4.0, 5.0, 6.0]
    assert v2.get("cc") is None


def test_file_layout_compat(tmp_dir):
    """emb.f32 = packed f32 LE；emb-ids.txt 每行一个 id，行号=下标。"""
    v = VectorIndex(tmp_dir)
    v.put("id0", [1.5, -2.5])
    raw = open(os.path.join(tmp_dir, F32_FILE), "rb").read()
    assert struct.unpack("<2f", raw) == (1.5, -2.5)
    ids = open(os.path.join(tmp_dir, IDS_FILE), encoding="utf-8").read().splitlines()
    assert ids == ["id0"]


def test_overwrite_in_place(tmp_dir):
    v = VectorIndex(tmp_dir)
    v.put("aa", [1.0, 2.0])
    v.put("bb", [3.0, 4.0])
    v.put("aa", [9.0, 9.0])  # 原地覆盖
    assert v.count() == 2
    raw = open(os.path.join(tmp_dir, F32_FILE), "rb").read()
    assert len(raw) == 16  # 不追加
    v2 = VectorIndex(tmp_dir)
    assert v2.get("aa") == [9.0, 9.0]
    assert v2.get("bb") == [3.0, 4.0]


def test_dim_mismatch_rejected(tmp_dir):
    v = VectorIndex(tmp_dir)
    v.put("aa", [1.0, 2.0])
    with pytest.raises(ValueError, match="维度不一致"):
        v.put("bb", [1.0, 2.0, 3.0])


def test_reads_rust_shaped_fixture(tmp_dir):
    """外部（Rust/JS）写入的文件直接可读。"""
    os.makedirs(tmp_dir, exist_ok=True)
    vecs = [[0.25, 0.5, 0.75], [-1.0, 0.0, 1.0]]
    with open(os.path.join(tmp_dir, F32_FILE), "wb") as f:
        for vec in vecs:
            f.write(struct.pack("<3f", *vec))
    with open(os.path.join(tmp_dir, IDS_FILE), "w", encoding="utf-8") as f:
        f.write("rustid1\nrustid2\n")
    v = VectorIndex(tmp_dir)
    assert v.dim == 3
    assert v.get("rustid1") == [0.25, 0.5, 0.75]
    assert v.get("rustid2") == [-1.0, 0.0, 1.0]
