"""共享测试工具。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# basetemp 固定到仓群共享目录：沙箱/缺 TEMP 环境下 tempfile 会回退 cwd（仓内）。
# 若共享目录仍在某 git 仓内（默认 risense 树），find_repo/sync 的「不在仓」用例
# 由测试内 guard 跳过；CI/真机 basetemp 在系统 Temp 时全量运行。
_base = os.environ.get("TEMP") or os.environ.get("TMP") or os.path.join(
    os.path.expanduser("~"), "AppData", "Local", "Temp"
)
_candidates = [
    os.path.join(_base, "memlite-py-tests"),
    os.path.join(str(PKG_ROOT.parent), ".tmp-tests", "memlite-py"),  # 沙箱回退：workspace 内
]
_test_base = None
for cand in _candidates:
    try:
        os.makedirs(cand, exist_ok=True)
        probe = os.path.join(cand, ".probe")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        _test_base = cand
        break
    except OSError:
        continue
assert _test_base is not None, "no writable basetemp for tests"
tempfile.tempdir = _test_base

from memlite_py.store import MemRecord  # noqa: E402


def rec(id: str, ts: float, title: str, content: str, project: str = "", tags=None, extra=None) -> MemRecord:
    return MemRecord(
        op="put", id=id, ts=ts, title=title, content=content,
        tags=list(tags or []), project=project, extra=dict(extra or {}),
    )


@pytest.fixture
def tmp_dir(tmp_path: Path) -> str:
    return str(tmp_path / "data")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """测试不读写真实用户级配置。"""
    for var in ("MEMLITE_DATA", "MEMLITE_CONFIG", "MEMLITE_API_BASE", "MEMLITE_API_KEY", "MEMLITE_EMBED_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
