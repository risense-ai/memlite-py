"""同步层：白名单、仓根发现、状态；远端同步在无网环境下只验错误路径。"""

import os

import pytest

from memlite_py.sync import find_repo, status, sync, whitelisted


def _tmp_outside_repo(tmp_path) -> bool:
    """basetemp 若不幸落在某 git 仓内（沙箱回退 workspace 时），仓外断言无从谈起。"""
    return find_repo(str(tmp_path)) is None


def test_whitelisted(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    for name in ("memory.jsonl", "memory-0001.jsonl", "meta.json", "README.md",
                 "emb.f32", "emb-ids.txt", "memory.querylog.jsonl", "random.txt"):
        open(os.path.join(tmp_dir, name), "w").close()
    out = whitelisted(tmp_dir)
    assert out == ["README.md", "memory-0001.jsonl", "memory.jsonl", "memory.querylog.jsonl", "meta.json"]


def test_whitelisted_missing_dir(tmp_dir):
    assert whitelisted(os.path.join(tmp_dir, "nope")) == []


def test_find_repo_upwards(tmp_dir, tmp_path):
    repo = os.path.join(tmp_dir, "repo")
    sub = os.path.join(repo, "sub", "deep")
    os.makedirs(sub)
    os.makedirs(os.path.join(repo, ".git"))
    assert find_repo(sub) == repo
    # .git 目录本身也是仓内路径 → 回到仓根
    assert find_repo(os.path.join(repo, ".git")) == repo
    if not _tmp_outside_repo(tmp_path):
        pytest.skip("basetemp 在某 git 仓内（沙箱回退），仓外断言不可行")
    assert find_repo(tmp_dir) is None


async def test_status_outside_repo(tmp_dir, tmp_path):
    if not _tmp_outside_repo(tmp_path):
        pytest.skip("basetemp 在某 git 仓内（沙箱回退），仓外断言不可行")
    st = await status(os.path.join(tmp_dir, "data"))
    assert st.repo is None and st.has_remote is False
    assert "不在任何 git 仓" in st.note


async def test_sync_outside_repo_raises(tmp_dir, tmp_path):
    if not _tmp_outside_repo(tmp_path):
        pytest.skip("basetemp 在某 git 仓内（沙箱回退），仓外断言不可行")
    with pytest.raises(RuntimeError, match="不是 git 仓"):
        await sync(os.path.join(tmp_dir, "data"))


async def test_sync_without_remote_raises(tmp_dir):
    repo = os.path.join(tmp_dir, "repo")
    os.makedirs(repo)
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "t@t")
    _run_git(repo, "config", "user.name", "t")
    with pytest.raises(RuntimeError, match="未配置 origin"):
        await sync(repo)


async def test_status_dirty_in_real_repo(tmp_dir):
    repo = os.path.join(tmp_dir, "repo")
    data = os.path.join(repo, "memlite-data")
    os.makedirs(data)
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "t@t")
    _run_git(repo, "config", "user.name", "t")
    with open(os.path.join(data, "memory.jsonl"), "w", encoding="utf-8") as f:
        f.write('{"op":"put","id":"aaaaaaaaaaaa","ts":1.0,"title":"","content":"x","tags":[],"project":""}\n')
    st = await status(data)
    assert st.repo == repo
    assert st.dirty is True  # 未提交 → 白名单文件脏
    assert st.local_sha == ""
    # 提交后不脏（无 remote 分支照常）
    _run_git(repo, "add", "--", os.path.join("memlite-data", "memory.jsonl"))
    _run_git(repo, "commit", "-m", "init", "--", os.path.join("memlite-data", "memory.jsonl"))
    st2 = await status(data)
    assert st2.dirty is False
    assert len(st2.local_sha) == 40


def _run_git(cwd, *args):
    import subprocess

    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)
