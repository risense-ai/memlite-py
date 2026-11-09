"""同步层：git 同步（数据仓本身，或工作区仓——项目级记忆文件夹随外层仓走）。

白名单 glob（memory*.jsonl/meta.json/README.md）+ push 后双通道核验 sha。
向量文件（emb.f32/emb-ids.txt）不在白名单——本机可重算（reembed），不入 git。
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
from dataclasses import dataclass


async def _git(repo: str, args: list[str]) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        repo,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} 失败: {msg}")
    return stdout.decode("utf-8", errors="replace").strip()


async def _git_ok(repo: str, args: list[str]) -> bool:
    """退出码语义版（diff --cached --quiet 用 0/1 表达有无变更）。"""
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        repo,
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    return proc.returncode == 0


async def _has_remote(repo: str) -> bool:
    try:
        s = await _git(repo, ["remote", "get-url", "origin"])
        return s != ""
    except Exception:
        return False


def whitelisted(data_dir: str) -> list[str]:
    """展开白名单：data_dir 下真实存在的文件（git pathspec 无匹配会报错）。"""
    try:
        names = os.listdir(data_dir)
    except OSError:
        return []
    out = [
        name
        for name in names
        if (name.startswith("memory") and name.endswith(".jsonl"))
        or name == "meta.json"
        or name == "README.md"
    ]
    out.sort()
    return out


def find_repo(data_dir: str) -> str | None:
    """找 data_dir 之上的 git 仓根：自身是仓 → 用它；否则向上找 .git（项目级记忆随工作区仓同步）。"""
    if os.path.exists(os.path.join(data_dir, ".git")):
        return data_dir
    cur = data_dir
    while True:
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent
        if os.path.exists(os.path.join(cur, ".git")):
            return cur


@dataclass
class SyncStatus:
    data_dir: str
    repo: str | None
    has_remote: bool
    dirty: bool
    local_sha: str
    note: str | None = None


async def status(data_dir: str) -> SyncStatus:
    repo = find_repo(data_dir)
    if repo is None:
        return SyncStatus(
            data_dir=data_dir,
            repo=None,
            has_remote=False,
            dirty=False,
            local_sha="",
            note="不在任何 git 仓内（项目级记忆须在 git 工作区内，或 git init 数据仓）",
        )
    remote = await _has_remote(repo)
    # data_dir 相对仓库根的前缀（仓本身时为空）
    prefix = os.path.relpath(data_dir, repo)
    prefix = "" if prefix == "." else prefix
    files = whitelisted(data_dir)
    dirty = False
    for f in files:
        rel = os.path.join(prefix, f) if prefix else f
        try:
            out = await _git(repo, ["status", "--porcelain", "--", rel])
        except Exception:
            out = ""
        if out != "":
            dirty = True
            break
    try:
        head = await _git(repo, ["rev-parse", "HEAD"])
    except Exception:
        head = ""
    return SyncStatus(
        data_dir=data_dir, repo=repo, has_remote=remote, dirty=dirty, local_sha=head
    )


@dataclass
class SyncReport:
    committed: bool
    local_sha: str
    remote_sha: str
    verified: bool
    """实际同步的仓库根（调试/展示用）"""
    repo: str


async def sync(data_dir: str) -> SyncReport:
    """一次完整同步。data_dir 可以是仓本身或仓内子目录（项目级记忆文件夹）——

    白名单文件在 data_dir 下找，git 操作在仓库根做，add 用相对仓库根的路径。"""
    repo = find_repo(data_dir)
    if repo is None:
        raise RuntimeError(f"{data_dir} 及其上层目录都不是 git 仓；先 git init 或 git clone")
    if not await _has_remote(repo):
        raise RuntimeError(
            "未配置 origin；先 git remote add origin <私有仓URL>（项目级可省——工作区仓已有 remote 则直接用）"
        )
    prefix = os.path.relpath(data_dir, repo)
    prefix = "" if prefix == "." else prefix

    # 1. 只 add 白名单（仓根视角的相对路径）
    files = whitelisted(data_dir)
    add_paths = [os.path.join(prefix, f) if prefix else f for f in files]
    if add_paths:
        await _git(repo, ["add", "--", *add_paths])

    # 2. 只提交白名单文件——commit 必须带路径限定！否则会把工作区里
    #    用户已暂存的无关改动一并卷进 memlite 的同步提交。
    #    用 `git diff --cached --quiet -- <paths>` 判「白名单自身有无变更」。
    committed = False
    if add_paths:
        staged = await _git_ok(repo, ["diff", "--cached", "--quiet", "--", *add_paths])
        committed = not staged
        if committed:
            msg = f"memlite sync {_chrono_lite()}"
            await _git(repo, ["commit", "-m", msg, "--", *add_paths])

    # 3. fetch；远端分支已存在才 rebase（首推时远端可能为空仓）
    await _git(repo, ["fetch", "origin"])
    branch = await _git(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    try:
        remote_ref = await _git(repo, ["ls-remote", "--heads", "origin", branch])
    except Exception:
        remote_ref = ""
    if remote_ref != "":
        # --autostash：工作区有用户未提交/已暂存改动时先 stash 再 rebase、完事还原——
        # 否则项目级记忆随工作区仓同步会被用户手头的半成品卡死
        await _git(repo, ["pull", "--rebase", "--autostash", "origin", branch])

    # 4. push -u
    await _git(repo, ["push", "-u", "origin", branch])

    # 5. 双通道核验
    local_sha = await _git(repo, ["rev-parse", "HEAD"])
    ls = await _git(repo, ["ls-remote", "origin", f"refs/heads/{branch}"])
    remote_sha = ls.split()[0] if ls.split() else ""
    verified = remote_sha != "" and local_sha == remote_sha
    if not verified:
        raise RuntimeError(f"push 核验失败: local={local_sha[:8]} remote={remote_sha[:8]}")
    return SyncReport(
        committed=committed, local_sha=local_sha, remote_sha=remote_sha, verified=verified, repo=repo
    )


def _chrono_lite() -> str:
    """本地时间格式化（避免引入 chrono 依赖）。"""
    d = _dt.datetime.now()
    return d.strftime("%Y-%m-%d %H:%M:%S")
