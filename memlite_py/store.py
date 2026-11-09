"""存储层：append-only JSONL + 16MB 分块轮转 + 内存索引。

归档旧→新、活跃文件恒末，末写胜出（LWW）。
与 Rust memlite store.rs / memlite-js store.ts 数据契约互通（compat.md §二）。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_FILE = "memory.jsonl"
CHUNK_BYTES = 16 * 1024 * 1024

_ARCHIVE_RE = re.compile(r"^memory-(\d+)\.jsonl$")

_KNOWN_FIELDS = {"op", "id", "ts", "title", "content", "tags", "project"}


@dataclass
class MemRecord:
    """一条记忆（或 del 墓碑）。parent_id / importance / kind 等经 flatten 落在 extra。"""

    op: str  # 'put' | 'del'
    id: str
    ts: float
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    project: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def parent_id(r: MemRecord) -> str:
    """因果树父指针：extra.parent_id（迁移库带此字段；无则空串=根）。"""
    v = r.extra.get("parent_id")
    return v if isinstance(v, str) else ""


def importance(r: MemRecord) -> str:
    """重要性（respire 迁移数据 important/trivial；新条可缺省）。"""
    v = r.extra.get("importance")
    return v if isinstance(v, str) else ""


def kind(r: MemRecord) -> str:
    """记忆形态（Context/Task/Skill/...；迁移库带，新条可缺省）。"""
    v = r.extra.get("kind")
    return v if isinstance(v, str) else ""


def now_ts() -> float:
    import time

    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def serialize_record(r: MemRecord) -> str:
    """序列化一行：固定字段在前，extra 平铺在后（对齐 serde flatten 形状）。"""
    obj: dict[str, Any] = {
        "op": r.op,
        "id": r.id,
        "ts": r.ts,
        "title": r.title,
        "content": r.content,
        "tags": r.tags,
        "project": r.project,
    }
    for k, v in r.extra.items():
        obj[k] = v
    return json.dumps(obj, ensure_ascii=False)


def parse_record(line: str) -> MemRecord | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("op") not in ("put", "del"):
        return None
    if not isinstance(obj.get("id"), str) or not isinstance(obj.get("ts"), (int, float)):
        return None
    if not isinstance(obj.get("content"), str):
        return None
    extra = {k: v for k, v in obj.items() if k not in _KNOWN_FIELDS}
    tags = obj.get("tags")
    return MemRecord(
        op=obj["op"],
        id=obj["id"],
        ts=float(obj["ts"]),
        title=obj.get("title") if isinstance(obj.get("title"), str) else "",
        content=obj["content"],
        tags=[t for t in tags if isinstance(t, str)] if isinstance(tags, list) else [],
        project=obj.get("project") if isinstance(obj.get("project"), str) else "",
        extra=extra,
    )


def _clean_tags(tags: list[str]) -> list[str]:
    return [t.strip() for t in tags if t.strip()]


def _sort_counts(m: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))


class Store:
    data_dir: str
    records: dict[str, MemRecord]
    deleted: set[str]
    bad_lines: int
    chunk_bytes: int

    def __init__(self, data_dir: str, chunk_bytes: int = CHUNK_BYTES):
        self.data_dir = data_dir
        self.chunk_bytes = chunk_bytes
        self.records = {}
        self.deleted = set()
        self.bad_lines = 0
        self.load()

    # ---- 文件布局 -------------------------------------------------------

    def _archive_paths(self) -> list[str]:
        """归档按序号升序（旧→新）。"""
        found: list[tuple[int, str]] = []
        try:
            names = os.listdir(self.data_dir)
        except OSError:
            return []
        for name in names:
            m = _ARCHIVE_RE.match(name)
            if m:
                found.append((int(m.group(1)), os.path.join(self.data_dir, name)))
        found.sort(key=lambda p: p[0])
        return [p for _, p in found]

    def _files(self) -> list[str]:
        """归档在前（旧），活跃在后（新）。"""
        paths = self._archive_paths()
        active = os.path.join(self.data_dir, DATA_FILE)
        if os.path.exists(active):
            paths.append(active)
        return paths

    # ---- 装载与写入 -----------------------------------------------------

    def load(self) -> None:
        for p in self._files():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            for raw in text.split("\n"):
                line = raw.strip()
                if not line:
                    continue
                rec = parse_record(line)
                if rec is None:
                    self.bad_lines += 1
                    continue
                self.apply(rec)

    def apply(self, op: MemRecord) -> None:
        if op.op == "put":
            self.records[op.id] = op
            self.deleted.discard(op.id)
        else:
            self.deleted.add(op.id)
            self.records.pop(op.id, None)

    def _append(self, rec: MemRecord) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        active = os.path.join(self.data_dir, DATA_FILE)
        line = serialize_record(rec)
        fd = os.open(active, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self._maybe_rotate()

    def _maybe_rotate(self) -> None:
        active = os.path.join(self.data_dir, DATA_FILE)
        try:
            size = os.path.getsize(active)
        except OSError:
            return
        if size < self.chunk_bytes:
            return
        archives = self._archive_paths()
        max_seq = 0
        if archives:
            m = _ARCHIVE_RE.match(Path(archives[-1]).name)
            if m:
                max_seq = int(m.group(1))
        seq = max_seq + 1
        os.rename(active, os.path.join(self.data_dir, f"memory-{seq:04d}.jsonl"))

    # ---- 写 API ---------------------------------------------------------

    def put(self, content: str, title: str, tags: list[str], project: str) -> str:
        return self.put_child(content, title, tags, project, "")

    def put_child(
        self, content: str, title: str, tags: list[str], project: str, parent: str
    ) -> str:
        """存一条并显式认因（parent 非空须已在库——不产生悬空父指针；空串=根）。"""
        if not content.strip():
            raise ValueError("content 为空")
        if parent != "" and parent not in self.records:
            raise ValueError(f"目标因（父）不存在: {parent}")
        rid = new_id()
        rec = MemRecord(
            op="put",
            id=rid,
            ts=now_ts(),
            title=title.strip(),
            content=content.strip(),
            tags=_clean_tags(tags),
            project=project.strip(),
            extra={"parent_id": parent},
        )
        self._append(rec)
        self.apply(rec)
        return rid

    def delete(self, rid: str) -> None:
        if rid not in self.records:
            raise ValueError(f"记忆不存在: {rid}")
        rec = MemRecord(op="del", id=rid, ts=now_ts(), title="", content="")
        self._append(rec)
        self.apply(rec)

    def put_with_id(
        self,
        rid: str,
        content: str,
        title: str,
        tags: list[str],
        project: str,
        extra_patch: dict[str, Any] | None = None,
    ) -> None:
        """就地改：同 id 追加新 put（末写胜出，id 不变）。

        extra 全量保留（parent_id/importance/kind 随迁——改前文不能丢因链）；
        extra_patch 非空时按键合并覆盖（parent_id 仍以库中原值为准，改挂走 reparent）。"""
        if not content.strip():
            raise ValueError("content 为空")
        old = self.records.get(rid)
        extra = dict(old.extra if old else {})
        if extra_patch:
            extra.update(extra_patch)
        extra["parent_id"] = parent_id(old) if old else ""
        rec = MemRecord(
            op="put",
            id=rid,
            ts=now_ts(),
            title=title.strip(),
            content=content.strip(),
            tags=_clean_tags(tags),
            project=project.strip(),
            extra=extra,
        )
        self._append(rec)
        self.apply(rec)

    def put_full(
        self,
        content: str,
        title: str,
        tags: list[str] | None = None,
        project: str = "",
        parent: str = "",
        id: str | None = None,
        ts: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """完整受控存入：显式 id/ts/extra（导入桥等客户端用）；parent 非空须已在库。

        id 已在库即拒（改条走 update 流）；extra 中夹带的 parent_id 以 parent 参数为准。"""
        if not content.strip():
            raise ValueError("content 为空")
        if id is not None and id in self.records:
            raise ValueError(f"id 已存在（改条走 update 流）: {id}")
        if parent != "" and parent not in self.records:
            raise ValueError(f"目标因（父）不存在: {parent}")
        rid = id if id is not None else new_id()
        full_extra = dict(extra or {})
        full_extra["parent_id"] = parent
        rec = MemRecord(
            op="put",
            id=rid,
            ts=ts if ts is not None else now_ts(),
            title=title.strip(),
            content=content.strip(),
            tags=_clean_tags(list(tags or [])),
            project=(project or "").strip(),
            extra=full_extra,
        )
        self._append(rec)
        self.apply(rec)
        return rid

    # ---- 因果结构 -------------------------------------------------------

    def reparent(self, rid: str, new_parent: str) -> None:
        """改挂因（新父）：追加同 id 新 put，仅换 parent_id（空串=升根）。

        ts 保留原值：改挂是结构调整，不该把老条顶进「最近」时间线。
        防环由调用方查证（reparent_checked）；本函数只保证两节点存在。"""
        old = self.records.get(rid)
        if old is None:
            raise ValueError(f"记忆不存在: {rid}")
        if new_parent != "" and new_parent not in self.records:
            raise ValueError(f"目标因（父）不存在: {new_parent}")
        rec = MemRecord(
            op=old.op,
            id=old.id,
            ts=old.ts,
            title=old.title,
            content=old.content,
            tags=list(old.tags),
            project=old.project,
            extra={**old.extra, "parent_id": new_parent},
        )
        self._append(rec)
        self.apply(rec)

    def reparent_checked(self, rid: str, new_parent: str) -> None:
        """改挂带防环：新父不得为本条、也不得在本条因链上（即新父是本条后代）。"""
        if rid == new_parent:
            raise ValueError("不能挂自己")
        if new_parent != "":
            for anc in self.ancestor_chain(new_parent):
                if anc == rid:
                    raise ValueError("防环：新父是本条的后代")
        self.reparent(rid, new_parent)

    def ancestor_chain(self, rid: str) -> list[str]:
        """因链回溯（父→祖父→…），返回自根至父顺序（不含本条）。

        防环封顶 64 层（脏数据不挂死）；父不在库即断链止步。"""
        chain: list[str] = []
        cur = rid
        for _ in range(64):
            rec = self.records.get(cur)
            if rec is None:
                break
            pid = parent_id(rec)
            if pid == "":
                break
            chain.append(pid)
            cur = pid
        chain.reverse()
        return chain

    def children(self, rid: str) -> list[MemRecord]:
        """直接子条（果），按 ts 升序（存入序，树渲染稳定）。"""
        v = [r for r in self.records.values() if parent_id(r) == rid]
        v.sort(key=lambda r: r.ts)
        return v

    def roots(self) -> list[MemRecord]:
        """根集：parent_id 为空（含老数据无此字段）。"""
        v = [r for r in self.records.values() if parent_id(r) == ""]
        v.sort(key=lambda r: r.ts)
        return v

    # ---- 查询 -----------------------------------------------------------

    def count(self) -> int:
        return len(self.records)

    def all(self) -> list[MemRecord]:
        return list(self.records.values())

    def get(self, rid: str) -> MemRecord | None:
        return self.records.get(rid)

    def resolve(self, prefix: str) -> MemRecord | None:
        """id / 纲名（精确标题唯一命中）/ 前缀 三级解析——与 respire resolve_prefix 同序。"""
        p = re.sub(r"^#+", "", prefix)
        exact = self.records.get(p)
        if exact is not None:
            return exact
        by_title = [r for r in self.records.values() if r.title == p]
        if len(by_title) == 1:
            return by_title[0]
        by_prefix = [r for r in self.records.values() if r.id.startswith(p)]
        if len(by_prefix) != 1:
            return None  # 无命中或歧义
        return by_prefix[0]

    def versions(self, rid: str) -> list[MemRecord]:
        """同 id 的全部历史版本（append-only 日志天然保历史）：按文件序（旧→新）。"""
        out: list[MemRecord] = []
        for p in self._files():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            for raw in text.split("\n"):
                line = raw.strip()
                if not line:
                    continue
                rec = parse_record(line)
                if rec is not None and rec.op == "put" and rec.id == rid:
                    out.append(rec)
        return out

    def tag_counts(self) -> list[tuple[str, int]]:
        """全库标签频次（降序，同次数按标签升序）。"""
        m: dict[str, int] = {}
        for r in self.records.values():
            for t in r.tags:
                m[t] = m.get(t, 0) + 1
        return _sort_counts(m)

    def project_counts(self) -> list[tuple[str, int]]:
        """全库 project 频次（降序，空 project 归「（无）」）。"""
        m: dict[str, int] = {}
        for r in self.records.values():
            k = "（无）" if r.project == "" else r.project
            m[k] = m.get(k, 0) + 1
        return _sort_counts(m)

    def line_count(self) -> int:
        n = 0
        for p in self._files():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    n += sum(1 for l in f if l.strip())
            except OSError:
                pass  # 读不到按 0 计
        return n

    def file_stats(self) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for p in self._files():
            try:
                size = os.path.getsize(p)
            except OSError:
                size = 0
            out.append((Path(p).name, size))
        return out

    def compact(self) -> int:
        """压实：只留存活重写回单文件，删归档。返回压实前行数。"""
        before = self.line_count()
        active = os.path.join(self.data_dir, DATA_FILE)
        tmp = os.path.join(self.data_dir, f"{DATA_FILE}.tmp")
        recs = sorted(self.records.values(), key=lambda r: (r.ts, r.id))
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            for rec in recs:
                os.write(fd, (serialize_record(rec) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        for p in self._archive_paths():
            os.remove(p)
        os.replace(tmp, active)  # Windows 下 rename 不覆盖，replace 原子覆盖
        return before
