"""检索日志（query_log）：一次 recall/检索落一行，外加按条聚合的召回次数。

训练与复盘取材用。文件：data_dir/memory.querylog.jsonl（追加式；名字过白名单
memory*.jsonl 随 sync 走，且不撞 Store 的 memory-<n>.jsonl 归档命名）。
次数不另设可变计数文件——全量聚合自日志，进程内常驻。
与 Rust memlite querylog.rs / memlite-js querylog.ts 行格式互通（compat.md §六）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

LOG_FILE = "memory.querylog.jsonl"
"""载入内存保留的最近条数（recent/stats 展示窗口；counts 按全量行聚合）"""
KEEP = 50_000


@dataclass
class Hit:
    id: str
    """原始分（未四舍五入）"""
    score: float
    title: str


@dataclass
class Entry:
    ts: float
    query: str
    """semantic | bm25 | rrf"""
    backend: str
    limit: int
    """空＝本库命中；"user"＝项目级无命中回退用户级"""
    fallback: str
    """cli | web | dsh | 其他宿主名"""
    source: str
    """端到端耗时（毫秒，含回退那次）"""
    ms: int
    hits: list[Hit] = field(default_factory=list)


def _entry_json(e: Entry) -> str:
    return json.dumps(
        {
            "ts": e.ts,
            "query": e.query,
            "backend": e.backend,
            "limit": e.limit,
            "fallback": e.fallback,
            "source": e.source,
            "ms": e.ms,
            "hits": [{"id": h.id, "score": h.score, "title": h.title} for h in e.hits],
        },
        ensure_ascii=False,
    )


def _round4(x: float) -> float:
    return round(x * 10000) / 10000


def _parse_entry(line: str) -> Entry | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if not isinstance(obj.get("ts"), (int, float)) or not isinstance(obj.get("query"), str) or not isinstance(obj.get("backend"), str):
        return None
    raw_hits = obj.get("hits")
    hits: list[Hit] = []
    if isinstance(raw_hits, list):
        for h in raw_hits:
            if isinstance(h, dict) and isinstance(h.get("id"), str) and isinstance(h.get("score"), (int, float)):
                hits.append(
                    Hit(
                        id=h["id"],
                        score=float(h["score"]),
                        title=h.get("title") if isinstance(h.get("title"), str) else "",
                    )
                )
    return Entry(
        ts=float(obj["ts"]),
        query=obj["query"],
        backend=obj["backend"],
        limit=obj.get("limit") if isinstance(obj.get("limit"), (int, float)) else 0,
        fallback=obj.get("fallback") if isinstance(obj.get("fallback"), str) else "",
        source=obj.get("source") if isinstance(obj.get("source"), str) else "",
        ms=obj.get("ms") if isinstance(obj.get("ms"), (int, float)) else 0,
        hits=hits,
    )


def _log_path(data_dir: str) -> str:
    return os.path.join(data_dir, LOG_FILE)


def append_line(data_dir: str, e: Entry) -> None:
    """只落盘、不载内存（CLI 短进程用）。"""
    os.makedirs(data_dir, exist_ok=True)
    with open(_log_path(data_dir), "a", encoding="utf-8") as f:
        f.write(_entry_json(e) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Log:
    """日志在进程内的常驻形态：尾窗条目 + 全量命中次数。"""

    def __init__(self, data_dir: str):
        self.entries: list[Entry] = []
        self.counts: dict[str, int] = {}
        self.total = 0
        self.bad_lines = 0
        try:
            with open(_log_path(data_dir), "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return
        lines = [l for l in text.split("\n") if l.strip()]
        start = max(0, len(lines) - KEEP)
        for i, line in enumerate(lines):
            e = _parse_entry(line)
            if e is None:
                self.bad_lines += 1
                continue
            for h in e.hits:
                self.counts[h.id] = self.counts.get(h.id, 0) + 1
            if i >= start:
                self.entries.append(e)
            self.total += 1

    def append(self, data_dir: str, e: Entry) -> None:
        """落盘一行并就地更新内存视图（memory 持有时）。"""
        append_line(data_dir, e)
        for h in e.hits:
            self.counts[h.id] = self.counts.get(h.id, 0) + 1
        self.total += 1
        self.entries.append(e)
        if len(self.entries) > KEEP:
            del self.entries[: len(self.entries) - KEEP]

    def count_of(self, rid: str) -> int:
        return self.counts.get(rid, 0)

    def total_count(self) -> int:
        return self.total

    def recent(self, n: int) -> list[Entry]:
        """最近 n 条（新→旧）。"""
        return list(reversed(self.entries[-n:]))

    def stats(self, top: int) -> dict[str, Any]:
        """汇总：次数、档位/来源分布、平均与零命中、命中最多的条目 top。"""
        since = float("inf")
        until = 0.0
        sem = 0
        rrf = 0
        bm = 0
        fb = 0
        cli = 0
        web = 0
        hit_sum = 0
        zero = 0
        for e in self.entries:
            since = min(since, e.ts)
            until = max(until, e.ts)
            if e.backend == "semantic":
                sem += 1
            elif e.backend == "rrf":
                rrf += 1
            else:
                bm += 1
            if e.fallback != "":
                fb += 1
            if e.source == "web":
                web += 1
            else:
                cli += 1
            if not e.hits:
                zero += 1
            hit_sum += len(e.hits)
        n = len(self.entries)
        # top 按次数排；标题取最近一次命中的
        title_of: dict[str, str] = {}
        for e in reversed(self.entries):
            for h in e.hits:
                if h.id not in title_of:
                    title_of[h.id] = h.title
        # 次数降序，同次数按 id 升序（Map 迭代无序，不钉死顺序输出会抖）
        topv = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_arr = [
            {"id": rid, "count": c, "title": title_of.get(rid, "")}
            for rid, c in topv[:top]
        ]
        return {
            "queries_window": n,
            "queries_total": self.total,
            "bad_lines": self.bad_lines,
            "since": since if n > 0 else 0.0,
            "until": until,
            "backend": {"semantic": sem, "rrf": rrf, "bm25": bm},
            "fallback_user": fb,
            "source": {"cli": cli, "web": web},
            "hits_avg": round(hit_sum / n * 100) / 100 if n > 0 else 0.0,
            "zero_hit": zero,
            "distinct_recalled": len(self.counts),
            "top": top_arr,
        }

    def recent_json(self, n: int) -> list[dict[str, Any]]:
        """最近条目的 JSON 视图（分数四舍五入到 4 位，别把 f32 噪声写进训练集）。"""
        return [
            {
                "ts": e.ts,
                "query": e.query,
                "backend": e.backend,
                "limit": e.limit,
                "fallback": e.fallback,
                "source": e.source,
                "ms": e.ms,
                "hits": [
                    {"id": h.id, "score": _round4(h.score), "title": h.title} for h in e.hits
                ],
            }
            for e in self.recent(n)
        ]


def export_log(data_dir: str, out: str, since: float = 0) -> int:
    """导出训练集：每行一次查询 {ts, query, backend, fallback, limit, positive(ids 按名次), hits}

    since 为 unix 秒下限（0＝全部）；返回导出条数。"""
    try:
        with open(_log_path(data_dir), "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise ValueError(f"读日志失败: {exc}") from exc
    n = 0
    buf: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        e = _parse_entry(line)
        if e is None:
            continue
        if e.ts < since:
            continue
        v = {
            "ts": e.ts,
            "query": e.query,
            "backend": e.backend,
            "fallback": e.fallback,
            "limit": e.limit,
            "positive": [h.id for h in e.hits],
            "hits": [
                {"id": h.id, "score": _round4(h.score), "title": h.title} for h in e.hits
            ],
        }
        buf.append(json.dumps(v, ensure_ascii=False))
        n += 1
    parent = os.path.dirname(out)
    if parent and parent != ".":
        os.makedirs(parent, exist_ok=True)
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(("\n".join(buf) + "\n") if buf else "")
    except OSError as exc:
        raise ValueError(f"写导出文件失败: {exc}") from exc
    return n
