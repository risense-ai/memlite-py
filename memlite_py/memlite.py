"""Memlite 主类：可嵌入本地记忆引擎的装配点（proposal §6 契约）。

组合 Store / VectorIndex / querylog / recall 主流程 / 因果树 / 敏感闸 / 同步 / 嵌入档。
用法：`Memlite(config, deps)` → `open()` → `await recall()/remember()/...` → `close()`。
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .bm25 import default_title
from .config import ConfigCtx, embedding_merged, resolve_data_dir, user_data_dir
from .embedding.none import NONE_EMBEDDER
from .embedding.openai import make_openai_embedder
from .embedding.provider import Embedder, EmbedderDeps, EmbeddingProvider
from .querylog import LOG_FILE, Entry, Hit, Log, export_log
from .recall import SearchOnceOpts, search_once
from .sensitive import SensitiveError, scan
from .store import MemRecord, Store, now_ts, parent_id
from .sync import SyncReport, SyncStatus, status as _sync_status, sync as _sync_run
from .tree import (
    ChainView,
    TreeScope,
    TreeNode,
    causal_lines,
    chain_view,
    outline,
    render_tree,
    tree_json,
    tree_node_to_dict,
)
from .vectors import VectorIndex

DEFAULT_CREDENTIAL_REF = "MEMLITE_EMBED_API_KEY"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_TIMEOUT_MS = 30_000


@dataclass
class MemliteConfig:
    """缺省 data_dir 走 resolve_data_dir（CLI > MEMLITE_DATA > 工作区 > 用户级 > ~/memlite-data）"""

    data_dir: str | None = None
    recall_limit: int = 10
    sync: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None
    """默认 True"""
    sensitive_gate: bool = True


@dataclass
class RememberOpts:
    title: str = ""
    tags: list[str] = field(default_factory=list)
    project: str = ""
    """认因挂载：本条之因（父）——id/前8位/纲名皆可；空=根"""
    parent: str = ""
    """显式 id（导入桥回填 frontmatter.id 用）；已在库即拒"""
    id: str | None = None
    """显式 ts 秒级时间戳（导入回填创建时间用）；缺省取当前"""
    ts: float | None = None
    """额外平铺字段（source_hash/obsidian_path/importance/kind 等）；parent_id 由 parent 参数管"""
    extra: dict[str, Any] | None = None
    """同步等待嵌入完成（缺省异步不阻塞主路径）"""
    await_embedding: bool = False


@dataclass
class RecallOpts:
    limit: int | None = None
    project: str = ""
    """强制词面档"""
    bm25_only: bool = False


@dataclass
class CausalNeighbor:
    id: str
    title: str


@dataclass
class RecallHit:
    score: float
    id: str
    ts: float
    title: str
    content: str
    tags: list[str]
    project: str
    ancestors: list[CausalNeighbor]
    children: list[CausalNeighbor]


@dataclass
class RecallResult:
    backend: str
    """空=本库命中；"user"=项目级无命中回退用户级"""
    fallback: str
    ms: int
    hits: list[RecallHit]


@dataclass
class ShowChild:
    id: str
    title: str
    summary: str


@dataclass
class ShowView:
    record: MemRecord
    ancestors: list[CausalNeighbor]
    children: list[ShowChild]
    causal_lines: list[str]


@dataclass
class UpdateResult:
    id: str
    title: str
    embedded: bool


@dataclass
class ReembedResult:
    total: int
    embedded: int
    failed: int
    failures: list[str]


class Logger(Protocol):
    def warn(self, msg: str) -> None: ...


class _NullLogger:
    def warn(self, msg: str) -> None:
        pass


@dataclass
class MemliteDeps(EmbedderDeps):
    """直接注入 Embedder（测试或宿主全托管）；优先于 config.embedding 解析"""

    embedder: Embedder | None = None
    """环境/工作区上下文（data_dir 与用户级回退目录的解析来源；缺省取真实进程）"""
    ctx: ConfigCtx | None = None


def _normalize(p: str) -> str:
    return os.path.abspath(p)


class Memlite:
    def __init__(self, config: MemliteConfig | None = None, deps: MemliteDeps | None = None):
        self.config = config or MemliteConfig()
        self.deps = deps or MemliteDeps()
        self._store_ref: Store | None = None
        self._vectors_ref: VectorIndex | None = None
        self._log_ref: Log | None = None
        self._embedder_ref: Embedder | None = None
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def logger(self) -> Logger:
        """告警出口（deps.logger 可随时替换——迟注入宿主 logger 也生效）。"""
        return self.deps.logger or _NullLogger()

    # ---- 生命周期 -------------------------------------------------------

    @property
    def data_dir(self) -> str:
        """数据目录（显式配置优先；否则按分层解析定型）。"""
        if self.config.data_dir:
            return self.config.data_dir
        return resolve_data_dir(None, self.deps.ctx)

    @property
    def store(self) -> Store:
        if self._store_ref is None:
            self.open()
        assert self._store_ref is not None
        return self._store_ref

    @property
    def vectors(self) -> VectorIndex:
        if self._vectors_ref is None:
            self._vectors_ref = VectorIndex(self.data_dir)
        return self._vectors_ref

    @property
    def log(self) -> Log:
        """检索日志常驻视图（counts 全量聚合）。"""
        if self._log_ref is None:
            self._log_ref = Log(self.data_dir)
        return self._log_ref

    def open(self) -> "Memlite":
        """打开（幂等；DSH 插件在 ready 钩子调）。"""
        if self._store_ref is None:
            self._store_ref = Store(self.data_dir)
        return self

    def close(self) -> None:
        """关闭（释放常驻索引；dispose 钩子调）。"""
        self._store_ref = None
        self._vectors_ref = None
        self._log_ref = None
        self._embedder_ref = None

    # ---- 嵌入档 ---------------------------------------------------------

    async def resolve_embedder(self) -> Embedder:
        """嵌入档解析（切档重建实例即生效；结果缓存）。"""
        if self.deps.embedder is not None:
            return self.deps.embedder
        if self._embedder_ref is not None:
            return self._embedder_ref
        emb = self.config.embedding
        provider: EmbeddingProvider = (emb or {}).get("provider") or "none"
        if provider == "none":
            return NONE_EMBEDDER
        embedder: Embedder
        if provider == "host":
            if self.deps.host_embedder is None:
                raise ValueError("embedding.provider=host 须在 deps.host_embedder 注入宿主 Embedder")
            embedder = self.deps.host_embedder
        else:
            api_base = (emb or {}).get("api_base")
            if not api_base:
                raise ValueError("embedding.provider=openai-compatible 须配 api_base")
            embedder = await make_openai_embedder(
                ResolvedEmbeddingConfig(
                    provider=provider,
                    api_base=api_base,
                    model=(emb or {}).get("model") or DEFAULT_MODEL,
                    credential_ref=(emb or {}).get("credential_ref") or DEFAULT_CREDENTIAL_REF,
                    timeout_ms=(emb or {}).get("timeout_ms") or DEFAULT_TIMEOUT_MS,
                ),
                self.deps,
            )
        self._embedder_ref = embedder
        return embedder

    def _emb_provider(self) -> EmbeddingProvider:
        return (self.config.embedding or {}).get("provider") or "none"

    # ---- 敏感闸 ---------------------------------------------------------

    def scan_or_reject(self, text: str) -> None:
        """写前敏感闸（config.sensitive_gate=True 时启用）：命中即拒，提示改存位置指针。"""
        if not self.config.sensitive_gate:
            return
        hits = scan(text)
        if hits:
            raise SensitiveError(
                f"拒绝入库：命中敏感信息（{'、'.join(hits)}）。改存位置指针（如「密码在 ~/.env」）后重试。"
            )

    # ---- 写 -------------------------------------------------------------

    async def remember(self, content: str, opts: RememberOpts | None = None) -> str:
        """存记忆：主路径只落 JSONL（立即可被 BM25 命中），嵌入异步执行，失败仅记日志。"""
        opts = opts or RememberOpts()
        title = opts.title or ""
        probe = f"{title}\n{content}"
        self.scan_or_reject(probe)
        final_title = default_title(content) if title.strip() == "" else title.strip()
        parent_full = ""
        if opts.parent:
            p = self.store.resolve(opts.parent)
            if p is None:
                raise ValueError(f"目标因（父）不存在: {opts.parent}")
            parent_full = p.id
        has_full = opts.id is not None or opts.ts is not None or opts.extra is not None
        if has_full:
            rid = self.store.put_full(
                content=content,
                title=final_title,
                tags=opts.tags or [],
                project=opts.project or "",
                parent=parent_full,
                id=opts.id,
                ts=opts.ts,
                extra=opts.extra,
            )
        else:
            rid = self.store.put_child(content, final_title, opts.tags or [], opts.project or "", parent_full)
        if self._emb_provider() != "none" or self.deps.embedder is not None:
            task = self._embed_record(rid, final_title, content)
            if opts.await_embedding:
                await task
            else:
                # 侧车不阻塞主路径：调度后台任务（Python 协程惰性，须显式让渡一拍，
                # 快速嵌入档即可在同一事件循环内完成；慢档失败只记日志）。
                self._track_task(asyncio.create_task(task))
                await asyncio.sleep(0)
        return rid

    async def _embed_record(self, rid: str, title: str, content: str) -> bool:
        """侧车异步写入：失败不拦存入，只记日志。"""
        try:
            embedder = await self.resolve_embedder()
            if (embedder.dim if embedder.dim is not None else 1) == 0:
                return False
            v = await embedder.embed(f"{title}\n{content}")
            if len(v) == 0:
                return False
            self.vectors.put(rid, v)
            return True
        except Exception as e:
            self.logger.warn(f"记忆 {rid} 嵌入失败，将被语义档跳过: {e}")
            return False

    def _track_task(self, task: "asyncio.Task") -> None:
        """持强引用防 GC，结束自动出册；异常已在 _embed_record 内消化。"""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ---- 读 -------------------------------------------------------------

    async def recall(self, query: str, opts: RecallOpts | None = None) -> RecallResult:
        """检索：BM25 必算 → 嵌入 → RRF；项目级无命中回退用户级；写 querylog；带因果视界。"""
        opts = opts or RecallOpts()
        t0 = time.perf_counter()
        limit = opts.limit if opts.limit is not None else self.config.recall_limit
        project = opts.project or ""
        bm25_only = opts.bm25_only
        embedder = await self._try_embedder()
        r = await search_once(
            self.store.all(),
            self.vectors,
            embedder,
            query,
            SearchOnceOpts(limit=limit, project=project, bm25_only=bm25_only),
            self.logger,
        )
        backend = r.backend
        hits = r.hits
        # 两档查序：解析库非用户级（项目级）且没搜到 → 自动回退用户级
        fallbacked = False
        causal_store = self.store
        user_dir = user_data_dir(self.deps.ctx)
        if not hits and _normalize(self.data_dir) != _normalize(user_dir):
            user_store = Store(user_dir)
            user_vec = VectorIndex(user_dir)
            r2 = await search_once(
                user_store.all(),
                user_vec,
                embedder,
                query,
                SearchOnceOpts(limit=limit, project=project, bm25_only=bm25_only),
                self.logger,
            )
            if r2.hits:
                backend = r2.backend
                hits = r2.hits
                fallbacked = True
                causal_store = user_store
        # 落检索日志：一次查询一行（含零命中与回退），失败只提示不拦检索
        ms = round((time.perf_counter() - t0) * 1000)
        entry = Entry(
            ts=now_ts(),
            query=query,
            backend=backend,
            limit=limit,
            fallback="user" if fallbacked else "",
            source="library",
            ms=ms,
            hits=[Hit(id=h.record.id, score=h.score, title=h.record.title) for h in hits],
        )
        try:
            self.log.append(self.data_dir, entry)
        except Exception as e:
            self.logger.warn(f"检索日志写入失败：{e}")
        return RecallResult(
            backend=backend,
            fallback="user" if fallbacked else "",
            ms=ms,
            hits=[self._to_recall_hit(causal_store, h.score, h.record) for h in hits],
        )

    def _to_recall_hit(self, causal_store: Store, score: float, r: MemRecord) -> RecallHit:
        return RecallHit(
            score=round(score * 10000) / 10000,
            id=r.id,
            ts=r.ts,
            title=r.title,
            content=r.content,
            tags=r.tags,
            project=r.project,
            ancestors=[
                CausalNeighbor(id=pid, title=causal_store.records[pid].title)
                for pid in causal_store.ancestor_chain(r.id)
                if pid in causal_store.records
            ],
            children=[
                CausalNeighbor(id=c.id, title=c.title) for c in causal_store.children(r.id)
            ],
        )

    async def _try_embedder(self) -> Embedder | None:
        """嵌入档解析失败（如 credential_ref 取不到）不算致命——回退 BM25 并告警。"""
        try:
            return await self.resolve_embedder()
        except Exception as e:
            self.logger.warn(f"嵌入档不可用，走纯 BM25: {e}")
            return None

    def show(self, rid: str) -> ShowView | None:
        """看全文：resolve（id/前缀/纲名）+ 因果邻居。"""
        r = self.store.resolve(rid)
        if r is None:
            return None
        s = self.store

        def title_of(pid: str) -> str:
            p = s.records.get(pid)
            return p.title if p and p.title != "" else pid[:8]

        return ShowView(
            record=r,
            ancestors=[
                CausalNeighbor(id=pid, title=title_of(pid))
                for pid in s.ancestor_chain(r.id)
                if pid in s.records
            ],
            children=[
                ShowChild(id=c.id, title=c.title, summary=c.content[:80])
                for c in s.children(r.id)
            ],
            causal_lines=causal_lines(s, r.id),
        )

    def list(self, limit: int = 20) -> list[MemRecord]:
        """列最近（ts 降序）。"""
        all_recs = sorted(self.store.all(), key=lambda r: -r.ts)
        return all_recs[:limit]

    def forget(self, rid: str) -> str:
        """删记忆（resolve 支持 id/前缀/纲名）。"""
        r = self.store.resolve(rid)
        if r is None:
            raise ValueError(f"记忆不存在: {rid}")
        self.store.delete(r.id)
        return r.id

    # ---- 因果结构操作 ---------------------------------------------------

    def attach(self, rid: str, parent: str) -> dict[str, str]:
        """收编挂链：子条挂到 parent 之下（防自挂防环）；parent='' 升根。"""
        return self._attach_like(rid, parent)

    def demote(self, rid: str, parent: str) -> dict[str, str]:
        """因果下沉：本条挂到 parent 下成为其果（同一 reparent 语义）。"""
        return self._attach_like(rid, parent)

    def _attach_like(self, rid: str, parent: str) -> dict[str, str]:
        s = self.store
        child = s.resolve(rid)
        if child is None:
            raise ValueError(f"记忆不存在: {rid}")
        if parent == "":
            s.reparent_checked(child.id, "")
            return {"child": child.id, "parent": ""}
        p = s.resolve(parent)
        if p is None:
            raise ValueError(f"目标因（父）不存在: {parent}")
        s.reparent_checked(child.id, p.id)
        return {"child": child.id, "parent": p.id}

    def promote(self, rid: str) -> dict[str, str]:
        """因果上浮：本条升为更上游之因（挂到祖父/根，果链随之）。"""
        s = self.store
        me = s.resolve(rid)
        if me is None:
            raise ValueError(f"记忆不存在: {rid}")
        old_parent = parent_id(me)
        if old_parent == "":
            return {"id": me.id, "new_parent": ""}
        parent_rec = s.records.get(old_parent)
        # 新因 = 原因的因（祖父）；祖父不在库则升根
        grandparent = parent_id(parent_rec) if parent_rec else ""
        s.reparent_checked(me.id, grandparent)
        return {"id": me.id, "new_parent": grandparent}

    def chain(self, rid: str, depth: int = 3) -> ChainView:
        """因果链深挖：因链全文（根在前）+ 本条 + 果链下探 depth 层。"""
        r = self.store.resolve(rid)
        if r is None:
            raise ValueError(f"记忆不存在: {rid}")
        return chain_view(self.store, r.id, depth)

    def tree(
        self,
        opts: dict[str, Any] | None = None,
        *,
        from_: str | None = None,
        depth: int | None = None,
        mode: Literal["render", "outline", "json"] | None = None,
    ) -> list[str] | dict[str, Any] | list[TreeNode]:
        """因果树渲染（mode 缺省 render；from 支持 id/前8位/纲名）。"""
        o = opts or {}
        real_depth = depth if depth is not None else o.get("depth", 3)
        from_arg = from_ if from_ is not None else o.get("from")
        from_full: str | None = None
        if from_arg:
            r = self.store.resolve(from_arg)
            if r is None:
                raise ValueError(f"记忆不存在: {from_arg}")
            from_full = r.id
        scope = TreeScope(self.store)
        real_mode = mode or o.get("mode") or "render"
        if real_mode == "outline":
            lines, count = outline(scope, from_full)
            return {"lines": lines, "count": count}
        if real_mode == "json":
            return tree_json(scope, from_full)
        return render_tree(scope, from_full, real_depth)

    # ---- 改与批 ---------------------------------------------------------

    async def update(
        self,
        rid: str,
        patch: dict[str, Any] | None = None,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> UpdateResult:
        """就地改一条（同 id 追加新版本；未传字段保留原值）。

        extra 按键合并，parent_id 不随 patch 变（改挂走 attach）。侧车向量跟着更新（失败不拦）。"""
        p = patch or {}
        new_title = title if title is not None else p.get("title")
        new_content = content if content is not None else p.get("content")
        new_tags = tags if tags is not None else p.get("tags")
        new_project = project if project is not None else p.get("project")
        new_extra = extra if extra is not None else p.get("extra")
        probe = f"{new_title or ''}\n{new_content or ''}"
        if probe.strip() != "":
            self.scan_or_reject(probe)
        old = self.store.resolve(rid)
        if old is None:
            raise ValueError(f"记忆不存在: {rid}")
        final_title = new_title if new_title is not None else old.title
        final_content = new_content if new_content is not None else old.content
        final_tags = new_tags if new_tags is not None else old.tags
        final_project = new_project if new_project is not None else old.project
        if not str(final_content).strip():
            raise ValueError("content 不能为空")
        extra_patch = dict(new_extra or {})
        extra_patch.pop("parent_id", None)  # 改挂走 attach，不随 patch
        self.store.put_with_id(
            old.id, str(final_content), str(final_title), list(final_tags or []), str(final_project), extra_patch
        )
        embedded = False
        embedder = await self._try_embedder()
        if embedder is not None and (embedder.dim if embedder.dim is not None else 1) != 0:
            try:
                v = await embedder.embed(f"{final_title}\n{final_content}")
                if len(v) > 0:
                    self.vectors.put(old.id, v)
                    embedded = True
            except Exception as e:
                self.logger.warn(f"记忆 {old.id} 嵌入失败，将被语义档跳过: {e}")
        return UpdateResult(id=old.id, title=str(final_title), embedded=embedded)

    async def reembed(self, opts: dict[str, Any] | None = None) -> ReembedResult:
        """给无向量的记忆批量回填（ts 升序；已有向量跳过；维度变更拒收并计入 failed）。"""
        del opts  # 预留 batchSize
        embedder = await self.resolve_embedder()
        if (embedder.dim if embedder.dim is not None else 0) == 0:
            raise ValueError("嵌入档未开启")
        all_recs = sorted(self.store.all(), key=lambda r: r.ts)
        failures: list[str] = []
        done = 0
        for r in all_recs:
            if self.vectors.has(r.id):
                continue
            try:
                v = await embedder.embed(f"{r.title}\n{r.content}")
                if len(v) > 0:
                    self.vectors.put(r.id, v)
                    done += 1
            except Exception as e:
                failures.append(f"{r.id}: {e}")
        return ReembedResult(total=len(all_recs), embedded=done, failed=len(failures), failures=failures)

    # ---- 同步与维护 -----------------------------------------------------

    async def sync_status(self) -> SyncStatus:
        """同步状态（dirty / local_sha / has_remote）。"""
        return await _sync_status(self.data_dir)

    async def sync(self) -> SyncReport:
        """一次完整 git 同步（白名单 add + pathspec commit + rebase + push + sha 核验）。"""
        return await _sync_run(self.data_dir)

    def compact(self) -> int:
        """压实：只留存活重写回单文件，删归档。返回压实前行数。"""
        return self.store.compact()

    async def status(self) -> dict[str, Any]:
        """全库状态概览。"""
        s = self.store
        embedder = self.deps.embedder
        emb = self.config.embedding
        files = [{"file": name, "bytes": size} for name, size in s.file_stats()]
        try:
            sync_status: Any = await self.sync_status()
            sync_dict: Any = {
                "data_dir": sync_status.data_dir,
                "repo": sync_status.repo,
                "has_remote": sync_status.has_remote,
                "dirty": sync_status.dirty,
                "local_sha": sync_status.local_sha,
            }
            if sync_status.note:
                sync_dict["note"] = sync_status.note
        except Exception:
            sync_dict = None
        provider = self._emb_provider()
        return {
            "data_dir": self.data_dir,
            "records": s.count(),
            "lines": s.line_count(),
            "bad_lines": s.bad_lines,
            "files": files,
            "vectors": self.vectors.count(),
            "querylog": LOG_FILE,
            "embedding": {
                "provider": provider,
                "configured": bool(self.deps.embedder) or provider != "none",
                "api_base": (emb or {}).get("api_base") or "",
                "model": (emb or {}).get("model") or DEFAULT_MODEL,
                "resolved_dim": embedder.dim if embedder is not None else None,
            },
            "sync": sync_dict,
        }

    # ---- 检索日志视图 ---------------------------------------------------

    def querylog_recent(self, n: int) -> list[dict[str, Any]]:
        """检索日志：最近 n 条流水。"""
        return self.log.recent_json(n)

    def querylog_stats(self, top: int = 10) -> dict[str, Any]:
        """检索日志：汇总。"""
        return self.log.stats(top)

    def querylog_export(self, out: str, since: float = 0) -> int:
        """检索日志：导出训练集。"""
        return export_log(self.data_dir, out, since)


def memlite_from_files(deps: MemliteDeps | None = None, ctx: ConfigCtx | None = None) -> Memlite:
    """从分层配置文件与环境变量解析 MemliteConfig（兼容 Rust/JS 写的 config.json，含 api_key 读取）。

    密钥策略：credential_resolver 直接回落 config.json 的 embedding.api_key（向后兼容）；
    宿主已注入 credential_resolver 时以其为准。"""
    deps = deps or MemliteDeps()
    ctx = ctx or deps.ctx
    merged = embedding_merged(ctx)
    api_base = merged.get("api_base") if isinstance(merged.get("api_base"), str) else None
    api_key = merged.get("api_key") if isinstance(merged.get("api_key"), str) else None
    model = merged.get("model") if isinstance(merged.get("model"), str) else None
    config = MemliteConfig(
        data_dir=resolve_data_dir(None, ctx),
        embedding=(
            {"provider": "openai-compatible", "api_base": api_base, "model": model}
            if api_base and api_key and model
            else {"provider": "none"}
        ),
    )
    if deps.credential_resolver is None:
        async def _resolver(ref: str) -> str | None:
            return api_key

        deps.credential_resolver = _resolver
    return Memlite(config, deps)
