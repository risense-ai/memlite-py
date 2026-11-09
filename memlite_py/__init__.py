"""memlite-py 公共出口——可嵌入本地记忆引擎，与 respire-memlite（Rust）、@rsrsai/memlite（TS）数据互通。"""

from .store import (
    CHUNK_BYTES,
    DATA_FILE,
    MemRecord,
    Store,
    importance,
    kind,
    new_id,
    now_ts,
    parent_id,
    serialize_record,
)

# 分词与检索
from .tokenize import is_cjk, tokenize
from .bm25 import Scored, bm25, default_title, doc_tokens
from .cosine import cosine, semantic
from .rrf import rrf_fuse

# 向量侧车
from .vectors import VectorIndex

# 因果树
from .tree import (
    PARENT_L1_CAP,
    ChainView,
    TreeNode,
    TreeScope,
    causal_lines,
    chain_view,
    outline,
    render_tree,
    tree_json,
)

# 敏感闸
from .sensitive import SensitiveError, scan

# 配置
from .config import (
    ConfigCtx,
    config_set,
    embedding_merged,
    find_project_dir,
    home_dir,
    load_config,
    project_config,
    resolve_data_dir,
    user_config_path,
    user_data_dir,
)

# 检索日志
from .querylog import LOG_FILE, Entry, Hit, Log, append_line, export_log

# 检索主流程
from .recall import RecallBackend, SearchOnceOpts, SearchOnceResult, recall_search, search_once

# 同步
from .sync import SyncReport, SyncStatus, find_repo, status, sync, whitelisted

# 嵌入档
from .embedding import (
    EmbedOptions,
    Embedder,
    EmbedderDeps,
    EmbedderFactory,
    EmbeddingProvider,
    EmbedError,
    NONE_EMBEDDER,
    ResolvedEmbeddingConfig,
    make_host_embedder,
    make_openai_embedder,
)

# 主类
from .memlite import (
    CausalNeighbor,
    Memlite,
    MemliteConfig,
    MemliteDeps,
    RecallHit,
    RecallOpts,
    RecallResult,
    RememberOpts,
    ShowView,
    UpdateResult,
    ReembedResult,
    memlite_from_files,
)

VERSION = "0.1.0"

__all__ = [
    "VERSION",
    # 存储
    "Store", "DATA_FILE", "CHUNK_BYTES", "MemRecord", "parent_id", "importance", "kind",
    "now_ts", "new_id", "serialize_record",
    # 分词与检索
    "tokenize", "is_cjk", "bm25", "doc_tokens", "default_title", "Scored",
    "cosine", "semantic", "rrf_fuse",
    # 向量侧车
    "VectorIndex",
    # 因果树
    "TreeScope", "render_tree", "outline", "tree_json", "chain_view", "causal_lines",
    "PARENT_L1_CAP", "ChainView", "TreeNode",
    # 敏感闸
    "SensitiveError", "scan",
    # 配置
    "home_dir", "user_config_path", "load_config", "find_project_dir", "project_config",
    "user_data_dir", "resolve_data_dir", "embedding_merged", "config_set", "ConfigCtx",
    # 检索日志
    "Log", "append_line", "export_log", "LOG_FILE", "Entry", "Hit",
    # 检索主流程
    "recall_search", "search_once", "RecallBackend", "SearchOnceOpts", "SearchOnceResult",
    # 同步
    "whitelisted", "find_repo", "status", "sync", "SyncReport", "SyncStatus",
    # 嵌入档
    "Embedder", "EmbedderFactory", "EmbedderDeps", "EmbedOptions", "EmbeddingProvider",
    "ResolvedEmbeddingConfig", "NONE_EMBEDDER", "make_host_embedder", "make_openai_embedder",
    "EmbedError",
    # 主类
    "Memlite", "memlite_from_files", "MemliteConfig", "MemliteDeps",
    "RememberOpts", "RecallOpts", "RecallResult", "RecallHit",
    "CausalNeighbor", "ShowView", "UpdateResult", "ReembedResult",
]
