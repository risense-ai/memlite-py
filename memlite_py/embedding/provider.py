"""Embedder 抽象接口（embedding.md §二）。

密钥永不入本包——api_key 通过 credential_ref 由宿主从 write-only credentials
取出后注入 Embedder 工厂。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

EmbeddingProvider = str  # 'none' | 'host' | 'openai-compatible'


@dataclass
class EmbedOptions:
    timeout_ms: int | None = None


class Embedder(Protocol):
    async def embed(self, text: str, opts: EmbedOptions | None = None) -> list[float]: ...

    """dim 0 = 纯词面档（none），调用方直接走 BM25"""
    dim: int | None


@dataclass
class ResolvedEmbeddingConfig:
    """解析后的嵌入配置（provider 已定、默认值已补）"""

    provider: EmbeddingProvider
    api_base: str | None = None
    """默认 'BAAI/bge-m3'"""
    model: str = "BAAI/bge-m3"
    """宿主 credentials 引用名"""
    credential_ref: str = "MEMLITE_EMBED_API_KEY"
    """默认 30000"""
    timeout_ms: int = 30_000


CredentialResolver = Callable[[str], Awaitable[str | None]]


@dataclass
class EmbedderDeps:
    credential_resolver: CredentialResolver | None = None
    """host 档时由宿主注入"""
    host_embedder: Embedder | None = None
    logger: "LoggerLike | None" = None


class LoggerLike(Protocol):
    def warn(self, msg: str) -> None: ...


EmbedderFactory = Callable[[ResolvedEmbeddingConfig, EmbedderDeps], Awaitable[Embedder]]
