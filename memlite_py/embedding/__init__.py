"""嵌入档公共出口。"""

from .host import HostEmbedder, make_host_embedder
from .none import NONE_EMBEDDER, NoneEmbedder
from .openai import EmbedError, OpenAIEmbedder, make_openai_embedder
from .provider import (
    EmbedOptions,
    Embedder,
    EmbedderDeps,
    EmbedderFactory,
    EmbeddingProvider,
    ResolvedEmbeddingConfig,
)

__all__ = [
    "EmbedOptions",
    "Embedder",
    "EmbedderDeps",
    "EmbedderFactory",
    "EmbeddingProvider",
    "ResolvedEmbeddingConfig",
    "NONE_EMBEDDER",
    "NoneEmbedder",
    "HostEmbedder",
    "make_host_embedder",
    "EmbedError",
    "OpenAIEmbedder",
    "make_openai_embedder",
]
