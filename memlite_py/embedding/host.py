"""host 档：核心包不实现嵌入本身，由宿主插件注入 Embedder（embedding.md §3.3）。

宿主（如 AstrBot 插件）实现示例：
```python
def make_host_embed(providers) -> Embedder:
    class _Host:
        async def embed(self, text, opts=None):
            return await providers.embed(text)
    return _Host()
```
注入方式：`Memlite(config, host_embedder=make_host_embed(ctx))`，
且 config.embedding.provider === 'host'。
"""

from __future__ import annotations

from typing import Awaitable, Callable

from .provider import EmbedOptions, Embedder

HostEmbedFn = Callable[[str, EmbedOptions | None], Awaitable[list[float]]]


class HostEmbedder:
    """把宿主的单个 embed 函数包成 Embedder（宿主已有完整 Embedder 时直接注入即可）。"""

    def __init__(self, embed_fn: HostEmbedFn, dim: int | None = None):
        self._embed_fn = embed_fn
        self.dim = dim

    async def embed(self, text: str, opts: EmbedOptions | None = None) -> list[float]:
        return await self._embed_fn(text, opts)


def make_host_embedder(embed_fn: HostEmbedFn, dim: int | None = None) -> Embedder:
    return HostEmbedder(embed_fn, dim)
