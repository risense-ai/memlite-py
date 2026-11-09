"""默认档：空 provider，调用方检测到 dim 0 直接走 BM25，不进 RRF（embedding.md §3.1）。"""

from __future__ import annotations

from .provider import EmbedOptions


class NoneEmbedder:
    dim: int | None = 0

    async def embed(self, text: str, opts: EmbedOptions | None = None) -> list[float]:
        return []


NONE_EMBEDDER = NoneEmbedder()
