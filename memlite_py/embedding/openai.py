"""OpenAI 兼容 /embeddings 实现（embedding.md §3.2）。

标准库 urllib（经 asyncio.to_thread）——零第三方依赖；错误分类：网络 / 4xx / 5xx / 响应形状不符。
关键不变量：调用方（recall/remember）把 embed 抛错当回退信号，永不阻塞主路径。
"""

from __future__ import annotations

import asyncio
import json as _json
import socket
import urllib.error
import urllib.request
from typing import Any

from .provider import EmbedOptions, Embedder, EmbedderDeps, ResolvedEmbeddingConfig


class EmbedError(Exception):
    """嵌入错误分类：调用方可按类记日志，不影响回退语义。"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # network | client | server | shape | credential | timeout

    def __str__(self) -> str:
        return f"{self.kind}: {super().__str__()}"


def _http_post_json(
    url: str, headers: dict[str, str], body: bytes, timeout_s: float
) -> tuple[int, bytes]:
    """同步 HTTP POST（测试可 monkeypatch 此函数注入假传输层）。"""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.status, resp.read()


class OpenAIEmbedder:
    def __init__(self, cfg: ResolvedEmbeddingConfig, api_key: str):
        self._cfg = cfg
        self._api_key = api_key
        self.dim: int | None = None

    async def embed(self, text: str, opts: EmbedOptions | None = None) -> list[float]:
        timeout_ms = opts.timeout_ms if opts is not None and opts.timeout_ms else self._cfg.timeout_ms
        url = f"{self._cfg.api_base.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = _json.dumps({"model": self._cfg.model, "input": text}).encode("utf-8")
        try:
            status, raw = await asyncio.wait_for(
                asyncio.to_thread(_http_post_json, url, headers, body, timeout_ms / 1000),
                timeout=timeout_ms / 1000 + 5,
            )
        except asyncio.TimeoutError:
            raise EmbedError("timeout", f"嵌入超时（{timeout_ms}ms）") from None
        except (socket.timeout, TimeoutError) as e:
            raise EmbedError("timeout", f"嵌入超时（{timeout_ms}ms）") from e
        except urllib.error.HTTPError as e:
            detail = b""
            try:
                detail = e.read()
            except Exception:
                pass
            text_err = detail.decode("utf-8", errors="replace")
            if e.code >= 500:
                raise EmbedError("server", f"/embeddings {e.code}: {text_err}") from e
            raise EmbedError("client", f"/embeddings {e.code}: {text_err}") from e
        except Exception as e:
            raise EmbedError("network", f"嵌入 API 不可达: {e}") from e

        try:
            j: Any = _json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            raise EmbedError("shape", "响应不是 JSON") from None
        data = j.get("data") if isinstance(j, dict) else None
        arr = data[0].get("embedding") if isinstance(data, list) and data and isinstance(data[0], dict) else None
        if not isinstance(arr, list):
            raise EmbedError("shape", "响应形状不符")
        v = [float(x) for x in arr]
        if self.dim is None:
            self.dim = len(v)
        return v


async def make_openai_embedder(
    cfg: ResolvedEmbeddingConfig, deps: EmbedderDeps
) -> Embedder:
    api_key = ""
    if deps.credential_resolver is not None:
        resolved = await deps.credential_resolver(cfg.credential_ref)
        api_key = resolved or ""
    if not api_key:
        # 初始化期报错（而非运行期）——credential_ref 取不到立刻暴露
        raise EmbedError("credential", f"credentials 中未找到 {cfg.credential_ref}")
    return OpenAIEmbedder(cfg, api_key)
