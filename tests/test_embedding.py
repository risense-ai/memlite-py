"""嵌入档：none/host/openai-compatible（假传输层）与 memlite 侧车嵌入。"""

import asyncio
import json

import pytest

from memlite_py.embedding.none import NONE_EMBEDDER
from memlite_py.embedding.openai import EmbedError, OpenAIEmbedder, _http_post_json, make_openai_embedder
from memlite_py.embedding.provider import EmbedOptions, EmbedderDeps, ResolvedEmbeddingConfig


async def test_none_embedder():
    assert NONE_EMBEDDER.dim == 0
    assert await NONE_EMBEDDER.embed("x") == []


class _DimHost:
    dim = 4

    async def embed(self, text, opts=None):
        return [0.1] * 4


def test_host_embedder_wrapper():
    from memlite_py.embedding.host import make_host_embedder

    calls = []

    async def fn(text, opts=None):
        calls.append(text)
        return [1.0, 0.0]

    emb = make_host_embedder(fn, dim=2)
    assert emb.dim == 2
    assert asyncio.run(emb.embed("甲")) == [1.0, 0.0]
    assert calls == ["甲"]


async def test_openai_embedder_missing_credential():
    with pytest.raises(EmbedError) as ei:
        await make_openai_embedder(
            ResolvedEmbeddingConfig(provider="openai-compatible", api_base="https://x/v1"),
            EmbedderDeps(),
        )
    assert ei.value.kind == "credential"


async def test_openai_embedder_success_shape(monkeypatch):
    captured = {}

    def fake_post(url, headers, body, timeout_s):
        captured["url"] = url
        captured["body"] = json.loads(body)
        captured["auth"] = headers["Authorization"]
        return 200, json.dumps({"data": [{"embedding": [0.5, 0.25, 0.25]}]}).encode()

    monkeypatch.setattr("memlite_py.embedding.openai._http_post_json", fake_post)
    emb = await make_openai_embedder(
        ResolvedEmbeddingConfig(provider="openai-compatible", api_base="https://x/v1/", model="m"),
        EmbedderDeps(credential_resolver=_resolver("k")),
    )
    v = await emb.embed("甲")
    assert v == [0.5, 0.25, 0.25]
    assert emb.dim == 3
    assert captured["url"] == "https://x/v1/embeddings"  # 尾斜杠已剥
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["model"] == "m" and captured["body"]["input"] == "甲"
    assert isinstance(captured["body"]["input"], str)


async def test_openai_embedder_error_classification(monkeypatch):
    import urllib.error

    def post_server_error(url, headers, body, timeout_s):
        raise urllib.error.HTTPError(url, 503, "unavailable", None, None)

    monkeypatch.setattr("memlite_py.embedding.openai._http_post_json", post_server_error)
    emb = OpenAIEmbedder(ResolvedEmbeddingConfig(provider="openai-compatible", api_base="https://x/v1"), "k")
    with pytest.raises(EmbedError) as ei:
        await emb.embed("x")
    assert ei.value.kind == "server"

    def post_client_error(url, headers, body, timeout_s):
        raise urllib.error.HTTPError(url, 401, "denied", None, None)

    monkeypatch.setattr("memlite_py.embedding.openai._http_post_json", post_client_error)
    with pytest.raises(EmbedError) as ei2:
        await emb.embed("x")
    assert ei2.value.kind == "client"

    def post_network(url, headers, body, timeout_s):
        raise ConnectionError("reset")

    monkeypatch.setattr("memlite_py.embedding.openai._http_post_json", post_network)
    with pytest.raises(EmbedError) as ei3:
        await emb.embed("x")
    assert ei3.value.kind == "network"

    def post_bad_shape(url, headers, body, timeout_s):
        return 200, b'{"data": [{"nope": 1}]}'

    monkeypatch.setattr("memlite_py.embedding.openai._http_post_json", post_bad_shape)
    with pytest.raises(EmbedError) as ei4:
        await emb.embed("x")
    assert ei4.value.kind == "shape"

    def post_bad_json(url, headers, body, timeout_s):
        return 200, b"not json"

    monkeypatch.setattr("memlite_py.embedding.openai._http_post_json", post_bad_json)
    with pytest.raises(EmbedError) as ei5:
        await emb.embed("x")
    assert ei5.value.kind == "shape"


def _resolver(key: str):
    async def r(ref):
        return key

    return r


def test_embed_options_timeout_default():
    o = EmbedOptions()
    assert o.timeout_ms is None
