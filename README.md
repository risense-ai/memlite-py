# memlite-py

> 可嵌入的本地记忆引擎（Python 版）。`memlite-js`（`@rsrsai/memlite`）的忠实移植，
> 与 [respire-memlite](https://github.com/risense-ai/respire-memlite)（Rust）数据契约互通。

零第三方依赖（纯标准库），Python ≥ 3.10。

## 特性

- **存储层**：append-only JSONL + 16MB 分块轮转 + 内存索引；末写胜出（LWW）；del 墓碑可复活
- **检索**：中英混合分词（ASCII 整词 + CJK 单字/bigram）→ BM25 必算 → 嵌入余弦 → RRF 融合；嵌入失败自动回退 BM25
- **因果树**：父=因/子=果；attach/promote/demote 防环改挂；chain 深挖；outline/render/json 三种树视图
- **向量侧车**：`emb.f32`（packed f32 LE）+ `emb-ids.txt`，dim 自适应，可由 Rust/JS 端写入
- **敏感闸**：写前扫描（口令/密钥/身份证/银行卡 Luhn/手机号/JWT/PEM），命中即拒
- **检索日志**：`memory.querylog.jsonl` 一次查询一行，聚合计数，可导出训练集
- **git 同步**：白名单（`memory*.jsonl`/`meta.json`/`README.md`）pathspec 提交 + push 双通道 sha 核验
- **嵌入档**：`none` / `host`（宿主注入）/ `openai-compatible`（标准库 HTTP）

## 快速上手

```python
import asyncio
from memlite_py import Memlite, MemliteConfig, RememberOpts, RecallOpts, memlite_from_files

mem = Memlite(MemliteConfig(data_dir="~/memlite-data"))

async def main():
    rid = await mem.remember("nginx 504 超时：上游 connect 超时配置过小", RememberOpts(title="nginx 504 排查"))
    child = await mem.remember("根因是 proxy_connect_timeout 5s 太小", RememberOpts(title="根因", parent=rid))
    res = await mem.recall("nginx 超时")
    for hit in res.hits:
        print(hit.score, hit.id[:8], hit.title, "← 因链:", [a.title for a in hit.ancestors])
    print(mem.tree(mode="outline"))
    mem.close()

asyncio.run(main())
```

分层配置（与 Rust CLI 共用）：`~/.memlite/config.json` / 工作区 `.memlite/config.json` / 环境变量 `MEMLITE_DATA` `MEMLITE_API_BASE` `MEMLITE_API_KEY` `MEMLITE_EMBED_MODEL`：

```python
mem = memlite_from_files()   # 从配置文件与环境变量解析
```

## 数据契约（compat.md 摘要）

任何一版（Rust/JS/Python）破坏下列契约即视为 bug：

| 契约 | 值 |
|---|---|
| 记忆行 | `{"op","id","ts","title","content","tags","project", ...extra 平铺}` |
| id | uuid v4 simple 前 12 位 hex |
| 向量文件 | `emb.f32` packed f32 LE + `emb-ids.txt`（行号=下标），无 header |
| 检索日志 | `memory.querylog.jsonl`：`{ts,query,backend,limit,fallback,source,ms,hits}` |
| BM25 | k1=1.2, b=0.75, title×3 权重 |
| CJK 词界 | U+4E00–U+9FFF（勿扩全 CJK） |
| RRF | k=60 |
| 因果链防环 | 封顶 64 层 |
| querylog 尾窗 | 50_000 条 |
| 同步白名单 | `memory*.jsonl` / `meta.json` / `README.md`（向量文件不入 git） |

## 与宿主集成

宿主（AstrBot 插件、DSH 等）通过 `MemliteDeps` 注入：

- `embedder`：完整 Embedder（`async embed(text) -> list[float]`，`dim`）
- `host_embedder` + `embedding={"provider": "host"}`：宿主托管嵌入
- `credential_resolver`：`credential_ref → api_key`（密钥永不入本包）
- `logger`：`warn(msg)` 告警出口
- `ctx`：`ConfigCtx(env, cwd, home)` 环境上下文（测试隔离用）

## 测试

```bash
python -m pytest -q   # 98 例：存储/分词/BM25/RRF/向量/敏感闸/因果树/配置/日志/同步/嵌入档/互通契约
```

互通契约测试内置手写 Rust 形状 fixtures（JSONL flatten、emb.f32 二进制、querylog 行），不依赖 Rust 二进制。

## 协议

MIT。
