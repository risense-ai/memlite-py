# Changelog

## 0.1.0 (2026-11-07)

首版：memlite-js（@rsrsai/memlite）packages/core 的 Python 忠实移植。

- 存储：append-only JSONL + 16MB 轮转 + LWW + del 墓碑复活 + 压实
- 检索：中英分词 / BM25 / 余弦语义 / RRF 融合 / 项目级→用户级回退
- 因果树：TreeScope / render / outline / json / chain 深挖 / causal 视界行
- 向量侧车：emb.f32 + emb-ids.txt（Rust/JS 互通），dim 自适应，原地覆盖
- 敏感闸：PEM/JWT/token 前缀/明文口令/身份证/银行卡 Luhn/手机号
- 配置：~/.memlite 与工作区项目级分层 + MEMLITE_* env（兼容 Rust 写的 config.json api_key）
- 检索日志：querylog 常驻聚合 + 训练集导出
- 同步：git 白名单 pathspec 提交 + push sha 双通道核验
- 嵌入档：none / host 注入 / openai-compatible（标准库 HTTP，错误分类）
- 主类：Memlite（remember/recall/show/chain/tree/update/reembed/sync/status/compact）+ memlite_from_files
- 98 例测试全绿，含手写 Rust 形状互通 fixtures
