"""中英混合分词：ASCII 整词小写；CJK 段取单字 + bigram。

与 memlite-js（tokenize.ts）和 Rust memlite search.rs tokenize 逐字对齐。
"""

from __future__ import annotations

import re

_ALNUM = re.compile(r"[A-Za-z0-9_]")


def is_cjk(ch: str) -> bool:
    """CJK 词界仅 U+4E00–U+9FFF（历史口径，勿扩全 CJK）：
    全角标点（，。 U+3000/FF00 段）是分隔符不是词——当词切会改变文档长度，
    BM25 分数随之漂移（对拍发现过 0.7% 差）。"""
    code = ord(ch[0]) if ch else 0
    return 0x4E00 <= code <= 0x9FFF


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    ascii_buf = ""
    cjk_buf = ""

    def flush_ascii() -> None:
        nonlocal ascii_buf
        if ascii_buf:
            out.append(ascii_buf.lower())
            ascii_buf = ""

    def flush_cjk() -> None:
        nonlocal cjk_buf
        chars = list(cjk_buf)
        out.extend(chars)
        for i in range(len(chars) - 1):
            out.append(chars[i] + chars[i + 1])
        cjk_buf = ""

    for ch in text:
        if _ALNUM.fullmatch(ch):
            flush_cjk()
            ascii_buf += ch
        elif is_cjk(ch):
            flush_ascii()
            cjk_buf += ch
        else:
            flush_ascii()
            flush_cjk()
    flush_ascii()
    flush_cjk()
    return out
