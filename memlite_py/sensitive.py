"""写前敏感扫描：密码/身份证/手机号/银行卡/token/私钥，默认拒绝入库。

命中即拒（不是脱敏改写）——提示用户改存位置指针（如「密码在 ~/.env」）。
想强制入库暂不提供口子：这是嵌入式明文库，宁严勿松。
规则与 Rust memlite sensitive.rs / memlite-js sensitive.ts 对齐（compat.md §七）。
"""

from __future__ import annotations

import re


class SensitiveError(ValueError):
    """敏感闸拒绝入库。message 即用户可读的拒绝文案（宿主原样透传）。"""


def scan(text: str) -> list[str]:
    """扫描文本，返回命中种类（空表 = 干净）。"""
    hits: list[str] = []
    t = text

    # PEM 私钥块
    if "-----BEGIN" in t and ("PRIVATE KEY" in t or "RSA PRIVATE" in t):
        hits.append("PEM私钥")
    # JWT（eyJ 开头的三段 base64url）
    if _find_jwt(t):
        hits.append("JWT")
    # 常见 token 前缀
    prefixes: list[tuple[str, str]] = [
        ("sk-", "API密钥(sk-)"),
        ("ghp_", "GitHub token"),
        ("gho_", "GitHub token"),
        ("github_pat_", "GitHub PAT"),
        ("AKIA", "AWS密钥"),
        ("xox", "Slack token"),
    ]
    for pat, name in prefixes:
        if pat in t:
            if pat == "sk-":
                # sk- 后须跟足够长的字母数字，防误伤普通词（只看首个出现，与 JS/Rust 对齐）
                p = t.find("sk-")
                if p >= 0:
                    tail = t[p + 3 :]
                    m = re.match(r"^[A-Za-z0-9]+", tail)
                    if m and len(m.group(0)) >= 16:
                        hits.append(name)
            else:
                hits.append(name)
    # key=value / password: xxx 赋值式口令（明文口令）
    if _has_plaintext_password(t):
        hits.append("明文口令")
    # 身份证 18 位（末位可 X）
    if _find_cn_id(t):
        hits.append("身份证号")
    # 银行卡 16-19 位连续数字（Luhn 校验，防误伤长订单号）
    if _find_bank_card(t):
        hits.append("银行卡号")
    # CN 手机号 1[3-9]xxxxxxxxx（11 位，带边界）
    if _find_cn_phone(t):
        hits.append("手机号")
    return list(dict.fromkeys(hits))  # 去重保序（对齐 JS Set 语义）


_B64 = re.compile(r"[A-Za-z0-9_\-]")


def _find_jwt(t: str) -> bool:
    """eyJ 开头，后接 ≥10 个 base64url 字符，再 '.' 再非空 base64url 段（≥5）。"""
    i = 0
    while i + 10 < len(t):
        if t.startswith("eyJ", i):
            j = i + 3
            while j < len(t) and _B64.match(t[j]):
                j += 1
            seg1 = j - (i + 3)
            if seg1 >= 10 and j < len(t) and t[j] == ".":
                k = j + 1
                while k < len(t) and _B64.match(t[k]):
                    k += 1
                if k - (j + 1) >= 5:
                    return True
        i += 1
    return False


_PLACEHOLDER = {"none", "null", "true", "false", "<password>", "${password}", "***", "xxx"}
_INTERROGATIVE = ("什么", "啥", "多少", "几位", "忘了", "不告诉你", "不知道", "没设", "没有")

_PASSWORD_KEYS = (
    "password=",
    "password：",
    "password:",
    "password is ",
    "passwd=",
    "passwd:",
    "pwd=",
    "pwd:",
    "密码=",
    "密码：",
    "密码:",
    "密码是",
    "密碼是",
    "口令=",
    "口令：",
    "口令:",
    "口令是",
)


def _has_plaintext_password(t: str) -> bool:
    """形如 password=xxx / passwd: xxx / 密码：xxx / 密码是xxx（赋值右侧非空、非纯占位符）。"""
    lower = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), t)
    for key in _PASSWORD_KEYS:
        ascii_key = bool(re.match(r"^[ -~]", key))
        hay = lower if ascii_key else t
        p = hay.find(key)
        if p < 0:
            continue
        val = hay[p + len(key) :].lstrip()
        val = val.lstrip("\"'")
        m = re.match(r"^[^\s\"');]+", val)
        # 排除占位符、疑问词与显然非密码的
        word = m.group(0) if m else ""
        if (
            len(word.encode("utf-8")) >= 4
            and word not in _PLACEHOLDER
            and not word.startswith(_INTERROGATIVE)
        ):
            return True
    return False


def _luhn_ok(num: str) -> bool:
    """简化 Luhn。"""
    digits = [int(c) for c in num if c.isdigit()]
    if len(digits) < 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _digit_runs(t: str, min_len: int, max_len: int) -> list[str]:
    """抽 [min,max] 位连续纯数字段（整段超长时按窗滑取）。"""
    out: list[str] = []
    for m in re.finditer(r"[0-9]+", t):
        run = m.group(0)
        if len(run) >= min_len:
            if len(run) > max_len:
                for s in range(len(run) - max_len + 1):
                    out.append(run[s : s + max_len])
            else:
                out.append(run)
    return out


def _find_bank_card(t: str) -> str | None:
    """抽连续数字段（≥16），Luhn 通过才认。"""
    for seg in _digit_runs(t, 16, 19):
        if _luhn_ok(seg):
            return seg
    return None


def _find_cn_id(t: str) -> str | None:
    """18 位身份证：17 数字 + 末位数字或 X；年份段 1900..=2100 弱校验。"""
    for seg in _digit_runs(t, 17, 18):
        if len(seg) == 17:
            continue  # 还要跟一个 X/x 才完整
        if len(seg) == 18:
            year = int(seg[6:10])
            if 1900 <= year <= 2100:
                return seg
    # 17 位数字紧跟 X
    for m in re.finditer(r"(?<![0-9])([0-9]{17})[Xx]", t):
        seg = m.group(1)
        year = int(seg[6:10])
        if 1900 <= year <= 2100:
            return seg + "X"
    return None


def _find_cn_phone(t: str) -> str | None:
    """11 位手机号：1[3-9]xxxxxxxxx，前后无其他数字（防长数字中段误中）。"""
    m = re.search(r"(?<![0-9])1[3-9][0-9]{9}(?![0-9])", t)
    return m.group(0) if m else None
