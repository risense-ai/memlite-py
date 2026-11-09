"""配置分层：项目级（工作区记忆文件夹内 config.json）> 用户级（~/.memlite/config.json）。

- data_dir：CLI flag > MEMLITE_DATA > 工作区记忆文件夹 > 用户级 data_dir > 默认
- embedding：env > 项目级 > 用户级（API 三件套写入默认落用户级）
- 工作区记忆文件夹 = 从 cwd 向上找：`<dir>/.memlite/config.json` 或 `<dir>/memlite-data/config.json`，
  文件夹本身即 data_dir（项目级记忆与工作区同生共死，随工作区 git 一起同步）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ConfigCtx:
    """可注入的环境上下文（测试用；缺省取真实进程环境）。"""

    env: dict[str, str] | None = None
    cwd: str | None = None
    home: str | None = None


def _env_of(ctx: ConfigCtx | None) -> dict[str, str]:
    return ctx.env if ctx is not None and ctx.env is not None else os.environ


def home_dir(ctx: ConfigCtx | None = None) -> str:
    """家目录：注入 home → HOME（Linux/macOS）→ USERPROFILE（Windows）→ 点号（当前目录）。"""
    env = _env_of(ctx)
    if ctx is not None and ctx.home is not None:
        return ctx.home
    return env.get("HOME") or env.get("USERPROFILE") or "."


def user_config_path(ctx: ConfigCtx | None = None) -> str:
    env = _env_of(ctx)
    override = env.get("MEMLITE_CONFIG")
    if override:
        return override
    return os.path.join(home_dir(ctx), ".memlite", "config.json")


def load_config(p: str) -> dict[str, Any] | None:
    """读 JSON 配置；不存在或坏 JSON 返回 None。"""
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def find_project_dir(ctx: ConfigCtx | None = None) -> str | None:
    """从 cwd 向上找工作区记忆文件夹（含 config.json 的 .memlite/ 或 memlite-data/）。

    找到的第一个即项目级 data_dir——不跨出工作区继续上爬（git 根或文件系统根为界）。
    用户级配置目录（~/.memlite）与项目级同名，必须排除——否则家目录跑任何命令都会
    把用户配置误认成项目库（data_dir 错指 ~/.memlite）。"""
    user_dir = os.path.dirname(user_config_path(ctx))
    cur = os.path.abspath(ctx.cwd if ctx is not None and ctx.cwd else os.getcwd())
    while True:
        for name in (".memlite", "memlite-data"):
            cand = os.path.join(cur, name)
            if os.path.isfile(os.path.join(cand, "config.json")):
                if cand == user_dir:
                    continue  # 用户级配置，不是项目库
                return cand
        # git 根不越过：工作区即仓库根
        parent = os.path.dirname(cur)
        if os.path.exists(os.path.join(cur, ".git")) or parent == cur:
            return None
        cur = parent


def project_config(ctx: ConfigCtx | None = None) -> tuple[str | None, dict[str, Any] | None]:
    d = find_project_dir(ctx)
    if d is None:
        return None, None
    return d, load_config(os.path.join(d, "config.json"))


def user_data_dir(ctx: ConfigCtx | None = None) -> str:
    """用户级 data_dir（不受项目级影响——两档回退的兜底目标）。"""
    user = load_config(user_config_path(ctx))
    d = (user or {}).get("data_dir")
    if isinstance(d, str) and d != "":
        return d
    return os.path.join(home_dir(ctx), "memlite-data")


def resolve_data_dir(cli: str | None = None, ctx: ConfigCtx | None = None) -> str:
    """data_dir 解析（四层）。"""
    if cli:
        return cli
    env_data = _env_of(ctx).get("MEMLITE_DATA")
    if env_data:
        return env_data
    proj = find_project_dir(ctx)
    if proj:
        return proj
    return user_data_dir(ctx)


def embedding_merged(ctx: ConfigCtx | None = None) -> dict[str, Any]:
    """embedding 合并：用户级为底、项目级覆盖、env 最后覆盖（操作者意图最优先）。"""
    user = load_config(user_config_path(ctx))
    _, proj = project_config(ctx)
    out: dict[str, Any] = {}
    for src in ((user or {}).get("embedding"), (proj or {}).get("embedding")):
        if isinstance(src, dict) and not isinstance(src, list):
            out.update(src)
    env = _env_of(ctx)
    api_base = env.get("MEMLITE_API_BASE")
    if api_base:
        out["api_base"] = api_base
    api_key = env.get("MEMLITE_API_KEY")
    if api_key:
        out["api_key"] = api_key
    model = env.get("MEMLITE_EMBED_MODEL")
    if model:
        out["model"] = model
    return out


def config_set(key: str, value: str, project: bool, ctx: ConfigCtx | None = None) -> str:
    """`memlite config set <key> <value>`——key 形如 data_dir / embedding.api_base；

    缺省写用户级（全局），project=True 写项目级（须已识别到工作区记忆文件夹）。
    返回写入的配置文件路径。"""
    if project:
        d, pc = project_config(ctx)
        if d is None:
            raise ValueError("未识别到工作区记忆文件夹（.memlite/ 或 memlite-data/ 含 config.json）")
        p = os.path.join(d, "config.json")
        cfg = pc or {}
    else:
        p = user_config_path(ctx)
        cfg = load_config(p) or {}
    keys = key.split(".")
    if len(keys) == 1:
        cfg[keys[0]] = _value_into(keys[0], value, ctx)
    elif len(keys) == 2:
        sec, fld = keys
        sub = cfg.get(sec)
        if not (isinstance(sub, dict) and not isinstance(sub, list)):
            cfg[sec] = {}
        cfg[sec][fld] = value
    else:
        raise ValueError(f"key 只支持一层或两层（如 embedding.api_base）: {key}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return p


def _value_into(key: str, value: str, ctx: ConfigCtx | None) -> str:
    """data_dir 这类路径值原样存（并展开 ~/），其余按字符串存（不猜类型，简单可预期）。"""
    if key == "data_dir":
        return _shellexpand_home(value, home_dir(ctx))
    return value


def _shellexpand_home(v: str, home: str) -> str:
    if v.startswith("~/"):
        return os.path.join(home, v[2:])
    return v
