"""配置分层：data_dir 四层解析、项目级目录发现、config set。"""

import json
import os

import pytest

from memlite_py.config import (
    ConfigCtx,
    config_set,
    embedding_merged,
    find_project_dir,
    home_dir,
    resolve_data_dir,
    user_config_path,
    user_data_dir,
)


def ctx(tmp: str, **kw) -> ConfigCtx:
    return ConfigCtx(env={"HOME": tmp, "USERPROFILE": tmp, **kw.pop("env", {})}, cwd=kw.pop("cwd", tmp), home=tmp)


def test_home_dir_windows_fallback(tmp_dir):
    c = ConfigCtx(env={"USERPROFILE": tmp_dir}, cwd=tmp_dir)
    assert home_dir(c) == tmp_dir
    c2 = ConfigCtx(env={"HOME": "/home/u"}, cwd=tmp_dir)
    assert home_dir(c2) == "/home/u"


def test_resolve_data_dir_cli_env_user(tmp_dir):
    c = ctx(tmp_dir)
    assert resolve_data_dir("C:/explicit", c) == "C:/explicit"
    c.env["MEMLITE_DATA"] = "D:/env-data"
    assert resolve_data_dir(None, c) == "D:/env-data"
    del c.env["MEMLITE_DATA"]
    assert resolve_data_dir(None, c) == os.path.join(tmp_dir, "memlite-data")  # 用户级默认


def test_find_project_dir_walks_up(tmp_dir):
    ws = os.path.join(tmp_dir, "ws")
    sub = os.path.join(ws, "src", "deep")
    os.makedirs(sub)
    proj = os.path.join(ws, ".memlite")
    os.makedirs(proj)
    with open(os.path.join(proj, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"data_dir": proj}, f)
    c = ctx(tmp_dir, cwd=sub)
    assert find_project_dir(c) == proj
    assert resolve_data_dir(None, c) == proj


def test_find_project_dir_stops_at_git_root(tmp_dir):
    ws = os.path.join(tmp_dir, "ws")
    sub = os.path.join(ws, "sub")
    os.makedirs(sub)
    os.makedirs(os.path.join(ws, ".git"))
    # 上层有 .memlite 但被 git 根隔开 → 不返回
    upper = os.path.join(tmp_dir, "upper")
    os.makedirs(os.path.join(upper, "memlite-data", ".x"), exist_ok=True)
    with open(os.path.join(upper, "memlite-data", "config.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    c = ctx(tmp_dir, cwd=sub)
    assert find_project_dir(c) is None


def test_user_dir_not_mistaken_for_project(tmp_dir):
    """~/.memlite/config.json 是用户级配置，不是项目库。"""
    home = os.path.join(tmp_dir, "home")
    os.makedirs(os.path.join(home, ".memlite"))
    with open(os.path.join(home, ".memlite", "config.json"), "w", encoding="utf-8") as f:
        json.dump({"data_dir": "X:/real"}, f)
    c = ConfigCtx(env={"HOME": home, "USERPROFILE": home}, cwd=home, home=home)
    assert find_project_dir(c) is None
    assert resolve_data_dir(None, c) == "X:/real"


def test_user_data_dir_from_config(tmp_dir):
    home = os.path.join(tmp_dir, "home")
    os.makedirs(os.path.join(home, ".memlite"))
    p = os.path.join(home, ".memlite", "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"data_dir": "D:/lib"}, f)
    c = ConfigCtx(env={"HOME": home, "USERPROFILE": home}, cwd=tmp_dir, home=home)
    assert user_data_dir(c) == "D:/lib"
    assert user_config_path(c) == p


def test_config_set_user_level(tmp_dir):
    c = ctx(tmp_dir)
    p = config_set("embedding.api_base", "https://api.example.com/v1", False, c)
    cfg = json.load(open(p, encoding="utf-8"))
    assert cfg["embedding"]["api_base"] == "https://api.example.com/v1"
    p2 = config_set("data_dir", "~/custom-lib", False, c)
    assert json.load(open(p2, encoding="utf-8"))["data_dir"] == os.path.join(tmp_dir, "custom-lib")
    with pytest.raises(ValueError, match="key 只支持"):
        config_set("a.b.c", "v", False, c)


def test_config_set_project_level(tmp_dir):
    ws = os.path.join(tmp_dir, "ws")
    proj = os.path.join(ws, "memlite-data")
    os.makedirs(proj)
    with open(os.path.join(proj, "config.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    c = ctx(tmp_dir, cwd=ws)
    p = config_set("embedding.model", "BAAI/bge-m3", True, c)
    assert p == os.path.join(proj, "config.json")
    # 无项目级目录时拒写
    c2 = ctx(os.path.join(tmp_dir, "nowhere"))
    with pytest.raises(ValueError, match="未识别到工作区"):
        config_set("data_dir", "x", True, c2)


def test_embedding_merge_priority(tmp_dir):
    home = os.path.join(tmp_dir, "home")
    proj = os.path.join(tmp_dir, "ws", ".memlite")
    os.makedirs(os.path.join(home, ".memlite"))
    os.makedirs(proj)
    with open(os.path.join(home, ".memlite", "config.json"), "w", encoding="utf-8") as f:
        json.dump({"embedding": {"api_base": "user-base", "model": "user-model"}}, f)
    with open(os.path.join(proj, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"embedding": {"model": "proj-model"}}, f)
    c = ConfigCtx(env={"HOME": home, "USERPROFILE": home}, cwd=os.path.join(tmp_dir, "ws"), home=home)
    merged = embedding_merged(c)
    assert merged["api_base"] == "user-base"
    assert merged["model"] == "proj-model"  # 项目级覆盖用户级
    # env 最优先
    c.env["MEMLITE_API_BASE"] = "env-base"
    c.env["MEMLITE_API_KEY"] = "env-key"
    merged2 = embedding_merged(c)
    assert merged2["api_base"] == "env-base"
    assert merged2["api_key"] == "env-key"
