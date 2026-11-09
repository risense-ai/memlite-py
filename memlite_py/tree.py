"""因果树与因果链：树作用域、子树渲染、outline 目录、chain 深挖、recall/show 的因果视界行。

语义对齐 respire（memocap CLI）：父=因/子=果，根在前，trivial 出树、父失效提根。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .store import MemRecord, Store, importance, kind, parent_id

PARENT_L1_CAP = 140


class TreeScope:
    """树作用域内的节点（trivial 出树；父不在作用域则提为根）。结构由 kids（含 '' 键=根集）完整表达。"""

    def __init__(self, store: Store):
        in_scope = {r.id for r in store.records.values() if importance(r) != "trivial"}
        self.nodes = [r for r in store.all() if r.id in in_scope]
        self.nodes.sort(key=lambda r: r.ts)
        self.kids: dict[str, list[str]] = {}
        self.roots: list[str] = []
        for r in self.nodes:
            p = parent_id(r)
            if p == "" or p not in in_scope:
                self.roots.append(r.id)
            else:
                self.kids.setdefault(p, []).append(r.id)

    def children_of(self, node_id: str) -> list[str]:
        return self.kids.get(node_id, [])

    def get(self, node_id: str) -> MemRecord | None:
        for r in self.nodes:
            if r.id == node_id:
                return r
        return None


@dataclass
class TreeNode:
    id: str
    title: str
    kind: str
    level: int
    leaf: bool
    children: list["TreeNode"] = field(default_factory=list)


def _label(scope: TreeScope, node_id: str) -> str:
    r = scope.get(node_id)
    t = r.title if r and r.title != "" else node_id[:8]
    return t


def _id8(node_id: str) -> str:
    return node_id[:8]


def render_tree(scope: TreeScope, from_id: str | None, depth: int) -> list[str]:
    """树渲染（对齐 respire 格式）：根头 `🌳 因果树根 [id8] title`，
    分支 `├─ `/`└─ `，竖线继承用 next_prefix（三层不串行）。"""
    out: list[str] = []
    seen: set[str] = set()

    def render(node: str, depth_left: int, prefix: str) -> None:
        if node not in seen:
            seen.add(node)
        else:
            out.append(f"{prefix}（环，截断）[{_id8(node)}]")
            return
        title = _label(scope, node)
        kids = scope.children_of(node)
        rec = scope.get(node)
        meta = kind(rec) if rec else ""
        leaf = " 🍃叶" if len(kids) == 0 else ""
        out.append(f"{prefix}{title} [{_id8(node)}] {meta}{leaf}")
        if depth_left == 0:
            return
        n = len(kids)
        for i, kid in enumerate(kids):
            last = i == n - 1
            branch = "└─ " if last else "├─ "
            # 子层前缀继承 next_prefix：非末子的后代要带竖线，否则三层渲染串行
            next_prefix = prefix + ("   " if last else "│  ")
            render(kid, depth_left - 1, next_prefix + branch)
        seen.discard(node)

    if from_id is not None:
        if scope.get(from_id) is None:
            raise ValueError(f"无此节点或已出树: {from_id}")
        out.append(f"🌳 因果树（自 {from_id}）")
        render(from_id, depth, "")
    else:
        if not scope.roots:
            return ["（无根节点——库内皆叶，因果树未成形）"]
        for r in scope.roots:
            out.append(f"🌳 因果树根 [{_id8(r)}] {_label(scope, r)}")
            kids = scope.children_of(r)
            n = len(kids)
            for i, kid in enumerate(kids):
                last = i == n - 1
                branch = "└─ " if last else "├─ "
                render(kid, max(depth - 1, 0), branch)
    return out


def outline(scope: TreeScope, from_id: str | None) -> tuple[list[str], int]:
    """outline 全深目录：`[id8] title` 逐层缩进（AI 梳理先读此），返回行 + 节点数。"""
    out: list[str] = []
    seen: set[str] = set()

    def walk(node: str, indent: int) -> None:
        if node in seen:
            return
        seen.add(node)
        out.append(f"{'  ' * indent}[{_id8(node)}] {_label(scope, node)}")
        for kid in scope.children_of(node):
            walk(kid, indent + 1)

    if from_id is not None:
        if scope.get(from_id) is None:
            raise ValueError(f"无此节点或已出树: {from_id}")
        walk(from_id, 0)
    else:
        for r in scope.roots:
            walk(r, 0)
    return out, len(out)


def tree_json(scope: TreeScope, from_id: str | None) -> list[TreeNode]:
    """tree --json 结构（全深树，from 定起点）。"""
    seen: set[str] = set()

    def build(node: str, level: int) -> TreeNode:
        is_new = node not in seen
        seen.add(node)
        kids = scope.children_of(node) if is_new else []
        rec = scope.get(node)
        return TreeNode(
            id=node,
            title=_label(scope, node),
            kind=kind(rec) if rec else "",
            level=level,
            leaf=len(kids) == 0,
            children=[build(k, level + 1) for k in kids],
        )

    if from_id is not None:
        if scope.get(from_id) is None:
            raise ValueError(f"无此节点或已出树: {from_id}")
        return [build(from_id, 0)]
    return [build(r, 0) for r in scope.roots]


@dataclass
class ChainView:
    """因果链深挖（对齐 respire chain）：因链全文（根在前）+ 本条 + 果链下探 depth 层。"""

    ancestors: list[MemRecord]
    entry: MemRecord
    """(层级 1..=depth, 记录)，遍历序与 Rust 一致（frontier pop）"""
    descendants: list[tuple[int, MemRecord]]


def chain_view(store: Store, rid: str, depth: int) -> ChainView:
    entry = store.records.get(rid)
    if entry is None:
        raise ValueError(f"记忆不存在: {rid}")
    ancestors = [
        store.records[pid] for pid in store.ancestor_chain(rid) if pid in store.records
    ]
    descendants: list[tuple[int, MemRecord]] = []
    frontier: list[tuple[str, int]] = [(rid, 1)]
    seen = {rid}
    while frontier:
        cur, lv = frontier.pop()
        if lv > depth:
            continue
        for c in store.children(cur):
            if c.id in seen:
                continue
            seen.add(c.id)
            descendants.append((lv, c))
            frontier.append((c.id, lv + 1))
    return ChainView(ancestors=ancestors, entry=entry, descendants=descendants)


def causal_lines(store: Store, rid: str) -> list[str]:
    """recall/show 因果视界行（对齐 respire print_causal）：
    🔺因链标题链 → ▸纲L1 父内容头 → 🔻果（3 子带摘要 + 各孙标题）→ 超量提示走 chain。
    返回行列表（无父无子返回空）。"""
    out: list[str] = []

    def title_of(pid: str) -> str:
        p = store.records.get(pid)
        return p.title if p and p.title != "" else pid[:8]

    chain = [pid for pid in store.ancestor_chain(rid) if pid in store.records]
    if chain:
        parts = [title_of(pid) for pid in chain]
        out.append(f"   🔺因链: {' ← '.join(parts)} → 本条为其果")
        last_pid = chain[-1]
        p = store.records.get(last_pid)
        if p:
            head = p.content[:PARENT_L1_CAP].replace("\n", " ")
            if head.strip() != "":
                out.append(f"   ▸纲L1 「{title_of(last_pid)}」 {head}")
    children = store.children(rid)
    if children:
        out.append(f"   🔻果: {len(children)} 个（本条又为下游之因）")
        for c in children[:3]:
            title = c.title if c.title != "" else c.id[:8]
            summary = c.content[:60].replace("\n", " ")
            sep = "" if summary == "" else " · "
            out.append(f"      └─ [{c.id[:8]}] {title}{sep}{summary}")
            grands = store.children(c.id)
            shown = min(3, len(grands))
            for g in grands[:shown]:
                gt = g.title if g.title != "" else g.id[:8]
                out.append(f"         └─ {gt} [{g.id[:8]}]")
            if len(grands) > shown:
                out.append(f"         …另 {len(grands) - shown} 个")
        if len(children) > 3:
            out.append(f"      …另 {len(children) - 3} 个果 → memlite chain {rid[:8]}")
    return out


def tree_node_to_dict(node: TreeNode) -> dict[str, Any]:
    """TreeNode → JSON 兼容 dict（宿主插件序列化用）。"""
    return {
        "id": node.id,
        "title": node.title,
        "kind": node.kind,
        "level": node.level,
        "leaf": node.leaf,
        "children": [tree_node_to_dict(c) for c in node.children],
    }
