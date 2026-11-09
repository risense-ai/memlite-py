"""向量侧车：emb.f32（packed float32 LE）+ emb-ids.txt（id 顺序对齐）。

dim 自适应：首写定维，读时按 文件字节数/id 数 推导。
与 Rust memlite vectors.rs / memlite-js vectors.ts 文件格式互通（compat.md §三）。
"""

from __future__ import annotations

import os
import struct

IDS_FILE = "emb-ids.txt"
F32_FILE = "emb.f32"


class VectorIndex:
    def __init__(self, data_dir: str):
        self.dir = data_dir
        self.dim: int | None = None
        self.ids: list[str] = []
        self.offset: dict[str, int] = {}
        self._cache: bytes | None = None

        text: str | None
        try:
            with open(os.path.join(data_dir, IDS_FILE), "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = None
        if text is not None:
            # 空行不占 id，但保留其行号作向量下标（与 JS/Rust 对齐）
            for i, line in enumerate(text.split("\n")):
                if line != "":
                    self.offset[line] = i
                    self.ids.append(line)
        if self.dim is None and self.ids:
            try:
                n = os.path.getsize(os.path.join(data_dir, F32_FILE)) / 4
                if self.ids and n % len(self.ids) == 0:
                    self.dim = int(n / len(self.ids))
            except OSError:
                pass  # 无 f32 文件保持 dim 未定

    def _f32_path(self) -> str:
        return os.path.join(self.dir, F32_FILE)

    def _load_cache(self) -> None:
        if self._cache is not None:
            return
        try:
            with open(self._f32_path(), "rb") as f:
                self._cache = f.read()
        except OSError:
            self._cache = None

    def get(self, rid: str) -> list[float] | None:
        idx = self.offset.get(rid)
        if idx is None:
            return None
        dim = self.dim
        if dim is None:
            return None
        self._load_cache()
        raw = self._cache
        if raw is None:
            return None
        start = idx * dim * 4
        end = start + dim * 4
        if end > len(raw):
            return None
        return list(struct.unpack_from(f"<{dim}f", raw, start))

    def put(self, rid: str, emb: list[float]) -> None:
        """首写定维；维度不符拒收；同 id 原地覆盖；写前失效缓存。"""
        os.makedirs(self.dir, exist_ok=True)
        if self.dim is None:
            self.dim = len(emb)
        if len(emb) != self.dim:
            raise ValueError(f"向量维度不一致: {len(emb)} != {self.dim}")
        self._cache = None
        buf = struct.pack(f"<{len(emb)}f", *emb)
        existing = self.offset.get(rid)
        if existing is not None:
            # 同 id 原地覆盖
            fd = os.open(self._f32_path(), os.O_RDWR)
            try:
                os.lseek(fd, existing * len(emb) * 4, os.SEEK_SET)
                os.write(fd, buf)
                os.fsync(fd)
            finally:
                os.close(fd)
            return
        fd = os.open(self._f32_path(), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            os.write(fd, buf)
            os.fsync(fd)
        finally:
            os.close(fd)
        ids_fd = os.open(
            os.path.join(self.dir, IDS_FILE), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o644
        )
        try:
            os.write(ids_fd, (rid + "\n").encode("utf-8"))
            os.fsync(ids_fd)
        finally:
            os.close(ids_fd)
        self.offset[rid] = len(self.ids)
        self.ids.append(rid)

    def has(self, rid: str) -> bool:
        return rid in self.offset

    def count(self) -> int:
        return len(self.ids)
