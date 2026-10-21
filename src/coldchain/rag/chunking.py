"""Heading-aware Markdown chunking.

SOPs are structured documents: a rule only makes sense next to the heading it sits
under. So chunks follow the heading tree, each chunk keeps its full section path
("Cold-Chain SOP > 1. Temperature Control"), and long sections are split on
paragraph/bullet boundaries with a small overlap.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True, slots=True)
class Chunk:
    document: str
    section: str
    chunk_index: int
    content: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.document}|{self.section}|{self.content}".encode()).hexdigest()


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    units = [u for u in re.split(r"\n(?=\s*[*\-]\s|\n)", text) if u.strip()]
    out, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > max_chars:
            out.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += "\n" + u
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_markdown(
    text: str, document: str, max_chars: int = 1200, overlap: int = 150
) -> list[Chunk]:
    path: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    buf: list[str] = []

    def close() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in path) or document, [body]))
        buf.clear()

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            close()
            level, title = len(m.group(1)), m.group(2).strip().strip("*")
            path[:] = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
        else:
            buf.append(line)
    close()

    chunks: list[Chunk] = []
    for section, (body,) in sections:
        for part in _split_long(body, max_chars, overlap):
            chunks.append(Chunk(document, section, len(chunks), part))
    return chunks
