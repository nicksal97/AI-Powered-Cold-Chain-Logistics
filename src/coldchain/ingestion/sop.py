"""Incremental SOP indexing.

Each chunk is keyed by a content hash, so re-running only embeds what changed
(embedding calls cost money), and chunks that no longer exist in a document are
deleted so retrieval never cites a superseded rule.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, delete, select

from coldchain.db.models import SopChunk
from coldchain.db.session import session_scope
from coldchain.logging import get_logger
from coldchain.rag.chunking import chunk_markdown
from coldchain.rag.embeddings import Embedder

log = get_logger(__name__)
SUPPORTED = {".md", ".markdown", ".txt"}


def ingest_sop_dir(directory: Path, embedder: Embedder, engine: Engine | None = None) -> dict:
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No SOP documents ({', '.join(SUPPORTED)}) under {directory}")

    stats = {"documents": 0, "added": 0, "unchanged": 0, "removed": 0}
    with session_scope(engine) as session:
        for path in files:
            doc = path.name
            chunks = chunk_markdown(path.read_text(encoding="utf-8"), document=doc)
            wanted = {c.content_hash: c for c in chunks}
            existing = set(
                session.execute(
                    select(SopChunk.content_hash).where(
                        SopChunk.document == doc, SopChunk.embedding_model == embedder.model_name
                    )
                ).scalars()
            )
            # delete first: stale chunks (or chunks from an old embedding model) would
            # otherwise collide with re-inserted content on the unique content_hash
            removed = session.execute(
                delete(SopChunk).where(
                    SopChunk.document == doc,
                    ~SopChunk.content_hash.in_(list(wanted))
                    | (SopChunk.embedding_model != embedder.model_name),
                )
            ).rowcount
            new = [c for h, c in wanted.items() if h not in existing]
            if new:
                vectors = embedder.embed_documents([f"{c.section}\n{c.content}" for c in new])
                session.add_all(
                    SopChunk(
                        document=c.document,
                        section=c.section,
                        chunk_index=c.chunk_index,
                        content=c.content,
                        content_hash=c.content_hash,
                        embedding_model=embedder.model_name,
                        embedding=v,
                    )
                    for c, v in zip(new, vectors, strict=True)
                )
            stats["documents"] += 1
            stats["added"] += len(new)
            stats["unchanged"] += len(wanted) - len(new)
            stats["removed"] += removed or 0
    log.info("sop.ingested", **stats)
    return stats
