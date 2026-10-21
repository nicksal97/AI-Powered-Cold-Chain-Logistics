"""Hybrid retrieval over SOP chunks: pgvector cosine + Postgres full-text, fused with
Reciprocal Rank Fusion in a single SQL round trip.

Why hybrid: SOP questions mix semantics ("what do we do when the reefer fails")
with exact tokens that embeddings handle poorly ("PRT_CNG_LVL", "0.65", "Tier 2").
RRF needs no score calibration between the two rankers, which is why it is the
usual default for this.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine, text

from coldchain.rag.embeddings import Embedder

RRF_K = 60

HYBRID_SQL = text(
    """
    WITH vec AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:qvec AS vector)) AS rnk
        FROM sop_chunks
        ORDER BY embedding <=> CAST(:qvec AS vector)
        LIMIT :n
    ),
    lex AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rnk
        FROM sop_chunks, websearch_to_tsquery('english', :qtext) AS q
        WHERE tsv @@ q
        ORDER BY ts_rank_cd(tsv, q) DESC
        LIMIT :n
    ),
    fused AS (
        SELECT COALESCE(vec.id, lex.id) AS id,
               COALESCE(1.0 / (:rrf + vec.rnk), 0) + COALESCE(1.0 / (:rrf + lex.rnk), 0) AS score
        FROM vec FULL OUTER JOIN lex ON vec.id = lex.id
    )
    SELECT c.id, c.document, c.section, c.content, f.score
    FROM fused f JOIN sop_chunks c ON c.id = f.id
    ORDER BY f.score DESC
    LIMIT :k
    """
)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: int
    document: str
    section: str
    content: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.document} > {self.section}"


class HybridRetriever:
    def __init__(self, engine: Engine, embedder: Embedder, k: int = 4, candidates: int = 20):
        self.engine, self.embedder, self.k, self.candidates = engine, embedder, k, candidates

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        qvec = self.embedder.embed_query(query)
        with self.engine.connect() as conn:
            rows = conn.execute(
                HYBRID_SQL,
                {
                    "qvec": str(qvec),
                    "qtext": query,
                    "n": self.candidates,
                    "k": k or self.k,
                    "rrf": RRF_K,
                },
            ).all()
        return [
            RetrievedChunk(r.id, r.document, r.section, r.content, float(r.score)) for r in rows
        ]
