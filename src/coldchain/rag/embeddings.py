"""Embedding providers behind one small protocol.

`HashEmbedder` is a deterministic feature-hashing embedder (token + bigram hashes,
L2-normalised). It is not a semantic model; it exists so the full pipeline, tests
and the retrieval eval run offline and in CI with zero API keys. Hybrid retrieval
(see retriever.py) means lexical matching still carries it.
"""

from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise
from typing import Protocol

from coldchain.config import Settings

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.model_name = f"hash-{dim}"

    def _vec(self, text: str) -> list[float]:
        toks = TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in pairwise(toks)]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.dim = settings.embedding_dim
        self.model_name = settings.embedding_model
        self._impl = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dim,
            api_key=settings.llm_api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._impl.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder(settings.embedding_dim)
    return OpenAIEmbedder(settings)
