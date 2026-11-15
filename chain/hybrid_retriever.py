"""Hybrid retrieval: dense similarity search fused with BM25 keyword search.

Both rankings are merged with weighted Reciprocal Rank Fusion (RRF):

    score(doc) = alpha / (RRF_K + rank_dense) + (1 - alpha) / (RRF_K + rank_bm25)

alpha = 1.0 is pure dense search, alpha = 0.0 pure BM25. The algorithm mirrors
the hybrid retrieval of the RAG benchmarking framework so results stay
comparable between the Blueprint and the framework's own pipeline.
"""
import hashlib
import math
import re
import threading
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

RRF_K = 60  # standard RRF dampening constant
_CACHE_SIZE = 4


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Minimal in-memory Okapi BM25 index over Documents (no extra dependency)."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = list(docs)
        self._k1 = k1
        self._b = b
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lens: list[int] = []
        self._inverted: dict[str, list[int]] = {}
        for idx, doc in enumerate(self._docs):
            freqs: dict[str, int] = {}
            tokens = _tokenize(doc.page_content or "")
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self._term_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            for token in freqs:
                self._inverted.setdefault(token, []).append(idx)
        self._avg_len = sum(self._doc_lens) / len(self._docs) if self._docs else 0.0

    def search(self, query: str, k: int) -> list[Document]:
        """Top-k documents by BM25 score, best first; documents without a match are dropped."""
        if not self._docs or k <= 0:
            return []
        n_docs = len(self._docs)
        scores = [0.0] * n_docs
        for token in _tokenize(query):
            postings = self._inverted.get(token)
            if not postings:
                continue
            idf = math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for idx in postings:
                freq = self._term_freqs[idx][token]
                norm = 1.0 - self._b + self._b * self._doc_lens[idx] / (self._avg_len or 1.0)
                scores[idx] += idf * freq * (self._k1 + 1.0) / (freq + self._k1 * norm)
        ranked = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
        return [self._docs[i] for i in ranked[:k] if scores[i] > 0]


_index_cache: dict[tuple, BM25Index] = {}
_cache_lock = threading.Lock()


def _bm25_index(vectorstore: Any, where: Optional[dict]) -> BM25Index:
    """BM25 index over the (filtered) collection, rebuilt only when its chunks change."""
    ids = vectorstore.get(where=where, include=[])["ids"]
    fingerprint = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    key = (id(vectorstore._client), vectorstore._collection.name, repr(where), fingerprint)
    with _cache_lock:
        cached = _index_cache.get(key)
    if cached is not None:
        return cached
    data = vectorstore.get(where=where, include=["documents", "metadatas"])
    docs = [
        Document(id=doc_id, page_content=text or "", metadata=meta or {})
        for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    index = BM25Index(docs)
    with _cache_lock:
        if len(_index_cache) >= _CACHE_SIZE:
            _index_cache.pop(next(iter(_index_cache)))
        _index_cache[key] = index
    return index


def _doc_key(doc: Document) -> tuple:
    if doc.id:
        return ("id", doc.id)
    return ("content", doc.page_content, str((doc.metadata or {}).get("source")))


class HybridRetriever(BaseRetriever):
    """Dense + BM25 retriever over one Chroma collection, honoring the role filter."""

    vectorstore: Any
    k: int = 5
    alpha: float = 0.5
    fetch_k: Optional[int] = None
    where: Optional[dict] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        fetch_k = self.fetch_k or self.k * 4
        dense_docs: list[Document] = []
        if self.alpha > 0.0:
            dense_docs = self.vectorstore.similarity_search(query, k=fetch_k, filter=self.where)
        if self.alpha >= 1.0:
            return dense_docs[: self.k]
        keyword_docs = _bm25_index(self.vectorstore, self.where).search(query, fetch_k)
        if self.alpha <= 0.0:
            return keyword_docs[: self.k]
        if not keyword_docs:
            return dense_docs[: self.k]

        def ranks(docs: list[Document]) -> dict[tuple, float]:
            return {_doc_key(doc): 1.0 / (RRF_K + rank + 1) for rank, doc in enumerate(docs)}

        dense_ranks = ranks(dense_docs)
        keyword_ranks = ranks(keyword_docs)
        doc_by_key = {_doc_key(doc): doc for doc in keyword_docs}
        doc_by_key.update({_doc_key(doc): doc for doc in dense_docs})
        # Dense candidates first, so score ties resolve the same way on every run.
        candidates = list(dense_ranks) + [key for key in keyword_ranks if key not in dense_ranks]
        fused = sorted(
            (
                -(self.alpha * dense_ranks.get(key, 0.0) + (1.0 - self.alpha) * keyword_ranks.get(key, 0.0)),
                position,
                key,
            )
            for position, key in enumerate(candidates)
        )
        return [doc_by_key[key] for _, _, key in fused[: self.k]]
