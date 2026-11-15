# Hybrid retrieval

By default the chain retrieves documents by dense vector similarity only (Chroma, embedding model from `EMBEDDING_MODEL`). Dense search finds paraphrases well but often misses exact terms such as error codes, product numbers or acronyms. Hybrid retrieval adds a BM25 keyword search over the same collection and fuses both rankings.

## Configuration

All settings are read per request (`chain/retriever.py`), for the legacy path and the MCP path:

| Variable | Default | Meaning |
|---|---|---|
| `RETRIEVER_SEARCH_TYPE` | `similarity` | `similarity` (dense only, previous behavior) or `hybrid` |
| `RETRIEVER_HYBRID_ALPHA` | `0.5` | Dense weight: `0` = pure BM25, `1` = pure dense |
| `RETRIEVER_HYBRID_FETCH_K` | `4 * RETRIEVER_TOP_K` | Candidates taken from each ranking before fusion |
| `RETRIEVER_TOP_K` | `5` | Documents returned after fusion |

Invalid values raise a `ValueError` instead of silently falling back.

## How it works

`chain/hybrid_retriever.py`:

1. Dense: `similarity_search` for `fetch_k` candidates.
2. Keyword: Okapi BM25 (`k1=1.5`, `b=0.75`, lowercase `\w+` tokens) over all chunks of the collection, `fetch_k` candidates. Chunks without any matching term are dropped.
3. Fusion with weighted Reciprocal Rank Fusion: `score = alpha / (60 + rank_dense) + (1 - alpha) / (60 + rank_bm25)`. Ties resolve deterministically (dense candidates first).

At `alpha = 1` or `0` only the enabled side runs. The algorithm matches the hybrid retrieval of the RAG benchmarking framework, so Blueprint and framework results stay comparable.

The role filter (`category_bitmask`) applies to both sides: BM25 only indexes chunks the user may see.

The BM25 index is built in memory from the Chroma collection on first use and cached per collection and filter. A fingerprint of the chunk IDs is checked on every request, so uploads, deletions and re-indexing rebuild the index automatically. For very large collections the first hybrid request after a change takes longer (all chunks are read from Chroma once).

## Tests

`chain/tests/test_hybrid_retrieval.py` runs against a real on-disk Chroma collection: exact-term matches that dense search misses, role filtering of the keyword side, alpha endpoints, invalid settings, cache invalidation after new uploads, and the RAG chain using the hybrid retriever.
