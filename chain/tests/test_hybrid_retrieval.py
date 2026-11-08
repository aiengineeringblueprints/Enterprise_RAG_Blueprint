"""Hybrid retrieval (dense + BM25) through the public create_retriever factory.

Uses a real on-disk Chroma collection with a tiny keyword-based embedding so
the dense ranking is known and differs from the BM25 ranking.
"""
import os
import sys

import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retriever as retriever_module
from retriever import TAG_BITMASK_MAPPING, create_retriever

INDEX = "hybrid-test-index"

GUIDE = "General troubleshooting guide for printers"
ERROR_CODE = "Error code XJ-4471 means the fuser unit failed"
DRIVER = "Printer driver installation steps"
QUESTION = "what does XJ-4471 mean on my printer"


class PrinterEmbeddings(Embeddings):
    """Puts every text mentioning 'printer' close together, everything else far away."""

    def _embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if "printer" in text.lower() else [0.0, 1.0]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    from langchain_chroma import Chroma

    monkeypatch.setattr(retriever_module, "VECTORDB_DIR", str(tmp_path))
    embeddings = PrinterEmbeddings()
    store = Chroma(
        collection_name=INDEX,
        embedding_function=embeddings,
        persist_directory=str(tmp_path),
    )
    store.add_texts(
        [GUIDE, ERROR_CODE, DRIVER],
        metadatas=[{"source": "guide"}, {"source": "error"}, {"source": "driver"}],
    )
    return embeddings


def _contents(retriever, query=QUESTION):
    return {doc.page_content for doc in retriever.invoke(query)}


def test_hybrid_finds_exact_term_match_that_dense_search_misses(corpus, monkeypatch):
    monkeypatch.delenv("RETRIEVER_SEARCH_TYPE", raising=False)
    dense = create_retriever(index_name=INDEX, returned_docs=2, existing_embeddings=corpus)
    assert ERROR_CODE not in _contents(dense)

    monkeypatch.setenv("RETRIEVER_SEARCH_TYPE", "hybrid")
    hybrid = create_retriever(index_name=INDEX, returned_docs=2, existing_embeddings=corpus)
    assert ERROR_CODE in _contents(hybrid)


def test_hybrid_keyword_search_respects_role_filter(tmp_path, monkeypatch):
    from langchain_chroma import Chroma

    monkeypatch.setattr(retriever_module, "VECTORDB_DIR", str(tmp_path))
    monkeypatch.setenv("RETRIEVER_SEARCH_TYPE", "hybrid")
    embeddings = PrinterEmbeddings()
    store = Chroma(collection_name=INDEX, embedding_function=embeddings, persist_directory=str(tmp_path))
    it_mask = TAG_BITMASK_MAPPING["IT & Technology"]
    legal_mask = TAG_BITMASK_MAPPING["Legal & Compliance"]
    store.add_texts(
        [GUIDE, ERROR_CODE, DRIVER],
        metadatas=[{"category_bitmask": it_mask}, {"category_bitmask": legal_mask}, {"category_bitmask": it_mask}],
    )

    hybrid = create_retriever(
        index_name=INDEX, returned_docs=3, user_roles=["IT & Technology"], existing_embeddings=embeddings
    )

    assert _contents(hybrid) == {GUIDE, DRIVER}


def test_hybrid_sees_chunks_added_after_earlier_queries(corpus, tmp_path, monkeypatch):
    from langchain_chroma import Chroma

    monkeypatch.setenv("RETRIEVER_SEARCH_TYPE", "hybrid")
    monkeypatch.setenv("RETRIEVER_HYBRID_ALPHA", "0")
    hybrid = create_retriever(index_name=INDEX, returned_docs=1, existing_embeddings=corpus)
    assert _contents(hybrid, "toner cartridge ZQ-9") == set()

    store = Chroma(collection_name=INDEX, embedding_function=corpus, persist_directory=str(tmp_path))
    store.add_texts(["Replace the toner cartridge ZQ-9 every 5000 pages"])

    assert _contents(hybrid, "toner cartridge ZQ-9") == {"Replace the toner cartridge ZQ-9 every 5000 pages"}


def test_alpha_one_is_pure_dense_search(corpus, monkeypatch):
    monkeypatch.delenv("RETRIEVER_SEARCH_TYPE", raising=False)
    dense = create_retriever(index_name=INDEX, returned_docs=2, existing_embeddings=corpus)
    monkeypatch.setenv("RETRIEVER_SEARCH_TYPE", "hybrid")
    monkeypatch.setenv("RETRIEVER_HYBRID_ALPHA", "1")
    hybrid = create_retriever(index_name=INDEX, returned_docs=2, existing_embeddings=corpus)

    assert [d.page_content for d in hybrid.invoke(QUESTION)] == [d.page_content for d in dense.invoke(QUESTION)]


def test_alpha_zero_is_pure_keyword_search(corpus, monkeypatch):
    monkeypatch.setenv("RETRIEVER_SEARCH_TYPE", "hybrid")
    monkeypatch.setenv("RETRIEVER_HYBRID_ALPHA", "0")
    hybrid = create_retriever(index_name=INDEX, returned_docs=3, existing_embeddings=corpus)

    # Only the error-code and driver chunks share terms with the question.
    assert [d.page_content for d in hybrid.invoke(QUESTION)] == [ERROR_CODE, DRIVER]


@pytest.mark.parametrize(
    "env",
    [
        {"RETRIEVER_SEARCH_TYPE": "mmr"},
        {"RETRIEVER_SEARCH_TYPE": "hybrid", "RETRIEVER_HYBRID_ALPHA": "1.5"},
        {"RETRIEVER_SEARCH_TYPE": "hybrid", "RETRIEVER_HYBRID_ALPHA": "abc"},
        {"RETRIEVER_SEARCH_TYPE": "hybrid", "RETRIEVER_HYBRID_FETCH_K": "0"},
    ],
)
def test_invalid_search_settings_fail_clearly(corpus, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="RETRIEVER_"):
        create_retriever(index_name=INDEX, returned_docs=2, existing_embeddings=corpus)


def test_rag_chain_answers_from_hybrid_context(corpus, monkeypatch):
    from langchain_core.runnables import RunnableLambda
    from load_chain import load_chain
    from prompts.promt_manager import PromptKey

    monkeypatch.setenv("RETRIEVER_SEARCH_TYPE", "hybrid")
    hybrid = create_retriever(index_name=INDEX, returned_docs=2, existing_embeddings=corpus)
    prompts = []
    model = RunnableLambda(lambda prompt: prompts.append(prompt.to_string()) or "answer")

    chain = load_chain(hybrid, PromptKey.SOURCE, model)

    assert chain.invoke(QUESTION) == "answer"
    assert ERROR_CODE in prompts[0]
