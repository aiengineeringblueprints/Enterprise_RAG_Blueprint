import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
INDEX_NAME = os.environ.get("INDEX_NAME", "langchain-test-index")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# Shared directory for local vector store persistence (mounted into loader + chain)
VECTORDB_DIR = os.environ.get("VECTORDB_DIR", "/data/vectordb")


def upload_documents_to_vectorstore(
    all_splits: List[Document], index_name: Optional[str] = None
) -> None:
    """Add document chunks to the configured vector store (local Chroma by default)."""
    if index_name is None:
        index_name = INDEX_NAME

    documents: List[Document] = []
    for i in range(len(all_splits)):
        metadata = (
            all_splits[i].metadata.copy()
            if all_splits[i].metadata
            else {"source": "unknown"}
        )
        split = Document(
            page_content=all_splits[i].page_content, metadata=metadata
        )
        documents.append(split)

    # Ensure persistence directory exists (shared volume across containers)
    os.makedirs(VECTORDB_DIR, exist_ok=True)

    print(
        f"Uploading {len(documents)} documents to local Chroma at {VECTORDB_DIR}, collection={index_name}"
    )
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL
    )
    vectorstore = Chroma(
        collection_name=index_name,
        embedding_function=embeddings,
        persist_directory=VECTORDB_DIR,
    )
    vectorstore.add_documents(documents)

    # Persist compatibility across Chroma versions:
    # - chromadb <0.5 used client.persist()
    # - chromadb >=0.5 persists automatically (no persist method on the VectorStore)
    persist_fn = getattr(vectorstore, "persist", None)
    client = getattr(vectorstore, "_client", None)
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            pass
    elif client is not None and callable(getattr(client, "persist", None)):
        try:
            client.persist()  # type: ignore[attr-defined]
        except Exception:
            pass
    print(
        f"Successfully uploaded {len(documents)} documents to local Chroma store."
    )


def get_full_document_content(document_source: str) -> Optional[str]:
    """
    Retrieve the full content of a document from the vectorstore by its source path.
    This function works with the local Chroma vector store.
    """
    try:
        return get_full_document_content_local(document_source)
    except Exception as e:
        print(f"Error retrieving full document: {e}")
        return None


def get_full_document_content_local(document_source: str) -> Optional[str]:
    """Retrieve full document content from local Chroma vectorstore."""
    try:
        if Chroma is None:
            raise RuntimeError("Chroma not installed")
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL, model="EMBEDDING_MODEL"
        )
        vectorstore = Chroma(
            collection_name=INDEX_NAME,
            embedding_function=embeddings,
            persist_directory=VECTORDB_DIR,
        )

        # Use underlying Chroma collection to fetch all entries with matching source
        where_filter = {"source": {"$eq": document_source}}
        data = vectorstore._collection.get(
            where=where_filter, include=["documents", "metadatas"]
        )  # type: ignore
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        if not docs:
            return None

        # Build chunks with ordering if available
        chunks = []
        for content, meta in zip(docs, metas):
            page = (meta or {}).get("page", 0)
            chunk_idx = (meta or {}).get("chunk_index", 0)
            chunks.append(
                {"content": content, "page": page, "chunk_index": chunk_idx}
            )

        chunks.sort(key=lambda x: (x["page"], x["chunk_index"]))
        return "\n".join([c["content"] for c in chunks if c.get("content")])
    except Exception as e:
        print(f"Error retrieving from local vectorstore: {e}")
        return None


def list_document_sources() -> list:
    """
    Debug function to list all unique document sources in the vectorstore.
    """
    try:
        return list_document_sources_local()
    except Exception as e:
        print(f"Error listing document sources: {e}")
        return []


def list_document_sources_local() -> list:
    """List document sources from local Chroma vectorstore."""
    try:
        if Chroma is None:
            return []
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL
        )
        vectorstore = Chroma(
            collection_name=INDEX_NAME,
            embedding_function=embeddings,
            persist_directory=VECTORDB_DIR,
        )
        # Get a sample of all metadatas to extract unique sources
        data = vectorstore._collection.get(include=["metadatas"], limit=1000)  # type: ignore
        metas = data.get("metadatas", []) or []
        sources = set()
        for m in metas:
            if isinstance(m, dict):
                sources.add(m.get("source", "unknown"))
        return list(sources)
    except Exception as e:
        print(f"Error listing from local vectorstore: {e}")
        return []


def debug_get_local_content_for_source(document_source: str) -> dict:
    """Return debug information for a given source from local Chroma."""
    try:
        if Chroma is None:
            return {"error": "Chroma not installed"}
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL
        )
        vectorstore = Chroma(
            collection_name=INDEX_NAME,
            embedding_function=embeddings,
            persist_directory=VECTORDB_DIR,
        )
        where_filter = {"source": {"$eq": document_source}}
        data = vectorstore._collection.get(
            where=where_filter, include=["documents", "metadatas", "ids"]
        )  # type: ignore
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        ids = data.get("ids", []) or []
        samples = []
        for i, (doc, meta) in enumerate(zip(docs[:3], metas[:3])):
            samples.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "metadata_keys": list((meta or {}).keys()),
                    "source": (meta or {}).get("source", "NO SOURCE"),
                    "page_content_preview": (doc or "")[:200]
                    + ("..." if doc and len(doc) > 200 else ""),
                    "all_metadata": meta,
                }
            )
        return {
            "searched_source": document_source,
            "matches_found": len(docs),
            "sample_matches": samples,
        }
    except Exception as e:
        return {"error": str(e)}
