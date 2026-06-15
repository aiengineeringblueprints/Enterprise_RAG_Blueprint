import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
INDEX_NAME = os.environ.get("INDEX_NAME", "langchain-test-index")
VECTORDB_DIR = os.environ.get("VECTORDB_DIR", "/data/vectordb")
DISABLE_FILTER = os.environ.get("RETRIEVER_DISABLE_FILTER", "").lower() in (
    "1",
    "true",
    "yes",
)  # defaults to false (filter active)


# Tag-to-bitmask mapping for role-based filtering
# NOTE: These values must stay in sync with loader/tag_management.py!
# The tag system is managed by the loader, the chain only needs this mapping for filtering
TAG_BITMASK_MAPPING = {
    "General": 1 << 0,  # Bit 0: 1
    "Research & Development": 1 << 1,  # Bit 1: 2
    "Marketing & Sales": 1 << 2,  # Bit 2: 4
    "Production & Manufacturing": 1 << 3,  # Bit 3: 8
    "Finance & Controlling": 1 << 4,  # Bit 4: 16
    "Human Resources": 1 << 5,  # Bit 5: 32
    "Legal & Compliance": 1 << 6,  # Bit 6: 64
    "IT & Technology": 1 << 7,  # Bit 7: 128
    "Quality Management": 1 << 8,  # Bit 8: 256
    "Project Documentation": 1 << 9,  # Bit 9: 512
}


def combine_tag_bitmasks(tags: list) -> int:
    """
    Combines multiple tags into a single bitmask for filtering.
    This function is a copy from tag_management.py for service isolation.
    """
    combined_mask = 0
    for tag in tags:
        combined_mask |= TAG_BITMASK_MAPPING.get(tag, 0)
    return combined_mask


def is_admin_user(user_roles: list) -> bool:
    """
    Checks if a user has admin privileges.
    This function is a copy from tag_management.py for service isolation.
    """
    return "admin" in user_roles


def create_retriever(
    index_name: str = None,
    returned_docs: int = 3,
    similarity_threshold: float = 0.5,
    user_roles: list = ["General"],
    existing_embeddings=None,  # New option (better performance?): reuse existing embeddings
    existing_vectorstore=None,  # New option (better performance?): reuse existing vector store
) -> object:
    """
    Creates a retriever with optional role-based filtering.

    Args:
        index_name (str): The name of the collection/index to use
        returned_docs (int): Number of documents to return
        similarity_threshold (float): Minimum similarity score threshold
        user_roles (list): List of roles/categories the user has for filtering
        existing_embeddings: Reuse existing embedding instance
        existing_vectorstore: Reuse existing vector store instance

    Returns:
        object: A configured retriever with optional role-based filtering
    """

    # If an existing vector store is provided, use it directly
    if existing_vectorstore is not None:
        print("🔄 Using existing vector store")

        # Apply filter if necessary
        if DISABLE_FILTER or is_admin_user(user_roles):
            # No filter for admin users
            retriever = existing_vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": returned_docs}
            )
        else:
            # Filter for specific roles
            role_bitmask = combine_tag_bitmasks(user_roles)
            where_filter = {"category_bitmask": role_bitmask}

            retriever = existing_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": returned_docs, "filter": where_filter},
            )

        return retriever
    # Use INDEX_NAME from environment if not provided
    if index_name is None:
        index_name = INDEX_NAME

    # Local Chroma retriever only
    # Normalize roles and numeric parameters
    roles = user_roles or ["General"]
    try:
        k = int(returned_docs)
    except Exception:
        k = 3
    try:
        thr = float(similarity_threshold)
    except Exception:
        thr = 0.5
    # clamp threshold
    thr = 0.0 if thr < 0.0 else (1.0 if thr > 1.0 else thr)

    return init_filtered_vectorstore(
        index_name=index_name,
        user_roles=roles,
        returned_docs=k,
        similarity_threshold=thr,
        existing_embeddings=existing_embeddings,
    )


def init_filtered_vectorstore(
    index_name: str = None,
    user_roles: list = ["General"],
    returned_docs: int = 3,
    similarity_threshold: float = 0.5,
    existing_embeddings=None,  # New option
):
    """Create a vector store retriever with role-based filtering for local Chroma."""
    if index_name is None:
        index_name = INDEX_NAME

    # Ensure vector DB directory exists to avoid backend reader errors
    try:
        os.makedirs(VECTORDB_DIR, exist_ok=True)
    except Exception:
        pass

    # Reuse embeddings or create new ones
    if existing_embeddings is not None:
        embeddings = existing_embeddings
    else:
        embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL
        )

    vectorstore = Chroma(
        collection_name=index_name,
        embedding_function=embeddings,
        persist_directory=VECTORDB_DIR,
    )

    # Optional: disable filtering entirely via environment variable
    if DISABLE_FILTER:
        # IMPORTANT: similarity_score_threshold doesn't work correctly
        # Use normal similarity search with post-processing instead
        # Use similarity_score_threshold if provided (Chroma supports this in search_kwargs)
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": returned_docs,
                # "similarity_score_threshold": similarity_threshold
            },
        )

    # Admin users bypass all filtering
    if is_admin_user(user_roles or ["General"]):
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": returned_docs},
        )

    user_bitmask = combine_tag_bitmasks(user_roles or ["General"])

    # IMPORTANT: For general users (bitmask = 1) the filter is problematic
    # because they should read both category_bitmask: 0 and 1,
    # but Chroma doesn't support OR filters.
    # Solution: General users get no filter (like admins)
    if user_bitmask == 1:  # Only "General" role
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": returned_docs},
        )

    # For specific roles: filter only on exact bitmask matches
    if user_bitmask > 1:  # Specific roles (not just general)
        # Simple filter: only documents with exactly this bitmask
        filter_dict = {"category_bitmask": user_bitmask}
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": filter_dict, "k": returned_docs},
        )

    # Fallback: no filter
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": returned_docs},
    )
