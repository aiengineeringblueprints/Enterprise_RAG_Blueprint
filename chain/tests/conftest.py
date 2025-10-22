"""
Shared fixtures and mocks for chain service unit tests.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def mock_ollama_embeddings():
    """Mock OllamaEmbeddings for testing."""
    mock = Mock()
    mock.embed_query.return_value = [0.1] * 768  # Typical embedding dimension
    mock.embed_documents.return_value = [[0.1] * 768]
    return mock


@pytest.fixture
def mock_vectorstore():
    """Mock Chroma vectorstore for testing."""
    mock = Mock()
    
    # Mock similarity search results
    mock_docs = [
        Document(
            page_content="Test document content 1",
            metadata={
                "source": "test_doc1.pdf",
                "page": 1,
                "category_bitmask": 1
            }
        ),
        Document(
            page_content="Test document content 2",
            metadata={
                "source": "test_doc2.pdf",
                "page": 2,
                "category_bitmask": 1
            }
        )
    ]
    
    mock.similarity_search.return_value = mock_docs
    mock.as_retriever.return_value = Mock(invoke=Mock(return_value=mock_docs))
    
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM (ChatOpenAI) for testing."""
    mock = Mock()
    mock.invoke.return_value = Mock(content="This is a test response from the LLM.")
    return mock


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={
                "source": "python_guide.pdf",
                "page": 1,
                "category_bitmask": 1,
                "chunk_index": 0
            }
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={
                "source": "ml_basics.pdf",
                "page": 5,
                "category_bitmask": 2,
                "chunk_index": 1
            }
        ),
        Document(
            page_content="RAG systems combine retrieval and generation.",
            metadata={
                "source": "rag_overview.pdf",
                "page": 3,
                "category_bitmask": 1,
                "chunk_index": 0
            }
        )
    ]


@pytest.fixture
def mock_prompt_manager():
    """Mock PromptManager for testing."""
    mock = Mock()
    mock.load_template.return_value = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    return mock


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("EMBEDDING_SOURCE", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("INDEX_NAME", "test-index")
    monkeypatch.setenv("VECTORDB_DIR", "/tmp/test_vectordb")
    monkeypatch.setenv("RETRIEVER_DISABLE_FILTER", "false")
    monkeypatch.setenv("LLM_SOURCE_ANSWER", "openai")
    monkeypatch.setenv("LLM_SOURCE_CHECK", "openai")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("RETRIEVER_SIMILARITY_THRESHOLD", "0.5")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11435/v1")


@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain for testing."""
    mock_chain = Mock()
    mock_chain.invoke.return_value = {
        "answer": "Test answer",
        "context": [
            Document(page_content="Context 1", metadata={"source": "doc1.pdf"}),
            Document(page_content="Context 2", metadata={"source": "doc2.pdf"})
        ]
    }
    return mock_chain


class MockChainComponent(Mock):
    """
    Mock that supports the pipe (|) operator for langchain components.
    """
    def __or__(self, other):
        """Support pipe operator for chaining."""
        return MockChainComponent()
    
    def __ror__(self, other):
        """Support reverse pipe operator."""
        return MockChainComponent()
