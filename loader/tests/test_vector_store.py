
import pytest
import os
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from loader.vector_store import (
    upload_documents_to_vectorstore,
    get_full_document_content,
    get_full_document_content_local,
    list_document_sources,
    list_document_sources_local,
    debug_get_local_content_for_source,
)

class TestUploadDocumentsToVectorstore:

    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_documents_basic(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_vectorstore = MagicMock()
        mock_chroma_class.return_value = mock_vectorstore
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings
        
        docs = [
            Document(page_content="Content 1", metadata={"source": "test1.txt"}),
            Document(page_content="Content 2", metadata={"source": "test2.txt"})
        ]
        
        upload_documents_to_vectorstore(docs)
        
        mock_makedirs.assert_called_once()
        mock_vectorstore.add_documents.assert_called_once()
        
        call_args = mock_vectorstore.add_documents.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].page_content == "Content 1"
        assert call_args[1].page_content == "Content 2"
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_documents_with_custom_index(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_vectorstore = MagicMock()
        mock_chroma_class.return_value = mock_vectorstore
        
        docs = [Document(page_content="Test", metadata={"source": "test.txt"})]
        
        upload_documents_to_vectorstore(docs, index_name="custom-index")
        
        assert mock_chroma_class.call_args[1]["collection_name"] == "custom-index"
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_documents_with_metadata(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_vectorstore = MagicMock()
        mock_chroma_class.return_value = mock_vectorstore
        
        docs = [
            Document(
                page_content="Test content",
                metadata={
                    "source": "test.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "tag_bitmask": 3
                }
            )
        ]
        
        upload_documents_to_vectorstore(docs)
        
        call_args = mock_vectorstore.add_documents.call_args[0][0]
        assert call_args[0].metadata["source"] == "test.pdf"
        assert call_args[0].metadata["page"] == 1
        assert call_args[0].metadata["chunk_index"] == 0
        assert call_args[0].metadata["tag_bitmask"] == 3
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_documents_without_metadata(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_vectorstore = MagicMock()
        mock_chroma_class.return_value = mock_vectorstore
        
        docs = [Document(page_content="Content without metadata")]
        
        upload_documents_to_vectorstore(docs)
        
        call_args = mock_vectorstore.add_documents.call_args[0][0]
        assert "source" in call_args[0].metadata
        assert call_args[0].metadata["source"] == "unknown"
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_documents_with_persist_method(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_vectorstore = MagicMock()
        mock_persist = MagicMock()
        mock_vectorstore.persist = mock_persist
        mock_chroma_class.return_value = mock_vectorstore
        
        docs = [Document(page_content="Test", metadata={"source": "test.txt"})]
        
        upload_documents_to_vectorstore(docs)
        
        mock_persist.assert_called_once()
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_documents_with_client_persist(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_client = MagicMock()
        mock_client.persist = MagicMock()
        mock_vectorstore = MagicMock()
        mock_vectorstore._client = mock_client
        mock_vectorstore.persist = None
        mock_chroma_class.return_value = mock_vectorstore
        
        docs = [Document(page_content="Test", metadata={"source": "test.txt"})]
        
        upload_documents_to_vectorstore(docs)
        
        mock_client.persist.assert_called_once()
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_empty_documents_list(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_vectorstore = MagicMock()
        mock_chroma_class.return_value = mock_vectorstore
        
        upload_documents_to_vectorstore([])
        
        mock_vectorstore.add_documents.assert_called_once_with([])

class TestGetFullDocumentContent:

    @patch('loader.vector_store.get_full_document_content_local')
    def test_get_full_document_content_success(self, mock_local_func):

        mock_local_func.return_value = "Full document content"
        
        result = get_full_document_content("test.pdf")
        
        assert result == "Full document content"
        mock_local_func.assert_called_once_with("test.pdf")
    
    @patch('loader.vector_store.get_full_document_content_local')
    def test_get_full_document_content_error(self, mock_local_func):

        mock_local_func.side_effect = Exception("Test error")
        
        result = get_full_document_content("test.pdf")
        
        assert result is None
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_get_full_document_content_local_success(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Content chunk 1", "Content chunk 2", "Content chunk 3"],
            "metadatas": [
                {"source": "test.pdf", "page": 1, "chunk_index": 0},
                {"source": "test.pdf", "page": 1, "chunk_index": 1},
                {"source": "test.pdf", "page": 2, "chunk_index": 0}
            ]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = get_full_document_content_local("test.pdf")
        
        assert result == "Content chunk 1\nContent chunk 2\nContent chunk 3"
        mock_collection.get.assert_called_once()
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_get_full_document_content_local_not_found(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = get_full_document_content_local("nonexistent.pdf")
        
        assert result is None
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_get_full_document_content_local_sorts_chunks(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Chunk C", "Chunk A", "Chunk B"],
            "metadatas": [
                {"source": "test.pdf", "page": 2, "chunk_index": 0},
                {"source": "test.pdf", "page": 1, "chunk_index": 0},
                {"source": "test.pdf", "page": 1, "chunk_index": 1}
            ]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = get_full_document_content_local("test.pdf")
        
        assert result == "Chunk A\nChunk B\nChunk C"
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_get_full_document_content_local_missing_metadata(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Chunk 1", "Chunk 2"],
            "metadatas": [
                {"source": "test.pdf"},
                {"source": "test.pdf", "page": 1}
            ]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = get_full_document_content_local("test.pdf")
        
        assert result is not None
        assert "Chunk 1" in result
        assert "Chunk 2" in result
    
    @patch('loader.vector_store.Chroma', None)
    def test_get_full_document_content_local_chroma_not_installed(self):

        result = get_full_document_content_local("test.pdf")
        
        assert result is None

class TestDebugListDocumentSources:

    @patch('loader.vector_store.list_document_sources_local')
    def test_list_document_sources_success(self, mock_local_func):

        mock_local_func.return_value = ["doc1.pdf", "doc2.txt", "doc3.md"]
        
        result = list_document_sources()
        
        assert result == ["doc1.pdf", "doc2.txt", "doc3.md"]
        mock_local_func.assert_called_once()
    
    @patch('loader.vector_store.list_document_sources_local')
    def test_list_document_sources_error(self, mock_local_func):

        mock_local_func.side_effect = Exception("Test error")
        
        result = list_document_sources()
        
        assert result == []
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_list_document_sources_local_success(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "doc1.pdf"},
                {"source": "doc2.txt"},
                {"source": "doc1.pdf"},
                {"source": "doc3.md"}
            ]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = list_document_sources_local()
        
        assert len(result) == 3
        assert "doc1.pdf" in result
        assert "doc2.txt" in result
        assert "doc3.md" in result
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_list_document_sources_local_empty(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"metadatas": []}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = list_document_sources_local()
        
        assert result == []
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_list_document_sources_local_with_unknown(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "doc1.pdf"},
                {},
                {"other_field": "value"}
            ]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = list_document_sources_local()
        
        assert "doc1.pdf" in result
        assert "unknown" in result
    
    @patch('loader.vector_store.Chroma', None)
    def test_list_document_sources_local_chroma_not_installed(self):

        result = list_document_sources_local()
        
        assert result == []

class TestDebugGetLocalContentForSource:

    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_debug_get_local_content_success(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Content 1", "Content 2", "Content 3", "Content 4"],
            "metadatas": [
                {"source": "test.pdf", "page": 1},
                {"source": "test.pdf", "page": 2},
                {"source": "test.pdf", "page": 3},
                {"source": "test.pdf", "page": 4}
            ],
            "ids": ["id1", "id2", "id3", "id4"]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = debug_get_local_content_for_source("test.pdf")
        
        assert result["searched_source"] == "test.pdf"
        assert result["matches_found"] == 4
        assert len(result["sample_matches"]) == 3
        assert result["sample_matches"][0]["id"] == "id1"
        assert result["sample_matches"][0]["source"] == "test.pdf"
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_debug_get_local_content_no_matches(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": [],
            "metadatas": [],
            "ids": []
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = debug_get_local_content_for_source("nonexistent.pdf")
        
        assert result["searched_source"] == "nonexistent.pdf"
        assert result["matches_found"] == 0
        assert result["sample_matches"] == []
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_debug_get_local_content_long_preview(self, mock_embeddings_class, mock_chroma_class):

        long_content = "A" * 300
        
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": [long_content],
            "metadatas": [{"source": "test.pdf"}],
            "ids": ["id1"]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = debug_get_local_content_for_source("test.pdf")
        
        preview = result["sample_matches"][0]["page_content_preview"]
        assert len(preview) == 203
        assert preview.endswith("...")
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_debug_get_local_content_with_metadata(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Content"],
            "metadatas": [{
                "source": "test.pdf",
                "page": 1,
                "chunk_index": 0,
                "custom_field": "custom_value"
            }],
            "ids": ["id1"]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        result = debug_get_local_content_for_source("test.pdf")
        
        sample = result["sample_matches"][0]
        assert "page" in sample["metadata_keys"]
        assert "chunk_index" in sample["metadata_keys"]
        assert "custom_field" in sample["metadata_keys"]
        assert sample["all_metadata"]["custom_field"] == "custom_value"
    
    @patch('loader.vector_store.Chroma', None)
    def test_debug_get_local_content_chroma_not_installed(self):

        result = debug_get_local_content_for_source("test.pdf")
        
        assert "error" in result
        assert result["error"] == "Chroma not installed"
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_debug_get_local_content_exception(self, mock_embeddings_class, mock_chroma_class):

        mock_chroma_class.side_effect = Exception("Connection error")
        
        result = debug_get_local_content_for_source("test.pdf")
        
        assert "error" in result
        assert "Connection error" in result["error"]

class TestEnvironmentConfiguration:

    @patch.dict(os.environ, {
        'EMBEDDING_SOURCE': 'ollama',
        'OLLAMA_BASE_URL': 'http://test-ollama:11434',
        'INDEX_NAME': 'test-index',
        'VECTORDB_DIR': '/test/vectordb'
    })
    def test_environment_variables_loaded(self):

        import importlib
        import loader.vector_store
        importlib.reload(loader.vector_store)
        
        assert loader.vector_store.EMBEDDING_SOURCE == 'ollama'
        assert loader.vector_store.OLLAMA_BASE_URL == 'http://test-ollama:11434'
        assert loader.vector_store.INDEX_NAME == 'test-index'
        assert loader.vector_store.VECTORDB_DIR == '/test/vectordb'

class TestIntegration:

    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    @patch('loader.vector_store.os.makedirs')
    def test_upload_and_retrieve_workflow(self, mock_makedirs, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        stored_docs = []
        stored_metas = []
        
        def mock_add_documents(docs):
            for doc in docs:
                stored_docs.append(doc.page_content)
                stored_metas.append(doc.metadata)
        
        def mock_get(where=None, include=None, limit=None):
            if where and "source" in where:
                source = where["source"]["$eq"]
                filtered_docs = []
                filtered_metas = []
                for doc, meta in zip(stored_docs, stored_metas):
                    if meta.get("source") == source:
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                return {"documents": filtered_docs, "metadatas": filtered_metas}
            return {"documents": stored_docs, "metadatas": stored_metas}
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.add_documents = mock_add_documents
        mock_collection.get = mock_get
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        docs = [
            Document(page_content="First chunk", metadata={"source": "test.pdf", "page": 1, "chunk_index": 0}),
            Document(page_content="Second chunk", metadata={"source": "test.pdf", "page": 1, "chunk_index": 1}),
            Document(page_content="Third chunk", metadata={"source": "test.pdf", "page": 2, "chunk_index": 0})
        ]
        
        upload_documents_to_vectorstore(docs)
        
        assert len(stored_docs) == 3
        
        result = get_full_document_content_local("test.pdf")
        
        assert result == "First chunk\nSecond chunk\nThird chunk"
    
    @patch('loader.vector_store.Chroma')
    @patch('loader.vector_store.OllamaEmbeddings')
    def test_list_and_debug_workflow(self, mock_embeddings_class, mock_chroma_class):

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Doc1 content", "Doc2 content"],
            "metadatas": [
                {"source": "doc1.pdf", "page": 1},
                {"source": "doc2.txt", "page": 1}
            ],
            "ids": ["id1", "id2"]
        }
        
        mock_vectorstore = MagicMock()
        mock_vectorstore._collection = mock_collection
        mock_chroma_class.return_value = mock_vectorstore
        
        sources = list_document_sources_local()
        assert len(sources) == 2
        assert "doc1.pdf" in sources
        
        debug_info = debug_get_local_content_for_source("doc1.pdf")
        assert debug_info["searched_source"] == "doc1.pdf"
