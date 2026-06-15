import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from loader_functions import (
    MultiEncodingLoader,
    load_docs_from_dict,
    upload_all_docs_to_vector_store,
)


class TestMultiEncodingLoader:
    def test_load_utf8_file(self, temp_test_dir):

        file_path = temp_test_dir / "utf8.txt"
        file_path.write_text(
            "UTF-8 content with special chars: äöü", encoding="utf-8"
        )

        loader = MultiEncodingLoader(str(file_path))
        docs = loader.load()

        assert len(docs) == 1
        assert "UTF-8 content" in docs[0].page_content
        assert docs[0].metadata["source"] == str(file_path)

    def test_load_latin1_file(self, temp_test_dir):

        file_path = temp_test_dir / "latin1.txt"
        with open(file_path, "w", encoding="latin-1") as f:
            f.write("Latin-1 content: café")

        loader = MultiEncodingLoader(str(file_path))
        docs = loader.load()

        assert len(docs) == 1
        assert "Latin-1 content" in docs[0].page_content

    def test_load_with_custom_encodings(self, temp_test_dir):

        file_path = temp_test_dir / "custom.txt"
        file_path.write_text("Test content", encoding="utf-8")

        loader = MultiEncodingLoader(str(file_path), encodings=["utf-8"])
        docs = loader.load()

        assert len(docs) == 1

    def test_load_nonexistent_file(self):

        loader = MultiEncodingLoader("nonexistent.txt")

        with pytest.raises((FileNotFoundError, RuntimeError)):
            loader.load()

    def test_load_with_all_encodings_fail(self, temp_test_dir):

        file_path = temp_test_dir / "binary.txt"
        file_path.write_bytes(b"\x80\x81\x82\x83")

        loader = MultiEncodingLoader(str(file_path), encodings=["ascii"])

        with pytest.raises(RuntimeError) as exc_info:
            loader.load()
        assert "Error loading" in str(exc_info.value)

    def test_lazy_load(self, temp_test_dir):

        file_path = temp_test_dir / "lazy.txt"
        file_path.write_text("Lazy load content")

        loader = MultiEncodingLoader(str(file_path))
        docs = list(loader.lazy_load())

        assert len(docs) == 1
        assert docs[0].page_content == "Lazy load content"


class TestLoadDocsFromDict:
    @patch("loader_functions.UnstructuredWordDocumentLoader")
    @patch("loader_functions.PyPDFLoader")
    @patch("loader_functions.MultiEncodingLoader")
    @patch("loader_functions.UnstructuredMarkdownLoader")
    def test_load_mixed_file_types(
        self,
        mock_md_loader,
        mock_txt_loader,
        mock_pdf_loader,
        mock_docx_loader,
        temp_test_dir,
    ):

        (temp_test_dir / "test.txt").write_text("Text content")
        (temp_test_dir / "test.pdf").write_bytes(b"PDF")
        (temp_test_dir / "test.docx").write_bytes(b"DOCX")
        (temp_test_dir / "test.md").write_text("# Markdown")

        mock_txt_loader.return_value.load.return_value = [
            Document(page_content="Text", metadata={"source": "test.txt"})
        ]
        mock_pdf_loader.return_value.load.return_value = [
            Document(page_content="PDF", metadata={"source": "test.pdf"})
        ]
        mock_docx_loader.return_value.load.return_value = [
            Document(page_content="DOCX", metadata={"source": "test.docx"})
        ]
        mock_md_loader.return_value.load.return_value = [
            Document(page_content="MD", metadata={"source": "test.md"})
        ]

        docs = load_docs_from_dict(str(temp_test_dir), document_tag="General")

        assert len(docs) == 4
        assert all(doc.metadata.get("category") == "General" for doc in docs)
        assert all(doc.metadata.get("category_bitmask") == 1 for doc in docs)

    def test_load_with_tag(self, temp_test_dir):

        (temp_test_dir / "test.txt").write_text("Test content")

        docs = load_docs_from_dict(
            str(temp_test_dir), document_tag="IT & Technology"
        )

        assert len(docs) > 0
        for doc in docs:
            assert doc.metadata["category"] == "IT & Technology"
            assert doc.metadata["category_bitmask"] == 128

    def test_load_without_tag(self, temp_test_dir):

        (temp_test_dir / "test.txt").write_text("Test content")

        docs = load_docs_from_dict(str(temp_test_dir), document_tag=None)

        assert len(docs) > 0

    def test_load_empty_directory(self, temp_test_dir):

        docs = load_docs_from_dict(str(temp_test_dir))

        assert docs == []

    def test_load_unsupported_file_type(self, temp_test_dir):

        (temp_test_dir / "test.exe").write_bytes(b"EXE")
        (temp_test_dir / "test.jpg").write_bytes(b"JPG")

        docs = load_docs_from_dict(str(temp_test_dir))

        assert docs == []

    def test_load_with_subdirectories(self, temp_test_dir):

        subdir = temp_test_dir / "subdir"
        subdir.mkdir()
        (subdir / "test.txt").write_text("Subdir content")
        (temp_test_dir / "root.txt").write_text("Root content")

        docs = load_docs_from_dict(str(temp_test_dir))

        assert len(docs) == 2

    @patch("loader_functions.PyPDFLoader")
    def test_load_with_error_handling(self, mock_pdf_loader, temp_test_dir):

        (temp_test_dir / "test.pdf").write_bytes(b"PDF")

        mock_pdf_loader.return_value.load.side_effect = Exception(
            "Loading failed"
        )

        docs = load_docs_from_dict(str(temp_test_dir))

        assert docs == []

    def test_load_csv_file(self, temp_test_dir):

        csv_path = temp_test_dir / "test.csv"
        csv_path.write_text("name,age\nJohn,30\nJane,25")

        docs = load_docs_from_dict(str(temp_test_dir))

        assert len(docs) == 1
        assert "John" in docs[0].page_content


class TestUploadAllDocsToVectorStore:
    @patch("loader_functions.upload_documents_to_vectorstore")
    @patch("loader_functions.load_docs_from_dict")
    @patch("loader_functions.RecursiveCharacterTextSplitter")
    def test_upload_with_default_params(
        self, mock_splitter_class, mock_load_docs, mock_upload, mock_documents
    ):

        mock_load_docs.return_value = mock_documents
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = mock_documents
        mock_splitter_class.return_value = mock_splitter

        upload_all_docs_to_vector_store(root_path="test_path")

        mock_load_docs.assert_called_once_with("test_path", document_tag=None)
        mock_splitter_class.assert_called_once_with(
            chunk_size=1000, chunk_overlap=200
        )
        mock_splitter.split_documents.assert_called_once_with(mock_documents)
        mock_upload.assert_called_once()

    @patch("loader_functions.upload_documents_to_vectorstore")
    @patch("loader_functions.load_docs_from_dict")
    @patch("loader_functions.RecursiveCharacterTextSplitter")
    def test_upload_with_custom_chunk_params(
        self, mock_splitter_class, mock_load_docs, mock_upload, mock_documents
    ):

        mock_load_docs.return_value = mock_documents
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = mock_documents
        mock_splitter_class.return_value = mock_splitter

        upload_all_docs_to_vector_store(
            root_path="test_path", chunk_size=1000, chunk_overlap=100
        )

        mock_splitter_class.assert_called_once_with(
            chunk_size=1000, chunk_overlap=100
        )

    @patch("loader_functions.upload_documents_to_vectorstore")
    @patch("loader_functions.load_docs_from_dict")
    def test_upload_with_tag(
        self, mock_load_docs, mock_upload, mock_documents
    ):

        mock_load_docs.return_value = mock_documents

        upload_all_docs_to_vector_store(
            root_path="test_path", document_tag="Finance & Controlling"
        )

        mock_load_docs.assert_called_once_with(
            "test_path", document_tag="Finance & Controlling"
        )

    @patch("loader_functions.upload_documents_to_vectorstore")
    @patch("loader_functions.load_docs_from_dict")
    def test_upload_no_documents(self, mock_load_docs, mock_upload):

        mock_load_docs.return_value = []

        upload_all_docs_to_vector_store(root_path="empty_path")

        mock_upload.assert_not_called()

    @patch("loader_functions.upload_documents_to_vectorstore")
    @patch("loader_functions.load_docs_from_dict")
    @patch("loader_functions.RecursiveCharacterTextSplitter")
    def test_upload_text_splitting(
        self, mock_splitter_class, mock_load_docs, mock_upload
    ):

        large_doc = Document(
            page_content="x" * 5000, metadata={"source": "large.txt"}
        )
        mock_load_docs.return_value = [large_doc]

        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [
            Document(
                page_content="x" * 2000, metadata={"source": "large.txt"}
            ),
            Document(
                page_content="x" * 2000, metadata={"source": "large.txt"}
            ),
            Document(
                page_content="x" * 1000, metadata={"source": "large.txt"}
            ),
        ]
        mock_splitter_class.return_value = mock_splitter

        upload_all_docs_to_vector_store(root_path="test_path")

        mock_splitter.split_documents.assert_called_once_with([large_doc])

        call_args = mock_upload.call_args
        assert len(call_args[1]["all_splits"]) == 3


class TestIntegration:
    @patch("loader_functions.upload_documents_to_vectorstore")
    def test_end_to_end_document_loading(self, mock_upload, temp_test_dir):

        (temp_test_dir / "doc1.txt").write_text("First document content")
        (temp_test_dir / "doc2.txt").write_text("Second document content")

        upload_all_docs_to_vector_store(
            root_path=str(temp_test_dir),
            document_tag="General",
            chunk_size=50,
            chunk_overlap=10,
        )

        assert mock_upload.called

        call_args = mock_upload.call_args
        splits = call_args[1]["all_splits"]
        assert len(splits) > 0

    def test_multiencoding_loader_integration(self, temp_test_dir):

        file_path = temp_test_dir / "test.txt"
        file_path.write_text("Test content with special chars: äöü")

        loader = MultiEncodingLoader(str(file_path))
        docs = loader.load()

        assert len(docs) == 1
        assert "special chars" in docs[0].page_content
        assert docs[0].metadata["source"] == str(file_path)
