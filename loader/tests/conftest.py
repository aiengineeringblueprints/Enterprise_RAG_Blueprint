"""
Shared pytest fixtures for loader tests.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_txt_file(temp_test_dir):
    """Create a sample text file for testing."""
    file_path = temp_test_dir / "sample.txt"
    file_path.write_text("This is a test document with some sample content.")
    return str(file_path)


@pytest.fixture
def sample_csv_file(temp_test_dir):
    """Create a sample CSV file for testing."""
    file_path = temp_test_dir / "sample.csv"
    file_path.write_text("name,age,city\nJohn,30,NYC\nJane,25,LA")
    return str(file_path)


@pytest.fixture
def sample_md_file(temp_test_dir):
    """Create a sample Markdown file for testing."""
    file_path = temp_test_dir / "sample.md"
    file_path.write_text("# Test Document\n\nThis is a markdown file.")
    return str(file_path)


@pytest.fixture
def mock_document():
    """Create a mock Document object."""
    from langchain_core.documents import Document
    return Document(
        page_content="Sample document content",
        metadata={"source": "test.txt", "tag_bitmask": 1}
    )


@pytest.fixture
def mock_documents():
    """Create a list of mock Document objects."""
    from langchain_core.documents import Document
    return [
        Document(
            page_content="First document content",
            metadata={"source": "doc1.txt", "tag_bitmask": 1}
        ),
        Document(
            page_content="Second document content",
            metadata={"source": "doc2.txt", "tag_bitmask": 2}
        ),
        Document(
            page_content="Third document content",
            metadata={"source": "doc3.txt", "tag_bitmask": 4}
        )
    ]


@pytest.fixture
def mock_upload_job():
    """Create a mock UploadJob for testing."""
    from loader_api import UploadJob
    return UploadJob(
        job_id="test-job-123",
        session_id="session-456",
        user_id="user-789",
        document_tag="General",
        total_files=5,
        file_names=["file1.txt", "file2.pdf", "file3.docx"],
        chunk_size=2000,
        chunk_overlap=200,
        max_workers=4,
        use_bulk_upload=True,
        created_at=datetime.now()
    )


@pytest.fixture
def mock_minio_client():
    """Create a mock MinIO client."""
    client = MagicMock()
    client.bucket_exists.return_value = True
    client.put_object.return_value = Mock(version_id="version-123")
    client.list_objects.return_value = []
    return client


@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store."""
    vectorstore = MagicMock()
    vectorstore.add_documents.return_value = ["id1", "id2", "id3"]
    return vectorstore


@pytest.fixture
def sample_pdf_file(temp_test_dir):
    """Create a sample PDF-like file for testing."""
    file_path = temp_test_dir / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\nSample PDF content")
    return str(file_path)


@pytest.fixture
def sample_docx_file(temp_test_dir):
    """Create a sample DOCX-like file for testing."""
    file_path = temp_test_dir / "sample.docx"
    file_path.write_bytes(b"PK\x03\x04Sample DOCX content")
    return str(file_path)


@pytest.fixture
def multiple_test_files(temp_test_dir):
    """Create multiple test files."""
    files = []
    for i in range(3):
        file_path = temp_test_dir / f"test_file_{i}.txt"
        file_path.write_text(f"Content of file {i}")
        files.append(str(file_path))
    return files


@pytest.fixture
def mock_minio_versioning_client():
    """Create a comprehensive mock MinIO client for versioning tests."""
    client = MagicMock()
    
    # Mock bucket operations
    client.bucket_exists.return_value = True
    client.make_bucket.return_value = None
    
    # Mock object storage operations
    client.fput_object.return_value = Mock(etag="test-etag", version_id="v1")
    client.put_object.return_value = Mock(etag="test-etag", version_id="v1")
    
    # Mock object retrieval
    mock_response = MagicMock()
    mock_response.read.return_value = b"test document content"
    mock_response.close.return_value = None
    mock_response.release_conn.return_value = None
    client.get_object.return_value = mock_response
    
    # Mock presigned URL generation
    client.presigned_get_object.return_value = "http://localhost:9000/bucket/test-object?signature=xyz"
    
    # Mock listing objects
    mock_object = MagicMock()
    mock_object.object_name = "documents/General/test_doc/v001/test_doc.txt"
    mock_object.size = 1024
    client.list_objects.return_value = [mock_object]
    
    return client


@pytest.fixture
def sample_metadata():
    """Create sample document metadata."""
    return {
        "document_name": "test_document",
        "document_tag": "General",
        "latest_version": 2,
        "last_updated": "2025-10-18T10:00:00",
        "versions": [
            {
                "version": 1,
                "file_hash": "abc123",
                "object_key": "documents/General/test_document/v001/test_document.txt",
                "file_size": 1024,
                "file_extension": ".txt",
                "upload_date": "2025-10-17T10:00:00",
                "user_id": "user1",
                "session_id": "session1",
                "processing_metadata": {},
                "storage_type": "minio_object"
            },
            {
                "version": 2,
                "file_hash": "def456",
                "object_key": "documents/General/test_document/v002/test_document.txt",
                "file_size": 2048,
                "file_extension": ".txt",
                "upload_date": "2025-10-18T10:00:00",
                "user_id": "user2",
                "session_id": "session2",
                "processing_metadata": {},
                "storage_type": "minio_object"
            }
        ]
    }
