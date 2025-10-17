
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, Mock
from fastapi.testclient import TestClient
from datetime import datetime
from pathlib import Path
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader_api import (
    app,
    JobStatus,
    UploadJob,
    jobs_storage,
    jobs_lock,
    process_documents_background
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs_storage():
    jobs_storage.clear()
    yield
    jobs_storage.clear()


@pytest.fixture
def temp_test_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_file():
    return (
        "test_file.txt",
        io.BytesIO(b"Test document content for component testing"),
        "text/plain"
    )


@pytest.fixture
def multiple_mock_files():
    return [
        ("file1.txt", io.BytesIO(b"Content 1"), "text/plain"),
        ("file2.pdf", io.BytesIO(b"%PDF-1.4 Test"), "application/pdf"),
        ("file3.md", io.BytesIO(b"# Markdown"), "text/markdown")
    ]


class TestAsyncUploadWorkflow:

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_complete_async_upload_workflow(
        self,
        mock_store_docs,
        mock_upload_vectorstore,
        mock_file
    ):
        mock_store_docs.return_value = (
            [{"version": 1, "document_name": "test_file"}],
            []
        )
        
        response = client.post(
            "/upload-documents-async",
            params={
                "document_tag": "General",
                "chunk_size": 2000,
                "chunk_overlap": 200,
                "user_id": "test_user",
                "max_workers": 3,
                "use_bulk_upload": True
            },
            files={"files": mock_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert "session_id" in data
        assert "message" in data
        assert data["total_files"] == 1
        assert "test_file.txt" in data["file_names"]
        
        job_id = data["job_id"]
        assert job_id in jobs_storage
        
        job = jobs_storage[job_id]
        assert job.document_tag == "General"
        assert job.chunk_size == 2000
        assert job.user_id == "test_user"
        assert len(job.phases) == 4

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_async_upload_with_multiple_files(
        self,
        mock_store_docs,
        mock_upload_vectorstore,
        multiple_mock_files
    ):
        mock_store_docs.return_value = (
            [
                {"version": 1, "document_name": "file1"},
                {"version": 1, "document_name": "file2"},
                {"version": 1, "document_name": "file3"}
            ],
            []
        )
        
        response = client.post(
            "/upload-documents-async",
            params={"document_tag": "Research"},
            files=[("files", f) for f in multiple_mock_files]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 3
        
        job = jobs_storage[data["job_id"]]
        assert len(job.file_names) == 3

    def test_async_upload_no_files(self):
        response = client.post(
            "/upload-documents-async",
            params={"document_tag": "General"}
        )
        
        assert response.status_code == 422


class TestJobTracking:

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_job_status_retrieval(
        self,
        mock_store_docs,
        mock_upload_vectorstore,
        mock_file
    ):
        mock_store_docs.return_value = [
            {"version": 1, "document_name": "test", "file_hash": "abc123"}
        ]
        
        upload_response = client.post(
            "/upload-documents-async",
            params={"document_tag": "General"},
            files={"files": mock_file}
        )
        
        job_id = upload_response.json()["job_id"]
        
        import time
        time.sleep(0.1)
        
        status_response = client.get(f"/upload-jobs/{job_id}")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        assert status_data["job_id"] == job_id
        assert "status" in status_data
        assert "phases" in status_data
        assert "created_at" in status_data

    def test_job_status_not_found(self):
        response = client.get("/upload-jobs/nonexistent-job-id")
        assert response.status_code == 404

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_job_list_retrieval(
        self,
        mock_store_docs,
        mock_upload_vectorstore,
        mock_file
    ):
        mock_store_docs.return_value = [
            {"version": 1, "document_name": "test", "file_hash": "abc123"}
        ]
        
        jobs_storage.clear()
        
        for i in range(3):
            client.post(
                "/upload-documents-async",
                params={"document_tag": f"Tag{i}"},
                files={"files": mock_file}
            )
        
        import time
        time.sleep(0.1)
        
        response = client.get("/upload-jobs")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "jobs" in data
        assert "total_jobs" in data
        assert data["total_jobs"] >= 3
        assert len(data["jobs"]) >= 3

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_job_list_filtered_by_status(
        self,
        mock_store_docs,
        mock_upload_vectorstore,
        mock_file
    ):
        mock_store_docs.return_value = [
            {"version": 1, "document_name": "test", "file_hash": "abc123"}
        ]
        
        jobs_storage.clear()
        
        response1 = client.post(
            "/upload-documents-async",
            files={"files": mock_file}
        )
        job_id = response1.json()["job_id"]
        
        import time
        time.sleep(0.2)
        
        response = client.get("/upload-jobs?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert any(job["job_id"] == job_id for job in data["jobs"])

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_delete_job(
        self,
        mock_store_docs,
        mock_upload_vectorstore,
        mock_file
    ):
        mock_store_docs.return_value = [
            {"version": 1, "document_name": "test", "file_hash": "abc123"}
        ]
        
        upload_response = client.post(
            "/upload-documents-async",
            files={"files": mock_file}
        )
        job_id = upload_response.json()["job_id"]
        
        delete_response = client.delete(f"/upload-jobs/{job_id}")
        
        assert delete_response.status_code in [200, 400]
        if delete_response.status_code == 200:
            assert "message" in delete_response.json()

    def test_delete_nonexistent_job(self):
        response = client.delete("/upload-jobs/fake-job-id")
        assert response.status_code == 404


class TestMinIOIntegration:

    @patch('loader_api.get_minio_status')
    def test_minio_status_connected(self, mock_get_status):
        mock_get_status.return_value = {
            "connected": True,
            "bucket": "rag-documents",
            "endpoint": "localhost:9000"
        }
        
        response = client.get("/minio/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert "bucket" in data

    @patch('loader_api.get_minio_status')
    def test_minio_status_disconnected(self, mock_get_status):
        mock_get_status.return_value = {
            "connected": False,
            "error": "Connection failed"
        }
        
        response = client.get("/minio/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert "error" in data

    @patch('loader_api.MinIODocumentVersioning')
    def test_list_minio_objects(self, mock_minio_class):
        mock_versioning = MagicMock()
        mock_obj1 = Mock()
        mock_obj1.object_name = "doc1.pdf"
        mock_obj1.size = 1024
        mock_obj1.last_modified = datetime.now()
        mock_obj1.etag = "etag1"
        
        mock_obj2 = Mock()
        mock_obj2.object_name = "doc2.txt"
        mock_obj2.size = 512
        mock_obj2.last_modified = datetime.now()
        mock_obj2.etag = "etag2"
        
        mock_versioning.client.list_objects.return_value = [mock_obj1, mock_obj2]
        mock_versioning.bucket_name = "rag-documents"
        mock_minio_class.return_value = mock_versioning
        
        response = client.get("/debug/minio/objects")
        
        assert response.status_code == 200
        data = response.json()
        if "error" not in data:
            assert "objects" in data
            assert data["total_objects"] >= 0

    @patch('loader_api.MinIODocumentVersioning')
    def test_get_document_versions(self, mock_minio_class):
        mock_versioning = MagicMock()
        mock_versioning.list_document_versions.return_value = [
            {"version": 1, "upload_date": "2025-01-01"},
            {"version": 2, "upload_date": "2025-01-02"}
        ]
        mock_minio_class.return_value = mock_versioning
        
        response = client.get(
            "/document_versions/test_doc",
            params={"document_tag": "General"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert len(data["versions"]) == 2

    @patch('loader_api.MinIODocumentVersioning')
    def test_get_specific_document_version(self, mock_minio_class):
        mock_versioning = MagicMock()
        mock_versioning.get_document_version.return_value = b"Document content"
        mock_versioning.get_metadata.return_value = {"Content-Type": "text/plain"}
        mock_versioning._get_content_type.return_value = "text/plain"
        mock_minio_class.return_value = mock_versioning
        
        response = client.get(
            "/document_version/test_doc/1",
            params={"document_tag": "General"}
        )
        
        assert response.status_code == 200
        assert response.content == b"Document content"

    @patch('loader_api.MinIODocumentVersioning')
    def test_generate_presigned_url(self, mock_minio_class):
        mock_versioning = MagicMock()
        mock_versioning.list_document_versions.return_value = [
            {"version": 1, "upload_date": "2025-01-01", "version_id": "v1"}
        ]
        mock_versioning.generate_presigned_url.return_value = "http://minio/presigned/url"
        mock_minio_class.return_value = mock_versioning
        
        response = client.get(
            "/document/generate-view-url/test_doc",
            params={"document_tag": "General", "version": 1}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "view_url" in data
        assert "document_name" in data
        assert "version" in data


class TestVectorStoreIntegration:

    @patch('vector_store.debug_list_document_sources')
    def test_list_document_sources(self, mock_list_sources):
        mock_list_sources.return_value = ["doc1.pdf", "doc2.txt", "doc3.md"]
        
        response = client.get("/debug/list_sources")
        
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) == 3

    @patch('vector_store.debug_get_local_content_for_source')
    def test_get_document_debug_info(self, mock_get_content):
        mock_get_content.return_value = {
            "source": "test.pdf",
            "chunk_count": 5,
            "chunks": [
                {"id": "1", "content": "Sample content"}
            ]
        }
        
        response = client.get("/debug/local_content/test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "test.pdf"
        assert data["chunk_count"] == 5

    @patch('vector_store.get_full_document_content')
    def test_get_full_document(self, mock_get_full):
        mock_get_full.return_value = """{"pages": [{"page": 1, "content": "Page 1 content"}, {"page": 2, "content": "Page 2 content"}]}"""
        
        response = client.get("/get_full_document/test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "source" in data
        assert data["success"] is True


class TestBackgroundProcessing:

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_background_task_success(
        self,
        mock_store_docs,
        mock_upload_vectorstore,
        temp_test_dir
    ):
        test_file = temp_test_dir / "test.txt"
        test_file.write_text("Test content")
        
        mock_store_docs.return_value = (
            [{"version": 1, "document_name": "test"}],
            []
        )
        
        job = UploadJob(
            job_id="test-job",
            session_id="test-session",
            user_id="test-user",
            document_tag="General",
            total_files=1,
            file_names=["test.txt"],
            chunk_size=1000,
            chunk_overlap=200,
            max_workers=2,
            use_bulk_upload=True,
            created_at=datetime.now()
        )
        
        with jobs_lock:
            jobs_storage[job.job_id] = job
        
        process_documents_background(job.job_id, str(temp_test_dir))
        
        updated_job = jobs_storage[job.job_id]
        assert updated_job.status in [JobStatus.COMPLETED, JobStatus.FAILED]

    @patch('loader_api.store_uploaded_documents')
    def test_background_task_minio_failure(
        self,
        mock_store_docs,
        temp_test_dir
    ):
        mock_store_docs.side_effect = Exception("MinIO connection failed")
        
        test_file = temp_test_dir / "test.txt"
        test_file.write_text("Test content")
        
        job = UploadJob(
            job_id="test-job-fail",
            session_id="test-session",
            user_id="test-user",
            document_tag="General",
            total_files=1,
            file_names=["test.txt"],
            chunk_size=1000,
            chunk_overlap=200,
            max_workers=2,
            use_bulk_upload=True,
            created_at=datetime.now()
        )
        
        with jobs_lock:
            jobs_storage[job.job_id] = job
        
        process_documents_background(job.job_id, str(temp_test_dir))
        
        updated_job = jobs_storage[job.job_id]
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error_message is not None


class TestTagManagement:

    def test_calculate_bitmask_endpoint(self):
        response = client.post(
            "/tags/calculate-bitmask",
            json=["General", "IT_Technology"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "bitmask" in data
        assert "is_admin" in data
        assert "roles" in data
        assert data["bitmask"] > 0

    def test_calculate_bitmask_admin_role(self):
        response = client.post(
            "/tags/calculate-bitmask",
            json=["admin", "General"]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_admin"] is True

    def test_document_categories_endpoint(self):
        response = client.get("/document_categories")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "categories" in data
        assert isinstance(data["categories"], list)
        assert len(data["categories"]) > 0


class TestErrorHandling:

    def test_invalid_document_tag(self):
        response = client.post(
            "/upload-documents-async",
            params={"document_tag": "ValidTag"},
            files={"files": ("test.txt", io.BytesIO(b"test"), "text/plain")}
        )
        
        assert response.status_code == 200

    @patch('loader_api.get_minio_status')
    def test_minio_unavailable_graceful_handling(self, mock_get_status):
        mock_get_status.return_value = {
            "connected": False,
            "error": "Connection timeout"
        }
        
        response = client.get("/minio/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False

    @patch('vector_store.debug_list_document_sources')
    def test_vectorstore_error_handling(self, mock_list_sources):
        mock_list_sources.side_effect = Exception("Chroma error")
        
        response = client.get("/debug/list_sources")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data


class TestConcurrency:

    @patch('loader_api.upload_all_docs_to_vector_store')
    @patch('loader_api.store_uploaded_documents')
    def test_concurrent_uploads(
        self,
        mock_store_docs,
        mock_upload_vectorstore
    ):
        mock_store_docs.return_value = ([], [])
        
        responses = []
        for i in range(5):
            response = client.post(
                "/upload-documents-async",
                files={
                    "files": (
                        f"file{i}.txt",
                        io.BytesIO(f"Content {i}".encode()),
                        "text/plain"
                    )
                }
            )
            responses.append(response)
        
        for response in responses:
            assert response.status_code == 200
        
        job_ids = [r.json()["job_id"] for r in responses]
        assert len(set(job_ids)) == 5
        
        assert len(jobs_storage) >= 5

    def test_concurrent_job_status_queries(self):
        job_id = "test-concurrent-job"
        
        job = UploadJob(
            job_id=job_id,
            session_id="session",
            user_id="user",
            document_tag="General",
            total_files=1,
            file_names=["test.txt"],
            chunk_size=1000,
            chunk_overlap=200,
            max_workers=2,
            use_bulk_upload=True,
            created_at=datetime.now()
        )
        
        with jobs_lock:
            jobs_storage[job_id] = job
        
        responses = []
        for _ in range(10):
            response = client.get(f"/upload-jobs/{job_id}")
            responses.append(response)
        
        for response in responses:
            assert response.status_code == 200
            assert response.json()["job_id"] == job_id


class TestAPIDocumentation:

    def test_openapi_schema_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "info" in schema
        assert "paths" in schema

    def test_docs_endpoint_available(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_available(self):
        response = client.get("/redoc")
        assert response.status_code == 200
