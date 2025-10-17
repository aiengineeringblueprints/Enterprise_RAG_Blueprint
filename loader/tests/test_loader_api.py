
from unittest.mock import patch
from fastapi.testclient import TestClient
from datetime import datetime
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader_api import (
    app,
    JobStatus,
    JobPhase,
    UploadJob,
    HealthResponse,
    ProcessUploadRequest,
    jobs_storage,
    jobs_lock
)

client = TestClient(app)

class TestPydanticModels:

    def test_job_status_enum_values(self):

        assert JobStatus.PENDING == "pending"
        assert JobStatus.UPLOADING_TO_MINIO == "uploading_to_minio"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"
        
    def test_job_phase_creation(self):

        phase = JobPhase(
            name="Test Phase",
            status=JobStatus.PENDING
        )
        assert phase.name == "Test Phase"
        assert phase.status == JobStatus.PENDING
        assert phase.progress == 0.0
        assert phase.files_processed == 0
        
    def test_upload_job_default_values(self):

        job = UploadJob(
            job_id="test-123",
            session_id="session-456",
            user_id="user-789",
            document_tag="General",
            total_files=5,
            file_names=["file1.txt", "file2.pdf"],
            chunk_size=2000,
            chunk_overlap=200,
            max_workers=4,
            use_bulk_upload=True,
            created_at=datetime.now()
        )
        
        assert job.status == JobStatus.PENDING
        assert job.started_at is None
        assert job.completed_at is None
        assert len(job.phases) == 4
        assert "minio_upload" in job.phases
        assert job.cancellation_requested is False
        
    def test_process_upload_request_defaults(self):

        request = ProcessUploadRequest()
        
        assert request.document_tag == "General"
        assert request.chunk_size == 1000
        assert request.chunk_overlap == 200
        assert request.max_workers == 5
        assert request.use_bulk_upload is True

class TestHealthEndpoint:

    def test_health_endpoint_success(self):

        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "loader"
        
    def test_health_endpoint_response_model(self):

        response = client.get("/health")
        
        assert response.status_code == 200
        health_response = HealthResponse(**response.json())
        assert health_response.status == "healthy"
        assert health_response.service == "loader"

class TestSupportedFormatsEndpoint:

    def test_supported_formats_returns_list(self):

        response = client.get("/supported_formats")
        
        assert response.status_code == 200
        data = response.json()
        assert "supported_formats" in data
        assert isinstance(data["supported_formats"], list)
        
    def test_supported_formats_includes_common_types(self):

        response = client.get("/supported_formats")
        
        formats = response.json()["supported_formats"]
        expected_formats = [".pdf", ".docx", ".txt", ".md", ".csv"]
        
        format_extensions = [f["extension"] for f in formats]
        for fmt in expected_formats:
            assert fmt in format_extensions

class TestDocumentCategoriesEndpoint:

    def test_document_categories_returns_list(self):

        response = client.get("/document_categories")
        
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert isinstance(data["categories"], list)
        
    def test_document_categories_includes_expected_tags(self):

        response = client.get("/document_categories")
        
        categories = response.json()["categories"]
        expected_categories = [
            "General",
            "IT & Technology",
            "Finance & Controlling",
            "Human Resources"
        ]
        
        for cat in expected_categories:
            assert cat in categories

class TestCalculateBitmaskEndpoint:

    def test_calculate_bitmask_single_tag(self):

        response = client.post(
            "/tags/calculate-bitmask",
            json=["General"]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["bitmask"] == 1
        assert data["roles"] == ["General"]
        assert data["is_admin"] is False
        
    def test_calculate_bitmask_multiple_tags(self):

        response = client.post(
            "/tags/calculate-bitmask",
            json=["General", "IT & Technology"]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["bitmask"] == 129
        
    def test_calculate_bitmask_empty_list(self):

        response = client.post(
            "/tags/calculate-bitmask",
            json=[]
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["bitmask"] == 0

class TestUploadDocumentsAsyncEndpoint:

    @patch('loader_api.BackgroundTasks.add_task')
    def test_upload_async_creates_job(self, mock_add_task):

        file_content = b"Test document content"
        files = {
            "files": ("test.txt", io.BytesIO(file_content), "text/plain")
        }
        data = {
            "document_tag": "General",
            "chunk_size": "1000",
            "chunk_overlap": "200"
        }
        
        response = client.post(
            "/upload-documents-async",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "job_id" in result
        assert "session_id" in result
        assert result["total_files"] == 1
        assert "test.txt" in result["file_names"]
        assert "Started async processing" in result["message"]
        
    def test_upload_async_no_files(self):

        data = {
            "document_tag": "General"
        }
        
        response = client.post(
            "/upload-documents-async",
            data=data
        )
        
        assert response.status_code == 422
        
    @patch('loader_api.BackgroundTasks.add_task')
    def test_upload_async_multiple_files(self, mock_add_task):

        files = [
            ("files", ("test1.txt", io.BytesIO(b"Content 1"), "text/plain")),
            ("files", ("test2.pdf", io.BytesIO(b"PDF content"), "application/pdf"))
        ]
        data = {"document_tag": "IT & Technology"}
        
        response = client.post(
            "/upload-documents-async",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["total_files"] == 2
        assert len(result["file_names"]) == 2

class TestJobStatusEndpoint:

    def test_get_job_status_not_found(self):

        response = client.get("/upload-jobs/nonexistent-job-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        
    def test_get_job_status_success(self):

        job = UploadJob(
            job_id="test-job-123",
            session_id="session-456",
            user_id="user-789",
            document_tag="General",
            total_files=3,
            file_names=["file1.txt", "file2.pdf"],
            chunk_size=2000,
            chunk_overlap=200,
            max_workers=4,
            use_bulk_upload=True,
            created_at=datetime.now(),
            status=JobStatus.COMPLETED
        )
        
        with jobs_lock:
            jobs_storage["test-job-123"] = job
        
        try:
            response = client.get("/upload-jobs/test-job-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-123"
            assert data["status"] == "completed"
            assert "progress" in data
            assert "current_phase" in data
        finally:
            with jobs_lock:
                jobs_storage.pop("test-job-123", None)

class TestJobListEndpoint:

    def test_list_jobs_empty(self):

        with jobs_lock:
            jobs_storage.clear()
        
        response = client.get("/upload-jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_jobs"] == 0
        assert data["jobs"] == []
        
    def test_list_jobs_with_jobs(self):

        job1 = UploadJob(
            job_id="job-1",
            session_id="session-1",
            user_id="user-1",
            document_tag="General",
            total_files=2,
            file_names=["file1.txt"],
            chunk_size=2000,
            chunk_overlap=200,
            max_workers=4,
            use_bulk_upload=True,
            created_at=datetime.now()
        )
        job2 = UploadJob(
            job_id="job-2",
            session_id="session-2",
            user_id="user-2",
            document_tag="IT & Technology",
            total_files=3,
            file_names=["file2.pdf"],
            chunk_size=2000,
            chunk_overlap=200,
            max_workers=4,
            use_bulk_upload=True,
            created_at=datetime.now()
        )
        
        with jobs_lock:
            jobs_storage.clear()
            jobs_storage["job-1"] = job1
            jobs_storage["job-2"] = job2
        
        try:
            response = client.get("/upload-jobs")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_jobs"] == 2
            assert len(data["jobs"]) == 2
        finally:
            with jobs_lock:
                jobs_storage.clear()
                
    def test_list_jobs_filter_by_status(self):

        job_pending = UploadJob(
            job_id="job-pending",
            session_id="session-1",
            user_id="user-1",
            document_tag="General",
            total_files=1,
            file_names=["file.txt"],
            chunk_size=2000,
            chunk_overlap=200,
            max_workers=4,
            use_bulk_upload=True,
            created_at=datetime.now(),
            status=JobStatus.PENDING
        )
        job_completed = UploadJob(
            job_id="job-completed",
            session_id="session-2",
            user_id="user-2",
            document_tag="General",
            total_files=1,
            file_names=["file.txt"],
            chunk_size=2000,
            chunk_overlap=200,
            max_workers=4,
            use_bulk_upload=True,
            created_at=datetime.now(),
            status=JobStatus.COMPLETED
        )
        
        with jobs_lock:
            jobs_storage.clear()
            jobs_storage["job-pending"] = job_pending
            jobs_storage["job-completed"] = job_completed
        
        try:
            response = client.get("/upload-jobs?status=completed")
            
            assert response.status_code == 200
            data = response.json()
            completed_jobs = [j for j in data["jobs"] if j["status"] == "completed"]
            assert len(completed_jobs) >= 1
        finally:
            with jobs_lock:
                jobs_storage.clear()

class TestDeleteJobEndpoint:

    def test_delete_job_not_found(self):

        response = client.delete("/upload-jobs/nonexistent-job")
        
        assert response.status_code == 404
        
    def test_delete_job_success(self):

        job = UploadJob(
            job_id="job-to-cancel",
            session_id="session-1",
            user_id="user-1",
            document_tag="General",
            total_files=1,
            file_names=["file.txt"],
            chunk_size=2000,
            chunk_overlap=200,
            max_workers=4,
            use_bulk_upload=True,
            created_at=datetime.now(),
            status=JobStatus.PENDING
        )
        
        with jobs_lock:
            jobs_storage["job-to-cancel"] = job
        
        try:
            response = client.delete("/upload-jobs/job-to-cancel")
            
            assert response.status_code == 200
            assert "cancellation requested" in response.json()["message"]
            
            with jobs_lock:
                assert jobs_storage["job-to-cancel"].status == JobStatus.CANCELLED
        finally:
            with jobs_lock:
                jobs_storage.pop("job-to-cancel", None)

class TestMinIOStatusEndpoint:

    @patch('loader_api.get_minio_status')
    def test_minio_status_connected(self, mock_get_status):

        mock_get_status.return_value = {
            "connected": True,
            "bucket_exists": True,
            "bucket_name": "documents"
        }
        
        response = client.get("/minio/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["bucket_exists"] is True
        
    @patch('loader_api.get_minio_status')
    def test_minio_status_error(self, mock_get_status):

        def raise_exception():
            raise Exception("Connection failed")
        
        mock_get_status.side_effect = raise_exception
        
        try:
            response = client.get("/minio/status")
            assert response.status_code == 500
        except Exception as e:
            assert "Connection failed" in str(e)

class TestIntegration:

    def test_api_app_exists(self):

        assert app is not None
        assert app.title == "Document Loader API"
        
    def test_api_routes_exist(self):

        routes = [route.path for route in app.routes]
        
        expected_routes = [
            "/health",
            "/supported_formats",
            "/document_categories",
            "/tags/calculate-bitmask",
            "/upload-documents-async",
            "/upload-jobs/{job_id}",
            "/upload-jobs",
            "/minio/status"
        ]
        
        for expected_route in expected_routes:
            assert expected_route in routes
            
    @patch('loader_api.BackgroundTasks.add_task')
    def test_full_upload_workflow(self, mock_add_task):

        files = {
            "files": ("test.txt", io.BytesIO(b"Test content"), "text/plain")
        }
        data = {"document_tag": "General"}
        
        upload_response = client.post(
            "/upload-documents-async",
            files=files,
            data=data
        )
        
        assert upload_response.status_code == 200
        job_id = upload_response.json()["job_id"]
        
        status_response = client.get(f"/upload-jobs/{job_id}")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        
        list_response = client.get("/upload-jobs")
        
        assert list_response.status_code == 200
        jobs = list_response.json()["jobs"]
        job_ids = [j["job_id"] for j in jobs]
        assert job_id in job_ids
        
        with jobs_lock:
            jobs_storage.pop(job_id, None)
