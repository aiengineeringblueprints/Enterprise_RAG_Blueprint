
import pytest
import json
import os
import hashlib
from unittest.mock import Mock, MagicMock, patch
from minio.error import S3Error

from loader.minio_versioning import (
    MinIODocumentVersioning,
    store_uploaded_documents,
    get_minio_status
)

class TestMinIODocumentVersioningInit:

    @patch('loader.minio_versioning.Minio')
    def test_init_with_all_parameters(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test-minio:9000",
            access_key="test_access",
            secret_key="test_secret",
            bucket_name="test-bucket",
            secure=True,
            max_workers=10
        )
        
        assert versioning.endpoint == "test-minio:9000"
        assert versioning.access_key == "test_access"
        assert versioning.secret_key == "test_secret"
        assert versioning.bucket_name == "test-bucket"
        assert versioning.secure is True
        assert versioning.max_workers == 10
        
        mock_minio_class.assert_called_once_with(
            "test-minio:9000",
            access_key="test_access",
            secret_key="test_secret",
            secure=True
        )
    
    @patch('loader.minio_versioning.Minio')
    @patch.dict(os.environ, {
        'MINIO_ENDPOINT': 'env-minio:9000',
        'MINIO_ACCESS_KEY': 'env_access',
        'MINIO_SECRET_KEY': 'env_secret',
        'MINIO_BUCKET_NAME': 'env-bucket',
        'MINIO_SECURE': 'true'
    }, clear=True)
    def test_init_with_environment_variables(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint=None,
            access_key=None,
            secret_key=None,
            bucket_name=None
        )
        
        assert versioning.endpoint == "env-minio:9000"
        assert versioning.access_key == "env_access"
        assert versioning.secret_key == "env_secret"
        assert versioning.bucket_name == "env-bucket"
        assert versioning.secure is True
    
    @patch('loader.minio_versioning.Minio')
    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_credentials_raises_error(self, mock_minio_class):

        with pytest.raises(ValueError, match="MinIO credentials must be provided"):
            MinIODocumentVersioning(
                endpoint="test:9000",
                access_key=None,
                secret_key=None
            )
    
    @patch('loader.minio_versioning.Minio')
    def test_ensure_bucket_exists_creates_bucket(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = False
        mock_minio_class.return_value = mock_client
        
        MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        mock_client.make_bucket.assert_called_once_with("rag-documents")
    
    @patch('loader.minio_versioning.Minio')
    def test_ensure_bucket_exists_handles_error(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.side_effect = S3Error(
            code="TestError",
            message="Test error",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        with pytest.raises(S3Error):
            MinIODocumentVersioning(
                endpoint="test:9000",
                access_key="key",
                secret_key="secret"
            )

class TestHelperMethods:

    @patch('loader.minio_versioning.Minio')
    def test_calculate_file_hash(self, mock_minio_class, sample_txt_file):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        file_hash = versioning._calculate_file_hash(sample_txt_file)
        
        assert len(file_hash) == 64
        assert all(c in '0123456789abcdef' for c in file_hash)
        
        file_hash2 = versioning._calculate_file_hash(sample_txt_file)
        assert file_hash == file_hash2
    
    @patch('loader.minio_versioning.Minio')
    def test_generate_object_key(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        key = versioning._generate_object_key("test doc", "Research", 5, ".pdf")
        
        assert key == "documents/Research/test_doc/v005/test_doc.pdf"
    
    @patch('loader.minio_versioning.Minio')
    def test_generate_object_key_with_special_characters(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        key = versioning._generate_object_key("test/doc name", "IT/Tech", 1, ".txt")
        
        assert " " not in key
        assert key == "documents/IT_Tech/test_doc_name/v001/test_doc_name.txt"
        assert "test_doc_name" in key
        assert "IT_Tech" in key
    
    @patch('loader.minio_versioning.Minio')
    def test_generate_metadata_key(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        key = versioning._generate_metadata_key("test doc", "General")
        
        assert key == "metadata/General/test_doc/versions.json"
    
    @patch('loader.minio_versioning.Minio')
    def test_get_content_type(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        assert versioning._get_content_type(".pdf") == "application/pdf"
        assert versioning._get_content_type(".PDF") == "application/pdf"
        assert versioning._get_content_type(".docx") == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert versioning._get_content_type(".txt") == "text/plain"
        assert versioning._get_content_type(".md") == "text/markdown"
        assert versioning._get_content_type(".csv") == "text/csv"
        assert versioning._get_content_type(".unknown") == "application/octet-stream"

class TestStoreDocumentVersion:

    @patch('loader.minio_versioning.Minio')
    def test_store_new_document(self, mock_minio_class, sample_txt_file):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        version_info = versioning.store_document_version(
            file_path=sample_txt_file,
            document_tag="General",
            user_id="test_user",
            session_id="test_session"
        )
        
        assert version_info["version"] == 1
        assert "file_hash" in version_info
        assert version_info["object_key"].startswith("documents/General/sample/v001/")
        assert version_info["file_extension"] == ".txt"
        assert version_info["user_id"] == "test_user"
        assert version_info["session_id"] == "test_session"
        assert version_info["storage_type"] == "minio_object"
        
        assert mock_client.fput_object.called
    
    @patch('loader.minio_versioning.Minio')
    def test_store_document_duplicate_hash(self, mock_minio_class, sample_txt_file):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        hash_sha256 = hashlib.sha256()
        with open(sample_txt_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        file_hash = hash_sha256.hexdigest()
        
        existing_metadata = {
            "versions": [
                {
                    "version": 1,
                    "file_hash": file_hash,
                    "object_key": "documents/General/sample/v001/sample.txt"
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(existing_metadata).encode('utf-8')
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None
        mock_client.get_object.return_value = mock_response
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        version_info = versioning.store_document_version(
            file_path=sample_txt_file,
            document_tag="General"
        )
        
        assert version_info["version"] == 1
        assert version_info["file_hash"] == file_hash
        
        mock_client.fput_object.assert_not_called()
    
    @patch('loader.minio_versioning.Minio')
    def test_store_document_file_not_found(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        with pytest.raises(FileNotFoundError):
            versioning.store_document_version(
                file_path="/nonexistent/file.txt",
                document_tag="General"
            )
    
    @patch('loader.minio_versioning.Minio')
    def test_store_document_s3_error(self, mock_minio_class, sample_txt_file):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        
        mock_client.fput_object.side_effect = S3Error(
            code="AccessDenied",
            message="Access denied",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        with pytest.raises(S3Error):
            versioning.store_document_version(
                file_path=sample_txt_file,
                document_tag="General"
            )

class TestBulkUpload:

    @patch('loader.minio_versioning.Minio')
    def test_store_document_versions_bulk(self, mock_minio_class, multiple_test_files):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret",
            max_workers=3
        )
        
        results = versioning.store_document_versions_bulk(
            file_paths=multiple_test_files,
            document_tag="TestTag",
            user_id="bulk_user",
            session_id="bulk_session"
        )
        
        assert len(results) == 3
        
        for result in results:
            assert result.get("upload_success") is True
            assert "version" in result
            assert result["user_id"] == "bulk_user"
            assert result["session_id"] == "bulk_session"
    
    @patch('loader.minio_versioning.Minio')
    def test_store_document_versions_bulk_empty_list(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        results = versioning.store_document_versions_bulk(file_paths=[])
        
        assert results == []
    
    @patch('loader.minio_versioning.Minio')
    def test_upload_single_document_with_error(self, mock_minio_class, sample_txt_file):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = Exception("Test error")
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        result = versioning._upload_single_document(
            file_path=sample_txt_file,
            document_tag="General"
        )
        
        assert result["upload_success"] is False
        assert "error" in result
        assert result["file_path"] == sample_txt_file

class TestDocumentRetrieval:

    @patch('loader.minio_versioning.Minio')
    def test_get_document_version_latest(self, mock_minio_class, sample_metadata):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        metadata_response = MagicMock()
        metadata_response.read.return_value = json.dumps(sample_metadata).encode('utf-8')
        metadata_response.close.return_value = None
        metadata_response.release_conn.return_value = None
        
        doc_response = MagicMock()
        doc_response.read.return_value = b"Document content version 2"
        doc_response.close.return_value = None
        doc_response.release_conn.return_value = None
        
        mock_client.get_object.side_effect = [metadata_response, doc_response]
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        content = versioning.get_document_version(
            document_name="test_document",
            document_tag="General",
            version=None
        )
        
        assert content == b"Document content version 2"
        assert mock_client.get_object.call_count == 2
    
    @patch('loader.minio_versioning.Minio')
    def test_get_document_version_specific(self, mock_minio_class, sample_metadata):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        metadata_response = MagicMock()
        metadata_response.read.return_value = json.dumps(sample_metadata).encode('utf-8')
        metadata_response.close.return_value = None
        metadata_response.release_conn.return_value = None
        
        doc_response = MagicMock()
        doc_response.read.return_value = b"Document content version 1"
        doc_response.close.return_value = None
        doc_response.release_conn.return_value = None
        
        mock_client.get_object.side_effect = [metadata_response, doc_response]
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        content = versioning.get_document_version(
            document_name="test_document",
            document_tag="General",
            version=1
        )
        
        assert content == b"Document content version 1"
    
    @patch('loader.minio_versioning.Minio')
    def test_get_document_version_not_found(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        content = versioning.get_document_version(
            document_name="nonexistent",
            document_tag="General"
        )
        
        assert content is None
    
    @patch('loader.minio_versioning.Minio')
    def test_get_document_version_fallback(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        mock_client.get_object.side_effect = [
            S3Error(
                code="NoSuchKey",
                message="Not found",
                resource="/test",
                request_id="123",
                host_id="456",
                response=Mock()
            ),
            MagicMock(
                read=lambda: b"Fallback document content",
                close=lambda: None,
                release_conn=lambda: None
            )
        ]
        
        mock_object = MagicMock()
        mock_object.object_name = "documents/General/test_doc/v001/test_doc.txt"
        mock_client.list_objects.return_value = [mock_object]
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        content = versioning.get_document_version(
            document_name="test_doc",
            document_tag="General",
            version=1
        )
        
        assert content == b"Fallback document content"
    
    @patch('loader.minio_versioning.Minio')
    def test_list_document_versions(self, mock_minio_class, sample_metadata):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        metadata_response = MagicMock()
        metadata_response.read.return_value = json.dumps(sample_metadata).encode('utf-8')
        metadata_response.close.return_value = None
        metadata_response.release_conn.return_value = None
        mock_client.get_object.return_value = metadata_response
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        versions = versioning.list_document_versions(
            document_name="test_document",
            document_tag="General"
        )
        
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2
    
    @patch('loader.minio_versioning.Minio')
    def test_list_document_versions_no_metadata(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        versions = versioning.list_document_versions(
            document_name="nonexistent",
            document_tag="General"
        )
        
        assert versions == []

class TestPresignedURL:

    @patch('loader.minio_versioning.Minio')
    def test_generate_presigned_url_latest(self, mock_minio_class, sample_metadata):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        metadata_response = MagicMock()
        metadata_response.read.return_value = json.dumps(sample_metadata).encode('utf-8')
        metadata_response.close.return_value = None
        metadata_response.release_conn.return_value = None
        mock_client.get_object.return_value = metadata_response
        
        mock_client.presigned_get_object.return_value = "http://minio:9000/bucket/test?signature=xyz"
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="minio:9000",
            access_key="key",
            secret_key="secret"
        )
        
        url = versioning.generate_presigned_url(
            document_name="test_document",
            document_tag="General"
        )
        
        assert "signature=xyz" in url
        assert mock_client.presigned_get_object.called
    
    @patch('loader.minio_versioning.Minio')
    @patch.dict(os.environ, {'MINIO_EXTERNAL_ENDPOINT': 'localhost:9000'})
    def test_generate_presigned_url_with_external_endpoint(self, mock_minio_class, sample_metadata):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        metadata_response = MagicMock()
        metadata_response.read.return_value = json.dumps(sample_metadata).encode('utf-8')
        metadata_response.close.return_value = None
        metadata_response.release_conn.return_value = None
        mock_client.get_object.return_value = metadata_response
        
        mock_client.presigned_get_object.return_value = "http://minio:9000/bucket/test?signature=xyz"
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="minio:9000",
            access_key="key",
            secret_key="secret"
        )
        
        url = versioning.generate_presigned_url(
            document_name="test_document",
            document_tag="General"
        )
        
        assert "localhost:9000" in url
        assert "minio:9000" not in url
    
    @patch('loader.minio_versioning.Minio')
    def test_generate_presigned_url_document_not_found(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        with pytest.raises(ValueError, match="Document not found"):
            versioning.generate_presigned_url(
                document_name="nonexistent",
                document_tag="General"
            )
    
    @patch('loader.minio_versioning.Minio')
    def test_generate_presigned_url_version_not_found(self, mock_minio_class, sample_metadata):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        metadata_response = MagicMock()
        metadata_response.read.return_value = json.dumps(sample_metadata).encode('utf-8')
        metadata_response.close.return_value = None
        metadata_response.release_conn.return_value = None
        mock_client.get_object.return_value = metadata_response
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        with pytest.raises(ValueError, match="Version 999 not found"):
            versioning.generate_presigned_url(
                document_name="test_document",
                document_tag="General",
                version=999
            )

class TestMetadataOperations:

    @patch('loader.minio_versioning.Minio')
    def test_store_document_metadata(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        metadata = {
            "document_name": "test",
            "versions": [{"version": 1}]
        }
        
        versioning._store_document_metadata("test", "General", metadata)
        
        assert mock_client.put_object.called
        call_args = mock_client.put_object.call_args
        
        assert call_args[0][0] == "rag-documents"
        assert "metadata/General/test/versions.json" in call_args[0][1]
        
        assert call_args[1]["content_type"] == "application/json"
    
    @patch('loader.minio_versioning.Minio')
    def test_load_document_metadata_success(self, mock_minio_class, sample_metadata):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        metadata_response = MagicMock()
        metadata_response.read.return_value = json.dumps(sample_metadata).encode('utf-8')
        metadata_response.close.return_value = None
        metadata_response.release_conn.return_value = None
        mock_client.get_object.return_value = metadata_response
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        metadata = versioning._load_document_metadata("test_document", "General")
        
        assert metadata["document_name"] == "test_document"
        assert metadata["latest_version"] == 2
        assert len(metadata["versions"]) == 2
    
    @patch('loader.minio_versioning.Minio')
    def test_load_document_metadata_not_found(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        metadata = versioning._load_document_metadata("nonexistent", "General")
        
        assert metadata == {}
    
    @patch('loader.minio_versioning.Minio')
    def test_load_document_metadata_s3_error(self, mock_minio_class):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="AccessDenied",
            message="Access denied",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        with pytest.raises(S3Error):
            versioning._load_document_metadata("test", "General")

class TestUtilityFunctions:

    @patch('loader.minio_versioning.MinIODocumentVersioning')
    def test_store_uploaded_documents_bulk(self, mock_versioning_class, temp_test_dir):

        files = []
        for i in range(3):
            file_path = temp_test_dir / f"file_{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(str(file_path))
        
        mock_versioning = MagicMock()
        mock_versioning.store_document_versions_bulk.return_value = [
            {"version": 1, "upload_success": True},
            {"version": 1, "upload_success": True},
            {"version": 1, "upload_success": True}
        ]
        mock_versioning_class.return_value = mock_versioning
        
        results = store_uploaded_documents(
            temp_dir=str(temp_test_dir),
            document_tag="TestTag",
            user_id="user1",
            session_id="session1",
            use_bulk_upload=True
        )
        
        assert len(results) == 3
        assert all(r["upload_success"] for r in results)
        assert mock_versioning.store_document_versions_bulk.called
    
    @patch('loader.minio_versioning.MinIODocumentVersioning')
    def test_store_uploaded_documents_sequential(self, mock_versioning_class, temp_test_dir):

        file_path = temp_test_dir / "single_file.txt"
        file_path.write_text("Single file content")
        
        mock_versioning = MagicMock()
        mock_versioning.store_document_version.return_value = {
            "version": 1,
            "file_hash": "abc123"
        }
        mock_versioning_class.return_value = mock_versioning
        
        results = store_uploaded_documents(
            temp_dir=str(temp_test_dir),
            document_tag="TestTag",
            use_bulk_upload=False
        )
        
        assert len(results) == 1
        assert results[0]["upload_success"] is True
        assert mock_versioning.store_document_version.called
    
    @patch('loader.minio_versioning.MinIODocumentVersioning')
    def test_store_uploaded_documents_empty_directory(self, mock_versioning_class, temp_test_dir):

        mock_versioning = MagicMock()
        mock_versioning_class.return_value = mock_versioning
        
        results = store_uploaded_documents(
            temp_dir=str(temp_test_dir),
            document_tag="TestTag"
        )
        
        assert results == []
    
    @patch('loader.minio_versioning.MinIODocumentVersioning')
    def test_get_minio_status_connected(self, mock_versioning_class):

        mock_versioning = MagicMock()
        mock_versioning.client.bucket_exists.return_value = True
        mock_versioning.endpoint = "localhost:9000"
        mock_versioning.bucket_name = "rag-documents"
        mock_versioning_class.return_value = mock_versioning
        
        status = get_minio_status()
        
        assert status["minio_enabled"] is True
        assert status["status"] == "connected"
        assert status["bucket_exists"] is True
        assert status["endpoint"] == "localhost:9000"
        assert status["bucket_name"] == "rag-documents"
    
    @patch('loader.minio_versioning.MinIODocumentVersioning')
    def test_get_minio_status_error(self, mock_versioning_class):

        mock_versioning_class.side_effect = Exception("Connection failed")
        
        status = get_minio_status()
        
        assert status["minio_enabled"] is True
        assert status["status"] == "error"
        assert "Connection failed" in status["message"]

class TestIntegration:

    @patch('loader.minio_versioning.Minio')
    def test_full_upload_download_workflow(self, mock_minio_class, sample_txt_file):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        
        stored_metadata = {}
        
        def mock_get_object(bucket, key):
            if key in stored_metadata:
                response = MagicMock()
                response.read.return_value = json.dumps(stored_metadata[key]).encode('utf-8')
                response.close.return_value = None
                response.release_conn.return_value = None
                return response
            else:
                raise S3Error(
                    code="NoSuchKey",
                    message="Not found",
                    resource="/test",
                    request_id="123",
                    host_id="456",
                    response=Mock()
                )
        
        def mock_put_object(bucket, key, data, **kwargs):
            if "metadata" in key:
                data.seek(0)
                stored_metadata[key] = json.loads(data.read().decode('utf-8'))
        
        mock_client.get_object.side_effect = mock_get_object
        mock_client.put_object.side_effect = mock_put_object
        
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret"
        )
        
        version1 = versioning.store_document_version(
            file_path=sample_txt_file,
            document_tag="General",
            user_id="user1"
        )
        
        assert version1["version"] == 1
        
        versions = versioning.list_document_versions("sample", "General")
        assert len(versions) == 1
        assert versions[0]["version"] == 1
    
    @patch('loader.minio_versioning.Minio')
    def test_thread_safety_concurrent_uploads(self, mock_minio_class, multiple_test_files):

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = S3Error(
            code="NoSuchKey",
            message="Not found",
            resource="/test",
            request_id="123",
            host_id="456",
            response=Mock()
        )
        mock_minio_class.return_value = mock_client
        
        versioning = MinIODocumentVersioning(
            endpoint="test:9000",
            access_key="key",
            secret_key="secret",
            max_workers=3
        )
        
        results = versioning.store_document_versions_bulk(
            file_paths=multiple_test_files,
            document_tag="TestTag"
        )
        
        assert len(results) == len(multiple_test_files)
        assert all(r.get("upload_success") for r in results)
