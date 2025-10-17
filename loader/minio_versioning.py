"""
This module provides document versioning capabilities using MinIO object storage.
It stores original documents with version control and maintains metadata about
document versions and their relationship to vector store chunks.
"""

import os
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

class MinIODocumentVersioning:
    """
    Handles document versioning using MinIO object storage.
    Stores original documents with version control and maintains metadata.
    """
    
    def __init__(self, 
                 endpoint: str = None,
                 access_key: str = None,
                 secret_key: str = None,
                 bucket_name: str = "rag-documents",
                 secure: bool = False,
                 max_workers: int = 5):
        """
        Initialize MinIO client for document versioning.
        
        Args:
            endpoint: MinIO server endpoint (e.g., 'minio:9000' for internal)
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name for document storage
            secure: Whether to use HTTPS
            max_workers: Maximum number of parallel upload workers for bulk operations
        """
        # Load from environment if not provided
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")
        self.bucket_name = bucket_name or os.getenv("MINIO_BUCKET_NAME", "rag-documents")
        self.secure = secure or os.getenv("MINIO_SECURE", "false").lower() == "true"
        self.max_workers = max_workers
        
        if not self.access_key or not self.secret_key:
            raise ValueError("MinIO credentials must be provided via parameters or environment variables")
        
        # Initialize MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # Thread lock for metadata operations
        self._metadata_lock = threading.Lock()
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error creating bucket: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_object_key(self, 
                           document_name: str, 
                           document_tag: str, 
                           version: int, 
                           file_extension: str) -> str:
        """
        Generate object key for MinIO storage.
        
        Format: documents/{tag}/{document_name}/v{version}/{document_name}{extension}
        """
        safe_name = document_name.replace(" ", "_").replace("/", "_")
        safe_tag = document_tag.replace(" ", "_").replace("/", "_")
        return f"documents/{safe_tag}/{safe_name}/v{version:03d}/{safe_name}{file_extension}"
    
    def _generate_metadata_key(self, document_name: str, document_tag: str) -> str:
        """Generate metadata object key."""
        safe_name = document_name.replace(" ", "_").replace("/", "_")
        safe_tag = document_tag.replace(" ", "_").replace("/", "_")
        return f"metadata/{safe_tag}/{safe_name}/versions.json"
    
    def store_document_version(self, 
                             file_path: str,
                             document_tag: str = "General",
                             user_id: str = None,
                             session_id: str = None,
                             processing_metadata: Dict = None) -> Dict:
        """
        Store a new version of a document in MinIO as an object.
        
        Args:
            file_path: Path to the document file
            document_tag: Document category tag
            user_id: ID of user uploading the document
            session_id: Upload session ID
            processing_metadata: Additional metadata from processing
            
        Returns:
            Dict containing version information and storage details
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        document_name = file_path.stem
        file_extension = file_path.suffix
        file_hash = self._calculate_file_hash(str(file_path))
        
        # Load existing metadata or create new
        metadata = self._load_document_metadata(document_name, document_tag)
        
        # Check if this exact version already exists (by hash)
        for version_info in metadata.get("versions", []):
            if version_info["file_hash"] == file_hash:
                logger.info(f"Document version already exists: {document_name} (hash: {file_hash[:8]})")
                return version_info
        
        # Determine new version number
        next_version = len(metadata.get("versions", [])) + 1
        
        # Generate object key
        object_key = self._generate_object_key(document_name, document_tag, next_version, file_extension)
        
        # Store document in MinIO
        try:
            # Prepare metadata for the object
            object_metadata = {
                "Content-Type": self._get_content_type(file_extension),
                "document-name": document_name,
                "document-tag": document_tag,
                "version": str(next_version),
                "file-hash": file_hash,
                "upload-date": datetime.now().isoformat(),
                "user-id": user_id or "unknown",
                "session-id": session_id or ""
            }
            
            # Upload file to MinIO
            self.client.fput_object(
                self.bucket_name,
                object_key,
                str(file_path),
                metadata=object_metadata
            )
            
            # Create version info
            version_info = {
                "version": next_version,
                "file_hash": file_hash,
                "object_key": object_key,
                "file_size": file_path.stat().st_size,
                "file_extension": file_extension,
                "upload_date": datetime.now().isoformat(),
                "user_id": user_id,
                "session_id": session_id,
                "processing_metadata": processing_metadata or {},
                "storage_type": "minio_object"  # Stored as MinIO object
            }
            
            # Update metadata
            if "versions" not in metadata:
                metadata["versions"] = []
            metadata["versions"].append(version_info)
            metadata["latest_version"] = next_version
            metadata["document_name"] = document_name
            metadata["document_tag"] = document_tag
            metadata["last_updated"] = datetime.now().isoformat()
            
            # Store updated metadata
            self._store_document_metadata(document_name, document_tag, metadata)
            
            logger.info(f"Stored document version {next_version}: {object_key}")
            return version_info
            
        except S3Error as e:
            logger.error(f"Error storing document: {e}")
            raise

    def _upload_single_document(self, 
                              file_path: str,
                              document_tag: str = "General",
                              user_id: str = None,
                              session_id: str = None,
                              processing_metadata: Dict = None) -> Dict:
        """
        Thread-safe single document upload helper for bulk operations.
        
        Args:
            file_path: Path to the document file
            document_tag: Document category tag
            user_id: ID of user uploading the document
            session_id: Upload session ID
            processing_metadata: Additional metadata from processing
            
        Returns:
            Dict containing version information and storage details
        """
        try:
            return self.store_document_version(
                file_path=file_path,
                document_tag=document_tag,
                user_id=user_id,
                session_id=session_id,
                processing_metadata=processing_metadata
            )
        except Exception as e:
            logger.error(f"Error uploading {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "upload_success": False
            }

    def store_document_versions_bulk(self, 
                                   file_paths: List[str],
                                   document_tag: str = "General",
                                   user_id: str = None,
                                   session_id: str = None,
                                   processing_metadata: Dict = None) -> List[Dict]:
        """
        Store multiple documents in parallel using bulk upload.
        
        Args:
            file_paths: List of file paths to upload
            document_tag: Document category tag
            user_id: ID of user uploading the documents
            session_id: Upload session ID
            processing_metadata: Additional metadata from processing
            
        Returns:
            List of version information for all uploaded documents
        """
        if not file_paths:
            return []
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(
                    self._upload_single_document,
                    file_path,
                    document_tag,
                    user_id,
                    session_id,
                    processing_metadata
                ): file_path for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    result["upload_success"] = "error" not in result
                    results.append(result)
                    
                    if result.get("upload_success", False):
                        logger.info(f"Successfully uploaded: {file_path}")
                    else:
                        logger.error(f"Failed to upload: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Exception during upload of {file_path}: {e}")
                    results.append({
                        "file_path": file_path,
                        "error": str(e),
                        "upload_success": False
                    })
        
        # Log summary
        successful_uploads = sum(1 for r in results if r.get("upload_success", False))
        logger.info(f"Bulk upload completed: {successful_uploads}/{len(file_paths)} files uploaded successfully")
        
        return results

    def generate_presigned_url(self, 
                             document_name: str, 
                             document_tag: str = "General", 
                             version: int = None,
                             expires_in_hours: int = 24,
                             external_endpoint: str = None) -> str:
        """
        Generate a presigned URL for accessing a document version.
        
        Args:
            document_name: Name of the document
            document_tag: Document category tag
            version: Version number (latest if None)
            expires_in_hours: URL expiration time in hours
            external_endpoint: External endpoint for URL (e.g., 'localhost:9000' for local docker-compose)
                             If None, uses MINIO_EXTERNAL_ENDPOINT env var or falls back to internal endpoint
            
        Returns:
            Presigned URL string
        """
        from datetime import timedelta
        
        metadata = self._load_document_metadata(document_name, document_tag)
        if not metadata or "versions" not in metadata:
            raise ValueError(f"Document not found: {document_name}")
        
        # Get version info
        if version is None:
            version = metadata.get("latest_version")
        
        version_info = None
        for v in metadata["versions"]:
            if v["version"] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version} not found for document: {document_name}")
        
        # Generate presigned URL
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                version_info["object_key"],
                expires=timedelta(hours=expires_in_hours)
            )
            
            # Replace internal endpoint with external endpoint for browser access
            if external_endpoint is None:
                external_endpoint = os.getenv("MINIO_EXTERNAL_ENDPOINT")
            
            if external_endpoint:
                # Replace the internal endpoint with the external one
                # Internal might be: http://minio:9000/...
                # External should be: http://localhost:9000/...
                internal_endpoint = f"http://{self.endpoint}" if not self.secure else f"https://{self.endpoint}"
                external_full = f"http://{external_endpoint}" if not self.secure else f"https://{external_endpoint}"
                url = url.replace(internal_endpoint, external_full)
                logger.info(f"Replaced endpoint in presigned URL: {internal_endpoint} -> {external_full}")
            
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

    def _get_content_type(self, file_extension: str) -> str:
        """Get content type based on file extension."""
        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv"
        }
        return content_types.get(file_extension.lower(), "application/octet-stream")
    
    def _load_document_metadata(self, document_name: str, document_tag: str) -> Dict:
        """Load document metadata from MinIO."""
        metadata_key = self._generate_metadata_key(document_name, document_tag)
        
        try:
            response = self.client.get_object(self.bucket_name, metadata_key)
            metadata = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            return metadata
        except S3Error as e:
            if e.code == "NoSuchKey":
                # Metadata doesn't exist yet
                return {}
            else:
                logger.error(f"Error loading metadata: {e}")
                raise
    
    def _store_document_metadata(self, document_name: str, document_tag: str, metadata: Dict):
        """Store document metadata in MinIO."""
        metadata_key = self._generate_metadata_key(document_name, document_tag)
        metadata_json = json.dumps(metadata, indent=2)
        
        try:
            from io import BytesIO
            metadata_bytes = BytesIO(metadata_json.encode('utf-8'))
            self.client.put_object(
                self.bucket_name,
                metadata_key,
                metadata_bytes,
                length=len(metadata_json),
                content_type="application/json"
            )
        except S3Error as e:
            logger.error(f"Error storing metadata: {e}")
            raise
    
    def get_document_version(self, document_name: str, document_tag: str, version: int = None) -> Optional[bytes]:
        """
        Retrieve a specific version of a document.
        
        Args:
            document_name: Name of the document
            document_tag: Document category tag
            version: Version number (latest if None)
            
        Returns:
            Document content as bytes, or None if not found
        """
        logger.info(f"get_document_version called: name={document_name}, tag={document_tag}, version={version}")
        
        metadata = self._load_document_metadata(document_name, document_tag)
        if not metadata or "versions" not in metadata:
            logger.warning(f"No metadata found for document: {document_name}, tag: {document_tag}")
            logger.info("Attempting direct object access as fallback...")
            
            # FALLBACK: Try to access the document directly if it exists without metadata
            # This handles documents uploaded with older versions of the code
            try:
                # Use version 1 as default if no version specified
                fallback_version = version if version is not None else 1
                safe_name = document_name.replace(" ", "_").replace("/", "_")
                safe_tag = document_tag.replace(" ", "_").replace("/", "_")
                
                # Try to determine file extension by listing objects
                prefix = f"documents/{safe_tag}/{safe_name}/v{fallback_version:03d}/"
                logger.info(f"Searching for objects with prefix: {prefix}")
                
                objects = self.client.list_objects(self.bucket_name, prefix=prefix)
                object_list = list(objects)
                
                if not object_list:
                    logger.error(f"No objects found with prefix: {prefix}")
                    return None
                
                # Take the first object found (should be the document)
                object_key = object_list[0].object_name
                logger.info(f"Found object via fallback: {object_key}")
                
                response = self.client.get_object(self.bucket_name, object_key)
                content = response.read()
                response.close()
                response.release_conn()
                logger.info(f"Successfully retrieved document via fallback: {len(content)} bytes")
                return content
                
            except Exception as e:
                logger.error(f"Fallback document retrieval failed: {e}")
                return None
        
        logger.info(f"Metadata loaded: {len(metadata['versions'])} versions available")
        
        # Get version info
        if version is None:
            version = metadata.get("latest_version")
            logger.info(f"Using latest version: {version}")
        
        version_info = None
        for v in metadata["versions"]:
            logger.debug(f"Checking version: {v.get('version')} against requested: {version}")
            if v["version"] == version:
                version_info = v
                break
        
        if not version_info:
            logger.error(f"Version {version} not found. Available versions: {[v['version'] for v in metadata['versions']]}")
            return None
        
        logger.info(f"Found version info: object_key={version_info.get('object_key')}")
        
        # Retrieve document from MinIO object storage
        try:
            object_key = version_info["object_key"]
            logger.info(f"Retrieving object from MinIO: bucket={self.bucket_name}, key={object_key}")
            response = self.client.get_object(self.bucket_name, object_key)
            content = response.read()
            response.close()
            response.release_conn()
            logger.info(f"Successfully retrieved document: {len(content)} bytes")
            return content
        except S3Error as e:
            logger.error(f"S3Error retrieving document: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving document: {e}")
            return None
    
    def list_document_versions(self, document_name: str, document_tag: str) -> List[Dict]:
        """
        List document versions.
        
        Args:
            document_name: Document name
            document_tag: Document tag
            
        Returns:
            List of document version information
        """
        metadata = self._load_document_metadata(document_name, document_tag)
        return metadata.get("versions", [])


def store_uploaded_documents(temp_dir: str, 
                           document_tag: str,
                           user_id: str = None,
                           session_id: str = None,
                           max_workers: int = 5,
                           use_bulk_upload: bool = True) -> List[Dict]:
    """
    Store all documents from a temporary directory to MinIO using bulk upload.
    
    Args:
        temp_dir: Temporary directory containing uploaded files
        document_tag: Document category tag
        user_id: User ID for tracking
        session_id: Session ID for tracking
        max_workers: Maximum number of parallel upload workers
        use_bulk_upload: Whether to use parallel bulk upload (recommended for multiple files)
        
    Returns:
        List of version information for stored documents
    """
    versioning = MinIODocumentVersioning(max_workers=max_workers)
    
    # Collect all files to upload
    file_paths = []
    for file_path in Path(temp_dir).rglob("*"):
        if file_path.is_file():
            file_paths.append(str(file_path))
    
    if not file_paths:
        logger.info("No files found to upload")
        return []
    
    logger.info(f"Found {len(file_paths)} files to upload")
    
    if use_bulk_upload and len(file_paths) > 1:
        # Use bulk upload for multiple files
        logger.info(f"Using bulk upload with {max_workers} workers")
        stored_versions = versioning.store_document_versions_bulk(
            file_paths=file_paths,
            document_tag=document_tag,
            user_id=user_id,
            session_id=session_id
        )
    else:
        # Use sequential upload for single file or when bulk is disabled
        logger.info("Using sequential upload")
        stored_versions = []
        for file_path in file_paths:
            try:
                version_info = versioning.store_document_version(
                    str(file_path),
                    document_tag=document_tag,
                    user_id=user_id,
                    session_id=session_id
                )
                version_info["upload_success"] = True
                stored_versions.append(version_info)
            except Exception as e:
                logger.error(f"Error storing {file_path}: {e}")
                stored_versions.append({
                    "file_path": str(file_path),
                    "error": str(e),
                    "upload_success": False
                })
                raise  # Raise exception to maintain original behavior
    
    return stored_versions


def get_minio_status() -> Dict:
    """Get MinIO connection and configuration status."""
    try:
        versioning = MinIODocumentVersioning()
        # Test connection by checking if bucket exists
        bucket_exists = versioning.client.bucket_exists(versioning.bucket_name)
        
        return {
            "minio_enabled": True,
            "status": "connected",
            "bucket_exists": bucket_exists,
            "endpoint": versioning.endpoint,
            "bucket_name": versioning.bucket_name,
            "message": "MinIO connection successful"
        }
    except Exception as e:
        return {
            "minio_enabled": True,
            "status": "error",
            "message": f"MinIO connection failed: {str(e)}"
        }