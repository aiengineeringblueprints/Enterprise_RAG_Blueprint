import json
import uuid
import logging
import threading
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from loader_functions import upload_all_docs_to_vector_store
import tempfile
from minio_versioning import store_uploaded_documents, get_minio_status, MinIODocumentVersioning
from tag_management import TAG_BITMASK_MAPPING, combine_tag_bitmasks, is_admin_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Loader API", version="1.0.0")

# Job Management System
class JobStatus(str, Enum):
    PENDING = "pending"
    UPLOADING_TO_MINIO = "uploading_to_minio"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    VECTOR_DB = "vector_db"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPhase(BaseModel):
    name: str
    status: JobStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 100.0
    files_processed: int = 0
    total_files: int = 0
    error_message: Optional[str] = None

class UploadJob(BaseModel):
    job_id: str
    session_id: str
    user_id: Optional[str]
    document_tag: str
    total_files: int
    file_names: List[str]
    chunk_size: int
    chunk_overlap: int
    max_workers: int
    use_bulk_upload: bool
    
    # Job tracking
    status: JobStatus = JobStatus.PENDING
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Phase tracking
    phases: Dict[str, JobPhase] = {
        "minio_upload": JobPhase(name="MinIO Upload", status=JobStatus.PENDING),
        "chunking": JobPhase(name="Text Chunking", status=JobStatus.PENDING),
        "embedding": JobPhase(name="Embedding Generation", status=JobStatus.PENDING),
        "vector_db": JobPhase(name="Vector DB Storage", status=JobStatus.PENDING)
    }
    
    # Results
    versions_stored: List[Dict] = []
    error_message: Optional[str] = None
    cancellation_requested: bool = False

# Global job storage (in production, use Redis or database)
jobs_storage: Dict[str, UploadJob] = {}
jobs_lock = threading.Lock()

# Background task function for complete upload process
def process_documents_background(job_id: str, temp_dir: str):
    """
    Background task that processes the complete document upload pipeline:
    1. MinIO Bulk Upload
    2. Text Chunking
    3. Embedding Generation
    4. Vector DB Storage
    """
    job = None
    try:
        with jobs_lock:
            job = jobs_storage.get(job_id)
            if not job:
                return
            
            job.status = JobStatus.UPLOADING_TO_MINIO
            job.started_at = datetime.now()
            job.phases["minio_upload"].status = JobStatus.UPLOADING_TO_MINIO
            job.phases["minio_upload"].start_time = datetime.now()
        
        # Phase 1: MinIO Upload
        print(f"[Job {job_id}] Starting MinIO upload phase...")
        versions_stored = store_uploaded_documents(
            temp_dir=temp_dir,
            document_tag=job.document_tag,
            user_id=job.user_id,
            session_id=job.session_id,
            max_workers=job.max_workers,
            use_bulk_upload=job.use_bulk_upload
        )
        
        # Check for cancellation
        if job.cancellation_requested:
            with jobs_lock:
                job.status = JobStatus.CANCELLED
            return
        
        with jobs_lock:
            job.phases["minio_upload"].status = JobStatus.COMPLETED
            job.phases["minio_upload"].end_time = datetime.now()
            job.phases["minio_upload"].progress = 100.0
            job.phases["minio_upload"].files_processed = len(versions_stored)
            job.versions_stored = versions_stored
            
            # Start chunking phase
            job.status = JobStatus.CHUNKING
            job.phases["chunking"].status = JobStatus.CHUNKING
            job.phases["chunking"].start_time = datetime.now()
        
        print(f"[Job {job_id}] Starting chunking and vector storage phase...")
        
        # Phase 2-4: Chunking, Embedding, Vector DB (handled by upload_all_docs_to_vector_store)
        upload_all_docs_to_vector_store(
            root_path=temp_dir,
            document_tag=job.document_tag,
            chunk_size=job.chunk_size,
            chunk_overlap=job.chunk_overlap
        )
        
        # Check for cancellation
        if job.cancellation_requested:
            with jobs_lock:
                job.status = JobStatus.CANCELLED
            return
        
        # Complete all phases
        with jobs_lock:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Mark all remaining phases as completed
            for phase_name, phase in job.phases.items():
                if phase.status != JobStatus.COMPLETED:
                    phase.status = JobStatus.COMPLETED
                    phase.end_time = datetime.now()
                    phase.progress = 100.0
                    phase.files_processed = job.total_files
        
        print(f"[Job {job_id}] All phases completed successfully!")
        
    except Exception as e:
        print(f"[Job {job_id}] Error during processing: {str(e)}")
        if job:
            with jobs_lock:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                
                # Mark current phase as failed
                for phase in job.phases.values():
                    if phase.status in [JobStatus.PENDING, JobStatus.UPLOADING_TO_MINIO, 
                                       JobStatus.CHUNKING, JobStatus.EMBEDDING, JobStatus.VECTOR_DB]:
                        phase.status = JobStatus.FAILED
                        phase.error_message = str(e)
                        phase.end_time = datetime.now()
                        break

class HealthResponse(BaseModel):
    status: str
    service: str

class ProcessUploadRequest(BaseModel):
    document_tag: str = "General" 
    chunk_size: int = 1000
    chunk_overlap: int = 200
    user_id: Optional[str] = None
    max_workers: int = 5  # Bulk upload workers
    use_bulk_upload: bool = True  # Enable bulk upload

class AsyncUploadResponse(BaseModel):
    job_id: str
    session_id: str
    message: str
    total_files: int
    file_names: List[str]

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float  # Overall progress 0-100
    current_phase: str
    phases: Dict[str, JobPhase]
    total_files: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    versions_stored: List[Dict] = []
    error_message: Optional[str] = None

class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]
    total_jobs: int

class ProcessVolumeResponse(BaseModel):
    message: str
    session_id: str
    files_processed: int
    success: bool
    error_message: Optional[str] = None
    versions_stored: List[Dict] = []  # MinIO-Versionsinformationen


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="loader")

@app.get("/supported_formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": [
            {"extension": ".pdf", "description": "PDF documents"},
            {"extension": ".docx", "description": "Word documents"},
            {"extension": ".txt", "description": "Text files"},
            {"extension": ".md", "description": "Markdown files"},
            {"extension": ".csv", "description": "CSV files"}
        ]
    }

@app.get("/document_categories")
async def get_document_categories():
    """Get list of available document categories"""
    
    return {
        "categories": list(TAG_BITMASK_MAPPING.keys())
    }

@app.post("/tags/calculate-bitmask")
async def calculate_tag_bitmask(user_roles: List[str]):
    """
    Calculate combined bitmask for given user roles.
    This endpoint exposes the tag management logic owned by the loader service.
    
    Args:
        user_roles: List of role/tag names
        
    Returns:
        Combined bitmask value and admin status
    """
    return {
        "bitmask": combine_tag_bitmasks(user_roles),
        "is_admin": is_admin_user(user_roles),
        "roles": user_roles
    }

# New Async Upload Endpoints
@app.post("/upload-documents-async", response_model=AsyncUploadResponse)
async def upload_documents_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile],
    document_tag: str = "General",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    user_id: str = None,
    max_workers: int = 5,
    use_bulk_upload: bool = True
):
    """
    Start asynchronous document processing with complete pipeline:
    MinIO Upload -> Chunking -> Embedding -> Vector DB Storage
    """
    # Generate unique job and session IDs
    job_id = str(uuid.uuid4())
    session_id = f"async_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}"
    
    # Create temporary directory for uploaded files
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Save uploaded files to temporary directory
    file_names = []
    for file in files:
        file_path = temp_path / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        file_names.append(file.filename)
    
    # Create job object
    job = UploadJob(
        job_id=job_id,
        session_id=session_id,
        user_id=user_id,
        document_tag=document_tag,
        total_files=len(files),
        file_names=file_names,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_workers=max_workers,
        use_bulk_upload=use_bulk_upload,
        created_at=datetime.now()
    )
    
    # Store job
    with jobs_lock:
        jobs_storage[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_documents_background, job_id, str(temp_path))
    
    return AsyncUploadResponse(
        job_id=job_id,
        session_id=session_id,
        message=f"Started async processing of {len(files)} files",
        total_files=len(files),
        file_names=file_names
    )

@app.get("/upload-jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a specific upload job"""
    with jobs_lock:
        job = jobs_storage.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Calculate overall progress
        total_phases = len(job.phases)
        completed_phases = sum(1 for phase in job.phases.values() 
                             if phase.status == JobStatus.COMPLETED)
        overall_progress = (completed_phases / total_phases) * 100.0
        
        # Determine current phase
        current_phase = "pending"
        for phase_name, phase in job.phases.items():
            if phase.status in [JobStatus.UPLOADING_TO_MINIO, JobStatus.CHUNKING, 
                               JobStatus.EMBEDDING, JobStatus.VECTOR_DB]:
                current_phase = phase.name
                break
        
        # Estimate completion time (simple heuristic)
        estimated_completion = None
        if job.started_at and overall_progress > 10:
            elapsed = datetime.now() - job.started_at
            estimated_total = elapsed * (100 / overall_progress)
            estimated_completion = job.started_at + estimated_total
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=overall_progress,
            current_phase=current_phase,
            phases=job.phases,
            total_files=job.total_files,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            estimated_completion=estimated_completion,
            versions_stored=job.versions_stored,
            error_message=job.error_message
        )

@app.get("/upload-jobs", response_model=JobListResponse)
async def list_upload_jobs(limit: int = 50, status_filter: Optional[JobStatus] = None):
    """List all upload jobs with optional status filter"""
    with jobs_lock:
        jobs = list(jobs_storage.values())
        
        # Apply status filter
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        jobs = jobs[:limit]
        
        # Convert to response format
        job_responses = []
        for job in jobs:
            total_phases = len(job.phases)
            completed_phases = sum(1 for phase in job.phases.values() 
                                 if phase.status == JobStatus.COMPLETED)
            overall_progress = (completed_phases / total_phases) * 100.0
            
            current_phase = "pending"
            for phase_name, phase in job.phases.items():
                if phase.status in [JobStatus.UPLOADING_TO_MINIO, JobStatus.CHUNKING, 
                                   JobStatus.EMBEDDING, JobStatus.VECTOR_DB]:
                    current_phase = phase.name
                    break
            
            job_responses.append(JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                progress=overall_progress,
                current_phase=current_phase,
                phases=job.phases,
                total_files=job.total_files,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                estimated_completion=None,
                versions_stored=job.versions_stored,
                error_message=job.error_message
            ))
        
        return JobListResponse(
            jobs=job_responses,
            total_jobs=len(jobs_storage)
        )

@app.delete("/upload-jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running upload job"""
    with jobs_lock:
        job = jobs_storage.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled in current state")
        
        job.cancellation_requested = True
        job.status = JobStatus.CANCELLED
        
        return {"message": f"Job {job_id} cancellation requested"}

@app.post("/upload-documents", response_model=ProcessVolumeResponse)
async def upload_documents(
    files: List[UploadFile],
    document_tag: str = "General",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    user_id: str = None,
    max_workers: int = 5,
    use_bulk_upload: bool = True
):
    """
    Upload documents and process them with MinIO object storage.
    """
    session_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded files to temporary directory
            file_paths = []
            for file in files:
                file_path = temp_path / file.filename
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                file_paths.append(file_path)
            
            # Store documents in MinIO as objects using bulk upload
            versions_stored = store_uploaded_documents(
                temp_dir=str(temp_path),
                document_tag=document_tag,
                user_id=user_id,
                session_id=session_id,
                max_workers=max_workers,
                use_bulk_upload=use_bulk_upload
            )
            print(f"Stored {len(versions_stored)} document versions in MinIO")
            
            # Process files using existing loader function
            upload_all_docs_to_vector_store(
                root_path=str(temp_path),
                document_tag=document_tag,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            return ProcessVolumeResponse(
                message=f"Successfully processed {len(file_paths)} files",
                session_id=session_id,
                files_processed=len(file_paths),
                success=True,
                versions_stored=versions_stored
            )
            
    except Exception as e:
        print(f"Error processing upload: {e}")
        return ProcessVolumeResponse(
            message="Error during processing",
            session_id=session_id,
            files_processed=0,
            success=False,
            error_message=str(e)
        )


@app.get("/debug/local_content/{document_source:path}")
def debug_local_content(document_source: str):
    """Debug endpoint to inspect what's stored for a document in the local vector store."""
    try:
        from vector_store import debug_get_local_content_for_source
        return debug_get_local_content_for_source(document_source)
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/get_full_document/{document_source:path}")
async def get_full_document(document_source: str):
    """
    Get the full content of a document by its source path
    """
    try:
        from vector_store import get_full_document_content, debug_list_document_sources
        
        print(f"Requested document: {document_source}")
        
        # First, try to get the document with the exact path
        content = get_full_document_content(document_source)
        if content is not None and content.strip():  # Check for non-empty content
            return {
                "source": document_source,
                "content": content,
                "success": True
            }
        
        # If not found or empty, try to find similar document paths
        available_sources = debug_list_document_sources()
        print(f"Available sources: {available_sources}")
        
        # Extract just the filename from the requested path
        if '\\' in document_source:
            requested_filename = document_source.split('\\')[-1]
        elif '/' in document_source:
            requested_filename = document_source.split('/')[-1]
        else:
            requested_filename = document_source
        
        print(f"Looking for filename: {requested_filename}")
        
        # Find matching source by filename
        matching_source = None
        for source in available_sources:
            if source.endswith(requested_filename):
                matching_source = source
                break
        
        if matching_source:
            print(f"Found matching source: {matching_source}")
            content = get_full_document_content(matching_source)
            if content and content.strip():  # Check for non-empty content
                return {
                    "source": matching_source,
                    "original_request": document_source,
                    "content": content,
                    "success": True,
                    "note": f"Mapped {document_source} to {matching_source}"
                }
            else:
                # If content is empty, return what we have with a note
                return {
                    "source": matching_source,
                    "original_request": document_source,
                    "content": content or "Dokument gefunden, aber Inhalt ist leer oder sehr kurz.",
                    "success": True,
                    "note": f"Mapped {document_source} to {matching_source} (content may be empty)"
                }
        
        # If still not found, list similar filenames for debugging
        similar_files = [src for src in available_sources if requested_filename.lower() in src.lower()]
        error_detail = f"Document not found: {document_source}. "
        if similar_files:
            error_detail += f"Similar files found: {similar_files[:3]}"
        else:
            error_detail += f"Available files: {[src.split('/')[-1] for src in available_sources[:5]]}"
            
        raise HTTPException(status_code=404, detail=error_detail)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@app.get("/debug/list_sources")
async def list_document_sources():
    """
    Debug endpoint to list all available document sources
    """
    try:
        from vector_store import debug_list_document_sources
        sources = debug_list_document_sources()
        return {
            "total_sources": len(sources),
            "sources": sources[:50],  # Limit to first 50 for readability
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sources: {str(e)}")

@app.get("/minio/status")
async def get_minio_status_endpoint():
    """Check MinIO connection and configuration status."""
    return get_minio_status()

@app.get("/debug/minio/objects")
async def list_minio_objects():
    """Debug endpoint to list all objects in MinIO bucket."""
    try:
        from minio_versioning import MinIODocumentVersioning
        versioning = MinIODocumentVersioning()
        
        objects = []
        for obj in versioning.client.list_objects(versioning.bucket_name, recursive=True):
            objects.append({
                "object_name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "etag": obj.etag
            })
        
        return {
            "bucket_name": versioning.bucket_name,
            "total_objects": len(objects),
            "objects": objects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing MinIO objects: {str(e)}")

@app.get("/document_versions/{document_name}")
async def get_document_versions(document_name: str, document_tag: str = "General"):
    """
    Get all versions of a specific document from MinIO.
    """
    try:
        versioning = MinIODocumentVersioning()
        versions = versioning.list_document_versions(document_name, document_tag)
        
        return {
            "document_name": document_name,
            "document_tag": document_tag,
            "versions": versions,
            "total_versions": len(versions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving versions: {str(e)}")

@app.get("/document_version/{document_name}/{version}")
async def get_specific_document_version(
    document_name: str, 
    version: int, 
    document_tag: str = "General"
):
    """
    Download a specific version of a document from MinIO.
    """
    try:
        versioning = MinIODocumentVersioning()
        content = versioning.get_document_version(document_name, document_tag, version)
        
        if content is None:
            raise HTTPException(status_code=404, detail="Document version not found")
        
        # Get file extension for content type
        versions = versioning.list_document_versions(document_name, document_tag)
        file_extension = None
        for v in versions:
            if v["version"] == version:
                file_extension = v["file_extension"]
                break
        
        content_type = versioning._get_content_type(file_extension or "")
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={document_name}_v{version}{file_extension}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.get("/document/generate-view-url/{document_name}")
async def generate_view_url(
    document_name: str,
    version: int = None,
    document_tag: str = "General",
    request: Request = None
):
    """
    Generate a URL to view the document in browser.
    This mimics presigned URL functionality for our volume-based storage.
    """
    try:
        versioning = MinIODocumentVersioning()
        versions = versioning.list_document_versions(document_name, document_tag)
        
        if not versions:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Use latest version if not specified
        if version is None:
            version = max(v["version"] for v in versions)
        
        # Check if version exists
        version_exists = any(v["version"] == version for v in versions)
        if not version_exists:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
        # Generate URL
        base_url = str(request.base_url).rstrip('/')
        view_url = f"{base_url}/document/view/{document_name}?version={version}&document_tag={document_tag}"
        
        return {
            "view_url": view_url,
            "document_name": document_name,
            "version": version,
            "document_tag": document_tag,
            "expires_in": "never"  # No expiration for our API-based approach
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating view URL: {str(e)}")


@app.get("/document/view/{document_name}")
async def view_document(
    document_name: str,
    version: int = None,
    document_tag: str = "General"
):
    """
    Proxy endpoint to view a document directly in the browser.
    This endpoint fetches the document from MinIO and streams it to the client.
    Avoids signature issues with presigned URLs by proxying through the loader service.
    """
    try:
        logger.info(f"View document request: name={document_name}, tag={document_tag}, version={version}")
        versioning = MinIODocumentVersioning()
        
        # Try to get metadata (but allow fallback if not available)
        logger.info(f"Loading metadata for document: {document_name}, tag: {document_tag}")
        metadata = versioning._load_document_metadata(document_name, document_tag)
        
        has_metadata = metadata and "versions" in metadata
        
        if not has_metadata:
            logger.warning(f"No metadata found for document: {document_name}, tag: {document_tag}")
            logger.info("Will attempt direct access as fallback...")
        else:
            logger.info(f"Metadata found: {len(metadata.get('versions', []))} version(s)")
        
        # Get document content from MinIO (will use fallback if no metadata)
        content = versioning.get_document_version(
            document_name=document_name,
            document_tag=document_tag,
            version=version
        )
        
        if content is None:
            logger.error(f"Document content not found: {document_name}, tag: {document_tag}, version: {version}")
            raise HTTPException(status_code=404, detail=f"Document not found: {document_name}")
        
        # Determine file extension and content type
        if has_metadata:
            # Get version info from metadata
            if version is None:
                version = metadata.get("latest_version")
            
            version_info = None
            for v in metadata["versions"]:
                if v["version"] == version:
                    version_info = v
                    break
            
            if not version_info:
                raise HTTPException(status_code=404, detail=f"Version {version} not found")
            
            file_extension = version_info.get("file_extension", ".pdf")
        else:
            # FALLBACK: Try to infer extension from document name or default to .pdf
            logger.info("Using fallback: inferring file extension from document name")
            # Check if document_name already has an extension
            import os
            if "." in document_name:
                file_extension = os.path.splitext(document_name)[1]
            else:
                # Default to .pdf if no extension found
                file_extension = ".pdf"
            logger.info(f"Inferred file extension: {file_extension}")
        
        content_type = versioning._get_content_type(file_extension)
        
        # Return document as streaming response with inline display
        from fastapi.responses import Response
        
        # Build headers optimized for inline display in browser
        headers = {
            "Cache-Control": "public, max-age=3600",
            "Accept-Ranges": "bytes",
        }
        
        # IMPORTANT: For PDFs, DO NOT set Content-Disposition header at all!
        # Most browsers will display PDFs inline by default when:
        # 1. Content-Type is application/pdf
        # 2. No Content-Disposition header is present
        # Only set Content-Disposition for non-PDF files
        if file_extension.lower() != ".pdf":
            safe_filename = f"{document_name}{file_extension}".replace(" ", "_")
            headers["Content-Disposition"] = f"inline; filename=\"{safe_filename}\""
        
        return Response(
            content=content,
            media_type=content_type,
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error viewing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error viewing document: {str(e)}")


@app.get("/debug/list-documents")
async def debug_list_documents(document_tag: str = None):
    """
    Debug endpoint to list all documents and their metadata.
    """
    try:
        versioning = MinIODocumentVersioning()
        
        # List all objects in MinIO bucket
        objects = versioning.client.list_objects(versioning.bucket_name, recursive=True)
        
        all_objects = []
        metadata_objects = []
        
        for obj in objects:
            all_objects.append({
                "object_key": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
            })
            
            if obj.object_name.endswith("_metadata.json"):
                metadata_objects.append(obj.object_name)
        
        # Load metadata for all documents
        documents_metadata = []
        for meta_key in metadata_objects:
            try:
                response = versioning.client.get_object(versioning.bucket_name, meta_key)
                metadata = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
                
                # Filter by tag if provided
                if document_tag is None or metadata.get("document_tag") == document_tag:
                    documents_metadata.append({
                        "metadata_key": meta_key,
                        "document_name": metadata.get("document_name"),
                        "document_tag": metadata.get("document_tag"),
                        "latest_version": metadata.get("latest_version"),
                        "versions": metadata.get("versions", [])
                    })
            except Exception as e:
                logger.error(f"Error loading metadata from {meta_key}: {e}")
        
        return {
            "total_objects": len(all_objects),
            "metadata_objects": len(metadata_objects),
            "documents": documents_metadata,
            "all_objects": all_objects[:50]  # Limit to first 50 for readability
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
