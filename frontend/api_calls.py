import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CHAIN_URL = os.getenv("CHAIN_URL")
LOADER_URL = os.getenv("LOADER_URL")


def get_document_view_url_api(
    document_name: str, document_tag: str = "General", version: int = None
):
    """
    Get document view URL through the loader proxy endpoint.
    This function returns a URL that proxies the document through the loader service,
    avoiding signature issues with presigned URLs.

    Note: Uses localhost:8002 for browser access since the browser cannot resolve
    internal Docker hostnames like 'loader'.
    """
    try:
        # For browser access, we need to use localhost, not the internal docker hostname
        # The browser cannot resolve 'loader', but it can access localhost:8002
        browser_loader_url = "http://localhost:8002/"

        # Build URL for loader proxy endpoint
        if version:
            url = f"{browser_loader_url}document/view/{document_name}?document_tag={document_tag}&version={version}"
        else:
            url = f"{browser_loader_url}document/view/{document_name}?document_tag={document_tag}"

        return {
            "view_url": url,
            "document_name": document_name,
            "document_tag": document_tag,
            "version": version,
        }
    except Exception as e:
        st.error(f"Error generating view URL: {str(e)}")
        return None


def list_minio_documents_api():
    """
    List all documents available in MinIO through the loader API.
    """
    try:
        url = f"{LOADER_URL}minio/objects"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(
                f"Error listing documents: {response.status_code} - {response.text}"
            )
            return None
    except Exception as e:
        st.error(f"Error connecting to loader API: {str(e)}")
        return None


def llm_api(
    question,
    prompt_key="rag_source",
    show_sources=True,
    used_model="",
    user_roles=["Allgemein"],
    chat_history=None,
):
    url = CHAIN_URL + "call_llm"

    # Convert chat_history to API format if provided
    formatted_history = []
    if chat_history:
        for msg in chat_history:
            formatted_history.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )

    payload = {
        "question": question,
        "prompt_key": prompt_key,
        "used_model": used_model,
        "show_sources": show_sources,
        "user_roles": user_roles,
        "chat_history": formatted_history,
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return (
                data["result"],
                data["relevant_documents"],
                data["prompt_key"],
            )
        elif response.status_code == 400:
            # Guardrail blocked the input - return user-friendly message without showing error banner
            try:
                error_detail = response.json().get(
                    "detail",
                    "Ihre Eingabe wurde durch die Sicherheitsrichtlinien blockiert.",
                )
            except:
                error_detail = "Ihre Eingabe wurde durch die Sicherheitsrichtlinien blockiert."
            # Return the message directly without st.error()
            return error_detail, [], prompt_key
        else:
            st.error(
                f"LLM API Error: {response.status_code} - {response.text}"
            )
            return f"API Error: {response.status_code}", [], prompt_key
    except requests.exceptions.RequestException as e:
        st.error(f"Verbindungsfehler: {str(e)}")
        return "Es ist ein Verbindungsfehler aufgetreten.", [], prompt_key


def check_answer_api(
    question,
    answer,
    relevant_documents,
    question_prompt_key,
    prompt_template="rag_check",
    used_model="",
):
    url = CHAIN_URL + "check_answer"  # Adjust if your FastAPI runs elsewhere
    payload = {
        "question": question,
        "answer": answer,
        "relevant_documents": relevant_documents,
        "prompt_key": prompt_template,  # Changed from prompt_template to prompt_key
        "question_prompt_key": question_prompt_key,
        "used_model": used_model,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(
            f"Check Answer API Error: {response.status_code} - {response.text}"
        )
        return {"evaluation": f"API Error: {response.status_code}"}


def check_keywords_api(
    question, answer, expected_keywords=None, threshold=0.6
):
    url = CHAIN_URL + "check_keywords"  # Adjust if your FastAPI runs elsewhere
    payload = {
        "question": question,
        "answer": answer,
        "expected_keywords": expected_keywords or [],
        "threshold": threshold,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(
            f"Check Keywords API Error: {response.status_code} - {response.text}"
        )
        return {"evaluation": f"API Error: {response.status_code}"}


def upload_documents_api(
    files,
    document_tag="General",
    chunk_size=2000,
    chunk_overlap=200,
    max_workers=5,
    use_bulk_upload=True,
):
    """
    Upload documents to the loader service API via multipart upload with bulk upload support.
    Only use this for small files or when shared volume is not available

    Args:
        files: List of file objects from Streamlit file_uploader
        document_tag: Category tag for the documents
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        max_workers: Maximum number of parallel upload workers for bulk operations
        use_bulk_upload: Whether to use parallel bulk upload (recommended for multiple files)

    Returns:
        dict: API response or None on error
    """
    url = LOADER_URL + "upload-documents"

    # Prepare files for multipart upload
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), file.type)))

    # Prepare form data - use query parameters instead of form data
    params = {
        "document_tag": document_tag,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_workers": max_workers,
        "use_bulk_upload": use_bulk_upload,
    }

    try:
        response = requests.post(url, files=files_data, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Loader API Error: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error to loader service: {str(e)}")
        return None


# Async Upload Functions
def start_async_upload(
    files,
    document_tag="General",
    chunk_size=2000,
    chunk_overlap=200,
    max_workers=5,
    use_bulk_upload=True,
    user_id=None,
):
    """
    Start asynchronous document upload with complete processing pipeline.
    Returns job information for status tracking.
    """
    url = LOADER_URL + "upload-documents-async"

    # Prepare files for multipart upload
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), file.type)))

    # Prepare form data
    params = {
        "document_tag": document_tag,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_workers": max_workers,
        "use_bulk_upload": use_bulk_upload,
        "user_id": user_id or "streamlit_user",
    }

    try:
        response = requests.post(url, files=files_data, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Async Upload API Error: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error to loader service: {str(e)}")
        return None


def get_upload_job_status(job_id):
    """Get status of an upload job"""
    url = LOADER_URL + f"upload-jobs/{job_id}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"error": "Job not found"}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def list_upload_jobs(limit=20, status_filter=None):
    """List all upload jobs with optional status filter"""
    url = LOADER_URL + "upload-jobs"
    params = {"limit": limit}
    if status_filter:
        params["status_filter"] = status_filter

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def cancel_upload_job(job_id):
    """Cancel a running upload job"""
    url = LOADER_URL + f"upload-jobs/{job_id}"

    try:
        response = requests.delete(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def get_full_document(document_source: str) -> str:
    """
    Get the full content of a document by its source path
    """
    import urllib.parse

    # URL encode the document source to handle special characters and paths
    encoded_source = urllib.parse.quote(document_source, safe="")
    url = LOADER_URL + f"get_full_document/{encoded_source}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(
                f"Document API Error: {response.status_code} - {response.text}"
            )
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error to loader service: {str(e)}")
        return None


def get_supported_formats_api():
    """Get supported file formats from loader API"""
    url = LOADER_URL + "supported_formats"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def get_document_categories_api():
    """Get available document categories from loader API"""
    url = LOADER_URL + "document_categories"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def check_loader_health():
    """Check if loader service is healthy"""
    url = LOADER_URL + "health"
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
