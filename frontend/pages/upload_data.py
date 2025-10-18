import streamlit as st
import time
from datetime import datetime
from file_upload_manager import FileUploadManager
from api_calls import (
    upload_documents_api,
    get_supported_formats_api,
    get_document_categories_api,
    check_loader_health,
    # New async functions
    start_async_upload,
    get_upload_job_status,
    list_upload_jobs,
    cancel_upload_job
)

# Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please log in first to access this page.")
    st.stop()

st.markdown("# Data Upload 📁")
st.sidebar.markdown("# Data Upload 📁")

st.write("""
## Load Documents into Vectorstore

Here you can upload your documents and load them into the vectorstore. 
This allows the optimAIse Chat to access your documents and answer questions.
""")

# Check loader service health
if not check_loader_health():
    st.error("⚠️ Loader service is not available. Please try again later.")
    st.stop()

# --- UPLOAD SECTION ---
st.subheader("📤 Upload New Documents")

# Initialize upload manager
upload_manager = FileUploadManager()

# Get document categories from API
categories_response = get_document_categories_api()
if categories_response:
    document_tags = categories_response["categories"]
else:
    # Fallback tags
    document_tags = [
        "General", "Research & Development", "Marketing & Sales",
        "Production & Manufacturing", "Finance & Controlling", "Human Resources",
        "Legal & Compliance", "IT & Technology", "Quality Management", "Project Documentation"
    ]

# Tag selection
st.write("**Select category for the documents:**")
selected_tag = st.selectbox(
    "Choose a category for the documents to be uploaded:",
    document_tags,
    index=0,
    help="This category will be added to all uploaded documents as metadata"
)

# Upload mode selection
upload_mode = st.radio(
    "Upload Mode:",
    ["🚀 Asynchronous Upload (recommended)", "⚡ Synchronous Upload"],
    index=0,
    help="Asynchronous upload allows you to switch pages during processing."
)

# Advanced settings
with st.expander("🔧 Advanced Settings"):
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.number_input(
            "Chunk Size", 
            min_value=500, 
            max_value=4000, 
            value=2000,
            help="Size of text blocks for processing"
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, 
            max_value=500, 
            value=200,
            help="Overlap between consecutive text blocks"
        )
    
    with col2:
        max_workers = st.number_input(
            "Bulk Upload Workers",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of parallel upload workers for better performance with multiple files"
        )
        use_bulk_upload = st.checkbox(
            "Enable Bulk Upload",
            value=True,
            help="Parallel processing for faster uploads with multiple files"
        )

uploaded_files = st.file_uploader(
    "Select all files you want to upload",
    accept_multiple_files=True,
    help="Tip: Use CTRL+A in the file dialog to select all files from a folder"
)

if uploaded_files:
    # Calculate total size
    total_size = sum(len(file.getvalue()) for file in uploaded_files)
    total_size_mb = total_size / (1024 * 1024)
    
    st.write(f"**{len(uploaded_files)} files selected ({total_size_mb:.2f} MB)**")
    st.write(f"**Category:** {selected_tag}")
    st.write(f"**Mode:** {upload_mode}")
    
    # Show selected files
    with st.expander("Show selected files"):
        for file in uploaded_files:
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            st.write(f"- {file.name} ({file_size_mb:.2f} MB)")
    
    if st.button("🚀 Process Files", type="primary"):
        if "🚀 Asynchronous Upload" in upload_mode:
            # Async upload
            with st.spinner("Starting asynchronous upload..."):
                result = start_async_upload(
                    files=uploaded_files,
                    document_tag=selected_tag,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_workers=max_workers,
                    use_bulk_upload=use_bulk_upload,
                    user_id=st.session_state.get('user_id', 'streamlit_user')
                )
                
                if result and 'job_id' in result:
                    st.success("✅ Asynchronous upload started!")
                    st.info(f"**Job ID:** {result['job_id']}")
                    st.info(f"📁 {result['total_files']} files are being processed")
                    
                    # Store job_id in session state for easy access
                    if 'active_jobs' not in st.session_state:
                        st.session_state['active_jobs'] = []
                    st.session_state['active_jobs'].append(result['job_id'])
                    
                    # Force refresh to show status
                    st.rerun()
                else:
                    st.error("❌ Error starting asynchronous upload")
                    if result:
                        st.error(f"Details: {result}")
        else:
            # Sync upload (original behavior)
            with st.spinner("Processing files synchronously..."):
                try:
                    result = upload_documents_api(
                        files=uploaded_files,
                        document_tag=selected_tag,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        max_workers=max_workers,
                        use_bulk_upload=use_bulk_upload
                    )
                    
                    # Store result in session state so it persists
                    st.session_state['last_sync_upload'] = result
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.session_state['last_sync_upload'] = None

# Show sync upload result if available
if 'last_sync_upload' in st.session_state and st.session_state['last_sync_upload']:
    result = st.session_state['last_sync_upload']
    
    if result.get('success'):
        st.success(f"✅ {result['files_processed']} documents were successfully processed!")
        st.success(f"🏷️ Category '{result.get('document_tag', 'General')}' was assigned to all documents.")
        
        if 'versions_stored' in result:
            st.info(f"📦 {len(result['versions_stored'])} versions stored in MinIO")
        
        st.error("❌ Error uploading documents")
        if 'error_message' in result:
            st.error(f"Details: {result['error_message']}")
        
        if st.button("🔄 Try Again"):
            st.session_state['last_sync_upload'] = None
            st.rerun()

if not uploaded_files and 'last_sync_upload' not in st.session_state:
    st.info("Select files to load them into the vectorstore")

# --- STATUS SECTION ---
st.write("---")
st.subheader("📊 Upload Status")

# Show active upload jobs (if any)
if 'active_jobs' in st.session_state and st.session_state['active_jobs']:
    st.info("🔄 **Active uploads are being processed...**")
    
    active_jobs_to_remove = []
    
    for i, job_id in enumerate(st.session_state['active_jobs']):
        status = get_upload_job_status(job_id)
        
        if 'error' not in status:
            # Create compact status display
            with st.container():
                st.write(f"**📋 Upload #{i+1}:** `{job_id[:8]}...`")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Status indicator
                    status_icons = {
                        'pending': '⏳ Queue',
                        'uploading_to_minio': '📤 MinIO Upload', 
                        'chunking': '✂️ Text Chunking',
                        'embedding': '🧮 Embeddings',
                        'vector_db': '🗃️ Vector DB',
                        'completed': '✅ Completed',
                        'failed': '❌ Failed',
                        'cancelled': '❌ Cancelled'
                    }
                    
                    current_status = status_icons.get(status['status'], status['status'])
                    st.write(f"**Status:** {current_status}")
                    
                    # Progress bar
                    progress = status['progress'] / 100.0
                    st.progress(progress)
                    st.caption(f"{status['progress']:.1f}% completed")
                
                with col2:
                    st.metric("📁 Files", status['total_files'])
                    if status['started_at']:
                        try:
                            started = datetime.fromisoformat(status['started_at'].replace('Z', '+00:00'))
                            st.caption(f"⏰ Started: {started.strftime('%H:%M:%S')}")
                        except Exception:
                            st.caption("⏰ Running...")
                
                with col3:
                    if status['status'] in ['pending', 'uploading_to_minio', 'chunking', 'embedding', 'vector_db']:
                        if st.button("❌ Cancel", key=f"cancel_{job_id}"):
                            cancel_result = cancel_upload_job(job_id)
                            if 'error' not in cancel_result:
                                st.success("Job cancelled")
                                active_jobs_to_remove.append(job_id)
                                time.sleep(1)
                                st.rerun()
                    elif status['status'] == 'completed':
                        st.success("✅ Done!")
                        active_jobs_to_remove.append(job_id)
                    elif status['status'] in ['failed', 'cancelled']:
                        st.error("❌ Error")
                        if status.get('error_message'):
                            st.error(f"Details: {status['error_message']}")
                        active_jobs_to_remove.append(job_id)
            
            st.write("---")
        else:
            # Job not found or error - remove from active list
            active_jobs_to_remove.append(job_id)
    
    # Remove completed/failed jobs from active list
    for job_id in active_jobs_to_remove:
        if job_id in st.session_state['active_jobs']:
            st.session_state['active_jobs'].remove(job_id)
    
    # Auto-refresh every 3 seconds if there are still active jobs
    if st.session_state['active_jobs']:
        st.info("🔄 Auto-refresh in 3 seconds...")
        time.sleep(3)
        st.rerun()
else:
    st.info("💤 No active uploads")
    
    # Show recent jobs option
    if st.button("📋 Show Last 5 Jobs"):
        jobs_data = list_upload_jobs(limit=5)
        
        if 'error' not in jobs_data and jobs_data['jobs']:
            st.write("**📋 Recent Upload Jobs:**")
            
            for job in jobs_data['jobs']:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"`{job['job_id'][:8]}...`")
                    try:
                        created = datetime.fromisoformat(job['created_at'].replace('Z', '+00:00'))
                        st.caption(f"Created: {created.strftime('%d.%m.%Y %H:%M')}")
                    except Exception:
                        st.caption("Created: Unknown")
                
                with col2:
                    status_icon = {
                        'pending': '⏳',
                        'uploading_to_minio': '📤', 
                        'chunking': '✂️',
                        'embedding': '🧮',
                        'vector_db': '🗃️',
                        'completed': '✅',
                        'failed': '❌',
                        'cancelled': '❌'
                    }.get(job['status'], '⏳')
                    st.write(f"{status_icon} {job['status']}")
                
                with col3:
                    st.metric("Files", job['total_files'])
                
                st.write("---")
        else:
            st.info("No jobs found")

# --- FOOTER ---
st.write("---")
formats_response = get_supported_formats_api()
if formats_response:
    st.write("### 📄 Supported File Formats:")
    cols = st.columns(len(formats_response["supported_formats"]))
    for i, fmt in enumerate(formats_response["supported_formats"]):
        with cols[i]:
            st.write(f"**{fmt['extension']}**")
            st.caption(fmt['description'])
else:
    st.write("""
    ### 📄 Supported File Formats:
    **PDF** • **Word (.docx)** • **Text (.txt)** • **Markdown (.md)** • **CSV**
    """)

st.write("""
### ℹ️ How it works:
1. **📁 Select files** → Choose category → Start upload
2. **🔄 Track status** → Progress is displayed automatically below  
3. **✅ When completed** → Go directly to chat and ask questions

**💡 Tip:** With asynchronous upload, you can switch pages - the status is preserved!
""")