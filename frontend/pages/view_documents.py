import streamlit as st
from api_calls import get_document_view_url_api, list_minio_documents_api
import uuid

# Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please log in first to access this page.")
    st.stop()

st.markdown("# Document Access 📄")
st.sidebar.markdown("# Document Access 📄")

st.write("""
## Direct Links to Uploaded Documents

Here you can view your uploaded documents directly in the browser.
Documents are provided through the loader service.
""")

# List available documents
st.subheader("Available Documents")

documents_response = list_minio_documents_api()
if documents_response and documents_response.get("objects"):
    documents = documents_response["objects"]
    
    # Group documents by category and name
    doc_structure = {}
    for obj in documents:
        if obj["object_name"].startswith("documents/"):
            parts = obj["object_name"].split("/")
            if len(parts) >= 5:  # documents/category/name/version/file
                category = parts[1].replace("_", " ")
                doc_name = parts[2].replace("_", " ")
                
                if category not in doc_structure:
                    doc_structure[category] = {}
                if doc_name not in doc_structure[category]:
                    doc_structure[category][doc_name] = []
                
                doc_structure[category][doc_name].append({
                    "object_name": obj["object_name"],
                    "size": obj["size"],
                    "last_modified": obj["last_modified"]
                })
    
    # Display documents by category
    for category, documents in doc_structure.items():
        with st.expander(f"📁 {category} ({len(documents)} documents)"):
            for doc_index, (doc_name, versions) in enumerate(documents.items()):
                latest_version = max(versions, key=lambda x: x["last_modified"])
                file_size_mb = latest_version["size"] / (1024 * 1024)
                
                # Use category + doc_index for unique keys
                view_key = f"view_{category}_{doc_index}"
                
                col1, col2, col3= st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{doc_name}**")
                    st.caption(f"{file_size_mb:.2f} MB • {len(versions)} version(s)")
                
                with col2:
                    obj_parts = latest_version["object_name"].split("/")
                    original_doc_name = obj_parts[2]
                    view_response = get_document_view_url_api(
                        document_name=original_doc_name,
                        document_tag=category.replace(" ", "_")
                    )
                    if view_response:
                        # Create a button that opens document in new tab using JavaScript
                        st.markdown(
                            f'''
                            <a href="{view_response["view_url"]}" target="_blank" style="text-decoration:none;">
                                <button style="
                                    background-color: #FF6B6B;
                                    color: white;
                                    padding: 10px 15px;
                                    border: none;
                                    border-radius: 5px;
                                    cursor: pointer;
                                    font-size: 14px;
                                    font-weight: bold;
                                    width: 100%;
                                    transition: background-color 0.3s;
                                " onmouseover="this.style.backgroundColor='#FF5252'" onmouseout="this.style.backgroundColor='#FF6B6B'">
                                    View in Browser
                                </button>
                            </a>
                            ''',
                            unsafe_allow_html=True
                        )

                with col3:
                    st.metric("Version", len(versions))

else:
    st.info("No documents available. First upload documents via the upload page.")

st.write("---")

# Manual document access
st.subheader("Manual Document Access")

with st.form("manual_access"):
    col1, col2 = st.columns(2)
    
    with col1:
        manual_doc_name = st.text_input(
            "Document Name",
            placeholder="e.g. RAGOps__Operating_and_Managing_RAG_Pipelines",
            help="Use the exact name as it is stored in MinIO"
        )
    
    with col2:
        manual_category = st.selectbox(
            "Category",
            ["General", "Research_&_Development", "Marketing_&_Sales", 
             "Production_&_Manufacturing", "Finance_&_Controlling", "Human_Resources",
             "Legal_&_Compliance", "IT_&_Technology", "Quality_Management", "Project_Documentation"]
        )
    
    col1, col2 = st.columns(2)
    with col1:
        submit_view = st.form_submit_button("👁️ Open in Browser")
    with col2:
        submit_link = st.form_submit_button("📋 Show Link")
    
    if (submit_view or submit_link) and manual_doc_name:
        view_response = get_document_view_url_api(
            document_name=manual_doc_name,
            document_tag=manual_category
        )
        
        if view_response:
            if submit_view:
                st.success("✅ Document is being opened!")
                st.markdown(
                    f'<a href="{view_response["view_url"]}" target="_blank">🔗 Open document in new tab</a>',
                    unsafe_allow_html=True
                )
            
            if submit_link:
                st.success("✅ Direct link generated!")
                st.code(view_response["view_url"], language="text")
            
            # Display additional information
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dokument", view_response["document_name"])
            with col2:
                st.metric("Kategorie", view_response["document_tag"])
            
            st.info("💡 URL wird über Loader-Service geroutet")

st.write("---")
st.write("""
### ℹ️ Notes on Document Access

- **Direct Display**: Documents are opened directly in the browser
- **PDF Support**: PDFs are displayed inline in the browser
- **Proxy Service**: Access via loader service (avoids signature issues)
- **Original Documents**: Direct access to uploaded files from MinIO
- **No Time Restriction**: URLs remain valid as long as the service is running
""")