import streamlit as st
import api_calls
import html
import re
from chat.chat_manager import ChatManager


if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please log in to access the chat.")
    st.stop()

# Get user information including roles
user_info = st.session_state.get('user_info', {})
user_roles = user_info.get('roles', [])
username = user_info.get('username', st.session_state.get('username', 'unknown'))

# Display user role information
if user_roles:
    st.sidebar.write(f"**User:** {username}")
    st.sidebar.write(f"**Roles:** {', '.join(user_roles)}")
    st.sidebar.write("---")
else:
    st.sidebar.write(f"**User:** {username}")
    st.sidebar.write("**Roles:** None assigned")
    st.sidebar.write("---")


st.markdown("# optimAIse Chat 💬")
st.sidebar.markdown("# optimAIse Chat 💬")

# Create columns for chat and document display
col1, col2 = st.columns([2, 1])  # Chat nimmt 2/3, Dokumente 1/3
# Create columns for chat and document display
col1, col2 = st.columns([2, 1])  # Chat nimmt 2/3, Dokumente 1/3

with col1:
    # Chat Manager for chat history
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 1  # For automatically generated titles
    if "roles" not in st.session_state:
        st.session_state.roles = user_roles  # Store roles in session state
    if "current_relevant_docs" not in st.session_state:
        st.session_state.current_relevant_docs = []
    if "show_full_document" not in st.session_state:
        st.session_state.show_full_document = False
    if "full_document_content" not in st.session_state:
        st.session_state.full_document_content = ""
    if "selected_document_source" not in st.session_state:
        st.session_state.selected_document_source = None


    user = st.session_state.username
    chat_manager = st.session_state.chat_manager
    user = st.session_state.username
    chat_manager = st.session_state.chat_manager

with col2:
    st.markdown("### 📄 Relevant Documents")
    if st.session_state.current_relevant_docs:
        st.markdown("*Used for the last answer:*")
        
        # Group documents by filename
        docs_by_file = {}
        for doc_string in st.session_state.current_relevant_docs:
            # Parse the document string format: "source->content"
            if "->" in doc_string:
                doc_source, doc_content = doc_string.split("->", 1)
                # Extract filename from path
                doc_name = doc_source
                if '\\' in doc_name:
                    doc_name = doc_name.split('\\')[-1]
                elif '/' in doc_name:
                    doc_name = doc_name.split('/')[-1]
            else:
                # Fallback if format is different
                doc_name = f"Unknown Source"
                doc_content = doc_string
            
            # Group by filename
            if doc_name not in docs_by_file:
                docs_by_file[doc_name] = []
            docs_by_file[doc_name].append(doc_content)
        
        # Initialize selected document in session state if not exists
        if "show_full_document" not in st.session_state:
            st.session_state.show_full_document = False
        if "full_document_content" not in st.session_state:
            st.session_state.full_document_content = ""
        if "selected_document_source" not in st.session_state:
            st.session_state.selected_document_source = None
        
        # Display unique document files as clickable items
        for doc_name in docs_by_file.keys():
            # Create a button for each document to show full document
            col_doc1, col_doc2 = st.columns([3, 1])
            with col_doc1:
                # Sanitize key to avoid invalid DOM id characters
                safe_key = re.sub(r"[^A-Za-z0-9_-]", "_", doc_name)
                if st.button(f"📄 {doc_name}", key=f"full_doc_btn_{safe_key}", use_container_width=True, help="Show full document"):
                    # Get document source path for API call
                    doc_source = None
                    for doc_string in st.session_state.current_relevant_docs:
                        if "->" in doc_string:
                            source, _ = doc_string.split("->", 1)
                            current_name = source
                            if '\\' in current_name:
                                current_name = current_name.split('\\')[-1]
                            elif '/' in current_name:
                                current_name = current_name.split('/')[-1]
                            
                            if current_name == doc_name:
                                doc_source = source
                                break
                    
                    if doc_source:
                        with st.spinner("Loading full document..."):
                            doc_response = api_calls.get_full_document(doc_source)
                            if doc_response and doc_response.get("success"):
                                st.session_state.full_document_content = doc_response["content"]
                                st.session_state.selected_document_source = doc_source
                                st.session_state.show_full_document = True
                                st.rerun()
                            else:
                                st.error("Document could not be loaded.")
                                if doc_response:
                                    st.error(f"Error message: {doc_response.get('error', 'Unknown error')}")
                    else:
                        st.error("Document source not found.")
            with col_doc2:
                st.caption(f"({len(docs_by_file[doc_name])} excerpts)")
    else:
        st.info("No documents found for the current answer.")

# Show full document
if st.session_state.show_full_document and st.session_state.full_document_content:
    st.markdown("# 📖 Document Viewer")
    
    # Big close button at top
    if st.button("❌ BACK TO CHAT", key="back_to_chat_top", use_container_width=True, type="primary"):
        st.session_state.show_full_document = False
        st.session_state.selected_document_source = None
        st.rerun()
    
    st.markdown("---")
    
    # Get document name
    doc_name = "Unknown Document"
    if st.session_state.selected_document_source:
        source = st.session_state.selected_document_source
        if '\\' in source:
            doc_name = source.split('\\')[-1]
        elif '/' in source:
            doc_name = source.split('/')[-1]
    
    st.markdown(f"## 📄 {doc_name}")
    
    # Show document content with highlighting
    full_content = st.session_state.full_document_content.strip()
    if not full_content:
        st.error("Document content is empty or could not be loaded.")
    else:
        # Get excerpts for highlighting - find the right document excerpts
        excerpts_for_doc = []
        if hasattr(st.session_state, 'current_relevant_docs') and st.session_state.current_relevant_docs:
            for doc_string in st.session_state.current_relevant_docs:
                if "->" in doc_string:
                    source, content = doc_string.split("->", 1)
                    # Check if this excerpt belongs to the current document
                    current_name = source.strip()
                    if '\\' in current_name:
                        current_name = current_name.split('\\')[-1]
                    elif '/' in current_name:
                        current_name = current_name.split('/')[-1]
                    
                    if current_name == doc_name:
                        excerpts_for_doc.append(content.strip())
        
        # Escape content before injecting into HTML
        escaped_full_content = html.escape(full_content)
        highlighted_content = escaped_full_content
        highlight_count = 0
        
        if excerpts_for_doc:
            for excerpt in excerpts_for_doc:
                if len(excerpt) > 20:  # Only highlight meaningful excerpts
                    escaped_excerpt = html.escape(excerpt)
                    if escaped_excerpt in escaped_full_content:
                        highlighted_content = highlighted_content.replace(
                            escaped_excerpt,
                            f'<mark style="background-color: yellow; padding: 2px;">{escaped_excerpt}</mark>'
                        )
                        highlight_count += 1
        
        st.info(f"Document loaded: {len(full_content):,} characters | {highlight_count} sections highlighted")
        
        # Show highlighted content as HTML
        if highlight_count > 0:
            st.markdown(f"**📍 {highlight_count} relevant sections are highlighted in yellow:**")
            st.markdown(
                f'<div style="border: 1px solid #ddd; padding: 15px; background-color: #f9f9f9; '
                f'height: 400px; overflow-y: auto; white-space: pre-wrap; font-family: monospace;">'
                f'{highlighted_content}</div>',
                unsafe_allow_html=True
            )
        else:
            # Fallback to simple text area if no highlights
            st.text_area("Document content:", full_content, height=400, disabled=True)
    
    # Close button at bottom too
    st.markdown("---")
    if st.button("❌ BACK TO CHAT", key="back_to_chat_bottom", use_container_width=True, type="primary"):
        st.session_state.show_full_document = False
        st.session_state.selected_document_source = None
        st.rerun()

    st.stop()

# Continue with col1 for chat functionality
with col1:
    with st.sidebar:
        st.write("## Your Chats")
        # Add "New Chat" button at the top
        if st.button("➕ New Chat"):
            st.session_state.active_chat_id = None
            st.session_state.current_relevant_docs = []
            st.session_state.show_full_document = False
            st.session_state.full_document_content = ""
            st.session_state.selected_document_source = None
            st.rerun()
        
        # List existing chats
        chats = chat_manager.list_chats(user)
        for chat in chats:
            if st.button(chat["title"], key=chat["chat_id"]):
                st.session_state.active_chat_id = chat["chat_id"]
                st.rerun()

    # Main area: Show current chat and chat history
    chat_id = st.session_state.active_chat_id

    # Display messages (only if chat is selected)
    if chat_id:
        messages = chat_manager.get_messages(chat_id)
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat Input
    if question := st.chat_input("Chat with your documents"):
        # If no active chat: create a new one!
        if not chat_id:
            # Title by number, e.g. "Chat 1", "Chat 2", ...
            title = f"Chat {len(chat_manager.list_chats(user)) + 1}"
            chat_id = chat_manager.create_chat(user, title)
            st.session_state.active_chat_id = chat_id
            # Save and display message
        chat_manager.add_message(chat_id, "user", question)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
           
           # call LLM API 
            answer, relevant_documents, promt_key = api_calls.llm_api(
                question=question, 
                prompt_key="rag_source", 
                show_sources=True, 
                user_roles=user_roles)
            
            # Store relevant documents in session state for display
            st.session_state.current_relevant_docs = relevant_documents or []
            
            fact_checker_answer = api_calls.check_answer_api(
                question,
                answer=answer,
                prompt_template="rag_check",
                question_prompt_key="rag_source",
                relevant_documents=relevant_documents
            )

            keyword_check = api_calls.check_keywords_api(question, answer, expected_keywords=None, threshold=0.6)

            # Prepare the response text
            if "yes" in fact_checker_answer["evaluation"] and "yes" in keyword_check["evaluation"]:
                response_text = answer + " DOUBLE CHECKED FROM CONTEXT"
            else:
                response_text = answer + " CHECK: " + fact_checker_answer["evaluation"] + " KEYWORD: " + keyword_check["evaluation"]

            # Display the response
            st.markdown(response_text)
            
            # Store the response in the database
            chat_manager.add_message(chat_id, "assistant", response_text)
            
        st.rerun()

    elif not chat_id:
        st.write("This is the optimAIse Chat.")
        st.write("Here you can chat with your documents.")
        st.write("Select a chat on the left or write a message to start a new chat!")