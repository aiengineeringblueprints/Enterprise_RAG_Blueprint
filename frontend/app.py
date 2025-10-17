import streamlit as st

# Enable auto-reloading on file changes
st.set_page_config(
    page_title="optimAIse",
    page_icon="🏠",
    layout="wide"
)

# Configure auto-reload
st.config.set_option('server.runOnSave', True)

# call this function with "streamlit run .\frontend\app.py" in the terminal

main_page = st.Page("pages/main_page.py", title="optimAIse", icon="🏠")
rag_chat_page = st.Page("pages/rag_chat.py", title="optimAIse Chat", icon="💬")
datenupload_page = st.Page("pages/upload_data.py", title="Data Upload", icon="📁")
dokumentzugriff_page = st.Page("pages/view_documents.py", title="Document Access", icon="📄")
auth_page = st.Page("pages/auth_page.py", title="Login", icon="🔒")
# Set up navigation
pg = st.navigation([main_page, rag_chat_page, datenupload_page, dokumentzugriff_page, auth_page])

pg.run()

# note: run with streamlit run frontend/app.py --server.runOnSave=true
