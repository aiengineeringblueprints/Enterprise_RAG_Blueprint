import streamlit as st


st.write("# Welcome to RAG Blueprint")
st.write(
    "RAG Blueprint is an app that can automate your organizations. "
    "Simply upload your documents and let the app do the work. "
    "It extracts the most important information from your documents and automates your workflows. "
    "It also provides insights to improve decision-making. "
    "Experience the power of automation and data-driven insights!"
)

# Navigation
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.info("Please log in first!")
    if st.button("Login Page"):
        st.switch_page("pages/auth_page.py")  # make sure your login is named login_page.py
else:
    st.success(f"Logged in as {st.session_state.username} ({st.session_state.roles})")
    if st.button("Chat"):
        st.switch_page("pages/rag_chat.py")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.roles = None
        st.session_state.username = None
        st.success("Logged out!")
