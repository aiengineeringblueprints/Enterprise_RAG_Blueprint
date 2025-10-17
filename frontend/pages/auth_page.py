import streamlit as st
from authentication.auth_service import AuthService

auth_service = AuthService()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_info = None
    st.session_state.username = None
    st.session_state.roles = None

st.title("Login")

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_info = auth_service.authenticate(username, password)
        if user_info:
            st.session_state.logged_in = True
            st.session_state.user_info = user_info
            st.session_state.username = user_info['username']
            
            roles_display = ', '.join(user_info['roles']) if user_info['roles'] else 'No roles assigned'
            st.success(f"Logged in as {user_info['username']} (Roles: {roles_display})")
            st.switch_page("pages/main_page.py")
        else:
            st.error("Invalid credentials")

    st.markdown("---")
    if auth_service.user_count() == 0:
        st.info("No users found. Create the initial admin account:")
        new_user = st.text_input("Admin Username")
        new_pw = st.text_input("Admin Password", type="password")
        if st.button("Create Admin"):
            if auth_service.create_user(new_user, new_pw, "admin", admin=True):
                st.success("Admin user created!")
            else:
                st.error("Failed to create admin user (maybe username taken).")
else:
    user_info = st.session_state.user_info
    roles_display = ', '.join(user_info['roles']) if user_info['roles'] else 'No roles assigned'
    st.write(f"Welcome, {user_info['username']} (Roles: {roles_display})")
    
    if "admin" in user_info.get('roles', []) or user_info.get('role_string') == "admin":
        st.markdown("### Create new user")
        new_user = st.text_input("New Username")
        new_pw = st.text_input("New Password", type="password")
        
        # Role selection with multiple options
        st.markdown("**Available Roles (select multiple):**")
        available_roles = [
            "Marketing & Sales",
            "IT & Technology", 
            "Human Resources",
            "Finance & Controlling",
            "Project Documentation",
            "Production & Manufacturing",
            "Research & Development",
            "Legal & Compliance",
            "Quality Management",
            "General"
        ]
        
        selected_roles = []
        for role in available_roles:
            if st.checkbox(role, key=f"role_{role}"):
                selected_roles.append(role)
        
        # Admin role option
        is_admin = st.checkbox("Admin privileges")
        
        if st.button("Create User"):
            # Combine selected roles
            role_string = ", ".join(selected_roles)
            if is_admin:
                role_string = "admin" if not role_string else f"admin, {role_string}"
            
            if auth_service.create_user(new_user, new_pw, role_string, admin=True):
                st.success(f"User created with roles: {role_string}")
            else:
                st.error("Failed to create user.")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_info = None
        st.session_state.username = None
        st.session_state.roles = None
        st.rerun()
