import streamlit as st

# Function to handle login
def login():
    # Initialize session state for login if not already set
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Dummy credentials
    valid_username = "user"
    valid_password = "password"

    # If user is not logged in
    if not st.session_state.logged_in:
        st.title("Login")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")
        
        if login_button:
            if username == valid_username and password == valid_password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                st.rerun # Refresh the app to show the next page
            else:
                st.error("Invalid username or password.")
        return False
    else:
        return True
