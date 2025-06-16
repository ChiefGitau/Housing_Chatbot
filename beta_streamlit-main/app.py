import streamlit as st
import subprocess
import sys
import os

st.set_page_config(
    page_title="LAISA - Housing Assistant",
    page_icon=":robot_face:",
    layout="wide"
)

# Run setup check on first load
def check_setup():
    """Run setup test and return results."""
    try:
        # Run setup_test.py and capture output
        result = subprocess.run(
            [sys.executable, "setup_test.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Setup test timed out"
    except Exception as e:
        return False, "", str(e)

# Check if setup test should be run
if 'setup_checked' not in st.session_state:
    st.session_state.setup_checked = False

if not st.session_state.setup_checked:
    with st.spinner("Checking system setup..."):
        success, stdout, stderr = check_setup()
        st.session_state.setup_checked = True
        st.session_state.setup_success = success
        st.session_state.setup_output = stdout
        st.session_state.setup_error = stderr

# Show setup results
if not st.session_state.setup_success:
    st.error("⚠ Setup Check Failed")
    st.markdown("Please fix the following issues before using LAISA:")
    
    if st.session_state.setup_error:
        st.code(st.session_state.setup_error, language="text")
    
    if st.session_state.setup_output:
        with st.expander("Detailed Setup Output"):
            st.code(st.session_state.setup_output, language="text")

    
    if st.button("Re-run Setup Check"):
        st.session_state.setup_checked = False
        st.rerun()
    
    st.stop()
else:
    st.success("✅ System setup verified! LAISA is ready to use.")

st.write("# LAISA - Housing Assistant! :robot_face:")

st.sidebar.success("Navigation")

st.markdown(
    """
    Welcome to LAISA (Legal AI Support Assistant), your housing law and regulation assistant.
    
    ### Features:
    - Ask questions about housing rights and responsibilities
    - Get information about housing regulations
    - Receive guidance on housing-related legal matters
    
    ### How to use:
    1. Navigate to the **Chat** page to start a conversation
    2. Ask your housing-related questions
    3. Review the provided information and sources
    
    **Note**: This assistant provides informational guidance only and should not be considered as legal advice.
    """
)