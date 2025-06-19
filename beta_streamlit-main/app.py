import streamlit as st
import subprocess
import sys
import os

# Import global translation system
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from global_translations import t, display_language_selector

# Railway port configuration
PORT = int(os.environ.get("PORT", 8501))

st.set_page_config(
    page_title=f"LAISA - {t('housing_assistant')}",
    page_icon=":robot_face:",
    layout="wide"
)

# Display language selector in sidebar
with st.sidebar:
    display_language_selector()

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
    with st.spinner(f"{t('loading')}..."):
        success, stdout, stderr = check_setup()
        st.session_state.setup_checked = True
        st.session_state.setup_success = success
        st.session_state.setup_output = stdout
        st.session_state.setup_error = stderr

# Show setup results
if not st.session_state.setup_success:
    st.error(f"⚠ {t('system_check_failed')}")
    st.markdown("Please fix the following issues before using LAISA:")
    
    if st.session_state.setup_error:
        st.code(st.session_state.setup_error, language="text")
    
    if st.session_state.setup_output:
        with st.expander("Detailed Setup Output"):
            st.code(st.session_state.setup_output, language="text")

    
    if st.button(f"{t('check_system_status')}"):
        st.session_state.setup_checked = False
        st.rerun()
    
    st.stop()
else:
    st.success(f"✅ {t('all_systems_operational')}! LAISA is ready to use.")

st.write(f"# LAISA - {t('housing_assistant')}! :robot_face:")

st.sidebar.success("Navigation")

st.markdown(f"""
    Welcome to LAISA (Legal AI Support Assistant), your housing law and regulation assistant.
    
    ### Features:
    - Ask questions about housing rights and responsibilities
    - Get information about housing regulations
    - Receive guidance on housing-related legal matters
    
    ### How to use:
    1. Navigate to the **Chat** page to start a conversation
    2. Ask your housing-related questions
    3. Review the provided information and sources
    
    **{t('important_note')}**: This assistant provides informational guidance only and should not be considered as legal advice.
    """)