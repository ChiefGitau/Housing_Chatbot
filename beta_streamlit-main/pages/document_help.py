import streamlit as st
import tempfile
import os
from typing import Optional, Dict, List
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import global translation system
from global_translations import t, display_language_selector

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import language consistency functions
from core import get_session_language, ensure_response_language

# Page configuration
st.set_page_config(
    page_title=f"{t('document_help')} - LAISA",
    page_icon="📄",
    layout="wide"
)

# Display language selector in sidebar
with st.sidebar:
    display_language_selector()

st.header(t('document_help'))
st.markdown(f"**{t('upload_analysis')}**")

def load_pdf_document(uploaded_file) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Combine all pages
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Clean up
        os.unlink(tmp_path)
        
        return full_text
    except Exception as e:
        st.error(f"{t('failed_read_pdf')}: {e}")
        st.info(t('try_different_method'))
        return None

def analyze_document(document_text: str) -> Dict[str, str]:

    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        llm = ChatOpenAI(
            temperature=0, 
            model="gpt-4",
            api_key=api_key
        )
        
        # Get summary
        summary_prompt = f"""
        Analyze this document and provide:
        1. A brief summary (2-3 sentences)
        2. Document type (lease agreement, rental application, etc.)
        3. Any fields that need to be filled in by the user
        4. Important sections or clauses to pay attention to
        
        Document text:
        {document_text[:4000]}...
        
        Respond in JSON format:
        {{
            "summary": "Brief summary here",
            "document_type": "Type of document",
            "fields_to_fill": ["field1", "field2"],
            "important_sections": ["section1", "section2"]
        }}
        """
        
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        
        # Get section explanations
        explanation_prompt = f"""
        Break down this document into main sections and explain what each section means in simple terms.
        Focus on housing/legal terminology that might be confusing.
        
        Document text:
        {document_text[:4000]}...
        
        Provide explanations as a numbered list where each item explains a major section or clause.
        """
        
        explanation_response = llm.invoke([HumanMessage(content=explanation_prompt)])
        
        # Ensure responses are in the correct language for consistency
        session_lang = get_session_language()
        target_language = session_lang.get("language", "en")
        
        summary_content = ensure_response_language(summary_response.content, target_language)
        explanation_content = ensure_response_language(explanation_response.content, target_language)
        
        return {
            "summary": summary_content,
            "explanations": explanation_content
        }
        
    except Exception as e:
        st.error(f"Failed to analyze document: {e}")
        return {
            "summary": "Analysis unavailable - API error",
            "explanations": "Section explanations unavailable - API error"
        }

def parse_json_response(response_text: str) -> Dict:

    try:
        import json
        # Try to extract JSON from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = response_text[start:end]
            return json.loads(json_str)
    except:
        pass
    
    # Fallback to manual parsing
    return {
        "summary": "Document uploaded successfully. AI analysis available below.",
        "document_type": "Document",
        "fields_to_fill": ["Check document for blank fields"],
        "important_sections": ["Review all sections carefully"]
    }

# File uploader
uploaded_file = st.file_uploader(
    t('upload_document'),
    type=['pdf'],
    help=t('pdf_only_supported')
)

if uploaded_file is not None:
    # Show language consistency info
    session_lang = get_session_language()
    if session_lang["language"] != "en":
        st.info(f"Analysis will be provided in: {session_lang['language'].upper()} (Source: {session_lang['source']})")
    
    # Load document
    with st.spinner(f"{t('loading_document')}..."):
        document_text = load_pdf_document(uploaded_file)
    
    if document_text:
        # Analyze document
        with st.spinner(f"{t('analyzing_document')}..."):
            analysis = analyze_document(document_text)

        summary_data = parse_json_response(analysis["summary"])
        

        st.subheader(t('document_summary'))
        st.info(summary_data.get("summary", "Document analysis completed"))
        

        col1, col2 = st.columns(2)
        with col1:
            st.metric(t('document_type'), summary_data.get("document_type", "Unknown"))
        with col2:
            fields_count = len(summary_data.get("fields_to_fill", []))
            st.metric(t('fields_to_fill'), fields_count)
        
        # Show fields to fill if any
        if summary_data.get("fields_to_fill"):
            with st.expander("Fields That Need Your Attention"):
                for field in summary_data["fields_to_fill"]:
                    st.write(f"• {field}")
        
        # Show important sections
        if summary_data.get("important_sections"):
            with st.expander("Important Sections to Review"):
                for section in summary_data["important_sections"]:
                    st.write(f"• {section}")
        
        st.divider()
        
        # Main content: Document on left, explanations on right
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("Document Content")
            # Display document in a scrollable container
            st.text_area(
                "Full Document Text",
                value=document_text,
                height=600,
                disabled=True,
                label_visibility="collapsed"
            )
        
        with col_right:
            st.subheader(t('section_explanations'))
            # Display explanations
            st.markdown(analysis["explanations"])
            
            # Add help section
            st.divider()
            st.subheader(t('need_more_help'))
            st.markdown(f"""
            **{t('questions_about_document')}**
            - {t('go_to_chat')}
            - {t('upload_to_kb')}
            - {t('consult_legal')}
            
            **{t('common_questions')}:**
            - "What does [specific clause] mean?"
            - "Are these terms fair and legal?"
            - "What are my rights regarding [specific section]?"
            - "What should I do before signing this?"
            """)

else:
    # Instructions when no document is uploaded
    st.info(t('upload_pdf_to_start'))
    


# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    {t('document_analysis')}
</div>
""", unsafe_allow_html=True)