import os
import streamlit as st
import tempfile
import datetime
from typing import List, Optional

# Import processing functions from backend
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import global translation system
from global_translations import t, display_language_selector

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from pinecone import Pinecone

# Page configuration
st.set_page_config(
    page_title=f"{t('admin_only')} {t('data_upload')} - LAISA",
    page_icon="🔒",
    layout="wide"
)

# Display language selector in sidebar
with st.sidebar:
    display_language_selector()

# Admin section layout
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(f"### 🔒 {t('admin_only')}")
    st.markdown(f"**{t('data_upload')} Panel**")
    st.markdown(f"*{t('restricted_access')}*")
    st.divider()
    
    # PIN Authentication
    st.markdown(f"**{t('enter_admin_pin')}:**")
    admin_pin = st.text_input("PIN", type="password", max_chars=4, help="Enter the 4-digit admin PIN")
    
    # Check PIN
    if admin_pin:
        if admin_pin == "0000":
            st.session_state.admin_authenticated = True
            st.success(f"✅ {t('access_granted')}")
        else:
            st.session_state.admin_authenticated = False
            st.error(f"❌ {t('invalid_pin')}")
    else:
        st.session_state.admin_authenticated = False
    
    st.divider()
    
    # Admin status and info
    if st.session_state.get('admin_authenticated', False):
        st.markdown(f"**📊 {t('admin_information')}:**")
        st.info("Authenticated")
        
        st.markdown("**🔧 Functions:**")
        st.markdown(f"""
        - {t('upload_document')}
        - System diagnostics
        - Knowledge base management
        """)
    else:
        st.markdown("**ℹ️ Access Requirements:**")
        st.markdown("""
        - Valid admin PIN required
        - Authorized personnel only
        - All actions are logged
        """)
    
    st.divider()
    st.caption("⚠️ Unauthorized access prohibited")
    st.caption("📞 Contact IT for access issues")

with col2:
    if st.session_state.get('admin_authenticated', False):
        st.header(f"📁 {t('upload_document')} to Knowledge Base")
        st.markdown(f"**{t('upload_analysis')}**")
    else:
        st.header(f"🔒 {t('access_denied')}")
        st.warning(f"{t('enter_admin_pin')} to access the document upload functionality.")
        st.info("This page is for system administrators only.")
        st.stop()

def process_uploaded_files(uploaded_files) -> bool:
    """Process uploaded files in the background with minimal UI."""
    
    # Fixed configuration (no user inputs needed)
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    INDEX_NAME = "small-blogs-emmbeddings-index"
    NAMESPACE = None  # Use default namespace
    
    try:
        # Initialize services
        with st.spinner(f"{t('loading')}..."):
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        all_chunks = []
        
        # Process all files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"{t('processing')} {uploaded_file.name}...")
            
            # Load document
            documents = load_document(uploaded_file)
            if not documents:
                continue
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "source_file": uploaded_file.name,
                    "file_type": uploaded_file.name.split('.')[-1].lower(),
                    "upload_timestamp": str(datetime.datetime.now())
                })
            
            all_chunks.extend(chunks)
            progress_bar.progress((i + 1) / len(uploaded_files) * 0.7)
        
        if not all_chunks:
            st.error("No documents could be processed")
            return False
        
        # Upload to Pinecone
        status_text.text(f"{t('upload_document')} {len(all_chunks)} chunks to Pinecone...")
        
        # Try LangChain first, fallback to direct upload
        try:
            vectorstore = PineconeLangChain.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                index_name=INDEX_NAME,
                namespace=NAMESPACE
            )
            upload_method = "LangChain"
        except Exception as e:
            logger.warning(f"LangChain upload failed: {e}")
            # Direct upload fallback
            index = pc.Index(INDEX_NAME)
            batch_size = 100
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                vectors = []
                
                for j, chunk in enumerate(batch):
                    chunk_id = f"upload-{int(datetime.datetime.now().timestamp())}-{i}-{j}"
                    embedding = embeddings.embed_query(chunk.page_content)
                    metadata = chunk.metadata.copy()
                    metadata['text'] = chunk.page_content
                    vectors.append((chunk_id, embedding, metadata))
                
                index.upsert(vectors=vectors, namespace=NAMESPACE)
            
            upload_method = "Direct"
        
        progress_bar.progress(1.0)
        status_text.text(f"{t('upload_completed')}")
        
        # Show success metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files", len(uploaded_files))
        with col2:
            st.metric("Chunks", len(all_chunks))
        with col3:
            st.metric("Method", upload_method)
        
        st.balloons()
        return True
        
    except Exception as e:
        st.error(f"{t('upload_failed')}: {e}")
        logger.error(f"Document upload error: {e}")
        return False

def load_document(uploaded_file) -> Optional[List]:
    """Load document based on file type."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_extension == 'txt':
            loader = TextLoader(tmp_path, encoding='utf-8')
        elif file_extension in ['doc', 'docx']:
            loader = UnstructuredWordDocumentLoader(tmp_path)
        elif file_extension == 'csv':
            loader = CSVLoader(tmp_path)
        else:
            return None
        
        documents = loader.load()
        os.unlink(tmp_path)
        return documents
    
    except Exception as e:
        logger.error(f"Error loading {uploaded_file.name}: {e}")
        return None

# Admin interface - only shown when authenticated in col2
if st.session_state.get('admin_authenticated', False):
    with col2:
        st.divider()
        
        # Main file upload interface
        uploaded_files = st.file_uploader(
            f"**{t('drag_drop_files')}**",
            type=['pdf', 'txt', 'doc', 'docx', 'csv'],
            accept_multiple_files=True,
            help=f"{t('supported_formats')}: PDF, TXT, DOC, DOCX, CSV"
        )

        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024 / 1024  # MB
                st.write(f"  • {file.name} ({file_size:.1f} MB)")
            
            if st.button(f"🚀 {t('upload_document')} to Knowledge Base", type="primary", use_container_width=True):
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Admin system status check
        st.subheader(f"🔧 {t('system_status')}")
        if st.button(f"🔍 {t('check_system_status')}"):
            with st.spinner(f"{t('loading')}..."):
                try:
                    # Quick API checks
                    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    
                    # Test calls
                    test_embedding = embeddings.embed_query("test")
                    pc.list_indexes()
                    
                    st.success(f"✅ {t('all_systems_operational')}")
                    st.info(f"📊 Ready to process documents → Index: small-blogs-emmbeddings-index | Dimensions: {len(test_embedding)}")
                    
                except Exception as e:
                    st.error(f"❌ {t('system_check_failed')}: {e}")
                    st.info("💡 Run `python setup_test.py` in terminal for detailed diagnostics")
        
        # Admin information
        st.divider()
        st.subheader(f"ℹ️ {t('admin_information')}")
        st.markdown(f"""
        **{t('document_processing')}:**
        - Chunk size: 1000 characters
        - Chunk overlap: 200 characters
        - Target index: small-blogs-emmbeddings-index
        - Embedding model: text-embedding-3-small
        
        **{t('supported_formats')}:**
        - PDF, TXT, DOC, DOCX, CSV
        
        **{t('security_note')}:**
        - All uploads are logged
        - Documents are processed and stored permanently
        - Only upload authorized content
        """)