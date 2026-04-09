"""
Legal Contract Analyzer - Streamlit Application
Multi-tab interface with document upload, Q&A chat, and evaluation dashboard.
Run with: streamlit run app/streamlit_app.py
"""
 
import streamlit as st
import sys
import os
 
# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from src.ingestion import ingest_documents, load_vector_store, load_document, split_documents, create_vector_store
from src.retriever import HybridRetriever
from src.generator import generate_answer
 
# --- Page Config ---
st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
)
 
st.title("⚖️ Legal Contract Analyzer")
st.markdown("Upload contracts and ask questions with AI-powered analysis.")
 
# --- Session State Init ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []
 
# --- Tabs ---
tab_upload, tab_chat, tab_eval, tab_settings = st.tabs([
    "📁 Upload", "💬 Chat", "📊 Evaluation", "⚙️ Settings"
])
 
# ======================================
# TAB 1: DOCUMENT UPLOAD
# ======================================
with tab_upload:
    st.header("Upload Contracts")
 
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX contracts",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
 
    if uploaded_files and st.button("Process Documents", use_container_width=True):
        all_chunks = []
        progress = st.progress(0, text="Processing documents...")
 
        for i, file in enumerate(uploaded_files):
            progress.progress((i + 1) / len(uploaded_files),
                              text=f"Processing {file.name}...")
 
            temp_path = f"/tmp/{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
 
            docs = load_document(temp_path)
            chunks = split_documents(docs)
            all_chunks.extend(chunks)
            st.session_state.documents_loaded.append(file.name)
 
        progress.progress(1.0, text="Building vector index...")
        vs = create_vector_store(all_chunks)
        st.session_state.vector_store = vs
        st.session_state.retriever = HybridRetriever(vs)
 
        st.success(f"Processed {len(uploaded_files)} documents into {len(all_chunks)} searchable chunks.")
 
    if st.session_state.documents_loaded:
        st.markdown("**Loaded documents:**")
        for doc_name in st.session_state.documents_loaded:
            st.markdown(f"✅ {doc_name}")
 
# ======================================
# TAB 2: CHAT Q&A
# ======================================
with tab_chat:
    st.header("Ask Questions")
 
    if st.session_state.retriever is None:
        st.warning("👉 Upload and process documents in the Upload tab first.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "citations" in msg:
                    with st.expander("📚 View Sources"):
                        for cite in msg["citations"]:
                            st.markdown(f"**{cite['source']}** — Page {cite['page']}")
 
        if user_query := st.chat_input("Ask about your contracts..."):
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
 
            with st.chat_message("assistant"):
                with st.spinner("Analyzing contracts..."):
                    chunks = st.session_state.retriever.retrieve(user_query)
                    result = generate_answer(user_query, chunks)
 
                st.markdown(result["answer"])
 
                confidence = result.get("confidence", "unknown")
                if confidence == "high":
                    st.success(f"Confidence: {confidence}")
                elif confidence == "medium":
                    st.warning(f"Confidence: {confidence}")
                else:
                    st.error(f"Confidence: {confidence}")
 
                if result["citations"]:
                    with st.expander("📚 View Sources"):
                        for cite in result["citations"]:
                            st.markdown(f"**{cite['source']}** — Page {cite['page']}")
 
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", []),
                })
 
# ======================================
# TAB 3: EVALUATION
# ======================================
with tab_eval:
    st.header("RAG Evaluation Dashboard")
    st.markdown("Run RAGAS evaluation to measure system quality.")
    st.info("Create a test set at eval/test_set.json with question-answer pairs, then run: python -m src.evaluator from the terminal.")
 
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Faithfulness", "---")
    with col2: st.metric("Answer Relevancy", "---")
    with col3: st.metric("Context Precision", "---")
    with col4: st.metric("Context Recall", "---")
 
# ======================================
# TAB 4: SETTINGS
# ======================================
with tab_settings:
    st.header("Settings")
    st.slider("Chunk Size", 500, 2000, 1000, step=100, key="chunk_size")
    st.slider("Chunk Overlap", 0, 500, 200, step=50, key="chunk_overlap")
    st.slider("Top-K Results", 1, 10, 5, key="top_k")
    st.selectbox("Search Method", ["hybrid", "vector", "bm25"], key="search_method")
