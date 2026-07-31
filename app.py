import asyncio
import sys
import os
import shutil
import tempfile

# python 3.13 + windows: default proactor event loop breaks chromadb's async code.
# switching to selector event loop fixes the "event loop is closed" crash.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from generator import generate_answer
from loader import load_and_chunk_pdf
from embedder import build_vector_store, PERSIST_DIRECTORY

st.set_page_config(
    page_title="PDF Study Assistant",
    page_icon="📚",
    layout="centered"
)

# custom css: light background + purple accent theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── force light theme on entire app ── */
    .stApp {
        background-color: #f0f2f5 !important;
        color: #1f2937 !important;
    }
    .main, .main .block-container {
        background-color: #f0f2f5 !important;
        padding-top: 2rem;
    }
    /* override every text element in main area */
    .main, .main * {
        color: #374151 !important;
    }
    .main h1, .main h2, .main h3 {
        color: #1e1b4b !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* ── sidebar (dark theme kept intentionally) ── */
    [data-testid="stSidebar"] {
        background-color: #2c2f3a !important;
    }
    [data-testid="stSidebar"] * {
        color: #d4d8e8 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #444860 !important;
    }

    /* sidebar file uploader - softer purple, not black */
    [data-testid="stSidebar"] [data-testid="stFileUploader"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] div {
        background-color: #3d4059 !important;
        border-color: #555a7a !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border-radius: 10px !important;
        padding: 0.5rem !important;
        border: 1px dashed #7c3aed !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #7c3aed !important;
        color: #ffffff !important;
        border-color: #7c3aed !important;
    }

    /* sidebar slider accent */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #7c3aed !important;
    }

    /* sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        background-color: #3d4059 !important;
        color: #d4d8e8 !important;
        border: 1px solid #555a7a !important;
        border-radius: 8px !important;
        transition: background-color 0.2s !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #4b5068 !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
        color: #ffffff !important;
    }

    /* ── welcome card ── */
    .welcome-card {
        background: linear-gradient(135deg, #ede9fe 0%, #f5f3ff 100%) !important;
        border: 2px dashed #c4b5fd;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .welcome-card *, .welcome-card h3, .welcome-card p {
        color: #1e1b4b !important;
    }
    .welcome-card p {
        color: #6b7280 !important;
        font-size: 0.95rem;
    }

    /* ── file uploader in main area ── */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        border: 2px dashed #c4b5fd !important;
        padding: 0.5rem !important;
    }
    .main [data-testid="stFileUploader"] *,
    .main [data-testid="stFileUploader"] section,
    .main [data-testid="stFileUploader"] div {
        background-color: #ffffff !important;
    }
    .main [data-testid="stFileUploader"] button {
        background-color: #7c3aed !important;
        color: #ffffff !important;
        border-color: #7c3aed !important;
        border-radius: 8px !important;
    }

    /* ── main area buttons ── */
    .main .stButton button {
        background-color: #7c3aed !important;
        color: #ffffff !important;
        border: 1px solid #7c3aed !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: background-color 0.2s, transform 0.1s !important;
    }
    .main .stButton button:hover {
        background-color: #6d28d9 !important;
        border-color: #6d28d9 !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }

    /* ── user message bubble ── */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #ede9fe !important;
        border-radius: 12px !important;
        border-left: 4px solid #7c3aed !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * {
        color: #1e1b4b !important;
    }

    /* ── assistant message bubble - soft lavender ── */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #f5f3ff !important;
        border-radius: 12px !important;
        border-left: 4px solid #a78bfa !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) * {
        color: #1f2937 !important;
        background-color: transparent !important;
    }

    /* ── chat input ── */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] *,
    [data-testid="stChatInput"] div,
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    [data-testid="stChatInput"] {
        border: 2px solid #c4b5fd !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(124,58,237,0.08) !important;
    }
    /* chat send button */
    [data-testid="stChatInput"] button {
        background-color: #7c3aed !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    /* ── source chunks expander ── */
    [data-testid="stExpander"],
    [data-testid="stExpander"] div {
        background-color: #f9fafb !important;
        border-radius: 10px;
    }
    [data-testid="stExpander"] * {
        color: #1f2937 !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: #6d28d9 !important;
        font-weight: 500;
    }

    /* ── caption ── */
    .stCaption, .stCaption * {
        color: #9ca3af !important;
        font-style: italic;
    }

    /* ── spinner and alert texts ── */
    .stSpinner, .stSpinner * {
        color: #374151 !important;
    }
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }
    [data-testid="stAlert"], [data-testid="stAlert"] * {
        color: #1f2937 !important;
        border-radius: 10px;
    }

    /* ── horizontal divider ── */
    hr {
        border-color: #ddd6fe;
    }

    /* ── pdf loaded badge ── */
    .pdf-badge {
        background-color: #ede9fe !important;
        border: 1px solid #c4b5fd;
        border-radius: 8px;
        padding: 0.4rem 0.75rem;
        font-size: 0.85rem;
        color: #5b21b6 !important;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* ── bottom chat input bar container ── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background-color: #f0f2f5 !important;
    }
</style>
""", unsafe_allow_html=True)


def process_uploaded_pdf(uploaded_file):
    """save, chunk, and embed an uploaded pdf. returns (chunk_count, error_str)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        chunks = load_and_chunk_pdf(tmp_path)
        
        # generate a unique collection name to avoid windows database file-locking errors
        import uuid
        collection_id = f"pdf_{uuid.uuid4().hex[:8]}"
        st.session_state.collection_name = collection_id
        
        # build the new collection directly in the vector store
        from embedder import get_embedding_model
        from langchain_chroma import Chroma
        import time
        import random
        
        embedding_model = get_embedding_model()
        vector_store = Chroma(
            collection_name=collection_id,
            embedding_function=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )

        # Batch the chunks to avoid hitting Google's free tier rate limits (100 requests per minute)
        batch_size = 20
        max_retries = 6
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            backoff = 2.0
            
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(batch)
                    break
                except Exception as e:
                    err_str = str(e).upper()
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "QUOTA" in err_str:
                        if attempt == max_retries - 1:
                            raise e
                        sleep_time = backoff + random.uniform(0.1, 1.0)
                        time.sleep(sleep_time)
                        backoff *= 2.0
                    else:
                        raise e
            
            # small delay between successful batches to pace calls
            time.sleep(0.5)

        return len(chunks), None
    except Exception as e:
        return 0, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)



pdf_ready = "pdf_name" in st.session_state and os.path.exists(PERSIST_DIRECTORY)

# sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.25rem 0 1rem 0;">
        <div style="font-size: 2rem; margin-bottom: 0.25rem;">📚</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #a78bfa !important;">PDF Assistant</div>
        <div style="font-size: 0.82rem; color: #9ca3af !important; margin-top: 0.25rem;">Ask anything about your document</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # pdf uploader
    if pdf_ready:
        pdf_label = st.session_state.pdf_name
        st.markdown(
            f'<div style="background:#2a3d2a;border:1px solid #4ade80;border-radius:8px;padding:0.5rem 0.75rem;margin-bottom:0.75rem;">'
            f'<span style="color:#86efac !important;font-size:0.8rem;font-weight:600;">✓ LOADED</span><br/>'
            f'<span style="color:#d4d8e8 !important;font-size:0.85rem;">{pdf_label}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown('<div style="font-size:0.82rem;color:#9ca3af !important;margin-bottom:0.4rem;">📂 Replace document:</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.9rem;font-weight:600;color:#d4d8e8 !important;margin-bottom:0.5rem;">📄 Upload your PDF</div>', unsafe_allow_html=True)

    sidebar_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        key="sidebar_uploader",
        label_visibility="collapsed"
    )

    if sidebar_file is not None:
        btn_label = "🔄 Re-index document" if pdf_ready else "⚙️ Process & embed PDF"
        if st.button(btn_label, use_container_width=True):
            with st.spinner("Embedding your PDF..."):
                n, err = process_uploaded_pdf(sidebar_file)
            if err:
                st.error(f"Error: {err}")
            else:
                st.session_state.pdf_name = sidebar_file.name
                st.session_state.messages = []
                st.rerun()

    st.markdown("---")

    # only show these controls when a pdf is loaded
    if pdf_ready:
        st.markdown('<div style="font-size:0.82rem;color:#9ca3af !important;margin-bottom:0.4rem;">🔍 Retrieval depth</div>', unsafe_allow_html=True)
        k = st.slider("Context chunks (k)", min_value=1, max_value=8, value=4, label_visibility="collapsed")
        st.markdown('<div style="font-size:0.75rem;color:#6b7280 !important;margin-bottom:0.75rem;">Higher = more context, slightly slower</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        k = 4  # default when no pdf loaded yet

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#6b7280 !important; line-height:1.8;">
        <div style="color:#a78bfa !important; font-weight:600; margin-bottom:0.4rem;">How it works</div>
        <div>① Upload a PDF document</div>
        <div>② Text is chunked &amp; embedded</div>
        <div>③ Ask questions in the chat</div>
        <div>④ Relevant passages are retrieved</div>
        <div>⑤ AI answers from your content</div>
    </div>
    """, unsafe_allow_html=True)


# main area
st.markdown("""
<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
    <span style="font-size:2rem;">📚</span>
    <span style="font-size:1.9rem; font-weight:800; color:#1e1b4b !important; letter-spacing:-0.5px;">PDF Study Assistant</span>
</div>
<div style="font-size:0.9rem; color:#6b7280 !important; margin-bottom:1.5rem;">
    Powered by Gemini · Ask questions, get answers grounded in your document
</div>
""", unsafe_allow_html=True)

if not pdf_ready:
    # welcome / upload prompt
    st.markdown("""
        <div class="welcome-card">
            <div style="font-size: 3.5rem;">📄</div>
            <h3>Drop in a PDF to get started</h3>
            <p>Summaries · Definitions · Comparisons · Deep dives<br/>
            Every answer is sourced directly from your document.</p>
        </div>
    """, unsafe_allow_html=True)

    # inline uploader
    main_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="main_uploader",
        label_visibility="collapsed"
    )
    if main_file is not None:
        if st.button("⚙️ Process & start chatting", use_container_width=True):
            progress_placeholder = st.empty()
            progress_placeholder.info("📖 Reading and chunking your PDF...")
            n, err = process_uploaded_pdf(main_file)
            progress_placeholder.empty()
            if err:
                st.error(f"Something went wrong: {err}")
            else:
                st.session_state.pdf_name = main_file.name
                st.session_state.messages = []
                st.rerun()

    st.stop()



# pdf is loaded — show chat interface
if st.session_state.get("pdf_name"):
    st.markdown(
        f'<div class="pdf-badge">📄 {st.session_state.pdf_name}</div>',
        unsafe_allow_html=True
    )

st.markdown('<div style="font-size:0.88rem; color:#6b7280 !important; margin-bottom:0.5rem;">💬 Ask anything about your document. Follow-up questions are understood in context.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    # sources: list of (page, text) tuples
    with st.expander("📄 View source chunks used"):
        for i, (page, text) in enumerate(sources):
            st.markdown(f"**Source {i+1} (page {page})**")
            st.write(text)
            st.markdown("---")


# replay full conversation on each rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask anything about your PDF...")

if question and question.strip():
    with st.chat_message("user"):
        st.write(question)

    # build (question, answer) pairs from conversation so far
    history = []
    messages = st.session_state.messages
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            history.append((messages[i]["content"], messages[i + 1]["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and generating answer..."):
            try:
                result = generate_answer(
                    question, 
                    k=k, 
                    history=history, 
                    collection_name=st.session_state.get("collection_name")
                )


                st.write(result["answer"])

                # show how the follow-up was rewritten for retrieval
                if result["standalone_question"] != question:
                    st.caption(f"Interpreted as: *{result['standalone_question']}*")

                # store as plain tuples — avoids keeping Document objects in session state
                sources = [
                    (doc.metadata.get("page_label", "unknown"), doc.page_content)
                    for doc in result["sources"]
                ]
                if sources:
                    render_sources(sources)

                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
