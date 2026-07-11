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

# custom css: grey background + purple accent theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* main background */
    .stApp {
        background-color: #f0f2f5;
    }

    /* main content area */
    .main .block-container {
        background-color: #f0f2f5;
        padding-top: 2rem;
    }

    /* sidebar */
    [data-testid="stSidebar"] {
        background-color: #2c2f3a;
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

    /* sidebar file uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #3d4059;
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px dashed #555a7a;
    }

    /* sidebar slider accent */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #7c3aed !important;
    }

    /* sidebar button */
    [data-testid="stSidebar"] .stButton button {
        background-color: #3d4059;
        color: #d4d8e8;
        border: 1px solid #555a7a;
        border-radius: 8px;
        transition: background-color 0.2s;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #4b5068;
    }

    /* process pdf button in sidebar - purple accent */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #7c3aed;
        border-color: #7c3aed;
        color: #ffffff;
    }

    /* page title */
    h1 {
        color: #1e1b4b !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* subtitle text */
    .stMarkdown p {
        color: #4b5563;
    }

    /* welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #ede9fe 0%, #f5f3ff 100%);
        border: 2px dashed #c4b5fd;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .welcome-card h3 {
        color: #1e1b4b;
        font-size: 1.4rem;
        margin: 1rem 0 0.5rem;
    }
    .welcome-card p {
        color: #6b7280;
        font-size: 0.95rem;
        margin: 0;
    }

    /* file uploader in main area */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 2px dashed #c4b5fd;
        padding: 0.5rem;
    }

    /* main process button */
    .stButton button[data-testid="baseButton-secondary"] {
        background-color: #7c3aed;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: background-color 0.2s, transform 0.1s;
    }
    .stButton button[data-testid="baseButton-secondary"]:hover {
        background-color: #6d28d9;
        transform: translateY(-1px);
    }

    /* user message bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #ede9fe;
        border-radius: 12px;
        border-left: 4px solid #7c3aed;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }

    /* assistant message bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #ffffff;
        border-radius: 12px;
        border-left: 4px solid #a78bfa;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }

    /* chat input */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 2px solid #c4b5fd;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(124,58,237,0.08);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #7c3aed;
    }

    /* source chunks expander */
    [data-testid="stExpander"] {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        color: #6d28d9 !important;
        font-weight: 500;
    }

    /* caption */
    .stCaption {
        color: #9ca3af !important;
        font-style: italic;
    }

    /* spinner */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }

    /* horizontal divider */
    hr {
        border-color: #ddd6fe;
    }

    /* alert */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* pdf loaded badge */
    .pdf-badge {
        background-color: #ede9fe;
        border: 1px solid #c4b5fd;
        border-radius: 8px;
        padding: 0.4rem 0.75rem;
        font-size: 0.85rem;
        color: #5b21b6;
        display: inline-block;
        margin-bottom: 1rem;
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
        from langchain_community.vectorstores import Chroma
        
        embedding_model = get_embedding_model()
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=collection_id,
            persist_directory=PERSIST_DIRECTORY
        )
        return len(chunks), None
    except Exception as e:
        return 0, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)



pdf_ready = "pdf_name" in st.session_state and os.path.exists(PERSIST_DIRECTORY)

# sidebar
with st.sidebar:
    st.header("📚 PDF Assistant")
    st.write(
        "Upload any PDF and ask questions about it. "
        "Answers are grounded **only** in your document — not the model's general knowledge."
    )
    st.markdown("---")

    # pdf uploader
    if pdf_ready:
        pdf_label = st.session_state.pdf_name
        st.success(f"📄 **{pdf_label}**")
        st.markdown("**Replace PDF:**")
    else:
        st.markdown("**📄 Upload your PDF:**")

    sidebar_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        key="sidebar_uploader",
        label_visibility="collapsed"
    )

    if sidebar_file is not None:
        btn_label = "🔄 Replace & reindex" if pdf_ready else "⚙️ Process PDF"
        if st.button(btn_label, use_container_width=True):
            with st.spinner("Reading and embedding..."):
                n, err = process_uploaded_pdf(sidebar_file)
            if err:
                st.error(f"Error: {err}")
            else:
                st.session_state.pdf_name = sidebar_file.name
                st.session_state.messages = []
                st.rerun()

    st.markdown("---")
    st.write("**Tech stack:**")
    st.write("- LangChain\n- ChromaDB\n- Gemini Embeddings\n- Gemini 1.5 Flash\n- Streamlit")
    st.markdown("---")

    # only show these controls when a pdf is loaded
    if pdf_ready:
        k = st.slider("Context chunks to retrieve (k)", min_value=1, max_value=8, value=4)
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
    else:
        k = 4  # default when no pdf loaded yet


# main area
st.title("📚 PDF Study Assistant")

if not pdf_ready:
    # welcome / upload prompt - shown in main area for mobile users
    st.markdown("""
        <div class="welcome-card">
            <div style="font-size: 3.5rem;">📄</div>
            <h3>Upload a PDF to get started</h3>
            <p>Ask anything about your document — summaries, definitions, comparisons.<br/>
            Answers are pulled only from the PDF, not hallucinated.</p>
        </div>
    """, unsafe_allow_html=True)

    # inline uploader (critical for mobile — sidebar is collapsed by default)
    main_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="main_uploader",
        label_visibility="collapsed"
    )
    if main_file is not None:
        if st.button("⚙️ Process & start chatting", use_container_width=True):
            with st.spinner("Reading and embedding your PDF... this may take a moment."):
                n, err = process_uploaded_pdf(main_file)
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

st.write("Ask anything about your PDF. Follow-up questions are understood in context.")

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
