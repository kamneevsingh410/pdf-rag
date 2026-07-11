# 📚 PDF Study Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application that lets you ask questions
about any PDF and receive answers grounded **only** in that document's content —
not the LLM's general training knowledge.

Built to demonstrate a full end-to-end understanding of the RAG pipeline:
chunking, embeddings, vector similarity search, and grounded generation — powered
by Google's Gemini API, ChromaDB, and LangChain.

**Current use case:** a curated AI/ML interview prep PDF (50 Q&As), turned into an
interactive study assistant.


**Example:**
> **Q:** What is the difference between bagging and boosting?
>
> **A:** Bagging trains multiple models in parallel on bootstrapped samples and averages
> their outputs to reduce variance. Boosting trains models sequentially, where each new
> model focuses on correcting the errors of the previous ones, primarily reducing bias.

Ask something *not* in the PDF (e.g. "What is a Support Vector Machine?") and the app
correctly responds: *"I don't have enough information in the document to answer that."*
— proof the answers are grounded, not hallucinated from the model's general knowledge.

---

## 🧠 How It Works

```
PDF → chunked text → embeddings (vectors) → stored in ChromaDB
                                                    ↓
Your question → embedding → similarity search → top-k relevant chunks
                                                    ↓
                     relevant chunks + question → Gemini → grounded answer
```

This follows the standard two-stage RAG pattern:
- **Retrieval** — find the most semantically relevant chunks from the source document
- **Generation** — feed those chunks + the question into an LLM, instructed to answer
  only from the provided context

---

## 🏗️ Architecture / File Structure

```
pdf-rag/
├── app.py          ← Streamlit UI + direct PDF upload + dynamic ingestion
├── loader.py       ← PDF loading & chunking utility
├── embedder.py     ← Embedding & ChromaDB creation module (called by app.py)
├── retriever.py    ← Vector similarity search interface
├── generator.py    ← Condensation, prompt formatting, & Gemini interface
├── requirements.txt
└── README.md
```

Each stage of the pipeline remains modular and decoupled. While `embedder.py` was originally run standalone, the new uploader workflow calls these components dynamically from `app.py`.
independently — e.g. replacing ChromaDB with Pinecone/FAISS, or the LLM with a different
provider, without touching the retrieval or UI logic.

---

## ⚙️ Tech Stack

- **LangChain** — orchestration (document loading, text splitting, vector store interface)
- **Google Gemini API** (free tier) — embeddings (`gemini-embedding-001`) + generation
  (`gemini-1.5-flash`)
- **ChromaDB** — local, on-disk vector database (no server, no external API key needed)
- **Streamlit** — web UI with chat interface and direct PDF upload support
- **pypdf** — PDF text extraction

---

## 🚀 Setup & Run Locally

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd pdf-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Launch the app
```bash
streamlit run app.py
```

### 5. Upload & Chat
Once the web interface opens, drag and drop any PDF directly into the main uploader (or sidebar uploader) to process it and start chatting immediately!

---

## 🧪 Design Decisions Worth Highlighting

- **Dynamic in-browser PDF processing** — PDFs are uploaded, chunked, and embedded entirely inside the application session. No manual vector database initialization needed.
- **Mobile-friendly layout** — uploader displays directly on the main screen for clean mobile layout, with sidebar uploader support for desktop views.
- **Chunking with overlap** (`chunk_size=800`, `chunk_overlap=150`) — prevents cutting a
  question off from its answer at a chunk boundary while keeping chunks focused enough
  for precise retrieval.
- **Low generation temperature (0.2)** — keeps answers grounded and consistent rather than
  creative, which matters for factual Q&A.
- **Explicit "say you don't know" instruction in the prompt** — a direct, practical
  hallucination-reduction technique (rather than just hoping the model behaves).
- **Adjustable `k` (chunks retrieved) exposed in the UI** — lets you see live how retrieval
  breadth trades off against answer focus.
- **Follow-up question condensation** — follow-up questions like "what about boosting?" are
  first rewritten into a standalone question via a separate LLM call, so vector retrieval
  has a meaningful query to work with.
- **Separation of retrieval and generation into distinct files** — retrieval logic (how
  chunks are found) is fully decoupled from generation logic (how answers are produced),
  so either can be swapped or upgraded independently.

---

## 🔧 Known Limitations / Future Improvements

- **No retrieval confidence threshold.** `similarity_search` always returns the top-k
  chunks even if none are truly relevant. Currently the LLM prompt is relied on to say
  "I don't know" when context is weak — this works well in practice, but a more robust
  approach would gate retrieval itself: if the best match's distance score exceeds a
  threshold, skip calling the LLM entirely and return "not found in document" directly.
- **`langchain_community.vectorstores.Chroma` is deprecated** in favor of the standalone
  `langchain-chroma` package. Still functional, but worth migrating.
- **No re-ranking step.** Retrieval is pure vector similarity; a production system might
  add a cross-encoder re-ranker on top of the initial top-k for higher precision.
- **Limited conversation memory.** Multi-turn context is supported via a condensation step,
  but only the last 4 turns are carried into the prompt.

---

## 📄 License

MIT — feel free to fork and adapt.