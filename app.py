"""
RAG Knowledge Assistant — Streamlit GUI
========================================
Theme: "Deep Space" Dark Mode
Primary Accent: Emerald Green (#50C878)
Secondary: Slate Gray (#708090)
Typography: Inter (Google Fonts)

Run with:  streamlit run app.py
"""

import os
import time
import datetime
import streamlit as st
import PyPDF2
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Deep Space Theme + Glassmorphism
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --emerald: #50C878;
    --emerald-dim: rgba(80,200,120,0.15);
    --emerald-glow: 0 0 18px rgba(80,200,120,0.35);
    --slate: #708090;
    --bg-deep: #0B0F19;
    --bg-card: rgba(30,36,50,0.65);
    --bg-glass: rgba(255,255,255,0.04);
    --border-glass: rgba(255,255,255,0.08);
    --text-primary: #E8ECF1;
    --text-secondary: #8892A4;
    --radius: 15px;
}

/* ── Global overrides ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1525 0%, #131B2E 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Headers ── */
h1, h2, h3, h4 { color: var(--text-primary) !important; font-family: 'Inter', sans-serif !important; }

/* ── Buttons with glow ── */
.stButton > button {
    background: linear-gradient(135deg, #50C878 0%, #3BA55D 100%) !important;
    color: #0B0F19 !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 0.55rem 1.6rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s ease !important;
    box-shadow: var(--emerald-glow) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 28px rgba(80,200,120,0.55) !important;
}
.stButton > button:active { transform: scale(0.97) !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div,
[data-testid="stChatInput"] textarea {
    background-color: rgba(20,26,40,0.8) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, [data-testid="stChatInput"] textarea:focus {
    border-color: var(--emerald) !important;
    box-shadow: 0 0 0 2px rgba(80,200,120,0.2) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--emerald) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-glass) !important;
    border: 2px dashed rgba(80,200,120,0.3) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--emerald) !important;
}

/* ── Expander (source attribution) ── */
.streamlit-expanderHeader {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 10px !important;
    color: var(--emerald) !important;
    font-weight: 500 !important;
}

/* ── Chat message glassmorphism ── */
[data-testid="stChatMessage"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.8rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--emerald) !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(80,200,120,0.25); border-radius: 3px; }

/* ── Glass card utility ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius);
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
}
.status-online { background: #50C878; box-shadow: 0 0 6px #50C878; }
.status-offline { background: #FF4D4D; box-shadow: 0 0 6px #FF4D4D; }

/* ── Pulse animation for processing ── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(80,200,120,0.3); }
    50% { box-shadow: 0 0 24px rgba(80,200,120,0.7); }
}
.processing-indicator {
    animation: pulse-glow 1.5s ease-in-out infinite;
    border: 1px solid rgba(80,200,120,0.4);
    border-radius: var(--radius);
    padding: 0.8rem 1.2rem;
    text-align: center;
    color: var(--emerald);
    font-weight: 500;
    background: var(--bg-glass);
}

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "messages": [],
    "indexed_docs": [],
    "collection": None,
    "embedder": None,
    "llm_client": None,
    "vector_store_ready": False,
    "temperature": 0.3,
    "top_k": 5,
    "total_chunks": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# CORE RAG FUNCTIONS (from your notebook)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load the sentence-transformer model once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    """Create or get the ChromaDB collection."""
    client = chromadb.Client()
    return client.get_or_create_collection(name="football_docs")


def get_llm_client():
    """Initialize the Groq-backed OpenAI client."""
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def load_pdf(file_bytes, filename):
    """Extract text from uploaded PDF bytes."""
    import io
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def index_document(file_bytes, filename):
    """Index a single uploaded document into ChromaDB."""
    embedder = st.session_state.embedder
    collection = st.session_state.collection

    text = load_pdf(file_bytes, filename)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    embeddings = embedder.encode(chunks).tolist()
    base_id = len(collection.get()["ids"]) if collection.get()["ids"] else 0
    ids = [f"chunk_{base_id + i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    return len(chunks)


def ask_question(question):
    """
    Query the vector store and generate an answer.
    Returns (answer_text, source_documents_list).
    """
    embedder = st.session_state.embedder
    collection = st.session_state.collection
    client = st.session_state.llm_client
    top_k = st.session_state.top_k
    temperature = st.session_state.temperature

    q_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if results.get("distances") else [0] * len(docs)
    context = "\n\n".join(docs)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant. "
                    "Answer the question using the provided context as your primary source. "
                    "If the context contains partial information, use it and expand with your knowledge. "
                    "Be helpful, detailed, and well-structured in your answers."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=temperature,
    )

    answer = response.choices[0].message.content

    sources = []
    for i, doc in enumerate(docs):
        sources.append({
            "text": doc[:300] + ("..." if len(doc) > 300 else ""),
            "source": metadatas[i].get("source", "Unknown"),
            "chunk": metadatas[i].get("chunk_index", i),
            "relevance": round((1 - distances[i] / 2) * 100, 1) if distances[i] else 0,
        })

    return answer, sources


# ─────────────────────────────────────────────
# INITIALIZE MODELS
# ─────────────────────────────────────────────
if st.session_state.embedder is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedder = load_embedder()

if st.session_state.collection is None:
    st.session_state.collection = get_chroma_collection()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR (Status & Settings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div style="font-size:2.2rem; margin-bottom:0.3rem;">🧠</div>
        <div style="font-size:1.3rem; font-weight:700; color:#50C878;">RAG Assistant</div>
        <div style="font-size:0.75rem; color:#8892A4;">Deep Space Edition</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Vector Store Status ──
    st.markdown("##### 💾 Vector Store Status")
    is_ready = st.session_state.vector_store_ready
    dot_class = "status-online" if is_ready else "status-offline"
    status_text = "Online" if is_ready else "No Documents"
    st.markdown(f"""
    <div class="glass-card" style="padding:0.8rem;">
        <div style="font-size:0.85rem;">
            <span class="status-dot {dot_class}"></span> {status_text}
        </div>
        <div style="font-size:0.75rem; color:#8892A4; margin-top:4px;">
            Chunks: {st.session_state.total_chunks} &nbsp;|&nbsp; Docs: {len(st.session_state.indexed_docs)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Indexed docs list ──
    if st.session_state.indexed_docs:
        st.markdown("**Indexed Documents:**")
        for doc in st.session_state.indexed_docs:
            st.markdown(f"""
            <div class="glass-card" style="padding:0.6rem 0.9rem; font-size:0.85rem;">
                📄 {doc}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Model Settings ──
    st.markdown("##### ⚙️ Model Settings")
    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        help="Lower = more focused, Higher = more creative"
    )
    st.session_state.top_k = st.slider(
        "Top-K Retrieval", 1, 15, st.session_state.top_k,
        help="Number of document chunks to retrieve"
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN PANEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Header ──
st.markdown("""
<div style="text-align:center; padding:1.5rem 0 1rem;">
    <h1 style="font-size:2rem; font-weight:700; margin:0;">
        🧠 <span style="color:#50C878;">RAG</span> Knowledge Assistant
    </h1>
    <p style="color:#8892A4; font-size:0.9rem; margin-top:0.3rem;">
        Upload documents · Ask questions · Get sourced answers
    </p>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SETUP PANEL (always visible on main page)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
setup_col1, setup_col2 = st.columns(2)

with setup_col1:
    st.markdown("##### 🔑 API Key")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Paste your Groq API key here (gsk_...)",
        key="api_key_input",
        value=st.session_state.get("api_key", ""),
        label_visibility="collapsed",
    )
    if api_key:
        st.session_state["api_key"] = api_key
        st.session_state.llm_client = get_llm_client()
        st.success("✅ API key set!")

with setup_col2:
    st.markdown("##### 📚 Upload PDF")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
    )

if uploaded_files:
    if st.button("🔄 Index Documents", use_container_width=True):
        total = 0
        progress = st.progress(0, text="Indexing...")
        for i, f in enumerate(uploaded_files):
            if f.name not in st.session_state.indexed_docs:
                chunks = index_document(f.read(), f.name)
                total += chunks
                st.session_state.indexed_docs.append(f.name)
            progress.progress((i + 1) / len(uploaded_files), text=f"Processing {f.name}...")
        progress.empty()
        st.session_state.total_chunks += total
        st.session_state.vector_store_ready = True
        st.success(f"✅ Indexed {total} chunks from {len(uploaded_files)} file(s)")
        st.rerun()

st.divider()

# ── Stats row ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Documents", len(st.session_state.indexed_docs))
with col2:
    st.metric("Chunks", st.session_state.total_chunks)
with col3:
    st.metric("Messages", len(st.session_state.messages))
with col4:
    st.metric("Model", "Groq / Llama 3.3")

st.markdown("<br>", unsafe_allow_html=True)

# ── Chat History ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 View Sources"):
                for j, src in enumerate(msg["sources"]):
                    relevance = src.get("relevance", 0)
                    bar_color = "#50C878" if relevance > 70 else "#F0AD4E" if relevance > 40 else "#FF4D4D"
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-weight:600; font-size:0.85rem; color:#50C878;">
                                📄 {src['source']} — Chunk #{src['chunk']}
                            </span>
                            <span style="font-size:0.75rem; color:#8892A4;">{relevance}% match</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:6px; margin-bottom:8px;">
                            <div style="width:{relevance}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
                        </div>
                        <div style="font-size:0.8rem; color:#8892A4; line-height:1.5;">
                            {src['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ── Chat Input ──
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.get("api_key"):
        st.error("⚠️ Please enter your Groq API key above.")
        st.stop()
    if not st.session_state.vector_store_ready:
        st.error("⚠️ Please upload and index documents first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner(""):
            st.markdown("""
            <div class="processing-indicator">⚡ Searching knowledge base & generating answer...</div>
            """, unsafe_allow_html=True)
            answer, sources = ask_question(prompt)

        st.markdown(answer)

        if sources:
            with st.expander("📎 View Sources"):
                for j, src in enumerate(sources):
                    relevance = src.get("relevance", 0)
                    bar_color = "#50C878" if relevance > 70 else "#F0AD4E" if relevance > 40 else "#FF4D4D"
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-weight:600; font-size:0.85rem; color:#50C878;">
                                📄 {src['source']} — Chunk #{src['chunk']}
                            </span>
                            <span style="font-size:0.75rem; color:#8892A4;">{relevance}% match</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:6px; margin-bottom:8px;">
                            <div style="width:{relevance}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
                        </div>
                        <div style="font-size:0.8rem; color:#8892A4; line-height:1.5;">
                            {src['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()
