 # ============================================================
#     UNIVERSAL TEXT + PDF + BBC SUMMARIZER (FLAN-T5)
#                    Streamlit Premium UI
#                         — For ARPIT —
# ============================================================

import os
import PyPDF2
import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ============================================================
#                    CORE MODEL SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 150

SUMMARY_MODES = {
    "short":   "give a short concise summary: ",
    "long":    "give a detailed multi-paragraph summary: ",
    "bullets": "summarize as clear bullet points: ",
    "normal":  "summarize this clearly: "
}


@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


# ============================================================
#                      CORE FUNCTIONS
# ============================================================

def summarize_text_block(text: str, mode_key: str) -> str:
    """Summarize a text block using the selected mode."""
    prompt = SUMMARY_MODES[mode_key] + text

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        padding="max_length"
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=MAX_OUTPUT_LEN,
            num_beams=4,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def summarize(text: str, mode_key: str) -> str:
    """Handles short + long text automatically using chunking."""
    text = text.strip()

    if len(text) == 0:
        return "❌ No content found."

    if len(text) < 500:
        return summarize_text_block(text, mode_key)

    chunks = []
    step = 1500

    for i in range(0, len(text), step):
        chunks.append(text[i:i + step])

    summaries = [summarize_text_block(ch, mode_key) for ch in chunks]

    final_text = " ".join(summaries)
    final_summary = summarize_text_block(final_text, mode_key)

    return final_summary


def read_pdf_streamlit(file) -> str:
    """Read an uploaded PDF file object in Streamlit."""
    text = ""
    reader = PyPDF2.PdfReader(file)
    for i, page in enumerate(reader.pages):
        try:
            content = page.extract_text()
            if content:
                text += f"\n\n--- PAGE {i + 1} ---\n\n"
                text += content
        except Exception:
            pass
    return text


def read_bbc_article(category: str, filename: str):
    """Read ONE local BBC article from your dataset path."""
    base_path = r"C:\Users\arpit\OneDrive\Desktop\AIML PROJECT\BBC News Summary\BBC News Summary\News Articles"
    path = os.path.join(base_path, category, filename)

    if not os.path.exists(path):
        return None, "❌ Article not found. Check filename/category."

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text, None


# ============================================================
#                     STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🧠",
    layout="wide"
)

# ---------------------- CUSTOM CSS --------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
        color: #e5e7eb;
    }

    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #000000 100%);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.2rem;
        max-width: 1100px;
    }

    /* Navbar */
    .top-nav {
        width: 100%;
        padding: 12px 0 20px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #f9fafb;
        font-family: 'Poppins';
    }
    .top-nav-left {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .top-nav-pill {
        font-size: 0.8rem;
        padding: 3px 10px;
        border-radius: 999px;
        background: rgba(148,163,184,0.18);
        border: 1px solid rgba(148,163,184,0.4);
        margin-left: 0.5rem;
    }
    .top-nav-right {
        font-size: 0.8rem;
        color: #9ca3af;
        text-align: right;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(15,23,42,0.85);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(148,163,184,0.32);
        box-shadow: 0 22px 45px rgba(0,0,0,0.7);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .glass-header {
        font-size: 1.08rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.3rem;
        font-family: 'Poppins';
    }
    .glass-sub {
        font-size: 0.88rem;
        color: #9ca3af;
        margin-bottom: 0.8rem;
    }

    /* Textarea styling */
    textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #e5e7eb !important;
        font-size: 0.95rem !important;
    }

    textarea:focus {
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Button */
    .stButton > button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        border: none;
        padding: 0.65rem 1.6rem;
        font-size: 0.95rem;
        font-weight: 600;
        font-family: 'Poppins';
        background: linear-gradient(135deg, #38bdf8, #6366f1);
        color: white;
        cursor: pointer;
        box-shadow: 0 14px 30px rgba(15,23,42,0.9);
        transition: all 0.18s ease-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 20px 40px rgba(15,23,42,1);
        background: linear-gradient(135deg, #0ea5e9, #4f46e5);
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: 0 10px 24px rgba(15,23,42,1);
    }

    /* Summary card */
    .summary-card {
        margin-top: 1.4rem;
        background: rgba(15,23,42,0.95);
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,0.45);
        padding: 1.1rem 1.3rem;
        box-shadow: 0 20px 45px rgba(0,0,0,0.85);
        animation: fadeIn 0.45s ease-out;
    }
    .summary-title {
        font-weight: 600;
        margin-bottom: 0.4rem;
        font-family: 'Poppins';
    }
    .summary-body {
        font-size: 0.94rem;
        color: #d1d5db;
        line-height: 1.55;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Inter';
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#0ea5e9, #6366f1);
        border-radius: 999px;
    }

    .footer {
        text-align: center;
        margin-top: 2.0rem;
        font-size: 0.8rem;
        color: #6b7280;
    }
    .footer span {
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
#                        UI LAYOUT
# ============================================================

# Top nav
st.markdown(
    """
    <div class="top-nav">
        <div class="top-nav-left">
            🧠 AI TEXT Summarizer <span class="top-nav-pill">Text · PDF · BBC</span>
        </div>
        
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='font-size:0.95rem; color:#9ca3af; margin-bottom:1.3rem;'>"
    "Paste your text, upload a PDF, or select a BBC article. Choose summary style from the sidebar."
    "</div>",
    unsafe_allow_html=True,
)

# Sidebar: summary mode
st.sidebar.markdown("### ⚙️ Summary Mode")
mode_label = st.sidebar.radio(
    "",
    ["Normal", "Short", "Detailed", "Bullet Points"],
    index=0
)

mode_key_map = {
    "Normal": "normal",
    "Short": "short",
    "Detailed": "long",
    "Bullet Points": "bullets"
}
CURRENT_MODE_KEY = mode_key_map[mode_label]

st.sidebar.markdown("---")
st.sidebar.write("arpitjaiswal26")
#st.sidebar.write(f"**Model:** `{MODEL_NAME}`")
#st.sidebar.caption("Tip: You can later swap to `google/flan-t5-large` for stronger summaries.")

# Tabs: Text / PDF / BBC
tab_text, tab_pdf, tab_bbc = st.tabs(
    ["📝 Text Summarizer", "📄 PDF Summarizer", "📰 BBC Article Summarizer"]
)

# ----------------------- TEXT TAB --------------------------

with tab_text:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">📝 Summarize Text</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-sub">Paste lecture notes, theory, or long paragraphs. '
        'The model will compress them based on your selected mode.</div>',
        unsafe_allow_html=True,
    )

    text_input = st.text_area(
        label="",
        height=230,
        placeholder="Enter or paste long text here…",
    )

    col_btn, _ = st.columns([0.3, 0.7])
    with col_btn:
        run_text = st.button("⚡ Summarize Text")

    if run_text:
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Summarizing text..."):
                summary = summarize(text_input, CURRENT_MODE_KEY)

            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.markdown('<div class="summary-title">Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-body">{summary}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- PDF TAB ---------------------------

with tab_pdf:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">📄 Summarize PDF</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-sub">Upload a PDF (notes, articles, assignments). '
        'Text will be extracted and summarized using your chosen mode.</div>',
        unsafe_allow_html=True,
    )

    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    col_btn2, _ = st.columns([0.3, 0.7])
    with col_btn2:
        run_pdf = st.button("⚡ Summarize PDF")

    if run_pdf:
        if pdf_file is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Reading PDF and extracting text..."):
                pdf_text = read_pdf_streamlit(pdf_file)

            if not pdf_text.strip():
                st.error("Could not extract any text from this PDF.")
            else:
                with st.spinner("Summarizing PDF content..."):
                    summary = summarize(pdf_text, CURRENT_MODE_KEY)

                st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                st.markdown('<div class="summary-title">Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-body">{summary}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- BBC TAB ---------------------------

with tab_bbc:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">📰 Summarize ONE BBC Article</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-sub">Reads a single article from your local BBC dataset. '
        'Choose category and filename like <code>001.txt</code>.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox(
            "Category",
            ["business", "tech", "sport", "politics", "entertainment"],
        )
    with col2:
        filename = st.text_input("Filename (e.g. 001.txt)")

    run_bbc = st.button("⚡ Summarize BBC Article")

    if run_bbc:
        if not filename.strip():
            st.warning("Enter an article filename, for example 001.txt")
        else:
            with st.spinner("Reading BBC article..."):
                text, err = read_bbc_article(category, filename)

            if err:
                st.error(err)
            else:
                with st.spinner("Summarizing article..."):
                    summary = summarize(text, CURRENT_MODE_KEY)

                st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                st.markdown('<div class="summary-title">Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-body">{summary}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- FOOTER ----------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:10px; font-size:1rem; color:#e5e7eb;
         font-family:Poppins; font-weight:500;'>
         👋 Welcome, Buddy! Ready to summarize anything you want.
    </div>
    """,
    unsafe_allow_html=True,
)
