import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import json
import tempfile
from ollama import chat
import difflib

st.set_page_config(page_title="OCR + Llama Chat", page_icon="🤖", layout="wide")
st.title("File upload with Chat(Ollama)")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def find_relevant_chunks(query, file_history, top_n=3):
    """Find top-N chunks most related to the query using simple keyword similarity."""
    all_chunks = []
    for file, pages in file_history.items():
        for page, chunks in pages.items():
            for chunk in chunks:
                all_chunks.append(chunk)

    if not all_chunks:
        return ["(No document uploaded yet.)"]

    scored = [(chunk, difflib.SequenceMatcher(None, query, chunk).ratio()) for chunk in all_chunks]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    return [c for c, _ in scored[:top_n]]

if "file_history" not in st.session_state:
    st.session_state.file_history = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2, tab3 = st.tabs(["📂 Upload & Process", "📜 History", "💬 Chat with Llama"])

with tab1:
    st.subheader("📂 Upload your PDF file")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        st.info("⏳ Processing your PDF, please wait...")

        pages = convert_from_path(pdf_path, dpi=300)

        ocr_results = {}
        for i, page in enumerate(pages, start=1):
            text = pytesseract.image_to_string(page, lang="eng").strip()
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            ocr_results[f"page_{i}"] = chunks

        st.session_state.file_history[uploaded_file.name] = ocr_results

        st.success(f"✅ File '{uploaded_file.name}' processed successfully!")

        st.subheader("🔎 Preview of Extracted Text (first 2 pages)")
        preview = {k: v for k, v in list(ocr_results.items())[:2]}
        st.json(preview)

        output_json = json.dumps(ocr_results, indent=4, ensure_ascii=False)
        st.download_button(
            label="📥 Download Extracted JSON",
            data=output_json,
            file_name=f"{uploaded_file.name}_output.json",
            mime="application/json"
        )

with tab2:
    st.subheader("📜 Uploaded Files History")

    if st.session_state.file_history:
        st.write("Here are the files you have uploaded in this session:")
        for file_name in st.session_state.file_history.keys():
            st.markdown(f"- 📄 **{file_name}**")
    else:
        st.info("No files uploaded yet. Upload a file in the first tab.")

MAX_CONTEXT = 1000  

with tab3:
    st.subheader("💬 Chat with Llama 3 (Ollama)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask something about the uploaded document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        chunks = find_relevant_chunks(prompt, st.session_state.file_history, top_n=3)
        context_text = "\n\n".join(chunks)[:MAX_CONTEXT]

        system_prompt = (
            "You are an assistant that answers questions based only "
            "on the following document excerpts:\n\n" + context_text
        )

        try:
            response = chat(
                model="llama3.1:8b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            assistant_msg = response["message"]["content"]

        except Exception as e:
            assistant_msg = f"⚠️ Error communicating with Ollama: {e}"

        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        st.chat_message("assistant").write(assistant_msg)
