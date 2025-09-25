import streamlit as st
import pandas as pd
import docx
from PyPDF2 import PdfReader
import os
from pdf2image import convert_from_path
import pytesseract
from docx import Document
import tempfile
import json
import ollama
import numpy as np
from numpy.linalg import norm
from io import BytesIO
# Configure Tesseract & Poppler
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
poppler_path = r"D:\\Release-25.07.0-0\\poppler-25.07.0\\Library\\bin"

# App title
st.set_page_config(page_title="📂 File Previewer", layout="wide")
st.title("📂 Smart File Upload & Previewer")
st.markdown("### Upload PDF files and preview + extract text with OCR")

# Sidebar upload
st.sidebar.header("📁 Upload Section")
uploaded_files = st.sidebar.file_uploader(
    "Choose a file", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "file_content" not in st.session_state:
    st.session_state["file_content"] = ""

# OCR function
def pdf_to_word(pdf_path, docx_path, json_path):
    pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
    doc = Document()
    extracted_data = []

    for i, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page, lang="eng", config="--psm 6")
        doc.add_paragraph(text)
        extracted_data.append({"page": i, "text": text})
        st.write(f"✅ Processed page {i}/{len(pages)}")

    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)

    json_buffer = BytesIO()
    json_buffer.write(json.dumps(extracted_data, ensure_ascii=False, indent=4).encode("utf-8"))
    json_buffer.seek(0)

    return extracted_data, docx_buffer, json_buffer
def chunk_text(text, chunk_size=500, overlap=100):

   #Splits long text into overlapping chunks.

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_json(extracted_data, chunk_size=500, overlap=100):
    """
    Takes OCR JSON and splits each page into smaller chunks.
    Returns a list of chunks with page info.
    """
    all_chunks = []
    for page in extracted_data:
        page_text = page["text"]
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        for i, chunk in enumerate(page_chunks, start=1):
            all_chunks.append({
                "page": page["page"],
                "chunk_id": i,
                "text": chunk
            })
    return all_chunks
def get_embedding(text):
    # Return embedding vector for text using Ollama embedding model.
    try:
        # Ollama embeddings call may vary; this is the expected format
        resp = ollama.embeddings(model="nomic-embed-text", prompt=text)
        emb = resp.get("embedding") or resp.get("data", [{}])[0].get("embedding")
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None
def build_vector_store(chunks):
    """Build list of (chunk, embedding) and store norms for fast similarity."""
    store = []
    for c in chunks:
        emb = get_embedding(c["text"])
        if emb is None:
            continue
        store.append({"chunk": c, "emb": emb, "norm": norm(emb)})
    return store
def search_similar(query, vector_store, top_k=5):
    """Return top_k chunks most similar to query (cosine similarity)."""
    q_emb = get_embedding(query)
    if q_emb is None or not vector_store:
        return []
    q_norm = norm(q_emb)
    sims = []
    for item in vector_store:
        score = float(np.dot(q_emb, item["emb"]) / (q_norm * item["norm"] + 1e-12))
        sims.append((score, item["chunk"]))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in sims[:top_k]]

def query_llama(user_input, vector_store):
    try:
        # Retrieve most relevant chunks
        top_chunks = search_similar(user_input, vector_store, top_k=5)

        context = "\n".join([c["text"] for c in top_chunks])        
        #st.write("🔄 Sending request to Ollama...")
        response = ollama.chat(
            model="llama3.1:8b", 
            messages=[
                {"role": "system", "content": "You are an assistant that helps answer questions about documents."},
                {"role": "user", "content": f"Here is the document:\n{context[:2000]}\n\nQuestion: {user_input}"}
            ]
        )
       # st.write("✅ Response received from Ollama.",response)
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Error communicating with LLaMA: {e}"

# Main app body
if uploaded_files:
    all_extracted_data = []
    all_chunks = []

    for uploaded_file in uploaded_files:
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_pdf_path = tmp.name

            output_docx = temp_pdf_path.replace(".pdf", ".docx")
            output_json = temp_pdf_path.replace(".pdf", ".json")

        # Avoid re-running OCR if file already processed
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = {}

        if uploaded_file.name not in st.session_state["processed_files"]:
            with st.spinner(f"🔍 Running OCR on {uploaded_file.name}..."):
                extracted_data, docx_buffer, json_buffer = pdf_to_word(temp_pdf_path, output_docx, output_json)

            st.session_state["processed_files"][uploaded_file.name] = {
                "extracted_data": extracted_data,
                "docx_buffer": docx_buffer,
                "json_buffer": json_buffer
            }

            # Save chunks JSON per file
            file_chunks = chunk_json(extracted_data)
            chunks_output_path = os.path.splitext(uploaded_file.name)[0] + "_chunks.json"
            with open(chunks_output_path, "w", encoding="utf-8") as f:
                json.dump(file_chunks, f, ensure_ascii=False, indent=4)

            st.success(f"✅ Chunks saved locally as {chunks_output_path}")

        else:
            extracted_data = st.session_state["processed_files"][uploaded_file.name]["extracted_data"]

        # Collect all files’ text + chunks
        all_extracted_data.extend(extracted_data)
        all_chunks.extend(chunk_json(extracted_data))

    # Save combined extracted text
    st.session_state["file_content"] = "\n".join([p["text"] for p in all_extracted_data])
    st.session_state["extracted_data"] = all_extracted_data
    st.session_state["chunks"] = all_chunks

    # Save one merged JSON file for all PDFs
    merged_json_output_path = "all_files_extracted.json"
    with open(merged_json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted_data, f, ensure_ascii=False, indent=4)

    # Build vector store from all chunks
    with st.spinner("⚡ Generating embeddings for all chunks..."):
        st.session_state["vector_store"] = build_vector_store(all_chunks)
    st.success("✅ Vector store created for all PDFs!")


    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Preview", "📊 File Info", "💬 Chat", "⬇️ Download"])

    with tab1:
        st.subheader("📑 File Preview (first 1000 chars)")
        st.text(st.session_state["file_content"][:1000])

    with tab2:
        st.subheader("📊 File Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("📂 Total Files", len(uploaded_files))
        col2.metric("📄 Pages Extracted", len(all_extracted_data))
        col3.metric("🔖 Total Chunks", len(all_chunks))

        st.markdown("---")

        with st.expander("📑 Detailed Info per File"):
            for file in uploaded_files:
                st.write(f"**📌 {file.name}**")
                st.write(f"- Size: `{file.size/1024:.2f} KB`")
                st.write(f"- Pages Extracted: `{len(st.session_state['processed_files'][file.name]['extracted_data'])}`")
                st.markdown("---")

    with tab3:
        st.header("💬 Chat about your file")

    # Chat input
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background-color:#DCF8C6; padding:10px; border-radius:10px; margin:5px 0;">
                🧑 <b>You:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#F1F0F0; padding:10px; border-radius:10px; margin:5px 0;">
                🤖 <b>Bot:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
        user_input = st.text_input("Type your question", key="chat_input")
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
                    # Use vector store instead of raw chunks
            bot_reply = query_llama(user_input, st.session_state["vector_store"])
            st.session_state["chat_history"].append({"role": "assistant", "content": bot_reply})

            # Show only the latest reply below the chatbox
            st.markdown(f"**🤖 Bot:** {bot_reply}")

    # Sidebar chat history
    st.sidebar.header("💬 Chat History")
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.sidebar.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.sidebar.markdown(f"**🤖 Bot:** {msg['content']}")
    st.write("Total chunks:", len(st.session_state["chunks"]))
    with tab4:
        st.subheader("⬇️ Download Extracted Files")
        st.info("Pick a file to download:")
        if "processed_files" in st.session_state and st.session_state["processed_files"]:
            # Let user pick which file to download
            file_choices = list(st.session_state["processed_files"].keys())
            selected_file = st.selectbox("Select a file to download", file_choices)

            if selected_file:
                file_data = st.session_state["processed_files"][selected_file]

                st.download_button(
                    "Download Word File (.docx)",
                    data=file_data["docx_buffer"],
                    file_name=f"{os.path.splitext(selected_file)[0]}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

                st.download_button(
                    "Download JSON File",
                    data=file_data["json_buffer"],
                    file_name=f"{os.path.splitext(selected_file)[0]}.json",
                    mime="application/json"
                )

            # Optional: download all merged JSON at once
            st.download_button(
                "⬇️ Download ALL Extracted Data (JSON)",
                data=json.dumps(st.session_state["extracted_data"], ensure_ascii=False, indent=4),
                file_name="all_files_extracted.json",
                mime="application/json"
            )

        else:
            st.info("⚠️ No processed files available yet. Upload a PDF first.")


else:
    st.info("📤 Please upload a PDF from the sidebar to begin.")
