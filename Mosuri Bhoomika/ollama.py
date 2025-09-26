import os
import shutil
import json
import requests
import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# --- CONFIGURATION ---
# Update these paths to match your system
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"

# Ollama settings
OLLAMA_MODEL = "llama3"  # Change to any model you have: "mistral", "phi3", "gemma:2b", etc.
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# --- CORE FUNCTIONS ---

def _check_dependencies():
    """Return tuple (tesseract_ok, poppler_ok, messages)"""
    msgs = []
    
    tesseract_path = shutil.which("tesseract") or pytesseract.pytesseract.tesseract_cmd
    poppler_path = shutil.which("pdftoppm", path=POPPLER_PATH)
    
    if tesseract_path and os.path.exists(tesseract_path):
        msgs.append(f"✔️ Tesseract found at: {tesseract_path}")
        t_ok = True
    else:
        msgs.append("❌ Tesseract not found. Please check the path in the script.")
        t_ok = False
        
    if poppler_path and os.path.exists(poppler_path):
        msgs.append(f"✔️ Poppler (pdftoppm) found at: {poppler_path}")
        p_ok = True
    else:
        msgs.append("❌ Poppler (pdftoppm) not found. Please check the POPPLER_PATH in the script.")
        p_ok = False
        
    return (t_ok, p_ok, msgs)

def ocr_pdf_to_text(uploaded_file, dpi=300):
    pdf_bytes = uploaded_file.getvalue()
    images = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=POPPLER_PATH)

    extracted_pages = []
    for i, page_image in enumerate(images):
        text = pytesseract.image_to_string(page_image)
        extracted_pages.append({
            "page_number": i + 1,
            "text_content": text
        })
    return extracted_pages

def chunk_text_from_json(pages_data, chunk_size, chunk_overlap):
    all_chunks = []
    chunk_id_counter = 1
    
    for page in pages_data:
        text = page["text_content"]
        page_num = page["page_number"]
        
        if not text.strip():
            continue

        start_index = 0
        while start_index < len(text):
            end_index = start_index + chunk_size
            chunk_text = text[start_index:end_index]
            
            all_chunks.append({
                "chunk_id": f"page_{page_num}_chunk_{chunk_id_counter}",
                "source_page": page_num,
                "chunk_text": chunk_text
            })
            chunk_id_counter += 1
            
            start_index += chunk_size - chunk_overlap
            if start_index >= len(text):
                break

    return all_chunks

def query_ollama(prompt: str, context: str = "", model: str = OLLAMA_MODEL) -> str:
    full_prompt = f"""You are a helpful assistant that answers questions based ONLY on the following document context.
If the answer is not in the context, say "I don't know based on the provided document."

Document context:
{context}

Question: {prompt}

Answer:"""

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"❌ Error: Ollama returned status {response.status_code}"
    except Exception as e:
        return f"❌ Failed to connect to Ollama: {str(e)}"

def retrieve_relevant_chunks(query: str, chunks: list, top_k: int = 3) -> str:
    """
    Simple retrieval: returns the first `top_k` chunks.
    Replace this with keyword search or vector similarity for better results.
    """
    selected = chunks[:top_k]
    return "\n\n".join([chunk["chunk_text"] for chunk in selected])

# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("📄 Public Policy Navigation using AI")
st.markdown("Upload a PDF, extract text, chunk it, and chat with your document using a local LLM (Ollama).")

# --- Dependency Check ---
st.subheader("Dependency Status")
t_ok, p_ok, dep_msgs = _check_dependencies()
for m in dep_msgs:
    st.caption(m)

if not t_ok or not p_ok:
    st.error(
        "A required dependency (Tesseract or Poppler) was not found. "
        "Please ensure they are installed and the paths at the top of the script are correct."
    )

# --- Step 1: PDF Upload and OCR ---
st.header("Step 1: Upload PDF and Extract Text")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", disabled=(not t_ok or not p_ok))

if uploaded_file is not None:
    if 'file_name' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        # Reset state on new file
        for key in ['extracted_data', 'chunked_data', 'chat_history', 'file_name']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.file_name = uploaded_file.name
        
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        dpi = st.number_input("OCR Image Quality (DPI)", min_value=100, max_value=600, value=300, step=50)
        run_ocr = st.button("✅ Extract Text from PDF", type="primary")

    if run_ocr:
        with st.spinner("Processing PDF... This may take a moment."):
            try:
                extracted_data = ocr_pdf_to_text(uploaded_file, dpi=dpi)
                st.session_state.extracted_data = extracted_data
                st.success("Text extraction complete!")
            except Exception as e:
                st.error(f"An error occurred during OCR: {e}")

    if 'extracted_data' in st.session_state:
        extracted_data = st.session_state.extracted_data
        final_json = {
            "document_name": uploaded_file.name,
            "total_pages": len(extracted_data),
            "pages": extracted_data
        }
        json_string = json.dumps(final_json, ensure_ascii=False, indent=4)
        base_name = os.path.splitext(uploaded_file.name)[0]
        download_filename = f"{base_name}_extracted.json"

        st.download_button(
            label="📥 Download Full Extracted JSON",
            data=json_string,
            file_name=download_filename,
            mime="application/json"
        )
        
        # --- Step 2: Chunking ---
        st.header("Step 2: Chunk the Extracted Text")
        st.markdown("Break the text into smaller pieces for the LLM.")
        
        col_chunk1, col_chunk2, col_chunk3 = st.columns(3)
        with col_chunk1:
            chunk_size = st.number_input("Chunk Size (characters)", min_value=100, max_value=8000, value=1000)
        with col_chunk2:
            chunk_overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=1000, value=200)
        
        if chunk_overlap >= chunk_size:
            st.warning("Chunk overlap should be smaller than chunk size.")
        
        run_chunking = st.button("🔪 Chunk Text", type="primary")

        if run_chunking:
            with st.spinner("Chunking text..."):
                chunked_data = chunk_text_from_json(st.session_state.extracted_data, chunk_size, chunk_overlap)
                st.session_state.chunked_data = chunked_data
                st.success(f"Text successfully divided into {len(chunked_data)} chunks.")

        if 'chunked_data' in st.session_state:
            chunked_json_string = json.dumps(st.session_state.chunked_data, ensure_ascii=False, indent=4)
            chunked_filename = f"{base_name}_chunked.json"
            
            st.download_button(
                label="📥 Download Chunked JSON",
                data=chunked_json_string,
                file_name=chunked_filename,
                mime="application/json"
            )

            st.subheader("Preview of Chunked Data")
            st.json(st.session_state.chunked_data[:5])  # Show first 5 chunks

            # --- Step 3: Chatbot ---
            st.header("Step 3: Chat with Your Document 💬")
            st.markdown("Ask questions about the uploaded PDF. The AI uses the extracted text to answer.")

            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User input
            if user_query := st.chat_input("Ask a question about the document..."):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Retrieve context (simple: first 3 chunks)
                context = retrieve_relevant_chunks(user_query, st.session_state.chunked_data, top_k=3)

                # Get LLM response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = query_ollama(user_query, context=context, model=OLLAMA_MODEL)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

else:
    st.info("👆 Upload a PDF to get started.")
