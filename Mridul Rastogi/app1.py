import io
import time
import hashlib
import pandas as pd
import streamlit as st
import fitz
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import json
import os
import numpy as np
import math
from typing import List, Dict, Any

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
TOP_K = 4
MAX_CONTEXT_CHARS = 3000

st.set_page_config(page_title="📁 File Upload + Chatbot (Ollama RAG)", page_icon="📂", layout="wide")
st.title("📂 File Upload App with 🤖 Chatbot (Ollama)")

if "files" not in st.session_state:
    st.session_state["files"] = []
if "last_upload" not in st.session_state:
    st.session_state["last_upload"] = []
if "selected_from_history" not in st.session_state:
    st.session_state["selected_from_history"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vector_index" not in st.session_state:
    st.session_state["vector_index"] = []

def make_record(uploaded_file):
    data = uploaded_file.getvalue()
    fid = hashlib.sha1(data).hexdigest()[:12]
    return {
        "id": fid,
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "data": data,
        "size": len(data),
    }

def add_to_history(records):
    existing = {f["id"] for f in st.session_state["files"]}
    for r in records:
        if r["id"] not in existing:
            st.session_state["files"].append(r)
            existing.add(r["id"])

def human_size(nbytes):
    for unit in ["B","KB","MB","GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"

def perform_ocr_from_image_bytes(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"(OCR failed: {e})"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c] # Filter out empty chunks

def ollama_embeddings(texts: List[str], model: str = EMBED_MODEL) -> List[np.ndarray]:
    """
    Generates embeddings for a list of texts using Ollama.
    This function has been corrected to handle texts individually to match the Ollama API,
    and its error handling has been made more robust to prevent silent failures.
    """
    if not isinstance(texts, list):
        texts = [texts]

    all_embeddings = []

    if OLLAMA_AVAILABLE:
        try:
            # ANNOTATION: The primary fix is to iterate through each text and call the
            # embeddings function individually. This respects the API contract that
            # the 'prompt' parameter must be a string, not a list.
            for text in texts:
                resp = ollama.embeddings(model=model, prompt=text)
                all_embeddings.append(np.array(resp["embedding"]))
            return all_embeddings
        except Exception as e:
            st.warning(f"Ollama python client embeddings call failed: {e} — trying REST fallback")

    try:
        import requests
        embs = []
        for t in texts:
            # ANNOTATION: The second fix is in the REST fallback payload. The key must be
            # 'prompt', not 'input', as per the Ollama REST API documentation.
            payload = {"model": model, "prompt": t}
            r = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "embedding" in data:
                embs.append(np.array(data["embedding"]))
            else:
                raise ValueError("Unexpected embeddings response format from REST API.")
        return embs
    except Exception as e:
        # ANNOTATION: The third fix is to avoid returning zero-vectors on failure.
        # Instead, we display a clear error and return an empty list. This prevents
        # the "zero-vector collapse" and allows calling functions to handle the
        # failure gracefully.
        st.error(f"Failed to generate embeddings via both client and REST API: {e}")
        return

def ollama_chat(messages, model=CHAT_MODEL, stream=False):
    if OLLAMA_AVAILABLE:
        try:
            resp = ollama.chat(model=model, messages=messages, stream=stream)
            if not stream:
                return resp['message']['content']
            full_response = ""
            for chunk in resp:
                full_response += chunk['message']['content']
            return full_response
        except Exception as e:
            st.warning(f"Ollama python client chat error: {e} — trying REST fallback")
    try:
        import requests
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={"model": model, "messages": messages, "stream": False}, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data['message']['content']
    except Exception as e:
        st.error(f"Ollama REST chat request failed: {e}")
        return "(Ollama unavailable)"

def index_chunks_for_record(record, chunked_data):
    entries = []
    for page_key, chunks in chunked_data.items():
        for i, chunk in enumerate(chunks, start=1):
            chunk_id = f"{record['id']}_{page_key}_{i}"
            entries.append({
                "doc_id": record["id"],
                "doc_name": record["name"],
                "chunk_id": chunk_id,
                "chunk_text": chunk
            })
    texts = [e["chunk_text"] for e in entries]
    if not texts:
        return

    embeddings = ollama_embeddings(texts)
    # Ensure we got embeddings back before proceeding
    if not embeddings or len(embeddings)!= len(entries):
        st.error("Could not generate embeddings for the document. Indexing aborted.")
        return

    # Clear old index entries for this document ID before adding new ones
    st.session_state["vector_index"] = [
        entry for entry in st.session_state["vector_index"] 
        if entry["doc_id"]!= record["id"]
    ]

    for e, emb in zip(entries, embeddings):
        st.session_state["vector_index"].append({
            "doc_id": e["doc_id"],
            "doc_name": e["doc_name"],
            "chunk_id": e["chunk_id"],
            "chunk_text": e["chunk_text"],
            "embedding": emb
        })

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    # Handle potential zero vectors gracefully
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    result = np.dot(a, b) / (norm_a * norm_b)
    return result.item()

def retrieve_top_k(query, k=TOP_K):
    # ANNOTATION: The call to ollama_embeddings now returns a list with one item or an empty list.
    q_embs = ollama_embeddings([query]) # Pass query as a list
    if not q_embs:
        st.warning("Could not embed query. Retrieval failed.")
        return
    q_emb = q_embs[0]
    
    if not st.session_state["vector_index"]:
        st.info("No documents have been indexed yet. Please upload and process a file.")
        return
        
    scored = []
    for entry in st.session_state["vector_index"]:
        score = cosine_sim(q_emb, entry.get("embedding"))
        scored.append((score, entry))
    
    scored.sort(key=lambda x: x, reverse=True)
    return [item for score, item in scored[:k]]

def save_extracted_text(record, extracted_text):
    chunked_data = {}
    for key, text in extracted_text.items():
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        chunked_data[key] = chunks
    data = {
        "id": record["id"],
        "name": record["name"],
        "type": record["type"],
        "size": record["size"],
        "chunks": chunked_data
    }
    os.makedirs("extracted", exist_ok=True)
    file_path = os.path.join("extracted", f"{record['id']}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Index the chunks immediately after extraction
    index_chunks_for_record(record, chunked_data)
    return data

def preview_file(rec, show_text_expander=True, show_thumbs_expander=True, key_prefix=""):
    """
    Displays a preview of the file, extracts its text content, and indexes it.
    This updated version includes handlers for TXT, CSV, and XLSX files.
    """
    save_data = None
    st.markdown(f"### {rec['name']}")
    st.caption(f"Type: `{rec['type']}` • Size: {human_size(rec['size'])}")

    # --- IMAGE FILE HANDLER ---
    if rec["type"].startswith("image/"):
        st.image(rec["data"], caption=rec["name"], use_container_width=True)
        with st.spinner("Extracting text with OCR..."):
            ocr_text = perform_ocr_from_image_bytes(rec["data"])
            save_data = save_extracted_text(rec, {"page_1": ocr_text})
        st.success(f"Indexed text from {rec['name']}.")
        if show_text_expander and save_data:
            with st.expander("📄 OCR Extracted Text (Chunks)"):
                for i, chunk in enumerate(save_data["chunks"]["page_1"], start=1):
                    st.text_area(f"Chunk {i}", chunk, height=150, key=f"{key_prefix}chunk_{rec['id']}_{i}")

    # --- PDF FILE HANDLER ---
    elif rec["type"] == "application/pdf":
        try:
            doc = fitz.open(stream=rec["data"], filetype="pdf")
        except Exception as e:
            st.error(f"Couldn't open PDF: {e}")
            return

        with st.spinner("Extracting and indexing PDF text..."):
            extracted_text = {}
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if not text.strip(): # If no text, try OCR on the page image
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        image_bytes = pix.tobytes("png")
                        text = perform_ocr_from_image_bytes(image_bytes)
                    except Exception as ocr_e:
                        text = f"(OCR on page {i} failed: {ocr_e})"
                extracted_text[f"page_{i}"] = text
            save_data = save_extracted_text(rec, extracted_text)
        st.success(f"Indexed {len(doc)} pages from {rec['name']}.")

        num_pages = len(doc)
        page_num_to_show = 1
        if num_pages > 1:
            page_num_to_show = st.slider("Select page to view", 1, num_pages, 1, key=f"{key_prefix}page_{rec['id']}")

        page = doc[page_num_to_show - 1]
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            st.image(pix.tobytes("png"), caption=f"Page {page_num_to_show}", use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't render page image: {e}")

        if show_text_expander and save_data:
            with st.expander("Show extracted text in chunks for this page"):
                chunks = save_data["chunks"].get(f"page_{page_num_to_show}")
                if chunks:
                    for j, chunk in enumerate(chunks, start=1):
                        st.text_area(f"Page {page_num_to_show} - Chunk {j}", chunk, height=150, key=f"{key_prefix}pdf_chunk_{rec['id']}_{page_num_to_show}_{j}")
                else:
                    st.info("No text extracted for this page.")

    # --- TXT AND CSV FILE HANDLER ---
    elif rec["type"] in ["text/plain", "text/csv"]:
        try:
            text = rec["data"].decode("utf-8")
            if rec["type"] == "text/csv":
                # For previewing, show the table
                df = pd.read_csv(io.StringIO(text))
                st.dataframe(df)
                # For RAG, convert the entire dataframe to a single text string
                text = df.to_string()
            else:
                st.text_area("File Content", text, height=400)

            save_data = save_extracted_text(rec, {"content": text})
            st.success(f"Indexed content from {rec['name']}.")
        except Exception as e:
            st.error(f"Error processing plain text/CSV file: {e}")

    # --- EXCEL FILE HANDLER ---
    elif "spreadsheetml" in rec["type"]: # Handles .xlsx
        try:
            with st.spinner(f"Processing Excel file: {rec['name']}..."):
                xls = pd.ExcelFile(io.BytesIO(rec["data"]))
                sheet_names = xls.sheet_names

                # For RAG, combine all sheets into one text block
                all_sheets_text = []
                for sheet in sheet_names:
                    sheet_df = pd.read_excel(xls, sheet_name=sheet)
                    # Add a header for each sheet to provide context
                    all_sheets_text.append(f"--- SHEET: {sheet} ---\n{sheet_df.to_string()}")
                text = "\n\n".join(all_sheets_text)
                save_data = save_extracted_text(rec, {"content": text})
            st.success(f"Indexed {len(sheet_names)} sheet(s) from {rec['name']}.")

            # For previewing, let the user select a sheet
            chosen_sheet = sheet_names[0]
            if len(sheet_names) > 1:
                chosen_sheet = st.selectbox("Select a sheet to preview", sheet_names, key=f"{key_prefix}sheet_{rec['id']}")
            df_to_show = pd.read_excel(xls, sheet_name=chosen_sheet)
            st.dataframe(df_to_show)

        except Exception as e:
            st.error(f"Error processing Excel file: {e}")

    # --- FALLBACK FOR OTHER FILE TYPES ---
    else:
        st.warning(f"Preview and indexing not implemented for file type: {rec['type']}")

    # --- DOWNLOAD BUTTONS ---
    if save_data:
        json_string = json.dumps(save_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download Extracted Text (JSON)",
            data=json_string,
            file_name=f"{rec['id']}_extracted.json",
            mime="application/json",
            key=f"{key_prefix}dl_json_{rec['id']}",
            use_container_width=True,
        )

    st.download_button("⬇️ Download Original", data=rec["data"], file_name=rec["name"], key=f"{key_prefix}dl_{rec['id']}", use_container_width=True)
    st.divider()

# --- UI LAYOUT ---
st.sidebar.header("📂 History")
if st.session_state["files"]:
    options = {f"{f['name']} ({human_size(f['size'])})": f["id"] for f in st.session_state["files"]}
    chosen = st.sidebar.selectbox("Open from history", list(options.keys()))
    if st.sidebar.button("Open in Preview", use_container_width=True):
        st.session_state["selected_from_history"] = options[chosen]
else:
    st.sidebar.info("No files uploaded yet.")

if st.sidebar.button("🗑️ Clear history & Index", use_container_width=True):
    st.session_state["files"] = []
    st.session_state["last_upload"] = []
    st.session_state["selected_from_history"] = None
    st.session_state["messages"] = []
    st.session_state["vector_index"] = []
    st.rerun()

tab_upload, tab_preview, tab_chat, tab_history = st.tabs(
    ["📤 Upload & Process", "👀 Preview", "🤖 Chatbot", "📜 History"]
)

with tab_upload:
    st.subheader("Upload and Process Files")
    st.info("Files uploaded here will be automatically chunked and indexed for the chatbot.")
    files = st.file_uploader(
        "Drag & drop or browse",
        type=["csv", "txt", "png", "jpg", "pdf", "xlsx"],
        accept_multiple_files=True,
        key="uploader",
    )
    if files:
        records = [make_record(f) for f in files]
        st.session_state["last_upload"] = records
        add_to_history(records)
        
        # Process files immediately upon upload
        for r in records:
            with st.status(f"Processing {r['name']}..."):
                preview_file(r, show_text_expander=False, show_thumbs_expander=False, key_prefix="upload_")
        st.success("All new files have been processed and indexed.")


with tab_preview:
    st.subheader("Preview Files")
    if not st.session_state["files"]:
        st.info("History is empty. Upload files in the first tab.")
    else:
        file_map = {f["id"]: f for f in st.session_state["files"]}
        default_id = st.session_state["selected_from_history"] or st.session_state["files"][-1]["id"]
        ids = [f["id"] for f in st.session_state["files"]]
        try:
            idx_default = ids.index(default_id)
        except ValueError:
            idx_default = len(ids) - 1
            
        label_map = [f"{f['name']} ({human_size(f['size'])})" for f in st.session_state["files"]]
        chosen_idx = st.selectbox("Pick a file from history", range(len(ids)),
                                  index=idx_default,
                                  format_func=lambda i: label_map[i],
                                  key="history_select_in_preview")
        if chosen_idx is not None:
            preview_file(st.session_state["files"][chosen_idx], key_prefix="preview_")


with tab_chat:
    st.subheader("Chat with 🤖 Ollama-powered Bot")
    if not st.session_state["vector_index"]:
        st.warning("The document index is empty. Please upload and process a file first.")
    
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant document chunks..."):
                top_chunks = retrieve_top_k(prompt, k=TOP_K)
            
            if not top_chunks:
                st.write("I couldn't find any relevant information in the uploaded documents to answer your question.")
                st.session_state["messages"].append({"role": "assistant", "content": "I couldn't find any relevant information in the uploaded documents to answer your question."})
            else:
                context_pieces = []
                total_len = 0
                for ent in top_chunks:
                    piece = f"Source: {ent['doc_name']} (chunk {ent['chunk_id']})\n{ent['chunk_text']}\n---\n"
                    if total_len + len(piece) > MAX_CONTEXT_CHARS:
                        break
                    context_pieces.append(piece)
                    total_len += len(piece)
                
                context_text = "\n".join(context_pieces)
                system_msg = {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided CONTEXT from documents to answer the user's question. If the answer is not in the context, state that based on the provided documents, you cannot answer the question."
                }
                user_msg_content = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{prompt}"
                user_msg = {"role": "user", "content": user_msg_content}
                messages_for_model = [system_msg, user_msg]
                
                with st.spinner("Generating answer from Ollama..."):
                    assistant_response = ollama_chat(messages_for_model, model=CHAT_MODEL)
                
                st.markdown(assistant_response)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

                with st.expander("Chunks used for context (retrieved by similarity)"):
                    for i, ent in enumerate(top_chunks, start=1):
                        st.write(f"**Rank {i} — {ent['doc_name']}** (chunk id: {ent['chunk_id']})")
                        st.code(ent['chunk_text'], language=None)

with tab_history:
    st.subheader("📜 Upload History")
    st.info("This table shows all unique files that have been uploaded and processed in the current session.")

    if not st.session_state["files"]:
        st.warning("No files have been uploaded yet.")
    else:
        # Prepare data for a clean table display
        history_data = [
            {
                "File Name": f["name"],
                "Size": human_size(f["size"]),
                "Type": f["type"],
                "Unique ID": f["id"]
            }
            for f in st.session_state["files"]
        ]
        # Use a Pandas DataFrame for a nice table view
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

