import streamlit as st
import os
import io
import json
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import pandas as pd
import docx
import ollama
import re

CHUNK_SIZE = 3000

def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size
    return chunks

def extract_text_from_pdf(file_bytes):
    images = convert_from_bytes(file_bytes)
    texts = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        texts.append({"page": i+1, "text": text})
    return {"type": "pdf", "pages": texts}

def extract_text_from_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return {"type": "image", "text": text}

def extract_text_from_excel(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    records = df.to_dict(orient='records')
    return {"type": "excel", "rows": records}

def extract_text_from_docx(file_bytes):
    file = io.BytesIO(file_bytes)
    doc = docx.Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return {"type": "docx", "text": paragraphs}

def get_file_type(file_name):
    ext = file_name.split('.')[-1].lower()
    if ext in ['pdf']:
        return 'pdf'
    elif ext in ['jpg', 'jpeg', 'png']:
        return 'image'
    elif ext in ['xlsx']:
        return 'excel'
    elif ext in ['docx']:
        return 'docx'
    else:
        return None

def process_file(uploaded_file):
    file_type = get_file_type(uploaded_file.name)
    if not file_type:
        return {"error": "Unsupported file type"}, None, None
    file_bytes = uploaded_file.read()
    if file_type == "pdf":
        result = extract_text_from_pdf(file_bytes)
    elif file_type == "image":
        result = extract_text_from_image(file_bytes)
    elif file_type == "excel":
        result = extract_text_from_excel(file_bytes)
    elif file_type == "docx":
        result = extract_text_from_docx(file_bytes)
    else:
        result = {"error": "File type not supported"}
    
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    json_file_name = uploaded_file.name + ".json"
    json_path = os.path.join(os.getcwd(), json_file_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    return result, json_str, json_path

def query_ollama(prompt, model='llama3'):
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']

def get_top_chunks(chunks, query, top_k=2):
    scored = [(chunk, len(re.findall(query.lower(), chunk.lower()))) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k] if x[1] > 0] or chunks[:top_k]

st.title("Search and upload file (Ollama)")

col1, col2 = st.columns([2, 2])

with col1:
    search_query = st.text_input("Search box", placeholder="Type your question about the file here...")
    search_button = st.button("Enter")

with col2:
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "xlsx", "jpg", "jpeg", "png"])

extracted_text = ""
summary_result = None
result = None

if "show_extracted" not in st.session_state:
    st.session_state["show_extracted"] = False

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    st.write({
        "File name": uploaded_file.name,
        "File size": uploaded_file.size,
        "Type": uploaded_file.type
    })
    result, json_str, json_path = process_file(uploaded_file)

    st.info(f"JSON file saved to your project folder: {json_path}")
    with open(json_path, "rb") as f:
        st.download_button(
            label="Download extracted JSON",
            data=f,
            file_name=os.path.basename(json_path),
            mime="application/json"
        )

    if "pages" in result:
        extracted_text = "\n".join([p["text"] for p in result["pages"]])
    elif "text" in result:
        if isinstance(result["text"], list):
            extracted_text = "\n".join(result["text"])
        else:
            extracted_text = result["text"]
    elif "rows" in result:
        extracted_text = "\n".join([str(r) for r in result["rows"]])

    # Unhide/Hide buttons for Extracted Text/Data
    if not st.session_state["show_extracted"]:
        if st.button("Unhide Extracted Text/Data"):
            st.session_state["show_extracted"] = True
    else:
        if st.button("Hide Extracted Text/Data"):
            st.session_state["show_extracted"] = False
    if st.session_state["show_extracted"]:
        st.subheader("Extracted Text/Data")
        st.json(result)

    # Button for document summary
    if extracted_text:
        if st.button("Summarize Document"):
            summary_prompt = f"Summarize the following document concisely:\n\n{extracted_text[:12000]}"
            with st.spinner("Creating document summary..."):
                summary_result = query_ollama(summary_prompt)
            st.write(summary_result)
else:
    st.info("No file uploaded yet.")

# Q&A "Enter" button logic
if search_query and search_button:
    if extracted_text:
        text_chunks = chunk_text(extracted_text)
        relevant_chunks = get_top_chunks(text_chunks, search_query, top_k=2)
        context = "\n".join(relevant_chunks)
        qa_prompt = f"Answer the following question based on this document context:\nQuestion: {search_query}\nContext:\n{context}"
        with st.spinner("Searching your document for the answer..."):
            final_answer = query_ollama(qa_prompt)
        st.write(final_answer)
    else:
        with st.spinner("Ollama responding..."):
            ollama_response = query_ollama(search_query)
        st.write(ollama_response)
