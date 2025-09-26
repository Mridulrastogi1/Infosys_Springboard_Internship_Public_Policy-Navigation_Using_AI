import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import re
import ollama
from io import BytesIO
import json
import os
import platform
import traceback

# ---------------- Paths ---------------- #
system_os = platform.system()

if system_os == "Windows":
    tess_path = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    poppler_path = r"C:/poppler-25.07.0/Library/bin"
    if os.path.exists(tess_path):
        pytesseract.pytesseract.tesseract_cmd = tess_path
    else:
        st.error("⚠️ Tesseract not found at default path.")
    POPPLER_PATH = poppler_path if os.path.exists(poppler_path) else None
elif system_os in ["Linux", "Darwin"]:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    POPPLER_PATH = None
else:
    POPPLER_PATH = None
    st.error("Unsupported OS")

# ---------------- Helpers ---------------- #
def read_bytes(file):
    if hasattr(file, "getvalue"):
        return file.getvalue()
    pos = file.tell() if hasattr(file, "tell") else None
    if pos is not None:
        file.seek(0)
    data = file.read()
    if pos is not None:
        file.seek(pos)
    return data

def extract_pdf_text_no_ocr(data):
    try:
        reader = PdfReader(BytesIO(data))
        return [p.extract_text() or "" for p in reader.pages]
    except:
        return []

def is_real_text(txt: str) -> bool:
    if not txt or not txt.strip():
        return False
    clean = re.sub(r"[^A-Za-z0-9]", "", txt)
    if len(clean) < 50:
        return False
    ratio = len(clean) / max(len(txt), 1)
    if ratio < 0.2:
        return False
    return True

def run_ocr_on_pdf(data, page_texts, dpi=100):
    updated_pages, ocr_pages = [], 0
    try:
        reader = PdfReader(BytesIO(data))
        total_pages = len(reader.pages)
        for i in range(total_pages):
            page_txt = page_texts[i] if i < len(page_texts) else ""
            if not is_real_text(page_txt):
                images = convert_from_bytes(
                    data,
                    dpi=dpi,
                    first_page=i + 1,
                    last_page=i + 1,
                    poppler_path=POPPLER_PATH,
                )
                ocr_text = "".join(
                    pytesseract.image_to_string(img, lang="eng") for img in images
                )
                updated_pages.append(ocr_text)
                if ocr_text.strip():
                    ocr_pages += 1
            else:
                updated_pages.append(page_txt)
    except Exception as e:
        print("OCR error:", e)
        return "", {"pages": 0, "ocr": 0, "text": 0}

    full_text = "\n".join(updated_pages)
    text_pages = sum(1 for p in updated_pages if p.strip())
    return full_text, {
        "pages": len(updated_pages),
        "ocr": ocr_pages,
        "text": text_pages,
    }

def run_ocr_on_image(data):
    try:
        img = Image.open(BytesIO(data))
        text = pytesseract.image_to_string(img, lang="eng")
    except:
        text = ""
    stats = {
        "pages": 1,
        "ocr": 1 if text.strip() else 0,
        "text": 1 if text.strip() else 0,
    }
    return text, stats

def read_txt(file):
    try:
        return file.read().decode("utf-8")
    except:
        return file.read().decode("latin-1", errors="ignore")

def read_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

# ---------------- Section splitting ---------------- #
def split_sections_full_text(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "", []
    policy_title = lines[0]
    sections = []
    current_area = ""
    current_details = []
    for line in lines[1:]:
        if re.match(r"^[A-Z][A-Za-z\s]{2,}$", line) or line.endswith(":"):
            if current_area or current_details:
                details_text = "\n".join(current_details).strip()
                if not details_text:
                    details_text = (
                        f"Details for '{current_area}' are not explicitly provided in the document."
                    )
                sections.append({"area": current_area, "details": details_text})
            current_area = line.strip().rstrip(":")
            current_details = []
        else:
            current_details.append(line)
    if current_area or current_details:
        details_text = "\n".join(current_details).strip()
        if not details_text:
            details_text = (
                f"Details for '{current_area}' are not explicitly provided in the document."
            )
        sections.append(
            {"area": current_area if current_area else "General", "details": details_text}
        )
    return policy_title, sections

def save_json_full(text, filename="extracted_text.json"):
    policy_title, sections = split_sections_full_text(text)
    data = {"policy": policy_title, "Focus": sections}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return filename

# ---------------- Chunking ---------------- #
def chunk_text(text, chunk_size=1200, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ---------------- Ollama Chatbot ---------------- #
def chat_with_document(prompt, text, chunk_size=1200):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chunks = chunk_text(text, chunk_size=chunk_size)
    context = "\n\n".join(chunks)  # use all chunks for maximum coverage

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant. First try to answer using the document below. "
                "If the answer cannot be found in the document, then provide a helpful general answer "
                "based on your knowledge.\n\n"
                f"Document:\n{context}"
            ),
        },
        *st.session_state.chat_history,
        {"role": "user", "content": prompt},
    ]

    try:
        response = ollama.chat(model="llama3.2:1b", messages=messages)
        answer = response["message"]["content"].strip()

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        return answer
    except Exception as e:
        st.error("❌ Ollama Error: Did you install `llama3.2:1b`? Run `ollama pull llama3.2:1b`.")
        return str(e)

# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="📘 Policy Chatbot OCR", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>📘 Policy Chatbot with OCR & QnA</h1>",
    unsafe_allow_html=True,
)

for key in ["uploaded_file_bytes", "file_type", "extracted_text", "stats", "ocr_applied"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "stats" else {"pages": 0, "ocr": 0, "text": 0}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

left, right = st.columns([2, 2])

with right:
    st.subheader("📂 Upload & OCR")
    uploaded = st.file_uploader(
        "Upload File (PDF, TXT, DOCX, JPG, PNG)", type=["pdf", "txt", "docx", "jpg", "jpeg", "png"]
    )

    if uploaded:
        try:
            data = read_bytes(uploaded)
            st.session_state.uploaded_file_bytes = data
            st.session_state.file_type = uploaded.type
            st.session_state.ocr_applied = False

            if uploaded.type == "application/pdf":
                page_texts = extract_pdf_text_no_ocr(data)
                full_text, stats = run_ocr_on_pdf(data, page_texts)
            elif uploaded.type in ["image/jpeg", "image/png"]:
                full_text, stats = run_ocr_on_image(data)
                st.image(Image.open(BytesIO(data)), width=240)
            elif uploaded.type == "text/plain":
                full_text = read_txt(uploaded)
                stats = {"pages": 1, "ocr": 0, "text": 1 if full_text.strip() else 0}
            elif uploaded.type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ]:
                full_text = read_docx(uploaded)
                stats = {"pages": 1, "ocr": 0, "text": 1 if full_text.strip() else 0}
            else:
                st.error("Unsupported file type")
                full_text, stats = "", {"pages": 0, "ocr": 0, "text": 0}

            st.session_state.extracted_text = full_text
            st.session_state.stats = stats
            st.session_state.ocr_applied = True

            st.caption(
                f"Pages: {stats['pages']} | OCR pages: {stats['ocr']} | Text pages: {stats['text']} | Chars: {len(full_text)}"
            )

            if full_text:
                chunks = chunk_text(full_text, chunk_size=1200)
                st.success(f"✅ Document split into {len(chunks)} chunks for QnA")

                with st.expander("🧾 Full Extracted Text"):
                    st.text_area("Extracted text", full_text, height=300)
                filename = save_json_full(full_text)
                st.download_button("⬇ Download JSON", filename, filename)

        except Exception:
            st.error("Failed to process file.")
            traceback.print_exc()

with left:
    st.subheader("💬 Chatbot (QnA)")
    if st.session_state.extracted_text:
        q = st.text_input("Ask a question about the document:")
        if st.button("Ask") and q:
            ans = chat_with_document(q, st.session_state.extracted_text)
            st.markdown(f"**Bot:** {ans}")

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📝 Conversation History")
            for msg in st.session_state.chat_history:
                role = "**You:**" if msg["role"] == "user" else "**Bot:**"
                st.markdown(f"{role} {msg['content']}")
    else:
        st.info("Upload a policy document to extract text and start QnA.")
