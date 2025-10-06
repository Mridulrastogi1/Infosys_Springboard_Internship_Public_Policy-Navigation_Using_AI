**📄 Public Policy Navigation Using AI**

This project is an AI-powered tool designed to simplify the exploration and understanding of complex public policy documents.
Policy documents, legislative texts, and government reports are often dense and lengthy, making it difficult for analysts, researchers, and students to find specific information quickly. This application addresses that challenge by providing a powerful, intuitive, and entirely local conversational interface for your document library. It leverages a Retrieval-Augmented Generation (RAG) pipeline powered by Ollama, allowing you to "chat" with your documents. Simply upload your files, and the system will process and index their content. You can then ask questions in plain English, and the AI will retrieve the most relevant passages to generate accurate, context-aware answers. Because the entire process runs locally, your data remains completely private and secure.

**✨ Key Features**

**Multi-Format Upload:** Supports a wide range of document types including PDF (both text-based and scanned), TXT, CSV, XLSX, PNG, and JPG.

**Intelligent Text Extraction:** Automatically extracts text from files, using Optical Character Recognition (OCR) for images and scanned PDFs.

**Local-First AI:** Powered by Ollama to run large language models on your own machine. No need for API keys or internet dependency.

**Interactive Chatbot:** A user-friendly Streamlit interface to ask questions and receive answers.

**Transparent Sourcing:** The chatbot shows you which specific chunks of text from the source documents were used to generate its answer, ensuring you can verify the information.

**In-Memory Vector Index:** A lightweight and fast in-session vector store for efficient similarity searches.

**🔧 Technologies Used**

This project is built with a modern, open-source stack.

Application Framework

**Streamlit:** For building and running the interactive web application UI.

AI & Language Models

**Ollama:** As the server for running local large language models.

ollama Python Client: For interacting with the Ollama API.

**Embedding Model:** nomic-embed-text (for converting text chunks into numerical vectors).

Chat Model: llama3.2 (for understanding context and generating answers).

Data Processing & Extraction

**Pandas:** For parsing and handling .csv and .xlsx files.

**PyMuPDF (fitz) or Py2PDF:** For extracting text and images from .pdf documents.

**Pytesseract:** The OCR engine used to extract text from images.

**Pillow (PIL):** For handling image data before it's passed to the OCR engine.

Core Python Libraries

**NumPy:** For high-performance numerical operations on embeddings.

**hashlib:** For generating unique, content-based IDs for uploaded files.
