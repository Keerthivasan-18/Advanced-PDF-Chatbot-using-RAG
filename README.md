# 🤖 Advanced PDF Chatbot using RAG (Retrieval-Augmented Generation)

This project is an advanced PDF-based chatbot built using **RAG (Retrieval-Augmented Generation)** from Hugging Face, FAISS for fast similarity search, Sentence Transformers for embedding, and Gradio for an interactive UI. It allows users to upload any PDF and then ask natural language questions about its contents, with answers generated intelligently using context-relevant information retrieved from the document.

---

## 📌 Features

- 🔍 **Chunk-based document retrieval**
- 🧠 **Sentence embeddings** via `intfloat/e5-large-v2`
- ⚡ **Fast semantic search** using `FAISS`
- 📟 **Question answering** powered by `facebook/rag-token-nq`
- 🌐 **Gradio UI** for user-friendly interactions
- 📄 Supports **any custom PDF** upload
- 🚀 Easily deployable in **Google Colab** or locally

---

## 👩‍💻 Tech Stack

| Component                 | Description                         |
| ------------------------- | ----------------------------------- |
| PyMuPDF (`fitz`)          | PDF text extraction                 |
| Sentence Transformers     | Text embedding                      |
| FAISS                     | Semantic vector search              |
| Hugging Face Transformers | RAG model (`facebook/rag-token-nq`) |
| Gradio                    | Web-based UI interface              |
| Python                    | Programming language                |

---

## 🧑‍💻 How it Works

1. **PDF Upload**  
   Text is extracted using `PyMuPDF`.

2. **Chunking**  
   The extracted text is split into overlapping chunks to preserve context.

3. **Embedding & Indexing**  
   Chunks are embedded with `e5-large-v2` and indexed using FAISS.

4. **Query Processing**  
   When a question is asked, the top matching chunks are retrieved.

5. **Answer Generation**  
   RAG generates an answer using the relevant document chunks.

---

## 🚀 Getting Started

### 1. Clone the Repo & Install Dependencies

```bash
!pip install -q transformers sentence-transformers faiss-cpu pymupdf gradio
```

### 2. Run the App

Paste the full code in a Python script or Jupyter/Colab notebook and run it. The Gradio app will open with:

```
✅ PDF loaded. Ask your questions below.
```

---

## 📷 Sample Use Case

1. Upload a research paper PDF.
2. Ask:
   - *"What is the main contribution of this paper?"*
   - *"Explain the algorithm proposed."*

---

## 📁 File Structure

```bash
.
├── app.py  # (your main code if using as script)
├── README.md
└── requirements.txt (optional)
```

---

## 📬 Contact

For queries, reach out at: **keerthivasang50@gmail.com**
