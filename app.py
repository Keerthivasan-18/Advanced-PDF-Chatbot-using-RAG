# 📦 Install all required packages
!pip install -q transformers sentence-transformers faiss-cpu pymupdf gradio

import fitz  # PyMuPDF
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# 📄 Step 1: Extract text from PDF using PyMuPDF
def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# 🔪 Step 2: Chunk text with overlap
def chunk_with_overlap(text, max_tokens=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = words[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

# 🧠 Step 3: Embed chunks using SentenceTransformer and FAISS
embedder = SentenceTransformer("intfloat/e5-large-v2")

# 🔍 Step 4: Build retriever with FAISS index
class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.embeddings = embedder.encode(chunks, show_progress_bar=True).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def get_top_chunks(self, query, top_k=5):
        query_emb = embedder.encode([query]).astype("float32")
        D, I = self.index.search(query_emb, top_k)
        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]

# 🔁 Step 5: RAG LLM pipeline setup
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")
rag_model = rag_model.to("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Step 6: Generate answers using RAG with stronger context retrieval
def generate_answer(question, retriever):
    if retriever is None:
        return "⚠️ Please upload a PDF first."
    top_chunks = retriever.get_top_chunks(question, top_k=7)
    context_docs = [{"title": "PDF Chunk", "text": chunk} for chunk in top_chunks]

    input_dict = rag_tokenizer.prepare_seq2seq_batch(
        question,
        context_docs=context_docs,
        return_tensors="pt"
    )
    input_dict = {k: v.to(rag_model.device) for k, v in input_dict.items()}

    outputs = rag_model.generate(
        input_ids=input_dict["input_ids"],
        attention_mask=input_dict["attention_mask"],
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True
    )
    answer = rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer.strip()

# 🖼️ Step 7: Gradio UI
def process_pdf(file):
    global retriever
    text = extract_text(file.name)
    chunks = chunk_with_overlap(text)
    retriever = Retriever(chunks)
    return "✅ PDF loaded. Ask your questions below."

def chat(question):
    return generate_answer(question, retriever)

retriever = None  # global retriever object

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Advanced PDF Chatbot with Powerful RAG Retrieval")
    with gr.Row():
        pdf_input = gr.File(label="Upload your PDF")
        load_output = gr.Textbox(label="Status")
    pdf_input.change(process_pdf, inputs=pdf_input, outputs=load_output)

    chatbot = gr.Textbox(label="Ask a question")
    response = gr.Textbox(label="Answer")
    chatbot.submit(chat, inputs=chatbot, outputs=response)

    gr.Markdown("Made with ❤️ using RAG, FAISS & Gradio")

# 🚀 Launch
demo.launch()
