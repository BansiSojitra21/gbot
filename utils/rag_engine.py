import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def create_chunks(text, page_number, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page_number
            })
        start += chunk_size - overlap
    return chunks

def create_rag_index(uploaded_pdf):
    embedding_model = get_embedding_model()
    reader = PdfReader(uploaded_pdf)
    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text:
            continue
        page_chunks = create_chunks(page_text, page_number)
        all_chunks.extend(page_chunks)
    
    if not all_chunks:
        return None, []

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, all_chunks

def retrieve_documents(question, index, chunks, top_k=4):
    if index is None or not chunks:
        return []

    embedding_model = get_embedding_model()
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    question_embedding = question_embedding.astype("float32")
    
    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )
    
    retrieved_chunks = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue
        chunk = chunks[index_position].copy()
        chunk["score"] = float(score)
        retrieved_chunks.append(chunk)
    return retrieved_chunks
