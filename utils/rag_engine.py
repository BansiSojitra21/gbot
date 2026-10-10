import hashlib
import os
import re
import uuid
from pathlib import Path

import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
SUPPORTED_EXTENSIONS = {".pdf"}
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_RETRIEVAL_TOP_K = 4
DEFAULT_RELEVANCE_THRESHOLD = 0.25
GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening|thanks|thank you)"
    r"(?:\s+there)?[!.?,\s]*$",
    re.IGNORECASE,
)


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def ensure_storage_directories():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
    VECTORSTORE_DIR.mkdir(exist_ok=True, parents=True)


def get_session_identifier():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def sanitize_filename(filename):
    name = Path(filename).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe_name or "document.pdf"


def _hash_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_extracted_text(raw_text):
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def is_greeting_only(text):
    return bool(GREETING_PATTERN.fullmatch((text or "").strip()))


def create_chunks(text, page_number=None, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP, metadata=None):
    metadata = metadata or {}
    if not text or not text.strip():
        return []

    cleaned_text = clean_extracted_text(text)
    if not cleaned_text:
        return []

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(cleaned_text):
        end = start + chunk_size
        chunk_text = cleaned_text[start:end].strip()
        if chunk_text:
            chunk_metadata = {**metadata}
            chunk_metadata.update({
                "page": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            chunks.append(chunk_metadata)
            chunk_index += 1
        if end >= len(cleaned_text):
            break
        start += max(1, chunk_size - overlap)
    return chunks


def build_document_chunks(file_path, source_type="built_in", owner_id=None, document_id=None):
    if not file_path.exists():
        return []

    file_name = file_path.name
    document_id = document_id or hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:12]
    metadata_base = {
        "document_id": document_id,
        "filename": file_name,
        "source_type": source_type,
        "owner_id": owner_id,
    }

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF '{file_path.name}': {exc}") from exc

    all_chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
        except Exception as exc:
            raise ValueError(
                f"Could not extract text from page {page_number} of '{file_path.name}': {exc}"
            ) from exc
        if not page_text:
            continue
        page_chunks = create_chunks(page_text, page_number=page_number, metadata=metadata_base)
        all_chunks.extend(page_chunks)
    return all_chunks


def discover_builtin_documents():
    ensure_storage_directories()
    documents = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            path.relative_to(UPLOADS_DIR)
            continue
        except ValueError:
            documents.append(path)
    return documents


def discover_uploaded_documents(session_id=None):
    ensure_storage_directories()
    session_owner = session_id or get_session_identifier()
    owner_dir = UPLOADS_DIR / str(session_owner)
    if not owner_dir.exists():
        return []
    documents = []
    for path in sorted(owner_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(path)
    return documents


def _build_faiss_index(chunks):
    if not chunks:
        return None, []

    embedding_model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, chunks


def create_rag_index(uploaded_pdf):
    if uploaded_pdf is None:
        return None, []

    if isinstance(uploaded_pdf, (str, Path)):
        file_path = Path(uploaded_pdf)
        chunks = build_document_chunks(file_path, source_type="built_in", owner_id="system")
        return _build_faiss_index(chunks)

    temp_path = save_uploaded_document(uploaded_pdf, session_id=get_session_identifier())
    chunks = build_document_chunks(temp_path, source_type="uploaded", owner_id=get_session_identifier())
    return _build_faiss_index(chunks)


def build_hybrid_index(session_id=None):
    ensure_storage_directories()
    session_id = session_id or get_session_identifier()
    all_chunks = []

    for document_path in discover_builtin_documents():
        all_chunks.extend(build_document_chunks(document_path, source_type="built_in", owner_id="system"))

    for document_path in discover_uploaded_documents(session_id):
        all_chunks.extend(build_document_chunks(document_path, source_type="uploaded", owner_id=session_id))

    return _build_faiss_index(all_chunks)


def save_uploaded_document(uploaded_file, session_id=None):
    ensure_storage_directories()
    session_id = session_id or get_session_identifier()
    if uploaded_file is None:
        raise ValueError("No file was provided.")

    file_name = sanitize_filename(getattr(uploaded_file, "name", "document.pdf"))
    if Path(file_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a PDF.")

    owner_dir = UPLOADS_DIR / str(session_id)
    owner_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{Path(file_name).stem}_{uuid.uuid4().hex[:8]}{Path(file_name).suffix.lower()}"
    target_path = owner_dir / unique_name

    file_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if not file_bytes:
        raise ValueError("The uploaded document is empty.")

    if isinstance(file_bytes, str):
        file_bytes = file_bytes.encode("utf-8")

    with open(target_path, "wb") as output_file:
        output_file.write(file_bytes)

    return target_path


def retrieve_documents(question, index, chunks, top_k=DEFAULT_RETRIEVAL_TOP_K, relevance_threshold=None):
    if index is None or not chunks:
        return []

    if relevance_threshold is None:
        relevance_threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", str(DEFAULT_RELEVANCE_THRESHOLD)))

    embedding_model = get_embedding_model()
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    question_embedding = question_embedding.astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    seen = set()
    retrieved_chunks = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue
        if float(score) < relevance_threshold:
            continue
        chunk = chunks[int(index_position)].copy()
        chunk_id = (chunk.get("document_id"), chunk.get("page"), chunk.get("chunk_index"), chunk.get("text"))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk["score"] = float(score)
        retrieved_chunks.append(chunk)

    retrieved_chunks.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return retrieved_chunks
