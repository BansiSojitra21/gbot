import os
import uuid

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        width: 310px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }


    /* ======================================================
       SIDEBAR TITLE
    ====================================================== */

    section[data-testid="stSidebar"] h2 {
        font-size: 21px !important;
        font-weight: 600 !important;
        margin-bottom: 10px !important;
    }


    /* ======================================================
       SIDEBAR SUBTITLE
    ====================================================== */

    section[data-testid="stSidebar"] h3 {
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-top: 5px !important;
        margin-bottom: 8px !important;
    }


    /* ======================================================
       SIDEBAR BUTTONS
    ====================================================== */

    section[data-testid="stSidebar"] button {
        font-size: 13px !important;
        min-height: 34px !important;
        height: 34px !important;
        width: fit-content;
        padding: 4px 9px !important;
        border-radius: 8px !important;
    }


    /* ======================================================
       NEW CHAT BUTTON
    ====================================================== */

    section[data-testid="stSidebar"] button[kind="secondary"] {
        font-size: 13px !important;
    }


    /* ======================================================
       CHAT LIST
    ====================================================== */

    .chat-list {
        margin-top: 3px;
        margin-bottom: 3px;
    }


    /* ======================================================
       THREE DOT BUTTON
    ====================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stHorizontalBlock"]
    button {
        font-size: 13px !important;
    }


    /* Make the three-dot column button compact */

    section[data-testid="stSidebar"]
    div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child
    button {
        min-width: 32px !important;
        width: 32px !important;
        padding: 0 !important;
        font-size: 16px !important;
    }


    /* ======================================================
       REDUCE SIDEBAR SPACING
    ====================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stVerticalBlock"] {
        gap: 0.25rem;
    }


    /* ======================================================
       DIVIDERS
    ====================================================== */

    section[data-testid="stSidebar"] hr {
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }


    /* ======================================================
       MAIN TITLE
    ====================================================== */

    .main-title {
        font-size: 30px;
        font-weight: 650;
        margin-top: 5px;
        margin-bottom: 3px;
    }


    /* ======================================================
       MAIN SUBTITLE
    ====================================================== */

    .main-subtitle {
        font-size: 13px;
        opacity: 0.60;
        margin-bottom: 18px;
    }


    /* ======================================================
       CHAT MESSAGE
    ====================================================== */

    [data-testid="stChatMessage"] {
        font-size: 14px !important;
    }


    /* ======================================================
       CHAT INPUT
    ====================================================== */

    [data-testid="stChatInput"] textarea {
        font-size: 14px !important;
    }


    /* ======================================================
       POPOVER
    ====================================================== */

    div[data-testid="stPopoverBody"] {
        padding: 8px !important;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API KEY
# ============================================================

if not api_key:

    st.error(
        "OPENROUTER_API_KEY is not configured."
    )

    st.info(
        "For local development, add OPENROUTER_API_KEY "
        "to your .env file."
    )

    st.stop()


# ============================================================
# OPENROUTER CLIENT
# ============================================================

@st.cache_resource
def get_client(key):

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key
    )


client = get_client(api_key)

# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text,page_number,chunk_size=1000,overlap=200):
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


# ============================================================
# CREATE RAG INDEX FROM PDF
# ============================================================

def create_rag_index(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    all_chunks = []

    for page_number,page in enumerate(reader.pages,start=1) :
        page_text = page.extract_text()

        if not page_text:
            continue

        page_chunks = create_chunks(page_text,page_number)

        all_chunks.extend(page_chunks)
    
    if not all_chunks:

        return None, []

    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    # --------------------------------------------------------
    # CREATE FAISS INDEX
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index, all_chunks

# ============================================================
# SESSION STATE
# ============================================================

if "chats" not in st.session_state:

    st.session_state.chats = {}


if "current_chat_id" not in st.session_state:

    st.session_state.current_chat_id = None


# ============================================================
# RETRIEVE RELEVANT DOCUMENT CHUNKS
# ============================================================

def retrieve_documents(
    question,
    index,
    chunks,
    top_k=4
):

    if index is None or not chunks:

        return []

    # Create embedding for user question

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    # Search FAISS

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    retrieved_chunks = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        chunk = chunks[index_position].copy()

        chunk["score"] = float(score)

        retrieved_chunks.append(
            chunk
        )

    return retrieved_chunks

# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat():

    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {

        "title": "New Chat",

        "messages": [],

        "rag_index": None,

        "rag_chunks": [],

        "document_name": None

    }

    st.session_state.current_chat_id = chat_id


# ============================================================
# CREATE FIRST CHAT
# ============================================================

if not st.session_state.chats:

    create_new_chat()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # SIDEBAR TITLE
    # --------------------------------------------------------

    st.markdown("## 💬 Chats")


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋  New Chat",
        use_container_width=True,
        key="new_chat"
    ):

        create_new_chat()

        st.rerun()


    st.divider()

    # --------------------------------------------------------
    # PREVIOUS CHATS
    # --------------------------------------------------------

    st.markdown("### Previous Chats")


    # Get chats
    chat_items = list(
        st.session_state.chats.items()
    )


    # Newest first
    chat_items.reverse()


    # --------------------------------------------------------
    # CHAT LIST
    # --------------------------------------------------------

    for chat_id, chat in chat_items:

        col1, col2 = st.columns(
            [8, 1],
            gap="small"
        )


        # ====================================================
        # CHAT NAME
        # ====================================================

        with col1:

            if st.button(
                chat["title"],
                key=f"open_chat_{chat_id}",
                use_container_width=True
            ):

                st.session_state.current_chat_id = chat_id

                st.rerun()


        # ====================================================
        # THREE DOT MENU
        # ====================================================

        with col2:

            with st.popover(
                "...",
                use_container_width=True
            ):

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_chat_{chat_id}",
                    use_container_width=True
                ):

                    # Delete selected chat
                    del st.session_state.chats[
                        chat_id
                    ]


                    # If current chat was deleted
                    if (
                        st.session_state.current_chat_id
                        == chat_id
                    ):

                        # If chats remain
                        if st.session_state.chats:

                            remaining_chats = list(
                                st.session_state.chats.keys()
                            )

                            # Select newest remaining chat
                            st.session_state.current_chat_id = (
                                remaining_chats[-1]
                            )

                        # No chats remain
                        else:

                            create_new_chat()


                    st.rerun()


    st.divider()


    # ========================================================
    # CLEAR ALL CHATS
    # ========================================================

    if st.button(
        "🧹  Clear All Chats",
        use_container_width=True,
        key="clear_all_chats"
    ):

        st.session_state.chats = {}

        create_new_chat()

        st.rerun()


# ============================================================
# CURRENT CHAT
# ============================================================

current_chat = st.session_state.chats[
    st.session_state.current_chat_id
]


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Powered by OpenRouter'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY CURRENT CHAT
# ============================================================

for message in current_chat["messages"]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask me anything...",
    accept_file=True,
    file_type=["pdf"]
)


# ============================================================
# SEND MESSAGE
# ============================================================

if prompt:

    # --------------------------------------------------------
    # GET USER TEXT
    # --------------------------------------------------------

    if hasattr(prompt, "text"):
        user_text = prompt.text or ""
    elif isinstance(prompt, dict) and "text" in prompt:
        user_text = prompt["text"] or ""
    else:
        user_text = str(prompt) if prompt else ""


    # --------------------------------------------------------
    # GET UPLOADED FILES
    # --------------------------------------------------------

    if hasattr(prompt, "files"):
        uploaded_files = prompt.files or []
    elif isinstance(prompt, dict) and "files" in prompt:
        uploaded_files = prompt["files"] or []
    else:
        uploaded_files = []


    # --------------------------------------------------------
    # PROCESS PDF(S)
    # --------------------------------------------------------

    if uploaded_files:

        for uploaded_pdf in uploaded_files:

            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                index, chunks = create_rag_index(
                    uploaded_pdf
                )

            if index is None:

                st.error(
                    "Could not extract readable text from the PDF."
                )

            else:

                current_chat["rag_index"] = index
                current_chat["rag_chunks"] = chunks
                current_chat["document_name"] = uploaded_pdf.name

                st.toast(
                    f"PDF '{uploaded_pdf.name}' processed successfully! "
                    f"{len(chunks)} chunks created."
                )


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    current_chat["messages"].append(
        {
            "role": "user",
            "content": user_text
        }
    )


    # --------------------------------------------------------
    # CREATE CHAT TITLE
    # --------------------------------------------------------

    if current_chat["title"] == "New Chat":

        title = user_text.strip()

        if not title:
            title = "Document Chat"

        if len(title) > 32:
            title = title[:32] + "..."

        current_chat["title"] = title


    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(user_text)


    # ========================================================
    # RETRIEVE RELEVANT DOCUMENT CHUNKS
    # ========================================================

    retrieved_chunks = retrieve_documents(
        user_text,
        current_chat["rag_index"],
        current_chat["rag_chunks"],
        top_k=4
    )


    # ========================================================
    # BUILD RAG CONTEXT
    # ========================================================

    rag_context = ""

    for chunk in retrieved_chunks:

        rag_context += (
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}\n\n"
        )


    # ========================================================
    # CREATE SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are a document-based AI assistant.

Answer the user's question using the provided document context.

Rules:

1. Use the document context as the primary source.
2. Do not invent information that is not supported by the document.
3. If the answer cannot be found in the provided context,
   clearly say that the information was not found in the document.
4. When possible, mention the page number of the relevant information.
5. Give clear and concise answers.
"""


    if retrieved_chunks:

        system_prompt += (
            "\n\nDOCUMENT CONTEXT:\n\n"
            + rag_context
        )

    else:

        system_prompt += (
            "\n\nNo relevant document context was found."
        )


    # ========================================================
    # GENERATE AI RESPONSE
    # ========================================================

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                messages_to_send = current_chat["messages"].copy()

                messages_to_send.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                )

                response = client.chat.completions.create(

                    model="openrouter/free",

                    messages=messages_to_send

                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                if not answer:

                    answer = (
                        "Sorry, I couldn't generate "
                        "a response."
                    )

                st.write(answer)


        # ====================================================
        # SAVE AI RESPONSE
        # ====================================================

        current_chat["messages"].append(
            {
                "role": "assistant",
                "content": answer
            }
        )


    except Exception as e:

        # Remove failed user message
        current_chat["messages"].pop()

        st.error(
            f"API Error: {e}"
        )
