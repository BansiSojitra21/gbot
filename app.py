import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from utils.api_client import get_client
from utils.rag_engine import create_rag_index, retrieve_documents
from utils.chat_manager import (
    init_session_state, create_new_chat, get_current_chat,
    delete_chat, clear_all_chats
)

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

if not api_key:
    st.error("OPENROUTER_API_KEY is not configured.")
    st.info("For local development, add OPENROUTER_API_KEY to your .env file.")
    st.stop()

client = get_client(api_key)

init_session_state()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 AI Chatbot</div>', unsafe_allow_html=True)
    
    if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
        create_new_chat()
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Previous Chats")
    
    chat_items = list(st.session_state.chats.items())
    chat_items.reverse()
    
    for chat_id, chat in chat_items:
        col1, col2 = st.columns([8, 1], gap="small")
        with col1:
            if st.button(chat["title"], key=f"open_chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_chat_{chat_id}", help="Delete chat", use_container_width=True):
                delete_chat(chat_id)
                st.rerun()
                    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧹  Clear All Chats", use_container_width=True, key="clear_all_chats"):
        clear_all_chats()
        st.rerun()

    # JS to enforce left alignment on chat titles
    components.html(
        """
        <script>
        const parentDocs = window.parent.document;
        
        // Ensure sidebar buttons align text to left
        const buttons = parentDocs.querySelectorAll('[data-testid="stSidebar"] button');
        buttons.forEach(btn => {
            btn.style.display = 'flex';
            btn.style.justifyContent = 'flex-start';
            btn.style.alignItems = 'center';
            btn.style.textAlign = 'left';
            const mContainers = btn.querySelectorAll('[data-testid="stMarkdownContainer"]');
            mContainers.forEach(m => {
                m.style.width = '100%';
                m.style.textAlign = 'left';
            });
            const paras = btn.querySelectorAll('p');
            paras.forEach(p => {
                p.style.width = '100%';
                p.style.textAlign = 'left';
            });
        });
        </script>
        """,
        height=0,
        width=0
    )

current_chat = get_current_chat()
if not current_chat:
    create_new_chat()
    current_chat = get_current_chat()
    st.rerun()

# ============================================================
# MAIN TITLE (Hidden since it's moved to sidebar, keeping subtitle)
# ============================================================
if current_chat["title"] == "New Chat":
    st.markdown('<div class="main-subtitle">How can I help you today?</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="main-subtitle">Powered by OpenRouter</div>', unsafe_allow_html=True)

# ============================================================
# DISPLAY CURRENT CHAT
# ============================================================
for message in current_chat["messages"]:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================
prompt = st.chat_input("Message AI Chatbot...", accept_file=True, file_type=["pdf"])

if prompt:
    # Handle different Streamlit versions for prompt input
    if hasattr(prompt, "text"):
        user_text = prompt.text or ""
    elif isinstance(prompt, dict) and "text" in prompt:
        user_text = prompt["text"] or ""
    else:
        user_text = str(prompt) if prompt else ""
        
    if hasattr(prompt, "files"):
        uploaded_files = prompt.files or []
    elif isinstance(prompt, dict) and "files" in prompt:
        uploaded_files = prompt["files"] or []
    else:
        uploaded_files = []
        
    # Process PDFs
    if uploaded_files:
        for uploaded_pdf in uploaded_files:
            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                index, chunks = create_rag_index(uploaded_pdf)
            if index is None:
                st.error("Could not extract readable text from the PDF.")
            else:
                current_chat["rag_index"] = index
                current_chat["rag_chunks"] = chunks
                current_chat["document_name"] = uploaded_pdf.name
                st.toast(f"PDF '{uploaded_pdf.name}' processed successfully! {len(chunks)} chunks created.")
                
    # Save & Display user message
    current_chat["messages"].append({"role": "user", "content": user_text})
    
    if current_chat["title"] == "New Chat":
        title = user_text.strip()
        if not title:
            title = "Document Chat"
        if len(title) > 32:
            title = title[:32] + "..."
        current_chat["title"] = title
        
    with st.chat_message("user"):
        st.write(user_text)
        
    # Retrieve relevant document chunks if available
    retrieved_chunks = retrieve_documents(user_text, current_chat["rag_index"], current_chat["rag_chunks"], top_k=4)
    
    rag_context = ""
    for chunk in retrieved_chunks:
        rag_context += f"[Page {chunk['page']}]\n{chunk['text']}\n\n"
        
    # Create system prompt
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
        system_prompt += "\n\nDOCUMENT CONTEXT:\n\n" + rag_context
    else:
        system_prompt += "\n\nNo relevant document context was found."

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                messages_to_send = current_chat["messages"].copy()
                messages_to_send.insert(0, {"role": "system", "content": system_prompt})
                
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=messages_to_send
                )
                
                answer = response.choices[0].message.content
                if not answer:
                    answer = "Sorry, I couldn't generate a response."
                st.write(answer)
                
        current_chat["messages"].append({"role": "assistant", "content": answer})
        
    except Exception as e:
        current_chat["messages"].pop()
        st.error(f"API Error: {e}")
