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

# Confirmation Dialogs
@st.dialog("Delete Chat")
def delete_chat_dialog(chat_id):
    st.write("Are you sure you want to delete this chat?")
    if st.button("Yes, delete", use_container_width=True):
        delete_chat(chat_id)
        if st.session_state.get("current_chat_id") == chat_id:
            st.session_state.current_chat_id = None
        st.rerun()

@st.dialog("Clear All Chats")
def clear_all_chats_dialog():
    st.write("Are you sure you want to clear all your chat history? This action cannot be undone.")
    if st.button("Yes, clear all", use_container_width=True):
        clear_all_chats()
        st.session_state.current_chat_id = None
        st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 AI Chatbot</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="new-chat-btn-wrapper">', unsafe_allow_html=True)
    if st.button("＋ New Chat", use_container_width=True, key="new_chat"):
        create_new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="sidebar-section-heading">Previous Chats</div>', unsafe_allow_html=True)
    
    chat_items = list(st.session_state.chats.items())
    chat_items.reverse()
    
    for chat_id, chat in chat_items:
        # Layout for chat history item
        col1, col2 = st.columns([7, 1], gap="small")
        with col1:
            title = chat["title"]
            if st.button(title, key=f"open_chat_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            with st.popover("⋮", use_container_width=True):
                if st.button("🗑️ Delete", key=f"delete_chat_{chat_id}", use_container_width=True):
                    delete_chat_dialog(chat_id)
                    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="clear-all-btn-wrapper">', unsafe_allow_html=True)
    if st.button("🧹 Clear All Chats", use_container_width=True, key="clear_all_chats"):
        clear_all_chats_dialog()
    st.markdown('</div>', unsafe_allow_html=True)

    # JS to attach classes to buttons based on their text
    components.html(
        """
        <script>
        const parentDocs = window.parent.document;
        
        function applyClasses() {
            const buttons = parentDocs.querySelectorAll('[data-testid="stSidebar"] button');
            buttons.forEach(btn => {
                const text = btn.innerText.trim();
                
                // Remove existing custom classes
                btn.classList.remove('btn-new-chat', 'btn-clear-all', 'btn-delete', 'btn-popover', 'btn-chat-history');
                
                if(text.includes("New Chat")) {
                    btn.classList.add("btn-new-chat");
                }
                else if(text.includes("Clear All")) {
                    btn.classList.add("btn-clear-all");
                }
                else if(text.includes("Delete") || text.includes("🗑️")) {
                    btn.classList.add("btn-delete");
                }
                else if(text === "⋮" || text === "...") {
                    btn.classList.add("btn-popover");
                }
                else {
                    btn.classList.add("btn-chat-history");
                }
            });
        }
        
        // Run immediately
        applyClasses();
        
        // Streamlit dynamically rebuilds the DOM on interactions.
        const observer = new MutationObserver((mutations) => {
            applyClasses();
        });
        
        const sidebar = parentDocs.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            observer.observe(sidebar, { childList: true, subtree: true });
        }
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
# DISPLAY CURRENT CHAT
# ============================================================
messages_displayed = False
for message in current_chat["messages"]:
    if message["role"] != "system":
        messages_displayed = True
        with st.chat_message(message["role"]):
            st.write(message["content"])

if not messages_displayed:
    st.markdown("""
        <div class="empty-chat-container">
            <h2>How can I help you today?</h2>
        </div>
    """, unsafe_allow_html=True)

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
    retrieved_chunks = retrieve_documents(user_text, current_chat.get("rag_index"), current_chat.get("rag_chunks"), top_k=4)
    
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
