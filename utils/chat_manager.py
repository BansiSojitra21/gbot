import uuid
import streamlit as st

def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if not st.session_state.chats:
        create_new_chat()

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

def get_current_chat():
    if st.session_state.current_chat_id in st.session_state.chats:
        return st.session_state.chats[st.session_state.current_chat_id]
    return None

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    
    if st.session_state.current_chat_id == chat_id:
        if st.session_state.chats:
            remaining_chats = list(st.session_state.chats.keys())
            st.session_state.current_chat_id = remaining_chats[-1]
        else:
            create_new_chat()

def clear_all_chats():
    st.session_state.chats = {}
    create_new_chat()
