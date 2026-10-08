import uuid
import streamlit as st


def has_user_messages(chat):
    return any(message.get("role") == "user" for message in chat.get("messages", []))


def get_saved_chats():
    return {
        chat_id: chat
        for chat_id, chat in st.session_state.chats.items()
        if has_user_messages(chat)
    }


def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

    current_chat = st.session_state.chats.get(st.session_state.current_chat_id)
    if current_chat:
        for chat_id, chat in list(st.session_state.chats.items()):
            if chat_id != st.session_state.current_chat_id and not has_user_messages(chat):
                del st.session_state.chats[chat_id]
    else:
        create_new_chat()


def create_new_chat():
    current_chat = st.session_state.chats.get(st.session_state.current_chat_id)
    if current_chat and not has_user_messages(current_chat):
        return st.session_state.current_chat_id

    for chat_id, chat in list(st.session_state.chats.items()):
        if not has_user_messages(chat):
            del st.session_state.chats[chat_id]

    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "rag_index": None,
        "rag_chunks": [],
        "document_name": None
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


def get_current_chat():
    if st.session_state.current_chat_id in st.session_state.chats:
        return st.session_state.chats[st.session_state.current_chat_id]
    return None


def activate_chat(chat_id):
    if chat_id not in st.session_state.chats or not has_user_messages(st.session_state.chats[chat_id]):
        return

    for draft_id, chat in list(st.session_state.chats.items()):
        if draft_id != chat_id and not has_user_messages(chat):
            del st.session_state.chats[draft_id]
    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    
    if st.session_state.current_chat_id == chat_id:
        saved_chats = get_saved_chats()
        if saved_chats:
            st.session_state.current_chat_id = next(reversed(saved_chats))
        else:
            create_new_chat()


def clear_all_chats():
    st.session_state.chats = {}
    create_new_chat()
