import os
import uuid

import streamlit as st
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
# SESSION STATE
# ============================================================

if "chats" not in st.session_state:

    st.session_state.chats = {}


if "current_chat_id" not in st.session_state:

    st.session_state.current_chat_id = None


# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat():

    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {

        "title": "New Chat",

        "messages": []

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

    if hasattr(prompt, "text"):
        user_text = prompt.text
    elif isinstance(prompt, dict) and "text" in prompt:
        user_text = prompt["text"]
    else:
        user_text = prompt

    if hasattr(prompt, "files"):
        uploaded_files = prompt.files
    elif isinstance(prompt, dict) and "files" in prompt:
        uploaded_files = prompt["files"]
    else:
        uploaded_files = []

    if uploaded_files:
        for uploaded_pdf in uploaded_files:
            reader = PdfReader(uploaded_pdf)
            pdf_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
            st.session_state.pdf_text = pdf_text
            st.toast(f"PDF '{uploaded_pdf.name}' loaded successfully! {len(reader.pages)} pages found.")

    if user_text:

        # ========================================================
        # SAVE USER MESSAGE
        # ========================================================

        current_chat["messages"].append(
            {
                "role": "user",
                "content": user_text
            }
        )


        # ========================================================
        # CREATE CHAT TITLE
        # ========================================================

        if current_chat["title"] == "New Chat":

            title = user_text.strip()


            # Limit title length
            if len(title) > 32:

                title = title[:32] + "..."


            current_chat["title"] = title


        # ========================================================
        # DISPLAY USER MESSAGE
        # ========================================================

        with st.chat_message("user"):

            st.write(user_text)


        # ========================================================
        # GENERATE AI RESPONSE
        # ========================================================

        try:

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    messages_to_send = current_chat["messages"].copy()
                    if "pdf_text" in st.session_state and st.session_state.pdf_text:
                        messages_to_send.insert(0, {
                            "role": "system",
                            "content": f"Context from uploaded PDF:\n{st.session_state.pdf_text}"
                        })

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


                    # Make sure response isn't empty
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