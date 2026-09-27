import os
import streamlit as st
from dotenv import load_dotenv
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
# CHECK API KEY
# ============================================================

if not api_key:

    st.error("OPENROUTER_API_KEY is not configured.")

    st.info(
        "For local development, add OPENROUTER_API_KEY to your .env file. "
        "For Streamlit Cloud, add it to App Settings → Secrets."
    )

    st.stop()


# ============================================================
# OPENROUTER CLIENT
# ============================================================

@st.cache_resource
def get_client(api_key):

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


client = get_client(api_key)


# ============================================================
# SESSION STATE
# ============================================================

# All conversations
if "chats" not in st.session_state:

    st.session_state.chats = {}


# Currently selected chat
if "current_chat_id" not in st.session_state:

    st.session_state.current_chat_id = None


# ============================================================
# CREATE NEW CHAT FUNCTION
# ============================================================

def create_new_chat():

    chat_id = str(len(st.session_state.chats) + 1)

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

    st.title("💬 Chats")

    # New chat button
    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()


    st.divider()

    st.subheader("Previous Chats")


    # Display chats
    for chat_id, chat in st.session_state.chats.items():

        col1, col2 = st.columns([5, 1])


        # Open chat
        with col1:

            if st.button(
                chat["title"],
                key=f"open_{chat_id}",
                use_container_width=True
            ):

                st.session_state.current_chat_id = chat_id

                st.rerun()


        # Delete chat
        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{chat_id}"
            ):

                del st.session_state.chats[chat_id]

                # If deleted chat was active
                if (
                    st.session_state.current_chat_id
                    == chat_id
                ):

                    if st.session_state.chats:

                        st.session_state.current_chat_id = (
                            list(st.session_state.chats.keys())[-1]
                        )

                    else:

                        create_new_chat()


                st.rerun()


    st.divider()


    # Clear all chats
    if st.button(
        "🧹 Clear All Chats",
        use_container_width=True
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
# CHAT TITLE
# ============================================================

st.title("🤖 AI Chatbot")

st.caption(
    f"Current chat: {current_chat['title']}"
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in current_chat["messages"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask me anything..."
)


# ============================================================
# SEND MESSAGE
# ============================================================

if prompt:

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    current_chat["messages"].append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # --------------------------------------------------------
    # Generate chat title
    # --------------------------------------------------------

    if current_chat["title"] == "New Chat":

        title = prompt.strip()

        if len(title) > 30:

            title = title[:30] + "..."

        current_chat["title"] = title


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(prompt)


    # --------------------------------------------------------
    # Generate AI response
    # --------------------------------------------------------

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = client.chat.completions.create(

                    model="openrouter/free",

                    messages=current_chat["messages"]

                )


                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )


                st.write(answer)


        # ----------------------------------------------------
        # Save AI response
        # ----------------------------------------------------

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