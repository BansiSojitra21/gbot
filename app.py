
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Chatbot")
st.caption("Powered by OpenRouter")

# Load API key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY not found in your .env file.")
    st.stop()


# Create OpenRouter client
@st.cache_resource
def get_client(key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key
    )


client = get_client(api_key)


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User input
prompt = st.chat_input("Ask me anything...")

if prompt:

    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    try:
        # Send conversation to OpenRouter
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=st.session_state.messages
                )

                answer = response.choices[0].message.content
                
                if not answer:
                    answer = "The model returned an empty response."

            st.write(answer)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        # Remove the failed user message so it can
        # be retried without duplicating it in history.
        st.session_state.messages.pop()

        st.error(f"API Error: {e}")