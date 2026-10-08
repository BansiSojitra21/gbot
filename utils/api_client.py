from openai import OpenAI
import streamlit as st

@st.cache_resource
def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=60.0,
        max_retries=0
    )
