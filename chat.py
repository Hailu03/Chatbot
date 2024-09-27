import streamlit as st
from llama_index.llms.openai import OpenAI
import openai
from src.conversation_engine import initialize_chatbot, chat_interface, load_chat_store
from llama_index.core import Settings
import os 
from src.global_settings import CONVERSATION_FILE

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
openai.api_key = st.secrets.openai.OPENAI_API_KEY


def main():
    
    st.header("HailuGPT - Đắc Nhân Tâm")

    # Load the chat store (will be empty if cleared)
    chat_store = load_chat_store()

    # Create a container for chat messages
    container = st.container()

    # Initialize the chatbot
    agent = initialize_chatbot(chat_store, container)

    # Run chat interface
    chat_interface(agent, chat_store, container)

if __name__ == "__main__":
    main()