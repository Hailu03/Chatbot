import os
from llama_index.core import load_index_from_storage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from datetime import datetime
import json
from src.global_settings import INDEX_STORAGE,SCORES_FILE,CONVERSATION_FILE
from src.prompts import CUSTOM_AGENT_SYSTEM_TEMPLATE
from llama_index.llms.openai import OpenAI
import openai
import streamlit as st


user_avatar = "data/images/user.png"
professor_avatar = "data/images/professor.jpg"
    
def load_chat_store():
    if os.path.exists(CONVERSATION_FILE) and os.path.getsize(CONVERSATION_FILE) > 0:
        try:
            chat_store = SimpleChatStore.from_persist_path(CONVERSATION_FILE) # Load the chat store from the file
        except:
            chat_store = SimpleChatStore() # If there is an error, create a new chat store
    else:
        chat_store = SimpleChatStore()

    return chat_store

def save_score(score, content, total_guess, username):
        """Write score and content to a file.

        Args:
            score (string): Score of the user's mental health.
            content (string): Content of the user's mental health.
            total_guess (string): Total guess of the user's mental health.
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {
            "username": username,
            "Time": current_time,
            "Score": score,
            "Content": content,
            "Total guess": total_guess
        }
        
        # Đọc dữ liệu từ file nếu tồn tại
        try:
            with open(SCORES_FILE, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        
        # Thêm dữ liệu mới vào danh sách
        data.append(new_entry)
        
        # Ghi dữ liệu trở lại file
        with open(SCORES_FILE, "w") as f:
            json.dump(data, f, indent=4)

def initialize_chatbot(chat_store, container):
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store = chat_store,
        chat_store_key= "user"
    )

    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_STORAGE
    )

    index = load_index_from_storage(storage_context,index_id="vector")

    engine = index.as_query_engine(similarity_top_k=3),

    tool = QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="book",
            description= (
                f"Cung cấp thông tin từ tập sách được lưu trữ về cách nói chuyện với mọi người"
                f"đắc nhân tâm của Dale Carnegie. Sử dụng câu hỏi văn bản thuần túy chi tiết làm đầu vào cho công cụ"
            )
        )
    )

    save_tool = FunctionTool.from_defaults(fn=save_score)
    agent = OpenAIAgent.from_tools(
        tools=[tool,save_tool],
        memory=memory,
        system_prompt=CUSTOM_AGENT_SYSTEM_TEMPLATE.format(user_info="Hai")
    )

    return agent

# Function to display chat interface with scrolling history
def chat_interface(agent, chat_store, container):  
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear history if delete button is clicked
    if st.button("🗑️ Xóa lịch sử"):
        # Clear chat history
        st.session_state.chat_history = []
        
        # Optionally, delete the conversation file if it exists
        if os.path.exists(CONVERSATION_FILE):
            os.remove(CONVERSATION_FILE)
        
        # Show a message that the chat has been cleared
        st.success("Lịch sử trò chuyện đã được xóa.")

    # Display all previous messages from session state
    with container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message(name="user", avatar=user_avatar):
                    st.markdown(chat["content"])
            else:
                with st.chat_message(name="assistant", avatar=professor_avatar):
                    st.markdown(chat["content"])
    
    # Initial message if no history exists
    if len(st.session_state.chat_history) == 0:
        with container:
            with st.chat_message(name="assistant", avatar=professor_avatar):
                st.markdown("Chào bạn, mình là HailuGPT được phát triển bởi Hailu. Mình sẽ giúp bạn hiểu về cách ứng xử thông minh và cải thiện giao tiếp của bạn. Hãy bắt đầu bằng cách viết tin nhắn vào ô bên dưới nhé!")
    
    # Accept user input
    prompt = st.chat_input("Viết tin nhắn tại đây ạ...")
    
    if prompt:
        # Add user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with container:
            with st.chat_message(name="user", avatar=user_avatar):
                st.markdown(prompt)

        # Get response from chatbot agent
        response = str(agent.chat(prompt))
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with container:
            with st.chat_message(name="assistant", avatar=professor_avatar):
                st.markdown(response)
        
        # Persist the conversation to the chat store
        chat_store.persist(CONVERSATION_FILE)
