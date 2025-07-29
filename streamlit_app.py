import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List
import logging
import sys
import os
import torch
torch.classes.__path__ = [] 

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Disaster Management RAG Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.7rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        font-size: 1.1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #222;
    }
    .bot-message {
        background-color: #f9fbe7;
        border-left: 4px solid #8bc34a;
        color: #222;
    }
    .source-box {
        background-color: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 0.25rem;
        padding: 0.7rem;
        margin: 0.5rem 0;
        font-size: 0.98rem;
        color: #222 !important;
    }
    .source-box * {
        color: #222 !important;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    # Initialize the RAG system (cached for performance)
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def display_chat_message(message: str, is_user: bool = True):
    # Display a chat message with styling
    css_class = "user-message" if is_user else "bot-message"
    role = "You" if is_user else "Assistant"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{role}:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_sources(sources: List[Dict[str, Any]]):
    # Display source information
    st.markdown("### Sources")
    for source in sources:
        st.markdown(f"""
        <div class="source-box">
            <strong>Source:</strong> {source['source']}<br>
            <strong>Type:</strong> {source['type']}<br>
            <strong>Relevance:</strong> {source['similarity_score']:.2%}<br>
            <strong>Content:</strong> {source['snippet']}
        </div>
        """, unsafe_allow_html=True)

def process_user_input(user_input: str):
    # Process user input when Enter key is pressed
    if user_input:
        st.session_state._input_submitted = True

def main():
    # Header
    st.markdown('<h1 class="main-header"> Disaster Management RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = initialize_rag_system()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Check if system is initialized
    if st.session_state.rag_system is None:
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        return
    
    # Check if vector store has data
    stats = st.session_state.rag_system.get_system_stats()
    if stats['vector_store'].get('total_documents', 0) == 0:
        st.error("No data found in the vector store. Please run setup_data.py first to initialize the data.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        # System stats
        st.markdown("### System Statistics")
        try:
            stats = st.session_state.rag_system.get_system_stats()
            
            st.metric("Total Documents", stats['vector_store'].get('total_documents', 0))
            st.metric("LLM Model", stats.get('llm_model', 'Unknown'))
            st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
            
            with st.expander("Advanced Settings"):
                st.write(f"**Chunk Size:** {stats.get('chunk_size', 'Unknown')}")
                st.write(f"**Top K Results:** {stats.get('top_k_results', 'Unknown')}")
                st.write(f"**Similarity Threshold:** {stats.get('similarity_threshold', 'Unknown')}")
            
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
        
        # Reset system
        if st.button("Reset System", type="secondary"):
            if st.session_state.rag_system.reset_system():
                st.session_state.chat_history = []
                st.success("System reset successfully!")
                st.rerun()
            else:
                st.error("Failed to reset system")
        
        # Model selection
        st.markdown("### Model Selection")
        available_models = [
            "mixtral-8x7b-32768",
            "llama-3.1-8b-instant", 
            "mixtral-8x7b-32768"
        ]
        
        current_model = st.selectbox(
            "Select LLM Model:",
            available_models,
            index=0 if st.session_state.rag_system.llm.model == "llama-3.1-70b-versatile" else 
                  1 if st.session_state.rag_system.llm.model == "llama-3.1-8b-instant" else 2
        )
        
        if st.button("Update Model"):
            if st.session_state.rag_system.llm.switch_model(current_model):
                st.success(f"Switched to {current_model}")
            else:
                st.error("Failed to switch model")

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Chat input
        user_input = st.text_input(
            "Ask me about disasters, emergency management, or safety measures:",
            placeholder="e.g., What should I do during an earthquake?",
            key="user_input"
        )
        
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process user input
        if send_button and user_input:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_system.query(user_input)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'user': user_input,
                        'assistant': response['answer'],
                        'sources': response['sources'],
                        'metadata': {
                            'model': response['model_used'],
                            'retrieved_docs': response['retrieved_documents'],
                            'tokens': response['token_usage']
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
        # Display chat history
        for message in reversed(st.session_state.chat_history):
            display_chat_message(message['user'], is_user=True)
            display_chat_message(message['assistant'], is_user=False)
            
            if message['sources']:
                display_sources(message['sources'])
    
    with col2:
        st.header("Quick Actions")
        
        # Sample questions
        st.markdown("### Sample Questions")
        sample_questions = [
            "What should I do during an earthquake?",
            "How can I prepare for floods?",
            "What are the warning signs of a flood?",
            "Emergency supplies for earthquake preparedness?",
            "How to create an emergency evacuation plan?",
            "What is the difference between earthquake magnitude and intensity?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        response = st.session_state.rag_system.query(question)
                        
                        st.session_state.chat_history.append({
                            'user': question,
                            'assistant': response['answer'],
                            'sources': response['sources'],
                            'metadata': {
                                'model': response['model_used'],
                                'retrieved_docs': response['retrieved_documents'],
                                'tokens': response['token_usage']
                            }
                        })
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # System information
        st.markdown("### About This System")
        st.info("""
        This RAG chatbot uses:
        - **GROQ** for fast LLM inference
        - **ChromaDB** for vector storage
        - **Sentence Transformers** for embeddings
        - **Cosine similarity** for retrieval
        - **Streamlit** for the interface
        
        Currently loaded with **Earthquake** and **Flood** data.
        """)

if __name__ == "__main__":
    main()