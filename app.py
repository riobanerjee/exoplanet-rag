"""
Streamlit application for the RAG system.
"""

import logging
import streamlit as st
from typing import Dict, Any

from src.utils import load_config
from src.ingest import run_ingestion_pipeline
from src.rag_chain import RAGChain

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Page configuration
st.set_page_config(
    page_title="Exoplanet RAG",
    page_icon="🪐",
    layout="wide"
)

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    
if "history" not in st.session_state:
    st.session_state.history = []


def check_ollama() -> bool:
    """Check if Ollama is available."""
    from langchain_ollama import OllamaLLM
    
    try:
        llm = OllamaLLM(model=config["llm"]["model_name"])
        llm.invoke("test")
        return True
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return False


def initialize_pipeline() -> None:
    """Initialize the RAG pipeline."""
    with st.spinner("Setting up the RAG pipeline. This may take a few minutes..."):
        try:
            vector_store = run_ingestion_pipeline(force_rebuild=False)
            st.session_state.rag_chain = RAGChain(vector_store)
            st.success("RAG pipeline initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")


def process_query(query: str) -> None:
    """Process a query and update history."""
    if st.session_state.rag_chain is None:
        st.error("RAG chain is not initialized. Please set up the pipeline first.")
        return
    
    with st.spinner("Generating response..."):
        try:
            rag_chain = st.session_state.rag_chain
            result = rag_chain.query(query, return_source_documents=True)
            st.session_state.history.append(result)
        except Exception as e:
            st.error(f"Error processing query: {e}")


# Sidebar
with st.sidebar:
    st.title("🪐 Exoplanet RAG")
    
    # Ollama status
    ollama_available = check_ollama()
    
    if ollama_available:
        st.success("✅ Ollama is available")
    else:
        st.error("❌ Ollama is not available")
        st.info("Please install Ollama and pull the model specified in config.yml")
    
    # Configuration display
    st.subheader("Configuration")
    st.write(f"**LLM:** {config['llm']['model_name']}")
    st.write(f"**Embeddings:** {config['embeddings']['model_name']}")
    st.write(f"**Papers to download:** {config['data_ingestion']['download_limit']}")
    
    # Initialize button
    if st.button("Initialize Pipeline"):
        if not ollama_available:
            st.error("Ollama is not available. Please install Ollama first.")
        else:
            initialize_pipeline()
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.history = []
    
    # About
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This RAG system uses ArXiv papers about exoplanets to answer your questions. "
        "It combines document retrieval with local LLM generation for accurate, source-based responses."
    )


# Main content
st.title("🪐 Exoplanet RAG")

# Query input
query = st.text_area(
    "Ask a question about exoplanets:", 
    placeholder="e.g., What are hot Jupiters? How are exoplanets detected?"
)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Submit"):
        if query:
            process_query(query)
            st.rerun()
        else:
            st.warning("Please enter a query.")

# Display history
if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.container():
            st.markdown("---")
            
            # Display query
            st.markdown(f"### Q: {item['query']}")
            
            # Display response
            st.markdown(f"### A:")
            st.markdown(item['response'])
            
            # View context
            with st.expander("View Context"):
                st.markdown("**Context used for generation:**")
                st.text(item.get('context', 'No context available'))
else:
    # Instructions for new users
    if st.session_state.rag_chain is None:
        st.info(
            "👈 To get started, check if Ollama is available in the sidebar and click **'Initialize Pipeline'**. "
            "This will download ArXiv papers on exoplanets and set up the RAG system."
        )
        
        st.markdown("### Example Questions")
        st.markdown("Once initialized, you can ask questions like:")
        st.markdown("- What are hot Jupiters?")
        st.markdown("- How are exoplanets detected?")  
        st.markdown("- What is the habitable zone?")
        st.markdown("- What methods are used for exoplanet characterization?")
    else:
        st.info(
            "Enter a question about exoplanets in the text area above and click **'Submit'**."
        )


# Footer
st.markdown("---")
st.markdown(
    "*This application uses Retrieval Augmented Generation (RAG) to answer questions about exoplanets "
    "using scientific papers from ArXiv. All processing is done locally.*"
)