"""
Streamlit application for the LangChain RAG system.
"""

import os
import sys
import logging
import streamlit as st
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
from src.utils import load_config, ensure_directories
from src.ingest import run_ingestion_pipeline
from src.retriever import load_vector_store
from src.rag_chain import RAGChain

# Load configuration
config = load_config()

# Page configuration
st.set_page_config(
    page_title="Exoplanet RAG",
    page_icon="🪐",
    layout="wide"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    
if "history" not in st.session_state:
    st.session_state.history = []


def check_ollama() -> bool:
    """
    Check if Ollama is available.
    
    Returns:
        Boolean indicating availability
    """
    from langchain_community.llms import Ollama
    
    try:
        llm = Ollama(model=config["llm"]["model_name"])
        llm.invoke("test")
        return True
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return False


def initialize_pipeline() -> None:
    """
    Initialize the RAG pipeline.
    """
    with st.spinner("Setting up the RAG pipeline. This may take a few minutes..."):
        try:
            # Check if we need to force rebuild
            force_rebuild = st.session_state.force_rebuild
            
            # Run ingestion pipeline
            vector_store = run_ingestion_pipeline(force_rebuild=force_rebuild)
            
            # Store in session state
            st.session_state.vector_store = vector_store
            
            # Create RAG chain
            st.session_state.rag_chain = RAGChain(vector_store)
            
            st.success("RAG pipeline initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")


def process_query(query: str) -> None:
    """
    Process a query and update history.
    """
    # Check if RAG chain is initialized
    if st.session_state.rag_chain is None:
        st.error("RAG chain is not initialized. Please set up the pipeline first.")
        return
    
    # Process query
    with st.spinner("Generating response..."):
        try:
            # Get RAG chain
            rag_chain = st.session_state.rag_chain
            
            # Process query
            result = rag_chain.query(query, return_source_documents=True)
            
            # Add to history
            st.session_state.history.append(result)
            
            # Clear query input
            st.session_state.query = ""
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
        st.info("Please install Ollama from https://ollama.com/ and pull the model specified in config.yml")
    
    # Config information
    st.subheader("Configuration")
    st.write(f"LLM: {config['llm']['model_name']}")
    st.write(f"Embeddings: {config['embeddings']['model_name']}")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        st.checkbox("Force Rebuild", value=False, key="force_rebuild")
    
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
        "Exoplanet RAG uses LangChain to build a Retrieval Augmented Generation system "
        "that answers your questions about exoplanets using ArXiv papers."
    )


# Main content
st.title("🪐 Exoplanet RAG")

# Query input
query = st.text_area("Ask a question about exoplanets:", key="query")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Submit"):
        if query:
            process_query(query)
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
            
            # Sources
            with st.expander("View Sources"):
                if "source_documents" in item and item["source_documents"]:
                    for j, doc in enumerate(item["source_documents"]):
                        source_type = doc.metadata.get("source", "full_text")
                        title = doc.metadata.get("title", "Unknown")
                        authors = doc.metadata.get("authors", "Unknown")
                        if isinstance(authors, list):
                            authors = ", ".join(authors)
                        published = doc.metadata.get("published", "Unknown")
                        
                        st.markdown(f"**Source {j+1}:** {title} by {authors} ({published})")
                        
                        if source_type == "abstract":
                            st.markdown("*From abstract:*")
                        
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown("---")
                else:
                    st.markdown("No source documents available.")
else:
    # First-time user instructions
    if st.session_state.rag_chain is None:
        st.info(
            "👈 To get started, check if Ollama is available in the sidebar and click 'Initialize Pipeline'. "
            "This will download ArXiv papers on exoplanets, process them, and set up the RAG system."
        )
    else:
        st.info(
            "Ask a question about exoplanets in the text area above and click 'Submit'. "
            "For example, you can ask 'What are hot Jupiters?' or 'How are exoplanets detected?'"
        )


# Footer
st.markdown("---")
st.markdown(
    "This application uses LangChain to implement Retrieval Augmented Generation (RAG) "
    "for answering questions about exoplanets using ArXiv papers."
)