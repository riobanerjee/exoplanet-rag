"""
Data ingestion pipeline for the LangChain RAG application.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .data_loader import search_arxiv_papers, download_pdfs, get_available_papers
from .utils import load_config, ensure_directories, time_function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
data_config = config["data_ingestion"]
embedding_config = config["embeddings"]
vector_store_config = config["vector_store"]


@time_function
def fetch_arxiv_papers() -> List[Dict[str, Any]]:
    """
    Fetch papers from ArXiv.
    
    Returns:
        List of paper metadata
    """
    logger.info("Fetching papers from ArXiv")
    
    # Search for papers
    papers = search_arxiv_papers(
        query=data_config["arxiv_query"],
        max_results=data_config["max_papers"]
    )
    
    # Download PDFs
    max_size_mb = data_config.get("max_pdf_size_mb", 10.0)
    download_pdfs(papers, limit=data_config["download_limit"], max_size_mb=max_size_mb)
    
    return papers


@time_function
def load_and_split_documents() -> List[Any]:
    """
    Load and split documents into chunks.
    
    Returns:
        List of document chunks
    """
    logger.info("Loading and splitting documents")
    
    # Get available papers
    available_papers = get_available_papers()
    
    if not available_papers:
        logger.error("No papers available. Fetch papers first.")
        return []
    
    # Configure text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=data_config["chunk_size"],
        chunk_overlap=data_config["chunk_overlap"],
        length_function=len,
        is_separator_regex=False
    )
    
    # Load and split documents
    all_chunks = []
    
    for paper in tqdm(available_papers, desc="Processing papers"):
        arxiv_id = paper["arxiv_id"]
        pdf_path = os.path.join(data_config["paper_dir"], f"{arxiv_id}.pdf")
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "paper_id": arxiv_id,
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "source": "full_text"
                })
            
            # Split documents
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            
            # Also add abstract as a separate document
            abstract_doc = {
                "page_content": paper["summary"],
                "metadata": {
                    "paper_id": arxiv_id,
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "source": "abstract"
                }
            }
            
            all_chunks.append(abstract_doc)
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
    
    logger.info(f"Created {len(all_chunks)} document chunks")
    return all_chunks


@time_function
def create_vector_store(documents: List[Any], recreate: bool = False) -> Any:
    """
    Create a vector store from documents.
    
    Args:
        documents: List of document chunks
        recreate: Whether to recreate the vector store
        
    Returns:
        Chroma vector store
    """
    logger.info("Creating vector store")
    
    # Configure embedding model
    embedding_model_name = embedding_config["model_name"]
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Configure vector store
    persist_directory = vector_store_config["persist_directory"]
    collection_name = vector_store_config["collection_name"]
    
    # Check if vector store exists
    if os.path.exists(persist_directory) and not recreate:
        # Load existing vector store
        logger.info(f"Loading existing vector store from {persist_directory}")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # If vector store is empty, we should add documents
        if vector_store._collection.count() == 0 and documents:
            logger.info("Vector store is empty, adding documents")
            vector_store.add_documents(documents)
            vector_store.persist()
    else:
        # Create new vector store
        if recreate and os.path.exists(persist_directory):
            logger.info(f"Recreating vector store at {persist_directory}")
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
        
        if not documents:
            logger.error("No documents provided for vector store creation")
            return None
        
        logger.info(f"Creating new vector store at {persist_directory}")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        vector_store.persist()
    
    logger.info(f"Vector store has {vector_store._collection.count()} documents")
    return vector_store


@time_function
def run_ingestion_pipeline(force_rebuild: bool = False) -> Any:
    """
    Run the complete ingestion pipeline.
    
    Args:
        force_rebuild: Whether to force rebuilding the vector store
        
    Returns:
        Chroma vector store
    """
    logger.info("Starting ingestion pipeline")
    
    # Ensure directories exist
    ensure_directories(config)
    
    # Check if we already have a vector store
    persist_directory = vector_store_config["persist_directory"]
    vector_store_exists = os.path.exists(persist_directory) and not force_rebuild
    
    # Fetch papers if necessary or requested
    if not vector_store_exists or force_rebuild:
        fetch_arxiv_papers()
    
    # Load and split documents if necessary or requested
    documents = []
    if not vector_store_exists or force_rebuild:
        documents = load_and_split_documents()
    
    # Create vector store
    vector_store = create_vector_store(documents, recreate=force_rebuild)
    
    logger.info("Ingestion pipeline complete")
    return vector_store


if __name__ == "__main__":
    run_ingestion_pipeline()