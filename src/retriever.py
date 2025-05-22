"""
Retrieval functions for the LangChain RAG application.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.retriever import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from .utils import load_config, time_function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
embedding_config = config["embeddings"]
vector_store_config = config["vector_store"]
rag_config = config["rag"]


def get_embeddings() -> Any:
    """
    Get embedding model.
    
    Returns:
        HuggingFace embeddings
    """
    embedding_model_name = embedding_config["model_name"]
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings


def load_vector_store() -> Optional[Chroma]:
    """
    Load the Chroma vector store.
    
    Returns:
        Chroma vector store
    """
    logger.info("Loading vector store")
    
    persist_directory = vector_store_config["persist_directory"]
    collection_name = vector_store_config["collection_name"]
    embeddings = get_embeddings()
    
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        count = vector_store._collection.count()
        logger.info(f"Vector store loaded with {count} documents")
        
        if count == 0:
            logger.warning("Vector store is empty. Run the ingestion pipeline first.")
            return None
            
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None


def create_basic_retriever(vector_store: Chroma) -> BaseRetriever:
    """
    Create a basic retriever from the vector store.
    
    Args:
        vector_store: Chroma vector store
        
    Returns:
        Retriever object
    """
    n_results = rag_config["n_results"]
    retriever = vector_store.as_retriever(search_kwargs={"k": n_results})
    return retriever


def create_enhanced_retriever(vector_store: Chroma) -> BaseRetriever:
    """
    Create an enhanced retriever with contextual compression.
    
    Args:
        vector_store: Chroma vector store
        
    Returns:
        Contextual compression retriever
    """
    # Create base retriever with higher k to allow filtering
    n_results = rag_config["n_results"] * 2
    base_retriever = vector_store.as_retriever(search_kwargs={"k": n_results})
    
    # Create embeddings filter
    embeddings = get_embeddings()
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.7  # Adjust this threshold as needed
    )
    
    # Create contextual compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )
    
    return compression_retriever


@time_function
def retrieve_documents(
    query: str,
    vector_store: Optional[Chroma] = None,
    enhanced: bool = False,
    n_results: Optional[int] = None
) -> List[Any]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: User query
        vector_store: Optional Chroma vector store
        enhanced: Whether to use enhanced retrieval
        n_results: Number of results to return
        
    Returns:
        List of retrieved documents
    """
    logger.info(f"Retrieving documents for query: '{query}'")
    
    # Set default n_results from config if not provided
    if n_results is None:
        n_results = rag_config["n_results"]
    
    # Load vector store if not provided
    if vector_store is None:
        vector_store = load_vector_store()
        
    if vector_store is None:
        logger.error("Vector store is not available")
        return []
    
    # Create retriever
    if enhanced:
        retriever = create_enhanced_retriever(vector_store)
    else:
        retriever = create_basic_retriever(vector_store)
    
    # Retrieve documents
    try:
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} documents")
        
        # Limit to n_results
        docs = docs[:n_results]
        
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def get_unique_sources(docs: List[Any]) -> Dict[str, Any]:
    """
    Get unique sources from retrieved documents.
    
    Args:
        docs: Retrieved documents
        
    Returns:
        Dictionary mapping paper IDs to metadata
    """
    sources = {}
    
    for doc in docs:
        paper_id = doc.metadata.get("paper_id")
        
        if paper_id and paper_id not in sources:
            sources[paper_id] = {
                "title": doc.metadata.get("title", "Unknown"),
                "authors": doc.metadata.get("authors", "Unknown"),
                "published": doc.metadata.get("published", "Unknown"),
                "paper_id": paper_id
            }
    
    return sources


def format_retrieved_documents(docs: List[Any]) -> str:
    """
    Format retrieved documents into a context string.
    
    Args:
        docs: Retrieved documents
        
    Returns:
        Formatted context string
    """
    if not docs:
        return ""
    
    context_parts = []
    
    for i, doc in enumerate(docs):
        # Format source
        authors = doc.metadata.get("authors", "Unknown")
        if isinstance(authors, list):
            authors = ", ".join(authors)
            
        title = doc.metadata.get("title", "Unknown")
        published = doc.metadata.get("published", "Unknown")
        source_type = doc.metadata.get("source", "full_text")
        
        # Format document header
        if source_type == "abstract":
            header = f"[{i+1}] Abstract of '{title}' by {authors} ({published})"
        else:
            header = f"[{i+1}] From '{title}' by {authors} ({published})"
        
        # Format content
        content = doc.page_content.strip()
        
        # Build context part
        context_part = f"{header}\n\n{content}\n\n"
        context_parts.append(context_part)
    
    # Join all context parts
    context = "\n".join(context_parts)
    
    return context