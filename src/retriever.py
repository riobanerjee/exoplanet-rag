"""
Retrieval functions for the RAG application.
"""

import logging
from typing import List, Any, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .utils import load_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
embedding_config = config["embeddings"]
vector_store_config = config["vector_store"]
rag_config = config["rag"]


def get_embeddings() -> Any:
    """Get embedding model."""
    embedding_model_name = embedding_config["model_name"]
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings


def load_vector_store() -> Optional[Chroma]:
    """Load the Chroma vector store."""
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


def retrieve_documents(
    query: str,
    vector_store: Optional[Chroma] = None,
    n_results: Optional[int] = None
) -> List[Any]:
    """Retrieve relevant documents for a query."""
    logger.info(f"Retrieving documents for query: '{query}'")
    
    if n_results is None:
        n_results = rag_config["n_results"]
    
    if vector_store is None:
        vector_store = load_vector_store()
        
    if vector_store is None:
        logger.error("Vector store is not available")
        return []
    
    try:
        docs = vector_store.similarity_search(query, k=n_results)
        logger.info(f"Retrieved {len(docs)} documents")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def format_retrieved_documents(docs: List[Any]) -> str:
    """Format retrieved documents into a context string."""
    if not docs:
        return ""
    
    context_parts = []
    
    for i, doc in enumerate(docs):
        authors = doc.metadata.get("authors", "Unknown")
        title = doc.metadata.get("title", "Unknown")
        published = doc.metadata.get("published", "Unknown")
        source_type = doc.metadata.get("source", "full_text")
        
        if source_type == "abstract":
            header = f"[{i+1}] Abstract of '{title}' by {authors} ({published})"
        else:
            header = f"[{i+1}] From '{title}' by {authors} ({published})"
        
        content = doc.page_content.strip()
        context_part = f"{header}\n\n{content}\n\n"
        context_parts.append(context_part)
    
    return "\n".join(context_parts)