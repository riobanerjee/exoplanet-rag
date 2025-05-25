"""
Data ingestion pipeline for the RAG application.
"""

import os
import logging
from typing import List, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .data_loader import search_arxiv_papers, download_pdfs, get_available_papers
from .utils import load_config, ensure_directories

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
data_config = config["data_ingestion"]
embedding_config = config["embeddings"]
vector_store_config = config["vector_store"]


def fetch_arxiv_papers() -> List[Any]:
    """Fetch papers from ArXiv."""
    logger.info("Fetching papers from ArXiv")
    
    papers = search_arxiv_papers(
        query=data_config["arxiv_query"],
        max_results=data_config["max_papers"]
    )
    
    max_size_mb = data_config.get("max_pdf_size_mb", 10.0)
    download_pdfs(papers, limit=data_config["download_limit"], max_size_mb=max_size_mb)
    
    return papers


def load_and_split_documents() -> List[Any]:
    """Load and split documents into chunks."""
    logger.info("Loading and splitting documents")
    
    available_papers = get_available_papers()
    
    if not available_papers:
        logger.error("No papers available. Fetch papers first.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=data_config["chunk_size"],
        chunk_overlap=data_config["chunk_overlap"],
        length_function=len,
        is_separator_regex=False
    )
    
    all_chunks = []
    
    for paper in available_papers:
        arxiv_id = paper["arxiv_id"]
        pdf_path = os.path.join(data_config["paper_dir"], f"{arxiv_id}.pdf")
        
        # Convert authors list to string for ChromaDB compatibility
        authors = paper["authors"]
        if isinstance(authors, list):
            authors_str = ", ".join(authors)
        else:
            authors_str = str(authors)
        
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Skip if no documents loaded or if text is too short
            if not documents:
                logger.warning(f"No documents loaded from {pdf_path}")
                continue
                
            # Check if we got meaningful content
            total_text = "".join([doc.page_content for doc in documents])
            if len(total_text.strip()) < 100:
                logger.warning(f"Very little text extracted from {pdf_path}, skipping")
                continue
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "paper_id": arxiv_id,
                    "title": paper["title"],
                    "authors": authors_str,
                    "published": paper["published"],
                    "source": "full_text"
                })
            
            # Split documents
            chunks = text_splitter.split_documents(documents)
            
            if chunks:
                all_chunks.extend(chunks)
                logger.debug(f"Added {len(chunks)} chunks from {arxiv_id}")
            else:
                logger.warning(f"No chunks created from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
        
        # Always add abstract as a separate document
        try:
            abstract_doc = Document(
                page_content=paper["summary"],
                metadata={
                    "paper_id": arxiv_id,
                    "title": paper["title"],
                    "authors": authors_str,
                    "published": paper["published"],
                    "source": "abstract"
                }
            )
            
            all_chunks.append(abstract_doc)
            
        except Exception as e:
            logger.error(f"Failed to add abstract for {arxiv_id}: {e}")
    
    logger.info(f"Created {len(all_chunks)} document chunks")
    return all_chunks


def create_vector_store(documents: List[Any], recreate: bool = False) -> Any:
    """Create a vector store from documents."""
    logger.info("Creating vector store")
    
    embedding_model_name = embedding_config["model_name"]
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    persist_directory = vector_store_config["persist_directory"]
    collection_name = vector_store_config["collection_name"]
    
    # Check if vector store exists
    if os.path.exists(persist_directory) and not recreate:
        logger.info(f"Loading existing vector store from {persist_directory}")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        if vector_store._collection.count() == 0 and documents:
            logger.info("Vector store is empty, adding documents")
            vector_store.add_documents(documents)
    else:
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
    
    logger.info(f"Vector store has {vector_store._collection.count()} documents")
    return vector_store


def run_ingestion_pipeline(force_rebuild: bool = False) -> Any:
    """Run the complete ingestion pipeline."""
    logger.info("Starting ingestion pipeline")
    
    ensure_directories(config)
    
    persist_directory = vector_store_config["persist_directory"]
    vector_store_exists = os.path.exists(persist_directory) and not force_rebuild
    
    if not vector_store_exists or force_rebuild:
        fetch_arxiv_papers()
        documents = load_and_split_documents()
    else:
        documents = []
    
    vector_store = create_vector_store(documents, recreate=force_rebuild)
    
    logger.info("Ingestion pipeline complete")
    return vector_store