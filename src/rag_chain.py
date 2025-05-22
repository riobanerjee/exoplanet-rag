"""
RAG chain implementation using LangChain.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .retriever import retrieve_documents, load_vector_store, format_retrieved_documents
from .utils import load_config, time_function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
llm_config = config["llm"]
rag_config = config["rag"]


def get_llm() -> Any:
    """
    Get the LLM.
    
    Returns:
        Ollama LLM
    """
    logger.info(f"Initializing LLM: {llm_config['model_name']}")
    
    try:
        llm = Ollama(
            model=llm_config["model_name"],
            temperature=llm_config["temperature"],
            num_predict=llm_config["max_tokens"]
        )
        
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise


def get_rag_prompt() -> PromptTemplate:
    """
    Get the RAG prompt template.
    
    Returns:
        PromptTemplate
    """
    system_prompt = rag_config["system_prompt"]
    
    template = f"""
{system_prompt}

Context information is below.

{{context}}

Given the context information and not prior knowledge, answer the query.
Query: {{query}}

Answer:
"""
    
    return PromptTemplate.from_template(template)


def create_rag_chain(vector_store: Optional[Any] = None) -> Any:
    """
    Create a RAG chain.
    
    Args:
        vector_store: Optional vector store
        
    Returns:
        RAG chain
    """
    logger.info("Creating RAG chain")
    
    # Load vector store if not provided
    if vector_store is None:
        vector_store = load_vector_store()
        
    if vector_store is None:
        logger.error("Vector store is not available")
        raise ValueError("Vector store is not available. Run the ingestion pipeline first.")
    
    # Get LLM
    llm = get_llm()
    
    # Get prompt
    prompt = get_rag_prompt()
    
    # Define retrieval function
    def retrieve_and_format(query: str) -> str:
        docs = retrieve_documents(query, vector_store=vector_store)
        return format_retrieved_documents(docs)
    
    # Create RAG chain
    rag_chain = (
        {"context": retrieve_and_format, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


@time_function
def process_query(
    query: str,
    vector_store: Optional[Any] = None,
    return_source_documents: bool = False
) -> Dict[str, Any]:
    """
    Process a query through the RAG chain.
    
    Args:
        query: Query string
        vector_store: Optional vector store
        return_source_documents: Whether to return source documents
        
    Returns:
        Dictionary with response and sources
    """
    logger.info(f"Processing query: {query}")
    
    # Load vector store if not provided
    if vector_store is None:
        vector_store = load_vector_store()
        
    if vector_store is None:
        return {
            "query": query,
            "response": "Error: Vector store is not available. Please run the ingestion pipeline first.",
            "source_documents": [],
            "context": ""
        }
    
    try:
        # Create chain or get LLM directly
        rag_chain = create_rag_chain(vector_store)
        
        # Retrieve documents
        docs = retrieve_documents(query, vector_store=vector_store)
        context = format_retrieved_documents(docs)
        
        # Generate response
        response = rag_chain.invoke(query)
        
        result = {
            "query": query,
            "response": response,
            "context": context
        }
        
        if return_source_documents:
            result["source_documents"] = docs
            
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "query": query,
            "response": f"Error processing query: {str(e)}",
            "source_documents": [],
            "context": ""
        }


class RAGChain:
    """
    RAG Chain class for easier usage.
    """
    
    def __init__(self, vector_store: Optional[Any] = None):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: Optional vector store
        """
        self.vector_store = vector_store if vector_store else load_vector_store()
        self.chain = None
        
        # Initialize chain if vector store is available
        if self.vector_store is not None:
            self.chain = create_rag_chain(self.vector_store)
    
    def query(
        self,
        query: str,
        return_source_documents: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query.
        
        Args:
            query: Query string
            return_source_documents: Whether to return source documents
            
        Returns:
            Dictionary with response and sources
        """
        return process_query(
            query,
            vector_store=self.vector_store,
            return_source_documents=return_source_documents
        )