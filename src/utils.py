"""
Utility functions for the LangChain RAG application.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Dictionary with configuration
    """
    logger.info(f"Loading config from {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info("Config loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Application configuration
    """
    logger.info("Ensuring directories exist")
    
    # Create data directories
    directories = [
        config["data_ingestion"]["paper_dir"],
        config["data_ingestion"]["processed_dir"],
        config["vector_store"]["persist_directory"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory {directory} created or already exists")


def time_function(func):
    """
    Decorator to time a function.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Function {func.__name__} took {elapsed_time:.2f} seconds to execute")
        
        return result
    
    return wrapper


def format_citation(metadata: Dict[str, Any]) -> str:
    """
    Format document metadata into a citation.
    
    Args:
        metadata: Document metadata
        
    Returns:
        Formatted citation
    """
    title = metadata.get("title", "Unknown Title")
    authors = metadata.get("authors", "Unknown Authors")
    published = metadata.get("published", "Unknown Date")
    
    if isinstance(authors, list):
        authors = ", ".join(authors)
    
    return f"{authors}. ({published}). {title}."


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
        
    return text[:max_length] + "..."