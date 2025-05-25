"""
Utility functions for the RAG application.
"""

import os
import yaml
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
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
    """Ensure all required directories exist."""
    logger.info("Ensuring directories exist")
    
    directories = [
        config["data_ingestion"]["paper_dir"],
        config["data_ingestion"]["processed_dir"],
        config["vector_store"]["persist_directory"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def format_citation(metadata: Dict[str, Any]) -> str:
    """Format document metadata into a citation."""
    title = metadata.get("title", "Unknown Title")
    authors = metadata.get("authors", "Unknown Authors")
    published = metadata.get("published", "Unknown Date")
    
    if isinstance(authors, list):
        authors = ", ".join(authors)
    
    return f"{authors}. ({published}). {title}."