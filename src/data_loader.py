"""
Functions for fetching and loading ArXiv papers.
"""

import os
import json
import time
import logging
import arxiv
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .utils import load_config, ensure_directories

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
data_config = config["data_ingestion"]

# Get data directories
PAPERS_DIR = data_config["paper_dir"]
PROCESSED_DIR = data_config["processed_dir"]


def search_arxiv_papers(
    query: Optional[str] = None,
    max_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Search ArXiv for papers matching the query.
    
    Args:
        query: ArXiv query string
        max_results: Maximum number of results to return
        
    Returns:
        List of paper metadata dictionaries
    """
    # Use config values if not specified
    if query is None:
        query = data_config["arxiv_query"]
    
    if max_results is None:
        max_results = data_config["max_papers"]
    
    logger.info(f"Searching ArXiv for: {query}")
    
    # Create search client
    client = arxiv.Client()
    
    # Create search
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in tqdm(client.results(search), desc="Fetching papers", total=max_results):
        paper = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'pdf_url': result.pdf_url,
            'published': result.published.strftime('%Y-%m-%d'),
            'arxiv_id': result.entry_id.split('/')[-1],
            'categories': result.categories
        }
        papers.append(paper)
        
    logger.info(f"Found {len(papers)} papers")
    
    # Save the metadata
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    metadata_path = os.path.join(PROCESSED_DIR, "paper_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(papers, f, indent=2)
        
    return papers


def download_pdfs(papers: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
    """
    Download PDFs for the given papers.
    
    Args:
        papers: List of paper metadata
        limit: Optional limit on number of papers to download
    """
    if limit is None:
        limit = data_config["download_limit"]
        
    if limit:
        papers = papers[:limit]
        
    logger.info(f"Downloading {len(papers)} PDFs")
    
    # Use the ArXiv API directly for downloads
    client = arxiv.Client()
    
    os.makedirs(PAPERS_DIR, exist_ok=True)
    
    for paper in tqdm(papers, desc="Downloading PDFs"):
        arxiv_id = paper['arxiv_id']
        pdf_path = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")
        
        # Skip if already downloaded
        if os.path.exists(pdf_path):
            continue
            
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper_obj = next(client.results(search))
            paper_obj.download_pdf(dirpath=PAPERS_DIR, filename=f"{arxiv_id}.pdf")
            
            # Be nice to the ArXiv API - don't hammer it
            time.sleep(3)
        except Exception as e:
            logger.error(f"Error downloading {arxiv_id}: {e}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text()
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def load_papers_metadata() -> List[Dict[str, Any]]:
    """
    Load paper metadata from the processed directory.
    
    Returns:
        List of paper metadata
    """
    metadata_path = os.path.join(PROCESSED_DIR, "paper_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return []
    
    with open(metadata_path, 'r') as f:
        papers = json.load(f)
    
    logger.info(f"Loaded metadata for {len(papers)} papers")
    return papers


def get_paper_text(arxiv_id: str) -> Optional[str]:
    """
    Get the text content of a paper.
    
    Args:
        arxiv_id: ArXiv ID of the paper
        
    Returns:
        Text content of the paper
    """
    pdf_path = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")
    
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF file not found: {pdf_path}")
        return None
    
    text = extract_text_from_pdf(pdf_path)
    return text


def get_available_papers() -> List[Dict[str, Any]]:
    """
    Get metadata for papers that have been downloaded.
    
    Returns:
        List of paper metadata
    """
    papers = load_papers_metadata()
    
    if not papers:
        return []
    
    available_papers = []
    
    for paper in papers:
        arxiv_id = paper['arxiv_id']
        pdf_path = os.path.join(PAPERS_DIR, f"{arxiv_id}.pdf")
        
        if os.path.exists(pdf_path):
            available_papers.append(paper)
    
    logger.info(f"Found {len(available_papers)} available papers out of {len(papers)}")
    return available_papers