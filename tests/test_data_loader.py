"""
Simple tests for data loader.
"""

import responses
from src.data_loader import _check_pdf_size


@responses.activate
def test_pdf_size_check_small_file():
    """Test PDF size check for small file."""
    responses.add(
        responses.HEAD,
        "http://test.com/paper.pdf",
        headers={'content-length': str(5 * 1024 * 1024)},  # 5MB
        status=200
    )
    
    result = _check_pdf_size("http://test.com/paper.pdf", max_size_mb=10.0)
    assert result is True


@responses.activate  
def test_pdf_size_check_large_file():
    """Test PDF size check for large file."""
    responses.add(
        responses.HEAD,
        "http://test.com/paper.pdf", 
        headers={'content-length': str(15 * 1024 * 1024)},  # 15MB
        status=200
    )
    
    result = _check_pdf_size("http://test.com/paper.pdf", max_size_mb=10.0)
    assert result is False