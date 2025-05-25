"""
Simple unit tests for utility functions.
"""

import pytest
from src.utils import format_citation


def test_format_citation():
    """Test citation formatting."""
    metadata = {
        "title": "Test Paper",
        "authors": "John Doe, Jane Smith",
        "published": "2023-01-01"
    }
    
    result = format_citation(metadata)
    expected = "John Doe, Jane Smith. (2023-01-01). Test Paper."
    assert result == expected


def test_format_citation_missing_data():
    """Test citation with missing data."""
    metadata = {}
    result = format_citation(metadata)
    expected = "Unknown Authors. (Unknown Date). Unknown Title."
    assert result == expected