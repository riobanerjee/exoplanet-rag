#!/usr/bin/env python3
"""
Simple ArXiv search test - search and print results.
"""

import arxiv

# Search for exoplanet papers
query = "cat:astro-ph.EP AND (ti:exoplanet OR abs:exoplanet OR ti:exoplanets OR abs:exoplanets)"
max_results = 20

print(f"Searching ArXiv for: {query}")
print(f"Max results: {max_results}")
print("-" * 60)
client = arxiv.Client()

search = arxiv.Search(
    query=query,
    max_results=max_results,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for i, result in enumerate(client.results(search), 1):
    print(f"{i}. {result.title}")
    print(f"   Authors: {', '.join([author.name for author in result.authors[:2]])}...")
    print(f"   Published: {result.published.strftime('%Y-%m-%d')}")
    print(f"   ID: {result.entry_id.split('/')[-1]}")
    print()