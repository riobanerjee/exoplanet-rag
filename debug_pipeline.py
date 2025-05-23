#!/usr/bin/env python3
"""
Debug script to check each step of the RAG pipeline.
"""

import os
import sys
from src.utils import load_config
from src.data_loader import search_arxiv_papers, get_available_papers
from src.ingest import run_ingestion_pipeline
from src.retriever import load_vector_store, retrieve_documents

def check_config():
    """Check if config loads properly."""
    print("🔧 Checking configuration...")
    try:
        config = load_config()
        print(f"   ✅ Config loaded successfully")
        print(f"   📊 Max papers: {config['data_ingestion']['max_papers']}")
        print(f"   📥 Download limit: {config['data_ingestion']['download_limit']}")
        print(f"   🤖 LLM model: {config['llm']['model_name']}")
        print(f"   🧠 Embedding model: {config['embeddings']['model_name']}")
        return True
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        return False

def check_directories():
    """Check if directories exist."""
    print("\n📁 Checking directories...")
    config = load_config()
    
    dirs_to_check = [
        config["data_ingestion"]["paper_dir"],
        config["data_ingestion"]["processed_dir"],
        config["vector_store"]["persist_directory"]
    ]
    
    for directory in dirs_to_check:
        if os.path.exists(directory):
            files = os.listdir(directory)
            print(f"   ✅ {directory} exists ({len(files)} files)")
        else:
            print(f"   ❌ {directory} does not exist")
            os.makedirs(directory, exist_ok=True)
            print(f"   🔧 Created {directory}")

def check_papers():
    """Check if papers are downloaded."""
    print("\n📄 Checking papers...")
    try:
        available_papers = get_available_papers()
        print(f"   📊 Found {len(available_papers)} available papers")
        
        if len(available_papers) == 0:
            print("   ⚠️  No papers available. Let's search for some...")
            papers = search_arxiv_papers(max_results=5)
            print(f"   🔍 Found {len(papers)} papers in search")
            return False
        else:
            for i, paper in enumerate(available_papers[:3], 1):
                print(f"   {i}. {paper['title'][:60]}... ({paper['published']})")
            return True
            
    except Exception as e:
        print(f"   ❌ Error checking papers: {e}")
        return False

def check_vector_store():
    """Check if vector store exists and has documents."""
    print("\n🗃️  Checking vector store...")
    try:
        vector_store = load_vector_store()
        
        if vector_store is None:
            print("   ❌ Vector store is None")
            return False
        
        count = vector_store._collection.count()
        print(f"   📊 Vector store has {count} documents")
        
        if count == 0:
            print("   ⚠️  Vector store is empty")
            return False
        else:
            print("   ✅ Vector store loaded successfully")
            return True
            
    except Exception as e:
        print(f"   ❌ Error loading vector store: {e}")
        return False

def test_retrieval():
    """Test document retrieval."""
    print("\n🔍 Testing retrieval...")
    try:
        test_query = "What are hot Jupiters?"
        print(f"   Query: {test_query}")
        
        docs = retrieve_documents(test_query, n_results=3)
        print(f"   📊 Retrieved {len(docs)} documents")
        
        if len(docs) == 0:
            print("   ❌ No documents retrieved")
            return False
        
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Unknown")[:50]
            source = doc.metadata.get("source", "unknown")
            print(f"   {i}. {title}... (source: {source})")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error during retrieval: {e}")
        return False

def run_full_pipeline():
    """Run the full ingestion pipeline."""
    print("\n🚀 Running full ingestion pipeline...")
    try:
        vector_store = run_ingestion_pipeline(force_rebuild=True)
        
        if vector_store is None:
            print("   ❌ Pipeline failed - vector store is None")
            return False
        
        count = vector_store._collection.count()
        print(f"   ✅ Pipeline completed - vector store has {count} documents")
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline failed: {e}")
        return False

def main():
    """Run all debug checks."""
    print("🔍 RAG Pipeline Debug Tool")
    print("=" * 50)
    
    # Check each component
    # config_ok = check_config()
    # if not config_ok:
    #     return
    
    # check_directories()
    # papers_ok = check_papers()
    vector_store_ok = check_vector_store()
    
    # # If vector store doesn't exist or is empty, run pipeline
    # if not vector_store_ok:
    #     print("\n🔧 Vector store issues detected. Running full pipeline...")
    #     pipeline_ok = run_full_pipeline()
        
    #     if pipeline_ok:
    #         # Test again after pipeline
    #         vector_store_ok = check_vector_store()
    
    # Test retrieval if vector store is working
    if vector_store_ok:
        retrieval_ok = test_retrieval()
    else:
        retrieval_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    # print(f"   Config: {'✅' if config_ok else '❌'}")
    # print(f"   Papers: {'✅' if papers_ok else '❌'}")
    print(f"   Vector Store: {'✅' if vector_store_ok else '❌'}")
    print(f"   Retrieval: {'✅' if retrieval_ok else '❌'}")
    
    # if all([config_ok, papers_ok, vector_store_ok, retrieval_ok]):
    #     print("\n🎉 All systems working! Your RAG pipeline should work now.")
    # else:
    #     print("\n⚠️  Some issues found. Check the output above for details.")

if __name__ == "__main__":
    main()