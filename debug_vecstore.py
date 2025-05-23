#!/usr/bin/env python3
"""
Targeted debug script for vector store issues.
"""

import os
import json
from src.utils import load_config
from src.ingest import load_and_split_documents, create_vector_store

def check_document_chunks():
    """Check the document chunks that were created."""
    print("🔍 Checking document chunks...")
    
    try:
        documents = load_and_split_documents()
        print(f"   📊 Total documents: {len(documents)}")
        
        if len(documents) == 0:
            print("   ❌ No documents found!")
            return None
        
        # Check first few documents
        for i, doc in enumerate(documents[:3]):
            print(f"\n   📄 Document {i+1}:")
            print(f"      Type: {type(doc)}")
            print(f"      Has page_content: {hasattr(doc, 'page_content')}")
            print(f"      Has metadata: {hasattr(doc, 'metadata')}")
            
            if hasattr(doc, 'page_content'):
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"      Content: {content_preview}")
            
            if hasattr(doc, 'metadata'):
                print(f"      Metadata keys: {list(doc.metadata.keys())}")
                print(f"      Paper ID: {doc.metadata.get('paper_id', 'N/A')}")
                print(f"      Source: {doc.metadata.get('source', 'N/A')}")
        
        return documents
        
    except Exception as e:
        print(f"   ❌ Error checking documents: {e}")
        return None

def test_embeddings():
    """Test if embeddings work."""
    print("\n🧠 Testing embeddings...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        config = load_config()
        embedding_model_name = config["embeddings"]["model_name"]
        
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embedding a simple text
        test_text = "This is a test sentence about exoplanets."
        embedding = embeddings.embed_query(test_text)
        
        print(f"   ✅ Embeddings working")
        print(f"   📊 Embedding dimension: {len(embedding)}")
        print(f"   📊 First 5 values: {embedding[:5]}")
        
        return embeddings
        
    except Exception as e:
        print(f"   ❌ Error testing embeddings: {e}")
        return None

def test_chroma_creation():
    """Test creating a minimal Chroma vector store."""
    print("\n🗃️ Testing Chroma creation...")
    
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        
        config = load_config()
        
        # Create test documents
        test_docs = [
            Document(
                page_content="Hot Jupiters are exoplanets that orbit very close to their stars.",
                metadata={"paper_id": "test1", "title": "Test Paper 1", "source": "test"}
            ),
            Document(
                page_content="Transit photometry is used to detect exoplanets.",
                metadata={"paper_id": "test2", "title": "Test Paper 2", "source": "test"}
            )
        ]
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=config["embeddings"]["model_name"],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create temporary vector store
        test_persist_dir = "test_chroma_temp"
        
        print("   🔧 Creating test vector store...")
        vector_store = Chroma.from_documents(
            documents=test_docs,
            embedding=embeddings,
            collection_name="test_collection",
            persist_directory=test_persist_dir
        )
        
        count = vector_store._collection.count()
        print(f"   ✅ Test vector store created with {count} documents")
        
        # Test query
        results = vector_store.similarity_search("hot jupiter", k=1)
        print(f"   🔍 Test query returned {len(results)} results")
        
        if results:
            print(f"   📄 First result: {results[0].page_content[:50]}...")
        
        # Cleanup
        import shutil
        if os.path.exists(test_persist_dir):
            shutil.rmtree(test_persist_dir)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing Chroma: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actual_vector_store_creation():
    """Test creating the actual vector store with real documents."""
    print("\n🚀 Testing actual vector store creation...")
    
    try:
        # Load real documents
        documents = load_and_split_documents()
        
        if not documents:
            print("   ❌ No documents to create vector store with")
            return False
        
        print(f"   📊 Creating vector store with {len(documents)} documents...")
        
        # Try to create vector store
        vector_store = create_vector_store(documents, recreate=True)
        
        if vector_store is None:
            print("   ❌ Vector store creation returned None")
            return False
        
        count = vector_store._collection.count()
        print(f"   ✅ Vector store created with {count} documents")
        
        # Test a query
        try:
            results = vector_store.similarity_search("exoplanet", k=3)
            print(f"   🔍 Test query returned {len(results)} results")
            
            for i, result in enumerate(results[:2]):
                title = result.metadata.get('title', 'Unknown')[:40]
                source = result.metadata.get('source', 'unknown')
                print(f"      {i+1}. {title}... (source: {source})")
            
        except Exception as e:
            print(f"   ⚠️ Query test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating actual vector store: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all vector store debug tests."""
    print("🔍 Vector Store Debug Tool")
    print("=" * 50)
    
    # Check documents
    documents = check_document_chunks()
    
    # Test embeddings
    embeddings_ok = test_embeddings() is not None
    
    # Test basic Chroma functionality
    chroma_basic_ok = test_chroma_creation()
    
    # Test actual vector store creation
    if documents:
        vector_store_ok = test_actual_vector_store_creation()
    else:
        vector_store_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VECTOR STORE DEBUG SUMMARY:")
    print(f"   Documents loaded: {'✅' if documents else '❌'}")
    print(f"   Embeddings working: {'✅' if embeddings_ok else '❌'}")
    print(f"   Chroma basic test: {'✅' if chroma_basic_ok else '❌'}")
    print(f"   Actual vector store: {'✅' if vector_store_ok else '❌'}")
    
    if not all([documents, embeddings_ok, chroma_basic_ok, vector_store_ok]):
        print("\n💡 NEXT STEPS:")
        if not documents:
            print("   - Run data ingestion again: python -c 'from src.ingest import load_and_split_documents; load_and_split_documents()'")
        if not embeddings_ok:
            print("   - Check embedding model installation: pip install sentence-transformers")
        if not chroma_basic_ok:
            print("   - Check Chroma installation: pip install langchain-chroma")
        if not vector_store_ok:
            print("   - Check the error details above for specific issues")
    else:
        print("\n🎉 All vector store components working!")

if __name__ == "__main__":
    main()