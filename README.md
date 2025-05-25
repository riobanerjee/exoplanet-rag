# Exoplanet RAG

A simple Retrieval Augmented Generation (RAG) system for querying exoplanets papers ArXiv papers, built with LangChain. This is a framework for RAG, that demonstrates the concept of a RAG in this field with a small document database. A larger document database and a better LLM would provide improved results.

Deployed at 

## Features

- Fetch and process ArXiv papers about exoplanets
- Create embeddings for efficient similarity search
- Answer questions using retrieved context and local LLM
- Web interface built with Streamlit

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and setup Ollama**:
   ```bash
   # Install from https://ollama.com/
   ollama pull gemma3:1b
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Initialize the system**: Click "Initialize Pipeline" in the sidebar

5. **Ask questions**: Enter questions about exoplanets like "What are sub Neptunes?"

## Configuration

Edit `config.yml` to change models or parameters:

```yaml
llm:
  model_name: "gemma3:1b"  # Change to your preferred model
embeddings:
  model_name: "all-MiniLM-L6-v2"
data_ingestion:
  max_papers: 50
  download_limit: 20
```

## Architecture

- **Data Ingestion**: Downloads and processes ArXiv papers
- **Embeddings**: Creates vector representations using Sentence Transformers
- **Vector Store**: Stores embeddings in ChromaDB for fast retrieval
- **LLM**: Uses Ollama for local language model inference
- **Interface**: Streamlit web app for user interaction

## Example Questions

- "What are sub Neptunes?"
- "How are exoplanets detected?"
- "What is the habitable zone?"
- "What is an atmospheric retrieval?"