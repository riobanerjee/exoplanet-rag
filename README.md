# Exoplanet RAG with LangChain

A minimalist Retrieval Augmented Generation (RAG) system for querying scientific knowledge about exoplanets from ArXiv papers, built with LangChain.

## Features

- Fetch and process ArXiv papers about exoplanets
- Create and store embeddings for efficient retrieval
- Query the system using natural language
- Generate responses based on retrieved context
- All components run locally - no API keys needed

## Project Structure

```
langchain-exoplanet-rag/
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
├── .gitignore               # Git ignore file
├── config.yml               # Configuration file
├── app.py                   # Streamlit application
└── src/                     # Source code
    ├── __init__.py
    ├── data_loader.py       # Functions for data collection
    ├── ingest.py            # Data ingestion pipeline
    ├── retriever.py         # Retrieval functions
    ├── rag_chain.py         # LangChain RAG chain
    └── utils.py             # Utility functions
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install Ollama from [https://ollama.com/](https://ollama.com/)
4. Pull a model: `ollama pull llama3`
5. Run the Streamlit app: `streamlit run app.py`

## Configuration

The `config.yml` file contains all settings for the application:

```yaml
# LLM settings
llm:
  provider: "ollama"
  model_name: "llama3"
  temperature: 0.7

# Embedding settings
embeddings:
  provider: "sentence_transformers"
  model_name: "all-MiniLM-L6-v2"

# Vector store settings
vector_store:
  provider: "chroma"
  collection_name: "exoplanet_papers"

# Data ingestion settings
data_ingestion:
  arxiv_query: "cat:astro-ph.EP AND (ti:exoplanet OR abs:exoplanet OR ti:exoplanets OR abs:exoplanets)"
  max_papers: 50
  download_limit: 20
  chunk_size: 500
  chunk_overlap: 50
```

## How it Works

1. **Data Ingestion**: The application fetches papers about exoplanets from ArXiv.
2. **Document Processing**: Papers are split into chunks and embedded.
3. **Retrieval**: When a query is received, relevant chunks are retrieved.
4. **Response Generation**: A LLM generates a response based on the retrieved context.

## Usage

After starting the Streamlit app, you can:
1. Click "Initialize Pipeline" to set up the RAG system
2. Enter your question about exoplanets in the query box
3. View the generated response and its source documents