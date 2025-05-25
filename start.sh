#!/bin/bash

# Start Ollama
ollama serve &
sleep 10

# Pull model
ollama pull gemma2:2b

# Start app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0