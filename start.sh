#!/bin/bash

# Start Ollama
ollama serve &
sleep 10

# Pull model
ollama pull gemma3:1b

# Start app
streamlit run app.py --server.port=8080 --server.address=0.0.0.0