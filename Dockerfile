FROM python:3.11-slim

WORKDIR /app

# Install system deps and Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Create data dirs
RUN mkdir -p data/papers data/processed data/chroma_db

EXPOSE 8501

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]