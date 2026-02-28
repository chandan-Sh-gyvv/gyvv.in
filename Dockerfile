FROM python:3.11-slim

WORKDIR /app

# System deps for faiss + sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create directories for indexes and DB
RUN mkdir -p /tmp/indexes

# Expose port
EXPOSE 8000

CMD ["uvicorn", "shsrs_rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
