FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common \
    sqlite3 \
    libsqlite3-dev \
    # GPU support libraries
    nvidia-cuda-toolkit \
    # PDF processing dependencies
    libpoppler-cpp-dev \
    poppler-utils \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt

# Install additional GPU-optimized packages
RUN pip3 install --no-cache-dir \
    faiss-gpu \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/cache/semantic_indexes && \
    mkdir -p /app/prompts && \
    mkdir -p /app/uploads && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Start Ollama service and pull models in background
RUN ollama serve & \
    sleep 10 && \
    ollama pull mistral:latest && \
    ollama pull llama3.1:latest && \
    ollama pull deepseek-r1:1.5b && \
    # Optional models (comment out if not needed)
    ollama pull gemma:7b && \
    ollama pull codellama:7b && \
    pkill ollama

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports
EXPOSE 8501 11434

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama service\n\
ollama serve &\n\
sleep 5\n\
\n\
# Start Streamlit application\n\
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableXsrfProtection=false --server.enableCORS=false\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the application
CMD ["/app/start.sh"]

# Build instructions and optimization notes:
#
# Build command:
# docker build -t resume-optimizer:latest .
#
# Run command (with GPU support):
# docker run --gpus all -p 8501:8501 -p 11434:11434 -v $(pwd)/data:/app/data resume-optimizer:latest
#
# Run command (CPU only):
# docker run -p 8501:8501 -p 11434:11434 -v $(pwd)/data:/app/data resume-optimizer:latest
#
# For production with persistent data:
# docker run --gpus all -p 8501:8501 -p 11434:11434 \
#   -v $(pwd)/data:/app/data \
#   -v $(pwd)/cache:/app/cache \
#   -v $(pwd)/models:/root/.ollama \
#   --restart unless-stopped \
#   resume-optimizer:latest
#
# Memory optimization:
# - Add --memory=8g --memory-swap=16g for memory limits
# - Use --shm-size=2g for shared memory
#
# GPU optimization notes:
# - Requires NVIDIA Docker runtime
# - GPU memory is automatically managed by PyTorch and FAISS
# - Models will automatically use GPU when available