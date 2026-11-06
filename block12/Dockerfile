# ── Sentinel-AI Production Dockerfile (Optimized for Render Free Tier - 512MB RAM) ──
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and generating .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    DEPLOYMENT_MODE=render_free_web

WORKDIR /app

# Install system dependencies:
# - libgomp1: Required by faiss-cpu (OpenMP shared library)
# - curl: Required for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Memory & disk optimization:
# Install CPU-only PyTorch wheel (~150MB vs ~900MB GPU/CUDA wheel) before requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code, pre-built React frontend, and pre-indexed FAISS vectors
COPY . .

# Expose container port (Render overrides PORT environment variable dynamically)
EXPOSE 7860

# Run as non-root user for security hardening
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Start application via python app.py (binds to 0.0.0.0 and dynamically respects $PORT)
CMD ["python", "app.py"]
