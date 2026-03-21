# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent AI System v2 — production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 : builder  — install all Python deps with layer-cache
# Stage 2 : runtime  — slim image, copy wheels from builder
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /app

# System build deps for sentence-transformers, chromadb native libs, torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY . .

# Directories required at runtime
RUN mkdir -p data/chroma_db logs

# ── Environment defaults (override via docker-compose / -e flags) ─────────────
ENV LLM_PROVIDER=ollama \
    LLM_MODEL=llama3.2:3b \
    OLLAMA_BASE_URL=http://ollama:11434 \
    EMBEDDING_DEVICE=cpu \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    FRONTEND_API_URL=http://api:8000 \
    LOG_LEVEL=INFO \
    MAX_AGENT_ITERATIONS=3 \
    PARALLEL_RESEARCH=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Health check for the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run both services (split via docker-compose for production)
CMD ["sh", "-c", \
    "uvicorn backend.api:app \
        --host 0.0.0.0 --port 8000 --workers 1 \
        --log-level info & \
     streamlit run frontend/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.fileWatcherType none; \
     wait"]
