FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn[standard] fastapi httpx

# Copy application code
COPY flare/ flare/
COPY dashboard/ dashboard/
COPY logs/ logs/

# Install the package itself
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -e ".[llm,api]"

ENV FLARE_HOST=0.0.0.0
ENV FLARE_PORT=8000
ENV FLARE_LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "flare.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
