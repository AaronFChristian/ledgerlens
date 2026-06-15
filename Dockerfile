FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    anthropic \
    neo4j \
    langgraph \
    langchain \
    langchain-core \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    pillow \
    loguru \
    python-multipart \
    python-dotenv

# Copy source
COPY src/ ./src/
COPY .env.example .env.example

# Create data directories
RUN mkdir -p data/extraction_results data/samples logs

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "ledgerlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
