FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/hf
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[models,ui]"
# Bake the embedding + reranker models into the image (no download at startup).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY app.py ./
COPY eval ./eval
RUN useradd --uid 1000 app && chown -R app /app
USER app
EXPOSE 8000 8501
CMD ["uvicorn", "secrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
