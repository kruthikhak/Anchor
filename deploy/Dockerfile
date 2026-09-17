# Image for a Hugging Face Space (Docker SDK). Build context is the repository root:
#   docker build -f deploy/Dockerfile -t anchor .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN useradd -m user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# bake the embedder and reranker into the image so a cold start doesn't wait on a 1.5 GB download
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY --chown=user rag rag
COPY --chown=user app app
COPY --chown=user data/corpus.json data/corpus.json
COPY --chown=user data/index data/index
COPY --chown=user eval/results eval/results

EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
