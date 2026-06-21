FROM python:3.11-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    sentence-transformers \
    numpy \
    chromadb \
    openai \
    python-dotenv \
    python-docx \
    requests \
    gunicorn \
    waitress \
    rank_bm25

COPY rag_api.py .
COPY chroma_db/ ./chroma_db/

ENV PYTHONUNBUFFERED=1
ENV PORT=80

EXPOSE 80

CMD ["python", "rag_api.py"]
