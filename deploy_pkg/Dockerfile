FROM python:3.11-slim-bookworm

WORKDIR /app

# 1. 先装 CPU 版 torch（节省 2GB）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. 装其他依赖
RUN pip install --no-cache-dir \
    -i https://pypi.org/simple \
    -r requirements.txt

# Copy source (从仓库根目录复制)
COPY ../rag_api.py .
COPY ../chroma_db/ ./chroma_db/
COPY ../.env .

ENV PYTHONUNBUFFERED=1

EXPOSE 80

CMD ["python", "rag_api.py"]
