FROM python:3.11-slim

WORKDIR /app

# 系统依赖：ffmpeg 给讯飞转写用，curl 给健康检查用
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制代码
COPY . .

# 预下载模型（在构建期下载，部署后免冷启动）
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-zh-v1.5')" || echo "embedding download failed"
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('BAAI/bge-reranker-base')" || echo "reranker download failed"

EXPOSE 80

# 启动：使用 waitress 启动（Windows/Linux 通用，云托管也支持）
# threads=4 表示每个 worker 4 线程处理请求
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=80", "--threads=4", "--ident=anti-fraud-ai", "rag_api:app"]
