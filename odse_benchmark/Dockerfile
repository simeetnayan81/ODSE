FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    CC=gcc \
    CXX=g++

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/start_training.sh

CMD ["/app/start_training.sh"]
