# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.5.1-cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ZipVoice

COPY requirements.txt /workspace/ZipVoice/requirements.txt
COPY runtime/fastapi_server/requirements.fastapi.txt /workspace/ZipVoice/runtime/fastapi_server/requirements.fastapi.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/ZipVoice/requirements.txt && \
    pip install --no-cache-dir -r /workspace/ZipVoice/runtime/fastapi_server/requirements.fastapi.txt

COPY . /workspace/ZipVoice

# Build args for packaging model files into the image.
# Example model dir in repo: models/zipvoice_distill
ARG MODEL_DIR=models/zipvoice_distill
ARG MODEL_NAME=zipvoice_distill

COPY ${MODEL_DIR} /opt/zipvoice_model

ENV PYTHONPATH=/workspace/ZipVoice
ENV ZIPVOICE_MODEL_NAME=${MODEL_NAME}
ENV ZIPVOICE_MODEL_DIR=/opt/zipvoice_model
ENV ZIPVOICE_DEVICE=cpu
ENV ZIPVOICE_MAX_CONCURRENCY=1
ENV ZIPVOICE_NUM_THREADS=1
ENV ZIPVOICE_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "runtime.fastapi_server.app:APP", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
