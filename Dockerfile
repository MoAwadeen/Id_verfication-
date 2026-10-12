FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (the default Linux wheel pulls ~2GB of CUDA libs)
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Bake the PaddleOCR model into the image so cold starts don't download it
RUN python -c "from paddleocr import TextRecognition; TextRecognition(model_name='arabic_PP-OCRv5_mobile_rec', device='cpu')"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
