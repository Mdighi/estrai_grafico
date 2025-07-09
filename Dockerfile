FROM python:3.11-slim

# 1) Installa Tesseract-OCR, libGL per OpenCV e distutils per la build di pacchetti
RUN apt-get update && \
    apt-get install -y \
      tesseract-ocr \
      libgl1-mesa-glx \
      python3-distutils \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copia e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copia il tuo script principale
COPY estrai_grafico.py .

EXPOSE 8501

# 4) Avvio su $PORT e bind a 0.0.0.0
CMD ["bash", "-c", "streamlit run estrai_grafico.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true"]
