# Dockerfile
FROM python:3.11-slim

# Installa Tesseract
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY estrai_grafico.py .

# Non serve EXPOSE: Railway espone automaticamente $PORT
# Se vuoi tenerlo, metti EXPOSE 8080

# Avvio su $PORT e bind a 0.0.0.0
CMD ["bash","-lc","streamlit run estrai_grafico.py \
      --server.port ${PORT:-8080} \
      --server.address 0.0.0.0 \
      --server.headless true"]
