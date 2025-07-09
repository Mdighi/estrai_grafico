# 1. Base Python slim
FROM python:3.11-slim

# 2. Tesseract di sistema
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# 3. Working dir
WORKDIR /app

# 4. Dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia codice
COPY estrai_grafico.py .

# 6. Esponi porta definita da Railway
EXPOSE  $PORT

# 7. Usa la variabile $PORT e bind su tutte le interfacce
ENV PORT 8501
CMD sh -c "streamlit run estrai_grafico.py \
    --server.port \$PORT \
    --server.address 0.0.0.0 \
    --server.headless true"
