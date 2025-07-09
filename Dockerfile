# 1. Usa un’immagine base Python + Debian
FROM python:3.11-slim

# 2. Installa Tesseract di sistema
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# 3. Imposta la working directory
WORKDIR /app

# 4. Copia requirements e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia il tuo script e, se lo usi, eventuali cartelle col codice
COPY estrai_grafico.py .

# 6. Espone la porta su cui gira Streamlit
EXPOSE 8501

# 7. Comando di avvio
CMD ["streamlit", "run", "estrai_grafico.py", "--server.port", "8501", "--server.headless", "true"]
