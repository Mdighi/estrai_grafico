FROM python:3.11-slim

# 1. Installa dipendenze di sistema
RUN apt-get update && \
    apt-get install -y \
      tesseract-ocr \
      libgl1-mesa-glx \
      python3-distutils \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. Setta la working directory
WORKDIR /app

# 3. Copia e installa le dipendenze Python
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia tutti i file dell'app
COPY . .

# 5. Espone la porta Streamlit
EXPOSE 8501

# 6. Avvia Streamlit sulla porta indicata
CMD ["bash", "-c", "streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
