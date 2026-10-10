# Dockerfile para a Plataforma ECG-AI (Diagnóstico Cardiológico Multimodal e Pesquisa Acadêmica)
FROM python:3.12-slim

# Evitar prompts interativos e escrita de bytecode
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalação de dependências de sistema essenciais (OpenCV headless, Tesseract OCR, compilação básica)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-por \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho na aplicação
WORKDIR /app

# Copia e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia todo o código-fonte, dados e documentação do projeto
COPY . .

# Expor a porta padrão do Streamlit
EXPOSE 8501

# Healthcheck do container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização do Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
