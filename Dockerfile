FROM python:3.10.11-slim

# Diretório da aplicação
WORKDIR /app

# Evitar criação de arquivos pyc e manter output imediato nos logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requisitos e instalar
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app /app/

# Expor porta do FastAPI
EXPOSE 8000

# Rodar FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

