FROM python:3.10.11-slim

# Instalar dependências do sistema para compilação de libs científicas
# gcc e gfortran são necessários para NumPy/SciPy em imagens slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc g++ gfortran \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Diretório da aplicação
WORKDIR /app

# Evitar criação de arquivos pyc e manter output imediato nos logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copiar requisitos e instalar
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app /app/

# Expor porta do FastAPI
EXPOSE 8000

# Rodar FastAPI
# Assumimos que o arquivo principal é 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
