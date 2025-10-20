# Use a imagem oficial do Python 3.11
FROM python:3.11

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo de dependências
COPY requirements.txt .

# Atualize pip e instale dependências de sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        gfortran \
        liblapack-dev \
        libatlas-base-dev \
        build-essential \
        python3-dev \
        libffi-dev \
        libssl-dev \
        curl \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc gfortran build-essential python3-dev libffi-dev libssl-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie o restante do código da aplicação
COPY . .

# Comando padrão para rodar a aplicação
CMD ["python", "main.py"]
