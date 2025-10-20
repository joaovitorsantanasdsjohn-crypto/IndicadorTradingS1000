# Use a imagem oficial do Python 3.11
FROM python:3.11

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo de dependências
COPY requirements.txt .

# Atualize pip e instale dependências do sistema necessárias
RUN pip install --upgrade pip && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        liblapack-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential gfortran liblapack-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie o restante do código da aplicação
COPY . .

# Comando padrão para rodar a aplicação
CMD ["python", "main.py"]
