# Base image completa com Python 3.11
FROM python:3.11

# Diretório de trabalho
WORKDIR /app

# Copiar arquivo de dependências
COPY requirements.txt .

# Atualizar pip
RUN pip install --upgrade pip

# Instalar dependências do sistema necessárias para pandas/numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential gfortran libatlas-base-dev liblapack-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar código da aplicação
COPY . .

# Comando para iniciar sua aplicação
CMD ["python", "main.py"]
