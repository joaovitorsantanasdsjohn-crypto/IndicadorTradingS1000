# Use Python 3.11 slim como base
FROM python:3.11.6-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o requirements.txt para o container
COPY requirements.txt .

# Atualiza pip e instala dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos da aplicação para o container
COPY main.py .
COPY ml_model.py .

# Expõe porta 5000 para Flask
EXPOSE 5000

# Define o comando de inicialização da aplicação
CMD ["python", "main.py"]
