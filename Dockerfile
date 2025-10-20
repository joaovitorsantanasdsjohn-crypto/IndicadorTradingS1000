# Usa Python 3.11 slim como base
FROM python:3.11.6-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos necessários
COPY requirements.txt .
COPY main.py .
COPY ml_model.py .

# Atualiza pip e instala dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expõe porta 5000 para Flask (Uptime Robot)
EXPOSE 5000

# Comando para rodar sua aplicação
CMD ["python", "main.py"]
