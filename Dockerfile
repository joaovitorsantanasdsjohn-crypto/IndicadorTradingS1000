# Usar imagem oficial do Python 3.10
FROM python:3.10-slim

# Define diretório de trabalho dentro do container
WORKDIR /app

# Copia o requirements.txt para o container
COPY requirements.txt .

# Instala dependências do Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do seu projeto para o container
COPY . .

# Define variável de ambiente para não gerar buffers nos prints
ENV PYTHONUNBUFFERED=1

# Expõe a porta do Flask (8080)
EXPOSE 8080

# Comando para iniciar o bot
CMD ["python", "main.py"]
