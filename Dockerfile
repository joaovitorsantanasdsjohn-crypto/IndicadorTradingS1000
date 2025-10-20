# Use imagem oficial do Python 3.10
FROM python:3.10-slim

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos do projeto
COPY requirements.txt .
COPY main.py .
COPY ml_model.py .
COPY .gitignore .

# Instala dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta 5000 para Flask
EXPOSE 5000

# Comando para rodar o bot
CMD ["python", "main.py"]
