# FORÇA Python 3.10 exato
FROM python:3.10.12-slim

# Diretório de trabalho
WORKDIR /app

# Copia arquivos do projeto
COPY requirements.txt .
COPY main.py .
COPY ml_model.py .

# Atualiza pip e instala dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expõe porta 5000 para Flask (Uptime Robot)
EXPOSE 5000

# Comando para rodar o bot
CMD ["python", "main.py"]
