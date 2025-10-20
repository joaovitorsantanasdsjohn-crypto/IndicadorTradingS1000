# Usa Python 3.10 slim, compatível com ta==0.12.2
FROM python:3.10-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Atualiza pip e instala dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos da aplicação para o container
COPY . .

# Comando padrão para rodar a aplicação
CMD ["python", "main.py"]
