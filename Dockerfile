# --------------------------
# Dockerfile para IndicadorTradingS1000
# --------------------------

# Imagem base
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Atualizar pip e instalar dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY . .

# Expor porta para Flask
EXPOSE 5000

# Comando padrão para rodar a aplicação
CMD ["python", "mine.py"]
