# Usar imagem oficial Python 3.10 (compatível com TensorFlow)
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos do projeto para o container
COPY . /app

# Atualizar pip e instalar dependências
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta que o Flask vai rodar
EXPOSE 5000

# Comando para iniciar o bot
CMD ["python", "main.py"]
