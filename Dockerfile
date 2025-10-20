# Use a imagem oficial do Python 3.11
FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        gfortran \
        liblapack-dev \
        build-essential \
        python3-dev \
        libffi-dev \
        libssl-dev \
        curl \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc gfortran build-essential python3-dev libffi-dev libssl-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "main.py"]
