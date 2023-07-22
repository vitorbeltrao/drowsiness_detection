FROM python:3.9.13

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos para o contêiner
COPY . /app

# Instalar os pacotes necessários
RUN pip install -r requirements.txt

# Rodar o script em questão
ENTRYPOINT python3 realtime_inferences.py