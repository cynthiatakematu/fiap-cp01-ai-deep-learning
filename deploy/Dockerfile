# Dockerfile — empacota o SERVIÇO (não o treino) em um contêiner.
#
# Construir e rodar, a partir da pasta cnn/:
#   docker build -t cnn-roupas -f deploy/Dockerfile .
#   docker run -p 8000:8000 cnn-roupas
#
# Observe o que entra na imagem: apenas o código do serviço e os artefatos
# (modelo + classes). Dataset, notebooks e scripts de treino ficam de fora —
# imagem de produção deve ser mínima.

FROM python:3.12-slim

WORKDIR /app

# torch em versão CPU: a imagem cai de ~2.5 GB para ~300 MB
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir \
    fastapi==0.115.* "uvicorn[standard]==0.32.*" python-multipart pillow

COPY deploy/api.py            /app/deploy/api.py
COPY outputs/modelo_scriptado.pt /app/outputs/modelo_scriptado.pt
COPY outputs/classes.json        /app/outputs/classes.json

EXPOSE 8000

# Health check: o Docker reinicia o contêiner se o endpoint parar de responder
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude')"

CMD ["python", "-m", "uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
