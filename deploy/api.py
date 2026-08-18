"""
api.py — ETAPA 7 (parte B) ESTENDIDA — Desafio final, Caminho C
----------------------------------------------------------------

Extensões implementadas sobre a API original da disciplina:

  (i)   POST /prever_lote  — classifica várias imagens em uma única requisição,
        empilhando os tensores em um único lote (um só forward, mais eficiente
        do que N chamadas a /prever).
  (ii)  Registro em arquivo (JSON Lines) de toda predição com confiança < 0,5
        — candidatas naturais a rotulagem manual e retreino (ver §7-D do README).
  (iii) GET /metricas — contagem de requisições, predições, latência média e
        taxa de baixa confiança, mantidas em memória desde o startup.
  (iv)  Compatível com o Dockerfile do projeto (mesmo caminho deploy/api.py;
        o diretório de logs é criado automaticamente e é configurável por
        variável de ambiente).

Decisões de engenharia preservadas da API original:
  1. Nada de src/ é importado — o serviço depende só dos artefatos
     modelo_scriptado.pt e classes.json.
  2. O modelo é carregado UMA vez, no lifespan.
  3. O pré-processamento é idêntico, byte a byte, ao do treino.

Como rodar (a partir da pasta cnn/):
    pip install fastapi "uvicorn[standard]" python-multipart pillow
    python -m uvicorn deploy.api:app --reload --port 8000
"""

import io
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps

# --- constantes que PRECISAM bater com o treino ----------------------------
TAMANHO = 28
MEDIA = 0.2860
DESVIO = 0.3530

# --- limiares e limites do serviço -----------------------------------------
LIMIAR_BAIXA_CONFIANCA = 0.5   # abaixo disso, a predição é registrada em log
MAX_IMAGENS_LOTE = 32          # protege o serviço contra lotes abusivos

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_MODELO = RAIZ / "outputs" / "modelo_scriptado.pt"
CAMINHO_CLASSES = RAIZ / "outputs" / "classes.json"

# Diretório de logs configurável por variável de ambiente (útil no Docker,
# para montar um volume:  docker run -v $(pwd)/logs:/app/logs ...)
DIR_LOGS = Path(os.environ.get("DIR_LOGS", RAIZ / "logs"))
CAMINHO_LOG_BAIXA_CONF = DIR_LOGS / "baixa_confianca.jsonl"

modelo = None
classes: list[str] = []


# ---------------------------------------------------------------------------
# (iii) MÉTRICAS — estado em memória, protegido por lock.
#
# Por que um Lock, se o serviço roda com 1 worker? Porque o FastAPI atende
# requisições async concorrentemente e os endpoints de lote fazem await entre
# leituras/escritas. O lock custa quase nada e elimina a corrida por completo.
# Em produção real, isso viraria Prometheus/StatsD — aqui o objetivo é
# demonstrar o conceito com o mínimo de dependências.
# ---------------------------------------------------------------------------
class Metricas:
    def __init__(self) -> None:
        self._lock = Lock()
        self.iniciado_em = datetime.now(timezone.utc)
        self.requisicoes_por_endpoint: dict[str, int] = {}
        self.total_predicoes = 0
        self.total_baixa_confianca = 0
        self.soma_latencia_ms = 0.0   # soma acumulada -> média = soma / n
        self.n_latencias = 0

    def registrar(self, endpoint: str, n_predicoes: int,
                  n_baixa_conf: int, latencia_ms: float) -> None:
        with self._lock:
            self.requisicoes_por_endpoint[endpoint] = \
                self.requisicoes_por_endpoint.get(endpoint, 0) + 1
            self.total_predicoes += n_predicoes
            self.total_baixa_confianca += n_baixa_conf
            self.soma_latencia_ms += latencia_ms
            self.n_latencias += 1

    def instantaneo(self) -> dict:
        with self._lock:
            media = (self.soma_latencia_ms / self.n_latencias
                     if self.n_latencias else None)
            return {
                "servico_iniciado_em": self.iniciado_em.isoformat(),
                "requisicoes_por_endpoint": dict(self.requisicoes_por_endpoint),
                "total_requisicoes_inferencia": self.n_latencias,
                "total_predicoes": self.total_predicoes,
                "predicoes_baixa_confianca": self.total_baixa_confianca,
                "taxa_baixa_confianca": (
                    round(self.total_baixa_confianca / self.total_predicoes, 4)
                    if self.total_predicoes else None),
                "latencia_media_ms": round(media, 2) if media is not None else None,
                "limiar_baixa_confianca": LIMIAR_BAIXA_CONFIANCA,
            }


metricas = Metricas()
_lock_log = Lock()  # serializa a escrita no arquivo de log


# ---------------------------------------------------------------------------
# (ii) LOG DE BAIXA CONFIANÇA — JSON Lines (1 objeto por linha, fácil de
# processar depois com pandas/jq). Predição com confiança < 0,5 provavelmente
# está fora do domínio de treino: é o insumo do ciclo de retreino do §7-D.
# ---------------------------------------------------------------------------
def registrar_baixa_confianca(registro: dict) -> None:
    linha = json.dumps(registro, ensure_ascii=False)
    with _lock_log:
        with open(CAMINHO_LOG_BAIXA_CONF, "a", encoding="utf-8") as f:
            f.write(linha + "\n")


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carrega o modelo uma única vez, na subida do servidor."""
    global modelo, classes
    if not CAMINHO_MODELO.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {CAMINHO_MODELO}.\n"
            "Rode antes:  python src/train.py  e  python src/export_model.py")
    modelo = torch.jit.load(str(CAMINHO_MODELO), map_location="cpu")
    modelo.eval()
    classes = json.loads(CAMINHO_CLASSES.read_text(encoding="utf-8"))
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[startup] modelo carregado | {len(classes)} classes | "
          f"log de baixa confiança em {CAMINHO_LOG_BAIXA_CONF}")
    yield
    print("[shutdown] encerrando serviço")


app = FastAPI(
    title="Classificador de Roupas (CNN) — Caminho C",
    description=("Serviço de inferência da CNN treinada no FashionMNIST, "
                 "estendido com lote, log de baixa confiança e métricas."),
    version="2.0.0",
    lifespan=ciclo_de_vida,
)


def preprocessar(bytes_imagem: bytes, inverter: bool = False) -> torch.Tensor:
    """bytes -> tensor (1,1,28,28). Mesma receita do treino, sem torchvision."""
    imagem = Image.open(io.BytesIO(bytes_imagem)).convert("L")
    if inverter:
        imagem = ImageOps.invert(imagem)
    imagem = imagem.resize((TAMANHO, TAMANHO))

    bruto = torch.frombuffer(bytearray(imagem.tobytes()), dtype=torch.uint8)
    tensor = bruto.float().reshape(1, 1, TAMANHO, TAMANHO) / 255.0  # = ToTensor()
    return (tensor - MEDIA) / DESVIO                                # = Normalize()


def montar_resposta(nome_arquivo: str, probabilidades: torch.Tensor) -> dict:
    """Formata a saída de uma predição e devolve também a confiança bruta."""
    valores, indices = probabilidades.topk(3)
    return {
        "arquivo": nome_arquivo,
        "predicao": classes[indices[0]],
        "confianca": round(valores[0].item(), 4),
        "top3": [{"classe": classes[i], "probabilidade": round(v.item(), 4)}
                 for v, i in zip(valores, indices)],
    }


def tratar_baixa_confianca(resultado: dict, endpoint: str) -> bool:
    """Se a predição ficou abaixo do limiar, registra no log. Devolve True/False."""
    if resultado["confianca"] >= LIMIAR_BAIXA_CONFIANCA:
        return False
    registrar_baixa_confianca({
        "quando": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "arquivo": resultado["arquivo"],
        "predicao": resultado["predicao"],
        "confianca": resultado["confianca"],
        "top3": resultado["top3"],
    })
    return True


@app.get("/saude")
def saude():
    """Health check — consultado pelo HEALTHCHECK do Docker."""
    return {"status": "ok", "modelo_carregado": modelo is not None,
            "n_classes": len(classes)}


# ---------------------------------------------------------------------------
# (iii) GET /metricas
# ---------------------------------------------------------------------------
@app.get("/metricas")
def obter_metricas():
    """Contadores do serviço desde o startup (em memória)."""
    return metricas.instantaneo()


@app.post("/prever")
async def prever(arquivo: UploadFile = File(...), inverter: bool = False):
    """Recebe UMA imagem e devolve as 3 classes mais prováveis."""
    if not arquivo.content_type or not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem.")

    conteudo = await arquivo.read()
    try:
        tensor = preprocessar(conteudo, inverter)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {e}")

    t0 = time.perf_counter()
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(tensor), dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000

    resultado = montar_resposta(arquivo.filename, probabilidades)
    resultado["tempo_inferencia_ms"] = round(ms, 2)

    baixa = tratar_baixa_confianca(resultado, "/prever")
    resultado["baixa_confianca"] = baixa
    metricas.registrar("/prever", n_predicoes=1,
                       n_baixa_conf=int(baixa), latencia_ms=ms)
    return resultado


# ---------------------------------------------------------------------------
# (i) POST /prever_lote
#
# Ponto pedagógico: em vez de N forwards, os N tensores são EMPILHADOS em um
# único lote (N,1,28,28) e passam pela rede de uma vez — exatamente como no
# treino. A latência por imagem cai porque o custo fixo (dispatch do
# interpretador, overhead do TorchScript) é pago uma vez só.
# ---------------------------------------------------------------------------
@app.post("/prever_lote")
async def prever_lote(arquivos: list[UploadFile] = File(...),
                      inverter: bool = False):
    """Recebe VÁRIAS imagens e devolve a predição de cada uma."""
    if not arquivos:
        raise HTTPException(status_code=400, detail="Envie ao menos uma imagem.")
    if len(arquivos) > MAX_IMAGENS_LOTE:
        raise HTTPException(
            status_code=413,
            detail=f"Lote muito grande: máximo de {MAX_IMAGENS_LOTE} imagens.")

    tensores, nomes, erros = [], [], []
    for arq in arquivos:
        conteudo = await arq.read()
        try:
            if not arq.content_type or not arq.content_type.startswith("image/"):
                raise ValueError("não é um arquivo de imagem")
            tensores.append(preprocessar(conteudo, inverter))
            nomes.append(arq.filename)
        except Exception as e:
            erros.append({"arquivo": arq.filename, "erro": str(e)})

    resultados = []
    ms = 0.0
    n_baixa = 0
    if tensores:
        lote = torch.cat(tensores, dim=0)          # (N, 1, 28, 28)
        t0 = time.perf_counter()
        with torch.no_grad():
            probabilidades = torch.softmax(modelo(lote), dim=1)
        ms = (time.perf_counter() - t0) * 1000

        for nome, probs in zip(nomes, probabilidades):
            resultado = montar_resposta(nome, probs)
            baixa = tratar_baixa_confianca(resultado, "/prever_lote")
            resultado["baixa_confianca"] = baixa
            n_baixa += int(baixa)
            resultados.append(resultado)

    metricas.registrar("/prever_lote", n_predicoes=len(resultados),
                       n_baixa_conf=n_baixa, latencia_ms=ms)
    return {
        "n_recebidas": len(arquivos),
        "n_classificadas": len(resultados),
        "n_baixa_confianca": n_baixa,
        "tempo_inferencia_ms": round(ms, 2),
        "tempo_medio_por_imagem_ms": (
            round(ms / len(resultados), 2) if resultados else None),
        "resultados": resultados,
        "erros": erros,
    }


@app.get("/", response_class=HTMLResponse)
def pagina_teste():
    """Página mínima para demonstração em sala, agora com envio em lote."""
    return """
<!doctype html><html lang="pt-br"><meta charset="utf-8">
<title>Classificador de Roupas — Caminho C</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem}
 #saida{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px;margin-top:1rem}
 button{padding:.6rem 1.2rem;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer;margin-right:.5rem}
</style>
<h1>Classificador de Roupas — CNN (Caminho C)</h1>
<p>Selecione uma ou mais imagens (peça de roupa clara sobre fundo escuro).</p>
<input type="file" id="arq" accept="image/*" multiple>
<label><input type="checkbox" id="inv"> inverter cores</label>
<p>
<button onclick="enviar()">Classificar</button>
<button onclick="verMetricas()">Ver métricas</button>
</p>
<div id="saida">aguardando…</div>
<script>
async function enviar(){
  const arquivos = document.getElementById('arq').files;
  const saida = document.getElementById('saida');
  if(!arquivos.length){ saida.textContent = 'Selecione ao menos um arquivo.'; return; }
  const inv = document.getElementById('inv').checked;
  const fd = new FormData();
  let url;
  if(arquivos.length === 1){
    fd.append('arquivo', arquivos[0]);
    url = '/prever?inverter=' + inv;
  }else{
    for(const f of arquivos) fd.append('arquivos', f);
    url = '/prever_lote?inverter=' + inv;
  }
  saida.textContent = 'processando…';
  const r = await fetch(url, {method:'POST', body: fd});
  saida.textContent = JSON.stringify(await r.json(), null, 2);
}
async function verMetricas(){
  const r = await fetch('/metricas');
  document.getElementById('saida').textContent =
    JSON.stringify(await r.json(), null, 2);
}
</script></html>
"""
