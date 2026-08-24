# CNN do dado ao deploy — Caminho C (Serviço completo)

> Desafio final da disciplina **Artificial Intelligence e Deep Learning Aplicada** (FIAP).
> Baseado no material do Prof. Dr. Alexandre Miguel de Carvalho
> ([carvalhoamc/fiap](https://github.com/carvalhoamc/fiap), pasta `2tsCPV-2/cnn`).

**Grupo:** Bruno Tomin (RM565037) e Cynthia Takematu (RM564100)

## O que este projeto faz

Classifica imagens de peças de roupa (Fashion-MNIST, 10 classes) com uma CNN
em PyTorch e serve o modelo como uma API REST em contêiner Docker. Sobre a
API original da disciplina, implementamos as quatro extensões do **Caminho C**:

1. **`POST /prever_lote`** — classifica várias imagens em uma única requisição
   (um só *forward* em lote, limite de 32 imagens);
2. **Log de baixa confiança** — toda predição com confiança < 0,5 é registrada
   em `logs/baixa_confianca.jsonl` (insumo para rotulagem e retreino);
3. **`GET /metricas`** — requisições por endpoint, total de predições, taxa de
   baixa confiança e latência média desde o startup;
4. **Docker** — imagem de produção mínima (282 MB comprimida) com HEALTHCHECK.

## Resultados

| Métrica | Valor |
|---|---|
| Acurácia de validação (12 épocas, CPU) | 91,00% |
| Acurácia de teste | 90,13% |
| Classe mais difícil | Camisa (F1 = 0,68; revocação 60,6%) |
| Modelo exportado (TorchScript) | 390 KB (dif. máx. 1,43e-06) |
| Teste de integração da API | 20/20 (100%), ~0,8 ms/imagem |
| Lote de 8 imagens | 8,19 ms (1,02 ms/imagem) |
| Imagem Docker | 282 MB comprimida (1,34 GB em disco) |

Achado interessante: ruído aleatório puro é classificado como "Bolsa" com
69–81% de confiança, e uma imagem cinza uniforme com 94,6% — evidência da
superconfiança do softmax em entradas fora do domínio. A discussão completa
está no relatório (`relatorio/`).

## Estrutura do repositório

```
.
├── README.md
├── CP1_AI___Deep_Learning_Caminho_C.ipynb   <- notebook executado (Colab)
├── deploy/
│   ├── api.py                       <- API estendida (Caminho C)
│   ├── testar_api.py                <- teste de integração (da disciplina)
│   └── Dockerfile                   <- imagem de produção
├── outputs/
│   ├── modelo_scriptado.pt          <- modelo TorchScript (390 KB)
│   ├── classes.json
│   └── *.png                        <- curvas, matriz de confusão, erros
├── logs/
│   └── baixa_confianca.jsonl        <- exemplo de log gerado
├── amostras/                        <- PNGs do Fashion-MNIST para teste
├── evidencias/                      <- prints (docker ps healthy, página web)
└── relatorio/                       <- relatório final (3 a 5 páginas)
```

> **Nota:** o dataset (`data/`) e o checkpoint de treino (`melhor_modelo.pt`)
> não são versionados: o dataset baixa automaticamente e o checkpoint é
> reproduzível pela semente fixa (42). Apenas os artefatos de produção
> (`modelo_scriptado.pt`, `classes.json`) estão no repositório, pois são
> pequenos e necessários para o Docker.

## Como reproduzir

### Opção A — Pipeline completo no Google Colab (~25 min)

1. Abra o notebook `CP1_AI___Deep_Learning_Caminho_C.ipynb` no
   [Colab](https://colab.research.google.com) e execute as células em ordem.
2. O notebook clona o repositório da disciplina, treina (12 épocas, ~16 min em
   CPU), avalia, exporta o TorchScript, sobe a API estendida em background e
   testa os quatro endpoints (`/saude`, `/prever`, `/prever_lote`,
   `/metricas`), incluindo o cenário de baixa confiança.
3. A última seção gera o `artefatos_caminho_c.zip` para download, que alimenta
   a reprodução local com Docker.

A semente fixa torna o treino reprodutível; os números podem variar
ligeiramente por versão de biblioteca ou hardware.

### Opção B — Somente o serviço, com Docker (local, ~10 min)

Pré-requisito: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(no Windows, com WSL 2).

```powershell
# 1. Obter este repositório (git clone ou Code -> Download ZIP)
git clone https://github.com/cynthiatakematu/CP1-AI-DEEP-LEARNING.git
cd CP1-AI-DEEP-LEARNING

# 2. Construir a imagem (o torch CPU baixa ~300 MB na primeira vez)
docker build -t cnn-roupas-caminho-c -f deploy/Dockerfile .

# 3. Subir o contêiner (o volume persiste o log de baixa confiança)
docker run -p 8000:8000 -v "${PWD}/logs:/app/logs" -e DIR_LOGS=/app/logs cnn-roupas-caminho-c
```

Espere a mensagem `[startup] modelo carregado | 10 classes`. Depois:

- **Navegador:** http://localhost:8000 (página de teste com upload múltiplo e
  botão de métricas) e http://localhost:8000/docs (Swagger).
- **Linha de comando** (em outro terminal, na pasta do repositório):

```powershell
curl.exe http://localhost:8000/saude
curl.exe http://localhost:8000/metricas
curl.exe -X POST http://localhost:8000/prever_lote `
  -F "arquivos=@amostras/amostra_000_Ankle boot.png" `
  -F "arquivos=@amostras/amostra_001_Pullover.png"

docker ps        # STATUS deve mostrar "Up ... (healthy)" apos ~30 s
type logs\baixa_confianca.jsonl
```

> Em Linux/macOS, troque `curl.exe` por `curl`, `type` por `cat` e
> `"${PWD}"` por `"$(pwd)"`.

### Dicas de teste

- As imagens de `amostras/` já estão no domínio do modelo — use-as **sem** a
  opção "inverter cores".
- Fotos reais de roupa (fundo claro) exigem a opção **"inverter cores"** e,
  mesmo assim, podem errar: o modelo só conhece 28×28 em escala de cinza com
  peça clara sobre fundo escuro. Esse deslocamento de domínio é discutido na
  seção 7 do relatório.

## Créditos

- Material didático, pipeline de treino e API original: Prof. Dr. Alexandre
  Miguel de Carvalho ([carvalhoamc/fiap](https://github.com/carvalhoamc/fiap)).
- Extensões do Caminho C (endpoints de lote e métricas, log de baixa
  confiança, ajustes de deploy): autoria do grupo.
- Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  (Xiao, Rasul & Vollgraf, 2017).
