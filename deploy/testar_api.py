"""
testar_api.py — cliente de teste do serviço
--------------------------------------------

Gera imagens PNG a partir do próprio conjunto de teste do FashionMNIST, envia
para a API e compara a resposta com o rótulo verdadeiro. É o teste de
integração mínimo: garante que o pré-processamento do SERVIDOR produz o mesmo
resultado do pré-processamento do TREINO. Se a acurácia aqui for muito menor
que a do evaluate.py, o bug está no deploy, não no modelo.

Uso (com o servidor já rodando em outro terminal):
    python deploy/testar_api.py
    python deploy/testar_api.py --n 20 --url http://127.0.0.1:8000
"""

import argparse
import json
import sys
import urllib.request
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from torchvision import datasets            # noqa: E402
from config import CLASSES, DATA_DIR, OUT_DIR   # noqa: E402


def enviar_imagem(url: str, caminho: Path) -> dict:
    """POST multipart/form-data usando apenas a biblioteca padrão do Python."""
    limite = uuid.uuid4().hex
    corpo = b"".join([
        f"--{limite}\r\n".encode(),
        f'Content-Disposition: form-data; name="arquivo"; '
        f'filename="{caminho.name}"\r\n'.encode(),
        b"Content-Type: image/png\r\n\r\n",
        caminho.read_bytes(),
        f"\r\n--{limite}--\r\n".encode(),
    ])
    requisicao = urllib.request.Request(
        f"{url}/prever", data=corpo, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    with urllib.request.urlopen(requisicao, timeout=30) as resposta:
        return json.loads(resposta.read())


def main():
    p = argparse.ArgumentParser(description="Teste de integração da API")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()

    # 1) Health check antes de qualquer coisa
    try:
        with urllib.request.urlopen(f"{args.url}/saude", timeout=10) as r:
            print("saúde:", json.loads(r.read()), "\n")
    except Exception as e:
        print(f"[erro] servidor não respondeu em {args.url}: {e}")
        print("Suba o servidor com:  python -m uvicorn deploy.api:app --port 8000")
        return

    # 2) Exporta n imagens de teste como PNG e envia uma a uma
    pasta = OUT_DIR / "amostras_api"
    pasta.mkdir(parents=True, exist_ok=True)
    conjunto_teste = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)

    acertos, tempos = 0, []
    for i in range(args.n):
        imagem, rotulo = conjunto_teste[i]
        caminho = pasta / f"amostra_{i:03d}.png"
        imagem.save(caminho)

        resposta = enviar_imagem(args.url, caminho)
        previsto = resposta["predicao"]
        verdadeiro = CLASSES[rotulo]
        acertou = previsto == verdadeiro
        acertos += int(acertou)
        tempos.append(resposta["tempo_inferencia_ms"])

        print(f"[{'OK  ' if acertou else 'ERRO'}] {caminho.name} | "
              f"previsto: {previsto:<16} ({resposta['confianca']:.1%}) | "
              f"verdadeiro: {verdadeiro:<16} | {resposta['tempo_inferencia_ms']:.1f} ms")

    print(f"\nAcertos: {acertos}/{args.n} ({acertos / args.n:.0%})")
    print(f"Latência média: {sum(tempos) / len(tempos):.1f} ms por imagem")
    print(f"Imagens usadas em: {pasta}")


if __name__ == "__main__":
    main()
