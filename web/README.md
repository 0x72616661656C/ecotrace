# Interface web

## Deploy estático na Vercel (processamento no cliente)

A versão de produção não usa a API FastAPI. O navegador baixa somente cada
GeoTIFF selecionado, lê B2/B3/B4/B8, normaliza os valores e roda a U-Net ONNX
localmente via WebAssembly. Portanto, a Vercel hospeda somente arquivos
estáticos; os GeoTIFFs e pesos grandes ficam em um bucket/CDN.

1. Exporte os pesos (uma única vez, em uma máquina com PyTorch e `onnx`):

   ```bash
   pip install onnx
   python models/export_onnx.py --checkpoint-dir web/backend/models/checkpoints
   ```

   O comando cria `web/frontend/public/models/unet.onnx` e
   `web/frontend/public/models/attn_unet.onnx`. Para evitar o limite de tamanho
   do deploy, envie-os ao mesmo CDN dos dados e use as URLs no próximo passo.

2. Gere os artefatos leves do mapa:

   ```bash
   python web/scripts/prepare_static_dataset.py \
     --input web/frontend/extract-gee \
     --output /tmp/ecotrace-static
   ```

   Envie os GeoTIFFs para `extract-gee/` no bucket e envie o conteúdo de
   `/tmp/ecotrace-static` para a raiz do mesmo prefixo. O bucket precisa expor
   CORS para `GET` e `HEAD`, permitir o cabeçalho `Range` e retornar
   `Accept-Ranges: bytes`; isso permite que `geotiff.js` leia COGs sem baixar o
   arquivo completo quando houver overviews.

3. Crie `web/frontend/.env.production` a partir de `.env.example`, trocando as
   URLs de exemplo. `VITE_DATASET_BASE_URL` deve apontar para o prefixo que
   contém `tile_index.json`, `previews/`, `sentinel_overview.png` e
   `extract-gee/`.

4. Na raiz do repositório, importe o projeto na Vercel. O `vercel.json` já usa
   `web/frontend` como build e publica `web/frontend/dist`. Cadastre as mesmas
   variáveis `VITE_*` no painel da Vercel antes do deploy.

   ```bash
   vercel --prod
   ```

Monte Carlo Dropout não é exposto nesta primeira versão estática: os modelos
ONNX são exportados em modo de avaliação. Para preservar memória, a versão
publicada usa somente a Attention U-Net e processa um tile de 256×256 por vez.
A segmentação e as máscaras são geradas integralmente no dispositivo do usuário.

O EcoTrace opera exclusivamente com GeoTIFFs Sentinel-2 locais, extraídos pelo
pipeline Extract-GEE. A inferência usa quatro bandas reais na ordem `B2, B3, B4,
B8`; não há integração com Google Satellite nem NIR sintético.

## Pré-requisitos

- Python 3.10 ou superior;
- Node.js 20 ou superior;
- pesos `unet_seed_2_best.pt` e/ou `attn_unet_seed_2_best.pt` em
  `web/backend/models/checkpoints/`;
- GeoTIFFs de quatro bandas em `web/frontend/extract-gee/`.

## Configuração

Os caminhos são fixos e relativos ao repositório. O índice
`web/frontend/tile_index.json` é criado automaticamente a partir dos arquivos `.tif`
quando necessário. Os GeoTIFFs precisam usar o CRS esperado pelo leitor
(`EPSG:32620`) e conter B2, B3, B4 e B8 nessa ordem. Dados, pesos, índices, caches e
builds não são versionados.

## Desenvolvimento

Em um terminal, inicie a API:

```bash
cd web/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Em outro terminal, inicie a interface:

```bash
cd web/frontend
npm ci
npm run dev
```

Abra `http://127.0.0.1:5173`. A interface usa o proxy Vite para acessar a API em
`http://127.0.0.1:8000`.

## Produção local

Para servir a interface pela própria FastAPI:

```bash
cd web/frontend
npm run build
cd ../backend
uvicorn app:app --host 127.0.0.1 --port 8000
```

Abra `http://127.0.0.1:8000`.
