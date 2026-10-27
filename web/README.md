# Interface web

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
