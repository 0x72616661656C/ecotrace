"""API local do EcoTrace para GeoTIFFs Sentinel-2 de quatro bandas reais.

Não há integração com Google Satellite nem criação de NIR sintético. A API opera
somente sobre os GeoTIFFs locais produzidos pelo pipeline Extract-GEE.
"""

import base64
import io
import logging
import os
import sys
from threading import Thread
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.architecture import load_model
from cog_reader import render_cog_xyz_tile, render_cog_xyz_tile_nir, read_cog_area
from local_tile_reader import (
    get_cached_tile_png,
    get_overview_png,
    load_tile_index,
    precache_all_tiles_parallel,
    read_and_process_selected,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SIZE = 512
OUTPUT_SIZE = 1536
CHECKPOINT_DIR = Path(__file__).resolve().parent / "models" / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_cache: dict[str, torch.nn.Module] = {}


def get_model(model_type: str) -> torch.nn.Module:
    if model_type not in model_cache:
        logger.info("Carregando %s em %s", model_type, DEVICE)
        model_cache[model_type] = load_model(model_type, CHECKPOINT_DIR, DEVICE)
    return model_cache[model_type]


def enable_dropout(module: torch.nn.Module) -> None:
    """Ativa apenas as camadas de dropout para inferência Monte Carlo."""
    if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
        module.train()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # A API pode subir sem pesos/dados para facilitar a configuração inicial.
    for model_type in ("attn_unet", "unet"):
        try:
            get_model(model_type)
        except Exception as error:
            logger.warning("Modelo %s não foi pré-carregado: %s", model_type, error)
    try:
        # A visão geral é pequena e evita abrir milhares de imagens no navegador.
        get_overview_png()
    except Exception as error:
        logger.warning("Prévia geral Sentinel-2 não foi criada: %s", error)
    Thread(target=precache_all_tiles_parallel, name="ecotrace-tile-cache", daemon=True).start()
    yield
    model_cache.clear()


app = FastAPI(
    title="EcoTrace API",
    description="Inferência local em GeoTIFF Sentinel-2 com B8/NIR real.",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BBox(BaseModel):
    west: float
    south: float
    east: float
    north: float


class DetectRequest(BaseModel):
    tile_ids: list[str] = Field(min_length=1, max_length=20)
    model_type: str = Field(default="unet", pattern="^(unet|attn_unet)$")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_mc_dropout: bool = False
    mc_samples: int = Field(default=5, ge=2, le=20)


class TileOverlay(BaseModel):
    tile_id: str
    mask_png_base64: str
    uncertainty_png_base64: Optional[str] = None
    bbox: BBox


class DetectResponse(BaseModel):
    overlays: list[TileOverlay]
    metadata: dict
    stats: dict


def png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preview_from_b4_b3_b2(image: np.ndarray) -> str:
    """Converte tensor B2,B3,B4,B8 normalizado em preview RGB natural."""
    rgb = np.moveaxis(image[[2, 1, 0]], 0, -1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return png_base64(Image.fromarray(rgb, mode="RGB"))


def heatmap_from_variance(variance: np.ndarray) -> str:
    """Mapa térmico: roxo (baixa variância) → vermelho → amarelo (alta)."""
    normalized = variance / (variance.max() + 1e-8)
    stops = np.array([
        [54, 20, 94],
        [117, 35, 128],
        [214, 48, 88],
        [248, 114, 44],
        [255, 232, 79],
    ], dtype=np.float32)
    positions = normalized * (len(stops) - 1)
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (positions - lower)[..., None]
    rgb = stops[lower] * (1 - fraction) + stops[upper] * fraction
    rgba = np.empty((*normalized.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = (95 + 135 * normalized).astype(np.uint8)
    return png_base64(Image.fromarray(rgba, mode="RGBA"))


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": sorted(model_cache),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "source": "Extract-GEE / GeoTIFF Sentinel-2 local",
        "nir": "B8 real",
    }


@app.get("/api/coverage")
async def get_coverage():
    try:
        return load_tile_index()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Não foi possível carregar os GeoTIFFs do Extract-GEE. "
                "Verifique web/frontend/extract-gee e, se necessário, remova "
                "web/frontend/tile_index.json para recriá-lo. "
                f"Detalhe: {error}"
            ),
        ) from error


@app.get("/api/tile_image/{filename}")
async def get_tile_image(filename: str):
    normalized = filename.replace(".png", ".tif") if filename.endswith(".png") else filename
    if not normalized.endswith(".tif"):
        normalized = f"{normalized}.tif"
    try:
        return Response(
            content=get_cached_tile_png(normalized),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as error:
        raise HTTPException(status_code=404, detail=f"Tile Sentinel-2 não encontrado: {error}") from error


@app.get("/api/overview.png")
async def get_overview():
    """Retorna o mosaico leve da cobertura completa para a visualização inicial."""
    try:
        return Response(
            content=get_overview_png(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Falha ao gerar mosaico Sentinel-2: {error}") from error


@app.get("/api/tiles/{z}/{x}/{y}.png")
async def get_rgb_tile(z: int, x: int, y: int):
    try:
        return Response(content=render_cog_xyz_tile(x, y, z), media_type="image/png")
    except Exception as error:
        logger.warning("Falha no tile RGB %s/%s/%s: %s", z, x, y, error)
        return Response(content=b"", media_type="image/png", status_code=204)


@app.get("/api/tiles/nir/{z}/{x}/{y}.png")
async def get_nir_tile(z: int, x: int, y: int):
    try:
        return Response(content=render_cog_xyz_tile_nir(x, y, z), media_type="image/png")
    except Exception as error:
        logger.warning("Falha no tile CIR %s/%s/%s: %s", z, x, y, error)
        return Response(content=b"", media_type="image/png", status_code=204)


@app.post("/api/detect", response_model=DetectResponse)
async def detect_non_forest(request: DetectRequest):
    try:
        model = get_model(request.model_type)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Falha ao carregar o modelo: {error}") from error

    overlays: list[TileOverlay] = []
    total_pixels = 0
    non_forest_pixels = 0
    variance_values: list[float] = []

    # Cada cena mantém sua própria normalização, inferência e georreferência.
    # Isso evita reduzir um mosaico de vários tiles a uma única imagem quadrada.
    for position, tile_id in enumerate(request.tile_ids, start=1):
        try:
            high_resolution, metadata = read_and_process_selected(
                tile_ids=[tile_id], target_size=OUTPUT_SIZE
            )
            model_input, _ = read_and_process_selected(
                tile_ids=[tile_id], target_size=MODEL_SIZE
            )
            tensor = torch.from_numpy(model_input).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if request.use_mc_dropout:
                    model.eval()
                    model.apply(enable_dropout)
                    samples = [torch.sigmoid(model(tensor)).squeeze().cpu().numpy() for _ in range(request.mc_samples)]
                    predictions = np.stack(samples, axis=0)
                    probabilities = predictions.mean(axis=0)
                    variance = predictions.var(axis=0)
                    scaled_variance = np.asarray(
                        Image.fromarray(variance).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BILINEAR)
                    )
                    uncertainty = heatmap_from_variance(scaled_variance)
                    variance_values.append(float(variance.mean()))
                else:
                    model.eval()
                    probabilities = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
                    uncertainty = None

            probability_image = Image.fromarray((np.clip(probabilities, 0, 1) * 255).astype(np.uint8))
            high_res_probabilities = np.asarray(
                probability_image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BILINEAR)
            ) / 255.0
            non_forest = (high_res_probabilities < request.threshold).astype(np.uint8)
            overlay = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 4), dtype=np.uint8)
            overlay[non_forest == 1] = [239, 68, 68, 190]

            tile_total = int(non_forest.size)
            tile_non_forest = int(non_forest.sum())
            total_pixels += tile_total
            non_forest_pixels += tile_non_forest
            overlays.append(TileOverlay(
                tile_id=tile_id,
                mask_png_base64=png_base64(Image.fromarray(overlay, mode="RGBA")),
                uncertainty_png_base64=uncertainty,
                bbox=BBox(**metadata["bbox"]),
            ))
            logger.info("Tile %s/%s processado: %s", position, len(request.tile_ids), tile_id)
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail=f"Falha ao processar o tile {position}/{len(request.tile_ids)} ({tile_id}): {error}",
            ) from error

    return DetectResponse(
        overlays=overlays,
        metadata={
            "source": "Extract-GEE / GeoTIFF Sentinel-2 local",
            "input_band_order": ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR real)"],
            "normalization": "mínimo-máximo por tile",
            "model_input_size": [MODEL_SIZE, MODEL_SIZE],
            "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "processing_mode": "sequencial, um GeoTIFF por vez",
            "processed_files": request.tile_ids,
        },
        stats={
            "total_pixels": total_pixels,
            "non_forest_pixels": non_forest_pixels,
            "forest_pixels": total_pixels - non_forest_pixels,
            "non_forest_pct": round(non_forest_pixels / total_pixels * 100, 2),
            "forest_pct": round((total_pixels - non_forest_pixels) / total_pixels * 100, 2),
            "threshold": request.threshold,
            "model_type": request.model_type,
            "mc_samples": request.mc_samples if request.use_mc_dropout else 0,
            "mean_variance": round(float(np.mean(variance_values)), 6) if variance_values else None,
            "processed_tiles": len(overlays),
        },
    )


FRONTEND_DIST = PROJECT_ROOT / "web" / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
