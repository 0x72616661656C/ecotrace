"""
Leitor e processador de GeoTIFFs com suporte a seleção estrita de tiles e multi-seleção.
"""

import os
import io
import json
import logging
import tempfile
from pathlib import Path
from threading import Lock
import numpy as np
import rasterio
from rasterio.merge import merge
from PIL import Image
from typing import Tuple, List, Dict, Optional
from pyproj import Transformer

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = WEB_DIR / "frontend"
EXTRACT_GEE_DIR = FRONTEND_DIR / "extract-gee"
INDEX_PATH = FRONTEND_DIR / "tile_index.json"
CACHE_DIR = FRONTEND_DIR / ".cache" / "tile_previews_v3"
OVERVIEW_PATH = CACHE_DIR / "sentinel_overview.png"
_overview_lock = Lock()

CACHE_DIR.mkdir(parents=True, exist_ok=True)

transformer_to_latlon = Transformer.from_crs("EPSG:32620", "EPSG:4326", always_xy=True)

_tile_index_data: Dict = None


def load_tile_index() -> Dict:
    """Carrega o índice dos 2.025 tiles."""
    global _tile_index_data
    if _tile_index_data is not None:
        return _tile_index_data

    if INDEX_PATH.is_file():
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            _tile_index_data = json.load(f)
        # O índice pode ter caminhos absolutos de outra máquina. A execução usa
        # sempre o diretório versionável definido para o projeto.
        for tile in _tile_index_data.get("tiles", []):
            tile["path"] = str(EXTRACT_GEE_DIR / tile["file"])
        return _tile_index_data

    tile_list = []
    min_lon, min_lat = 180.0, 90.0
    max_lon, max_lat = -180.0, -90.0

    if not EXTRACT_GEE_DIR.is_dir():
        raise FileNotFoundError(f"Diretório Extract-GEE não encontrado: {EXTRACT_GEE_DIR}")

    files = sorted(path.name for path in EXTRACT_GEE_DIR.glob("*.tif"))
    for filename in files:
        filepath = EXTRACT_GEE_DIR / filename
        with rasterio.open(filepath) as src:
            b = src.bounds
            l_min, la_min = transformer_to_latlon.transform(b.left, b.bottom)
            l_max, la_max = transformer_to_latlon.transform(b.right, b.top)

            min_lon = min(min_lon, l_min)
            min_lat = min(min_lat, la_min)
            max_lon = max(max_lon, l_max)
            max_lat = max(max_lat, la_max)

            tile_list.append({
                "file": filename,
                "path": str(filepath),
                "west": l_min,
                "south": la_min,
                "east": l_max,
                "north": la_max,
            })

    _tile_index_data = {
        "overall_bbox": {
            "west": min_lon,
            "south": min_lat,
            "east": max_lon,
            "north": max_lat,
        },
        "tiles": tile_list,
    }

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(_tile_index_data, f)

    return _tile_index_data


def find_overlapping_tiles_strict(bbox: Dict[str, float], min_overlap_pct: float = 0.15) -> List[Dict]:
    """
    Encontra estritamente os tiles que possuem pelo menos `min_overlap_pct` (15%)
    de área de sobreposição com o retângulo solicitado, ignorando toques acidentais na borda.
    """
    index = load_tile_index()
    tiles = index["tiles"]

    w_req, s_req, e_req, n_req = bbox["west"], bbox["south"], bbox["east"], bbox["north"]

    matching_tiles = []
    for tile in tiles:
        inter_w = max(w_req, tile["west"])
        inter_s = max(s_req, tile["south"])
        inter_e = min(e_req, tile["east"])
        inter_n = min(n_req, tile["north"])

        if inter_e > inter_w and inter_n > inter_s:
            inter_area = (inter_e - inter_w) * (inter_n - inter_s)
            tile_area = (tile["east"] - tile["west"]) * (tile["north"] - tile["south"])

            if (inter_area / tile_area) >= min_overlap_pct:
                matching_tiles.append(tile)

    # Fallback se a área for muito pequena (ex: clique pontual): pega apenas o tile que contém o centro
    if not matching_tiles:
        center_lon = (w_req + e_req) / 2.0
        center_lat = (s_req + n_req) / 2.0
        for tile in tiles:
            if tile["west"] <= center_lon <= tile["east"] and tile["south"] <= center_lat <= tile["north"]:
                matching_tiles.append(tile)
                break

    if not matching_tiles:
        best_tile = min(
            tiles,
            key=lambda t: ((t["west"] + t["east"]) / 2.0 - center_lon) ** 2
            + ((t["south"] + t["north"]) / 2.0 - center_lat) ** 2,
        )
        matching_tiles.append(best_tile)

    return matching_tiles


def get_tiles_by_ids(tile_ids: List[str]) -> List[Dict]:
    """Recupera os tiles estritamente pelos seus nomes de arquivo ID."""
    index = load_tile_index()
    tiles_map = {t["file"]: t for t in index["tiles"]}
    matched = []
    for tid in tile_ids:
        if tid in tiles_map:
            matched.append(tiles_map[tid])
    return matched


def read_and_process_selected(
    bbox: Optional[Dict[str, float]] = None,
    tile_ids: Optional[List[str]] = None,
    target_size: int = 512,
) -> Tuple[np.ndarray, Dict]:
    """
    Lê ESTRITAMENTE os tiles especificados por `tile_ids` ou `bbox`.
    """
    if tile_ids and len(tile_ids) > 0:
        tiles = get_tiles_by_ids(tile_ids)
        if not tiles:
            raise ValueError(f"Nenhum tile encontrado para os IDs: {tile_ids}")
    elif bbox:
        tiles = find_overlapping_tiles_strict(bbox)
    else:
        raise ValueError("Forneça um bbox ou uma lista de tile_ids.")

    logger.info(f"Processando estritamente {len(tiles)} tile(s): {[t['file'] for t in tiles]}")

    if len(tiles) == 1:
        # Caso 1 único tile: leitura direta sem merge!
        t = tiles[0]
        with rasterio.open(t["path"]) as src:
            raw_data = src.read()  # [4, 512, 512] uint16

        actual_bbox = {
            "west": t["west"],
            "south": t["south"],
            "east": t["east"],
            "north": t["north"],
        }
        file_info = t["file"]
    else:
        # Caso múltiplos tiles selecionados: mosaico estrito
        src_datasets = [rasterio.open(t["path"]) for t in tiles]
        merged_array, _ = merge(src_datasets)
        for ds in src_datasets:
            ds.close()

        raw_data = merged_array  # [4, H, W]

        w = min(t["west"] for t in tiles)
        s = min(t["south"] for t in tiles)
        e = max(t["east"] for t in tiles)
        n = max(t["north"] for t in tiles)
        actual_bbox = {"west": w, "south": s, "east": e, "north": n}
        file_info = f"Mosaico de {len(tiles)} tiles selecionados"

    # Redimensionar se necessário para 512x512
    _, h, w = raw_data.shape
    if h != target_size or w != target_size:
        resized_bands = []
        for b in range(raw_data.shape[0]):
            band_img = Image.fromarray(raw_data[b])
            band_resized = band_img.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
            resized_bands.append(np.array(band_resized, dtype=np.uint16))
        raw_data = np.stack(resized_bands, axis=0)

    # Normalização Min-Max por imagem (exata da rotina de avaliação do repositório)
    raw_float = raw_data.astype(np.float32)
    img_min, img_max = raw_float.min(), raw_float.max()
    image_array = (raw_float - img_min) / (img_max - img_min + 1e-8)

    metadata = {
        "file": file_info,
        "tiles_count": len(tiles),
        "selected_files": [t["file"] for t in tiles],
        "bbox": actual_bbox,
        "crs": "EPSG:32620",
        "bands": ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)"],
        "shape": image_array.shape,
    }

    return image_array, metadata


def create_rgb_preview(raw_data: np.ndarray) -> np.ndarray:
    """
    Gera imagem RGB True-Color (uint8 [H, W, 3]) com cores naturais do Sentinel-2.
    """
    if raw_data.ndim == 3 and raw_data.shape[0] >= 3:
        # Os GeoTIFFs Extract-GEE armazenam B2, B3, B4 e B8 nesta ordem.
        blue = raw_data[0].astype(np.float32)
        green = raw_data[1].astype(np.float32)
        red = raw_data[2].astype(np.float32)
    else:
        blue = raw_data[:, :, 0].astype(np.float32)
        green = raw_data[:, :, 1].astype(np.float32)
        red = raw_data[:, :, 2].astype(np.float32)

    if red.max() <= 1.0:
        red = red * 3000.0
        green = green * 3000.0
        blue = blue * 3000.0

    r_norm = np.power(np.clip(red / 2200.0, 0.0, 1.0), 0.85)
    g_norm = np.power(np.clip(green / 2500.0, 0.0, 1.0), 0.85)
    b_norm = np.power(np.clip(blue / 2000.0, 0.0, 1.0), 0.85)

    rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def get_overview_png(max_dimension: int = 2048) -> bytes:
    """Gera uma visão geral leve e persistente de toda a cobertura Sentinel-2."""
    if OVERVIEW_PATH.is_file():
        return OVERVIEW_PATH.read_bytes()

    with _overview_lock:
        if OVERVIEW_PATH.is_file():
            return OVERVIEW_PATH.read_bytes()

        index = load_tile_index()
        overall = index["overall_bbox"]
        width_degrees = overall["east"] - overall["west"]
        height_degrees = overall["north"] - overall["south"]
        width = max(1, max_dimension)
        height = max(1, round(max_dimension * height_degrees / width_degrees))
        canvas = Image.new("RGB", (width, height), (231, 236, 229))

        for tile in index["tiles"]:
            # Não use arredondamento independente para posição e dimensão: ele
            # cria fendas de 1–3 px entre tiles vizinhos. Calculamos as quatro
            # bordas e aplicamos uma pequena sobreposição visual, sem alterar os
            # GeoTIFFs ou os limites usados na inferência.
            left = int(np.floor(width * (tile["west"] - overall["west"]) / width_degrees))
            right = int(np.ceil(width * (tile["east"] - overall["west"]) / width_degrees))
            top = int(np.floor(height * (overall["north"] - tile["north"]) / height_degrees))
            bottom = int(np.ceil(height * (overall["north"] - tile["south"]) / height_degrees))
            bleed = 2
            x = max(0, left - bleed)
            y = max(0, top - bleed)
            right = min(width, right + bleed)
            bottom = min(height, bottom + bleed)
            pixel_width = max(1, right - x)
            pixel_height = max(1, bottom - y)
            try:
                with rasterio.open(tile["path"]) as source:
                    raw = source.read(out_shape=(4, pixel_height, pixel_width))
                preview = Image.fromarray(create_rgb_preview(raw), mode="RGB")
                canvas.paste(preview, (x, y))
            except Exception as error:
                logger.warning("Falha ao incluir %s no mosaico: %s", tile["file"], error)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", optimize=True)
        png = buffer.getvalue()
        OVERVIEW_PATH.write_bytes(png)
        return png


def get_cached_tile_png(filename: str) -> bytes:
    """
    Gera ou recupera do cache a versão PNG RGB de um tile GeoTIFF específico.
    """
    cache_png = os.path.join(CACHE_DIR, filename.replace(".tif", ".png"))
    if os.path.exists(cache_png):
        with open(cache_png, "rb") as f:
            return f.read()

    filepath = os.path.join(EXTRACT_GEE_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tile {filename} não encontrado.")

    with rasterio.open(filepath) as src:
        raw = src.read()

    rgb = create_rgb_preview(raw)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()
    img.close()

    with open(cache_png, "wb") as f:
        f.write(png_bytes)

    return png_bytes


def precache_all_tiles_parallel(max_workers: int = 8) -> int:
    """
    Pré-carrega e renderiza em paralelo todas as 2.025 imagens PNG dos tiles extract-gee na RAM/Disco.
    """
    import concurrent.futures

    index = load_tile_index()
    tiles = index.get("tiles", [])
    logger.info(f"Iniciando pré-carregamento em paralelo de {len(tiles)} tiles extract-gee...")

    def _process(t):
        try:
            return get_cached_tile_png(t["file"])
        except Exception as e:
            logger.warning(f"Erro ao pré-carregar tile {t['file']}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process, tiles))

    valid_count = sum(1 for r in results if r is not None)
    logger.info(f"Pré-carregamento paralelo concluído: {valid_count}/{len(tiles)} tiles prontos em cache.")
    return valid_count
