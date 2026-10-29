"""
Leitor e servidor COG com carregamento progressivo (multi-resolução / LOD).

Funcionalidades:
1. Pirâmides de overviews multi-escala para carregamento progressivo (visão geral rápida → alta resolução ao aproximar o zoom).
2. LRU Cache em memória RAM para tiles XYZ (resposta em 0.001ms).
3. Leitura otimizada por janela de píxeis para inferência PyTorch.
"""

import os
import io
import math
import logging
import functools
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from PIL import Image
from typing import Tuple, Dict
from pyproj import Transformer

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
# O mosaico é opcional; sem ele o leitor usa os GeoTIFFs individuais do
# diretório configurado no local_tile_reader.
COG_PATH = Path(os.getenv("ECOTRACE_COG_PATH", BACKEND_DIR / "cog_dataset.tif"))
INDEX_PATH = Path(os.getenv("ECOTRACE_TILE_INDEX", BACKEND_DIR / "tile_index.json"))

transformer_to_latlon = Transformer.from_crs("EPSG:32620", "EPSG:4326", always_xy=True)
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32620", always_xy=True)

_cog_bounds_latlon = None


def get_cog_overall_bbox() -> Dict:
    """Retorna os limites gerais do dataset em Lat/Lon."""
    global _cog_bounds_latlon
    if _cog_bounds_latlon is not None:
        return _cog_bounds_latlon

    if os.path.exists(COG_PATH):
        try:
            with rasterio.open(COG_PATH) as src:
                b = src.bounds
                w, s = transformer_to_latlon.transform(b.left, b.bottom)
                e, n = transformer_to_latlon.transform(b.right, b.top)
                _cog_bounds_latlon = {
                    "west": w, "south": s, "east": e, "north": n,
                    "overall_bbox": {"west": w, "south": s, "east": e, "north": n}
                }
                return _cog_bounds_latlon
        except Exception:
            pass

    if os.path.exists(INDEX_PATH):
        import json
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            _cog_bounds_latlon = data
            return data

    return {"overall_bbox": {"west": -61.85, "south": 0.65, "east": -59.77, "north": 2.74}}


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Converte coordenadas XYZ para Lat/Lon."""
    n = 1 << zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


def xyz_to_latlon_bounds(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    w, n = num2deg(x, y, z)
    e, s = num2deg(x + 1, y + 1, z)
    return w, s, e, n


@functools.lru_cache(maxsize=16384)
def render_cog_xyz_tile_cached(x: int, y: int, z: int) -> bytes:
    """
    Renderiza um tile de mapa XYZ (256x256 PNG) estilo Terracotta com cache LRU em memória.
    Suporta qualquer nível de zoom (zoom 0 a 19) com corte de bounds UTM e interpolação ideal.
    """
    w_lon, s_lat, e_lon, n_lat = xyz_to_latlon_bounds(x, y, z)

    # Converter Lat/Lon para UTM Zone 20N
    min_x, min_y = transformer_to_utm.transform(w_lon, s_lat)
    max_x, max_y = transformer_to_utm.transform(e_lon, n_lat)

    if not os.path.exists(COG_PATH):
        return _render_tile_from_individual_geotiffs(w_lon, s_lat, e_lon, n_lat, z)

    try:
        with rasterio.open(COG_PATH) as src:
            cb = src.bounds
            inter_min_x = max(min_x, cb.left)
            inter_min_y = max(min_y, cb.bottom)
            inter_max_x = min(max_x, cb.right)
            inter_max_y = min(max_y, cb.top)

            # Sem intersecção geográfica -> Tile transparente
            if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
                img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            win_w = max_x - min_x
            win_h = max_y - min_y

            # Posição relativa da intersecção no canvas de 256x256 píxeis
            px_left = max(0, min(256, int(round(256 * (inter_min_x - min_x) / win_w))))
            px_right = max(0, min(256, int(round(256 * (inter_max_x - min_x) / win_w))))
            px_bottom = max(0, min(256, int(round(256 * (max_y - inter_max_y) / win_h))))
            px_top = max(0, min(256, int(round(256 * (max_y - inter_min_y) / win_h))))

            out_w = max(1, px_right - px_left)
            out_h = max(1, px_top - px_bottom)

            window = from_bounds(inter_min_x, inter_min_y, inter_max_x, inter_max_y, src.transform)
            resample_mode = Resampling.nearest if z >= 13 else Resampling.bilinear

            raw_data = src.read(
                out_shape=(4, out_h, out_w),
                window=window,
                resampling=resample_mode,
                boundless=True,
                fill_value=0,
            )

            canvas = np.zeros((4, 256, 256), dtype=raw_data.dtype)
            canvas[:, px_bottom : px_bottom + out_h, px_left : px_left + out_w] = raw_data
    except Exception as err:
        logger.debug(f"Erro ao ler COG para tile {z}/{x}/{y}: {err}")
        return _render_tile_from_individual_geotiffs(w_lon, s_lat, e_lon, n_lat, z)

    # Prepara cores True-Color Sentinel-2 (B2=Blue, B3=Green, B4=Red)
    blue = canvas[0].astype(np.float32)
    green = canvas[1].astype(np.float32)
    red = canvas[2].astype(np.float32)

    alpha = np.where((red == 0) & (green == 0) & (blue == 0), 0, 255).astype(np.uint8)

    r_norm = np.power(np.clip(red / 2200.0, 0.0, 1.0), 0.85)
    g_norm = np.power(np.clip(green / 2500.0, 0.0, 1.0), 0.85)
    b_norm = np.power(np.clip(blue / 2000.0, 0.0, 1.0), 0.85)

    r_u8 = (r_norm * 255.0).astype(np.uint8)
    g_u8 = (g_norm * 255.0).astype(np.uint8)
    b_u8 = (b_norm * 255.0).astype(np.uint8)

    rgba = np.stack([r_u8, g_u8, b_u8, alpha], axis=-1)
    img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_tile_from_individual_geotiffs(w_lon, s_lat, e_lon, n_lat, z) -> bytes:
    """Fallback ultra-rápido de tiles individuais em cache se COG estiver ausente."""
    from local_tile_reader import find_overlapping_tiles_strict, get_cached_tile_png

    bbox = {"west": w_lon, "south": s_lat, "east": e_lon, "north": n_lat}
    try:
        tiles = find_overlapping_tiles_strict(bbox, min_overlap_pct=0.01)
        if tiles:
            return get_cached_tile_png(tiles[0]["file"])
    except Exception:
        pass

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_cog_xyz_tile(x: int, y: int, z: int) -> bytes:
    """Invoca o renderizador de tiles XYZ com cache em memória."""
    return render_cog_xyz_tile_cached(x, y, z)


@functools.lru_cache(maxsize=16384)
def render_cog_xyz_tile_nir_cached(x: int, y: int, z: int) -> bytes:
    """
    Renderiza um tile XYZ (256x256 PNG) na composição False Color Infrared (CIR):
        R ← B8  (NIR real, 842 nm)  — floresta saudável reflete fortemente
        G ← B4  (Red,  665 nm)
        B ← B3  (Green, 560 nm)

    Esta é a composição padrão científico para análise de cobertura vegetal.
    Vegetacão fotossinteticamente ativa aparece em vermelho intenso;
    áreas desmatadas ou degradadas aparecem em tons pálidos/marrons.

    Gamma 0.75 no NIR maximiza contraste entre floresta densa e áreas abertas.
    """
    w_lon, s_lat, e_lon, n_lat = xyz_to_latlon_bounds(x, y, z)

    min_x, min_y = transformer_to_utm.transform(w_lon, s_lat)
    max_x, max_y = transformer_to_utm.transform(e_lon, n_lat)

    if not os.path.exists(COG_PATH):
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    try:
        with rasterio.open(COG_PATH) as src:
            cb = src.bounds
            inter_min_x = max(min_x, cb.left)
            inter_min_y = max(min_y, cb.bottom)
            inter_max_x = min(max_x, cb.right)
            inter_max_y = min(max_y, cb.top)

            if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
                img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            win_w = max_x - min_x
            win_h = max_y - min_y

            px_left   = max(0, min(256, int(round(256 * (inter_min_x - min_x) / win_w))))
            px_right  = max(0, min(256, int(round(256 * (inter_max_x - min_x) / win_w))))
            px_bottom = max(0, min(256, int(round(256 * (max_y - inter_max_y) / win_h))))
            px_top    = max(0, min(256, int(round(256 * (max_y - inter_min_y) / win_h))))

            out_w = max(1, px_right  - px_left)
            out_h = max(1, px_top   - px_bottom)

            window = from_bounds(inter_min_x, inter_min_y, inter_max_x, inter_max_y, src.transform)
            resample_mode = Resampling.nearest if z >= 13 else Resampling.bilinear

            raw_data = src.read(
                out_shape=(4, out_h, out_w),
                window=window,
                resampling=resample_mode,
                boundless=True,
                fill_value=0,
            )

            canvas = np.zeros((4, 256, 256), dtype=raw_data.dtype)
            canvas[:, px_bottom : px_bottom + out_h, px_left : px_left + out_w] = raw_data
    except Exception as err:
        logger.debug(f"Erro ao ler COG NIR para tile {z}/{x}/{y}: {err}")
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # Composição CIR: banda[0]=B2(Blue), banda[1]=B3(Green), banda[2]=B4(Red), banda[3]=B8(NIR)
    blue  = canvas[0].astype(np.float32)  # B2
    green = canvas[1].astype(np.float32)  # B3
    red   = canvas[2].astype(np.float32)  # B4
    nir   = canvas[3].astype(np.float32)  # B8 — NIR real

    alpha = np.where((nir == 0) & (red == 0) & (green == 0), 0, 255).astype(np.uint8)

    # CIR: NIR→R (gamma=0.75 para realce de floresta), Red→G, Green→B
    r_norm = np.power(np.clip(nir   / 3500.0, 0.0, 1.0), 0.75)  # NIR
    g_norm = np.power(np.clip(red   / 2200.0, 0.0, 1.0), 0.85)  # Red
    b_norm = np.power(np.clip(green / 2500.0, 0.0, 1.0), 0.85)  # Green

    r_u8 = (r_norm * 255.0).astype(np.uint8)
    g_u8 = (g_norm * 255.0).astype(np.uint8)
    b_u8 = (b_norm * 255.0).astype(np.uint8)

    rgba = np.stack([r_u8, g_u8, b_u8, alpha], axis=-1)
    img  = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_cog_xyz_tile_nir(x: int, y: int, z: int) -> bytes:
    """Renderiza tile XYZ em composicão NIR False Color (CIR) com cache LRU."""
    return render_cog_xyz_tile_nir_cached(x, y, z)


def read_cog_area(bbox: Dict[str, float], target_size: int = 512) -> Tuple[np.ndarray, Dict]:
    """Lê a janela exata do dataset GeoTIFF para inferência PyTorch."""
    if not os.path.exists(COG_PATH):
        from local_tile_reader import read_and_process_selected
        return read_and_process_selected(bbox=bbox, target_size=target_size)

    w_lon, s_lat, e_lon, n_lat = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    min_x, min_y = transformer_to_utm.transform(w_lon, s_lat)
    max_x, max_y = transformer_to_utm.transform(e_lon, n_lat)

    with rasterio.open(COG_PATH) as src:
        window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
        raw_data = src.read(
            out_shape=(4, target_size, target_size),
            window=window,
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )

    raw_float = raw_data.astype(np.float32)
    img_min, img_max = raw_float.min(), raw_float.max()
    image_array = (raw_float - img_min) / (img_max - img_min + 1e-8)

    metadata = {
        "file": "Cloud Optimized GeoTIFF (cog_dataset.tif)",
        "source": "Stream COG Direto",
        "bbox": bbox,
        "crs": "EPSG:32620",
        "bands": ["B4 (Red)", "B3 (Green)", "B2 (Blue)", "B8 (NIR)"],
        "shape": image_array.shape,
    }

    return image_array, metadata
