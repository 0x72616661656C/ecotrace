"""Gera índice e PNGs de prévia para hospedar GeoTIFFs estáticos em um CDN."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from pyproj import Transformer


TO_LATLON = Transformer.from_crs("EPSG:32620", "EPSG:4326", always_xy=True)


def rgb_preview(data: np.ndarray) -> Image.Image:
    blue, green, red = (data[i].astype(np.float32) for i in range(3))
    rgb = np.stack([
        np.power(np.clip(red / 2200.0, 0, 1), 0.85),
        np.power(np.clip(green / 2500.0, 0, 1), 0.85),
        np.power(np.clip(blue / 2000.0, 0, 1), 0.85),
    ], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Diretório com os .tif do Extract-GEE")
    parser.add_argument("--output", type=Path, required=True, help="Diretório de artefatos para enviar ao CDN")
    parser.add_argument("--preview-size", type=int, default=512)
    args = parser.parse_args()

    tif_files = sorted(args.input.glob("*.tif"))
    if not tif_files:
        raise SystemExit(f"Nenhum .tif encontrado em {args.input}")
    preview_dir = args.output / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    west = south = float("inf")
    east = north = float("-inf")

    for position, path in enumerate(tif_files, start=1):
        with rasterio.open(path) as src:
            bounds = src.bounds
            tile_west, tile_south = TO_LATLON.transform(bounds.left, bounds.bottom)
            tile_east, tile_north = TO_LATLON.transform(bounds.right, bounds.top)
        preview_name = f"{path.stem}.png"
        preview_path = preview_dir / preview_name
        if not preview_path.is_file():
            with rasterio.open(path) as src:
                raw = src.read([1, 2, 3], out_shape=(3, args.preview_size, args.preview_size), resampling=rasterio.enums.Resampling.bilinear)
            rgb_preview(raw).save(preview_path, "PNG", optimize=True)
        tiles.append({"file": path.name, "west": tile_west, "south": tile_south, "east": tile_east, "north": tile_north})
        west, south = min(west, tile_west), min(south, tile_south)
        east, north = max(east, tile_east), max(north, tile_north)
        if position % 100 == 0:
            print(f"Previews: {position}/{len(tif_files)}", flush=True)

    index = {"overall_bbox": {"west": west, "south": south, "east": east, "north": north}, "tiles": tiles}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tile_index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")

    # A visão geral é feita a partir das prévias para não carregar todos os TIFFs no browser.
    width = 2048
    height = max(1, round(width * (north - south) / (east - west)))
    overview = Image.new("RGB", (width, height), (231, 236, 229))
    for tile in tiles:
        image = Image.open(preview_dir / f"{Path(tile['file']).stem}.png")
        left = round(width * (tile["west"] - west) / (east - west))
        right = round(width * (tile["east"] - west) / (east - west))
        top = round(height * (north - tile["north"]) / (north - south))
        bottom = round(height * (north - tile["south"]) / (north - south))
        overview.paste(image.resize((max(1, right - left), max(1, bottom - top))), (left, top))
    overview.save(args.output / "sentinel_overview.png", "PNG", optimize=True)
    print(f"Criados {len(tiles)} previews, tile_index.json e sentinel_overview.png em {args.output}")


if __name__ == "__main__":
    main()
