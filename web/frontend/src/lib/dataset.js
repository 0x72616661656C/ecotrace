const configuredBase = import.meta.env.VITE_DATASET_BASE_URL || '/data';

export const DATASET_BASE_URL = configuredBase.replace(/\/$/, '');

export function datasetUrl(path) {
  return `${DATASET_BASE_URL}/${path.replace(/^\//, '')}`;
}

export async function loadCoverage() {
  const response = await fetch(datasetUrl('tile_index.json'));
  if (!response.ok) throw new Error(`Índice dos tiles indisponível (${response.status})`);
  const coverage = await response.json();

  return {
    ...coverage,
    tiles: (coverage.tiles || []).map((tile) => ({
      ...tile,
      // URLs explícitas permitem usar um bucket/CDN diferente para cada asset.
      url: tile.url || datasetUrl(`extract-gee/${encodeURIComponent(tile.file)}`),
      preview_url: tile.preview_url || datasetUrl(`previews/${encodeURIComponent(tile.file.replace(/\.tiff?$/i, '.png'))}`),
    })),
  };
}

export function overviewUrl() {
  return import.meta.env.VITE_OVERVIEW_URL || datasetUrl('sentinel_overview.png');
}

export function modelUrl(modelType) {
  if (modelType === 'attn_unet') {
    return import.meta.env.VITE_ATTN_UNET_MODEL_URL || '/models/attn_unet.onnx';
  }
  return import.meta.env.VITE_UNET_MODEL_URL || '/models/unet.onnx';
}
