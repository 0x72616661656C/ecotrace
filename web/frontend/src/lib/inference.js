import * as ort from 'onnxruntime-web/wasm';
import { fromUrl } from 'geotiff';
import { modelUrl } from './dataset';

// A rede é totalmente convolucional e aceita 256×256. Essa resolução reduz a
// memória de ativações para cerca de 1/4 da usada em 512×512, sem acumular
// tiles na sessão do browser.
const MODEL_SIZE = 256;
const OUTPUT_SIZE = 768;
const sessions = new Map();

// `prebuild` copia os binários WASM para public/onnx. Assim a Vercel os entrega
// como estáticos e o runtime não depende de Python nem de node_modules em produção.
ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}onnx/`;
ort.env.wasm.numThreads = 1;

async function getSession(modelType) {
  if (!sessions.has(modelType)) {
    sessions.set(modelType, ort.InferenceSession.create(modelUrl(modelType), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }));
  }
  return sessions.get(modelType);
}

function normalizeBands(bands) {
  let min = Infinity;
  let max = -Infinity;
  for (const band of bands) {
    for (const value of band) {
      if (Number.isFinite(value)) {
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) throw new Error('GeoTIFF sem pixels válidos');

  const scale = 1 / (max - min + 1e-8);
  const tensor = new Float32Array(4 * MODEL_SIZE * MODEL_SIZE);
  for (let channel = 0; channel < 4; channel += 1) {
    const source = bands[channel];
    const offset = channel * MODEL_SIZE * MODEL_SIZE;
    for (let index = 0; index < source.length; index += 1) {
      tensor[offset + index] = Number.isFinite(source[index]) ? (source[index] - min) * scale : 0;
    }
  }
  return tensor;
}

async function readTile(tileUrl) {
  const tiff = await fromUrl(tileUrl);
  const image = await tiff.getImage();
  if (image.getSamplesPerPixel() < 4) throw new Error('O GeoTIFF precisa conter B2, B3, B4 e B8, nesta ordem.');
  const bands = await image.readRasters({
    samples: [0, 1, 2, 3],
    width: MODEL_SIZE,
    height: MODEL_SIZE,
    resampleMethod: 'bilinear',
    interleave: false,
  });
  return normalizeBands(bands);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function makeMask(probabilities, threshold) {
  const source = document.createElement('canvas');
  source.width = MODEL_SIZE;
  source.height = MODEL_SIZE;
  const sourceImage = source.getContext('2d').createImageData(MODEL_SIZE, MODEL_SIZE);
  for (let i = 0; i < probabilities.length; i += 1) {
    const intensity = Math.round(Math.min(1, Math.max(0, probabilities[i])) * 255);
    sourceImage.data[i * 4] = intensity;
    sourceImage.data[i * 4 + 1] = intensity;
    sourceImage.data[i * 4 + 2] = intensity;
    sourceImage.data[i * 4 + 3] = 255;
  }
  source.getContext('2d').putImageData(sourceImage, 0, 0);

  const scaled = document.createElement('canvas');
  scaled.width = OUTPUT_SIZE;
  scaled.height = OUTPUT_SIZE;
  const scaledContext = scaled.getContext('2d', { willReadFrequently: true });
  scaledContext.imageSmoothingEnabled = true;
  scaledContext.drawImage(source, 0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  const pixels = scaledContext.getImageData(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  let nonForest = 0;
  for (let i = 0; i < pixels.data.length; i += 4) {
    const isNonForest = pixels.data[i] / 255 < threshold;
    if (isNonForest) nonForest += 1;
    pixels.data[i] = 239;
    pixels.data[i + 1] = 68;
    pixels.data[i + 2] = 68;
    pixels.data[i + 3] = isNonForest ? 190 : 0;
  }
  scaledContext.putImageData(pixels, 0, 0);
  return {
    mask_png_base64: scaled.toDataURL('image/png').split(',')[1],
    totalPixels: OUTPUT_SIZE * OUTPUT_SIZE,
    nonForestPixels: nonForest,
  };
}

export async function detectTiles({ tiles, modelType, threshold, onProgress }) {
  const session = await getSession(modelType);
  const overlays = [];
  let totalPixels = 0;
  let nonForestPixels = 0;

  for (let index = 0; index < tiles.length; index += 1) {
    const tile = tiles[index];
    onProgress?.(index, tiles.length, tile.file);
    const input = await readTile(tile.url);
    const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 4, MODEL_SIZE, MODEL_SIZE]) };
    const output = await session.run(feeds);
    const logits = output[session.outputNames[0]].data;
    const probabilities = Float32Array.from(logits, sigmoid);
    const mask = makeMask(probabilities, threshold);
    totalPixels += mask.totalPixels;
    nonForestPixels += mask.nonForestPixels;
    overlays.push({
      tile_id: tile.file,
      mask_png_base64: mask.mask_png_base64,
      bbox: { west: tile.west, south: tile.south, east: tile.east, north: tile.north },
    });
  }

  return {
    overlays,
    metadata: {
      source: 'GeoTIFF Sentinel-2 remoto; processado no navegador',
      input_band_order: ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR real)'],
      normalization: 'mínimo-máximo por tile',
      model_input_size: [MODEL_SIZE, MODEL_SIZE],
      output_size: [OUTPUT_SIZE, OUTPUT_SIZE],
      processing_mode: 'cliente, sequencial, um GeoTIFF por vez',
    },
    stats: {
      total_pixels: totalPixels,
      non_forest_pixels: nonForestPixels,
      forest_pixels: totalPixels - nonForestPixels,
      non_forest_pct: Number((nonForestPixels / totalPixels * 100).toFixed(2)),
      forest_pct: Number(((totalPixels - nonForestPixels) / totalPixels * 100).toFixed(2)),
      threshold,
      model_type: modelType,
      processed_tiles: overlays.length,
    },
  };
}
