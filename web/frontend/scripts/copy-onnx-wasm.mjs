import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const source = resolve(root, 'node_modules/onnxruntime-web/dist');
const destination = resolve(root, 'public/onnx');
const files = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];

await mkdir(destination, { recursive: true });
await Promise.all(files.map((file) => copyFile(resolve(source, file), resolve(destination, file))));
