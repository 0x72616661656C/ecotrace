"""Exporta os checkpoints EcoTrace para ONNX fixo de 512×512, próprio para browsers."""

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.architecture import load_model


def export(model_type: str, checkpoint_dir: Path, output_dir: Path) -> Path:
    model = load_model(model_type, checkpoint_dir, "cpu").eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("attn_unet.onnx" if model_type == "attn_unet" else "unet.onnx")
    sample = torch.zeros((1, 4, 512, 512), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample,
            output,
            input_names=["sentinel_b2_b3_b4_b8"],
            output_names=["forest_logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"ONNX criado: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "web/backend/models/checkpoints")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "web/frontend/public/models")
    parser.add_argument("--model", choices=["unet", "attn_unet", "all"], default="all")
    args = parser.parse_args()
    for name in (["unet", "attn_unet"] if args.model == "all" else [args.model]):
        export(name, args.checkpoint_dir, args.output_dir)
