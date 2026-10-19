"""
Arquiteturas de inferência do EcoTrace.
Compatíveis com checkpoints portáteis em models/checkpoints/*.pt.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, n_input: int, n_output: int, dropout_probability: float = 0.5):
        super().__init__()
        self.linear1 = nn.Conv2d(n_input, n_output, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_probability)

        self.linear2 = nn.Conv2d(n_output, n_output, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return self.dropout2(x)


class Downsample(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        self.double_conv = DoubleConv(n_input, n_output)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        h = self.double_conv(x)
        x = self.pool(h)
        return h, x


class Upsample(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(n_input, n_output, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(n_input, n_output)

    def forward(self, h: torch.Tensor, x: torch.Tensor):
        x = self.conv_transpose(x)
        x = torch.cat([x, h], dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_inputs: int = 4, num_outputs: int = 1):
        super().__init__()
        self.downsample1 = Downsample(num_inputs, 64)
        self.downsample2 = Downsample(64, 128)
        self.downsample3 = Downsample(128, 256)
        self.downsample4 = Downsample(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.upsample1 = Upsample(1024, 512)
        self.upsample2 = Upsample(512, 256)
        self.upsample3 = Upsample(256, 128)
        self.upsample4 = Upsample(128, 64)
        self.conv2nd = nn.Conv2d(64, num_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, x = self.downsample1(x)
        h2, x = self.downsample2(x)
        h3, x = self.downsample3(x)
        h4, x = self.downsample4(x)
        x = self.bottleneck(x)
        x = self.upsample1(h4, x)
        x = self.upsample2(h3, x)
        x = self.upsample3(h2, x)
        x = self.upsample4(h1, x)
        x = self.conv2nd(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, n_input_x: int, n_input_g: int):
        super().__init__()
        self.conv_x = nn.Conv2d(n_input_x, n_input_x, kernel_size=1, stride=2, padding=0)
        self.conv_g = nn.Conv2d(n_input_g, n_input_x, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv_attn = nn.Conv2d(n_input_x, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        y = self.conv_x(x)
        g = self.conv_g(g)
        y = self.relu(y + g)
        attn = self.conv_attn(y)
        attn = self.sigmoid(attn)
        attn = self.upsample(attn)
        y = attn * x
        return y


class AttentionUNet(nn.Module):
    def __init__(self, num_inputs: int = 4, num_outputs: int = 1):
        super().__init__()
        self.downsample1 = Downsample(num_inputs, 16)
        self.downsample2 = Downsample(16, 32)
        self.downsample3 = Downsample(32, 64)
        self.downsample4 = Downsample(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.upsample1 = Upsample(256, 128)
        self.upsample2 = Upsample(128, 64)
        self.upsample3 = Upsample(64, 32)
        self.upsample4 = Upsample(32, 16)
        self.conv2nd = nn.Conv2d(16, num_outputs, kernel_size=1)
        self.ag1 = AttentionGate(128, 256)
        self.ag2 = AttentionGate(64, 128)
        self.ag3 = AttentionGate(32, 64)
        self.ag4 = AttentionGate(16, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1, x = self.downsample1(x)
        g2, x = self.downsample2(x)
        g3, x = self.downsample3(x)
        g4, x = self.downsample4(x)
        x = self.bottleneck(x)
        g4 = self.ag1(g4, x)
        x = self.upsample1(g4, x)
        g3 = self.ag2(g3, x)
        x = self.upsample2(g3, x)
        g2 = self.ag3(g2, x)
        x = self.upsample3(g2, x)
        g1 = self.ag4(g1, x)
        x = self.upsample4(g1, x)
        x = self.conv2nd(x)
        return x


def load_model(model_type: str, checkpoint_dir, device: str = "cpu") -> nn.Module:
    """Load a portable state dictionary produced by the training notebook."""
    from pathlib import Path

    if model_type == "unet":
        model = UNet(num_inputs=4, num_outputs=1)
        weights_path = Path(checkpoint_dir) / "unet_seed_2_best.pt"
    elif model_type in {"attn_unet", "attention_unet"}:
        model = AttentionUNet(num_inputs=4, num_outputs=1)
        weights_path = Path(checkpoint_dir) / "attn_unet_seed_2_best.pt"
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {weights_path}. "
            "Defina ECOTRACE_CHECKPOINT_DIR ou baixe os pesos conforme models/README.md."
        )

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
