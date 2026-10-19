import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class DoubleConv(nn.Module):
    def __init__(self, n_input: int, n_output: int, dropout_probability: float = 0.5):
        super().__init__()
        self.linear1 = nn.Conv2d(n_input, n_output, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_probability)

        self.linear2 = nn.Conv2d(n_output, n_output, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_probability)

        self.functions = nn.Sequential(
            self.linear1,
            self.relu1,
            self.dropout1,
            self.linear2,
            self.relu2,
            self.dropout2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.functions(x)


class Downsample(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.double_conv = DoubleConv(n_input, n_output)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        h = self.double_conv(x)
        x = self.pool(h)
        return h, x


class Upsample(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(n_input, n_output, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(n_input, n_output)

    def forward(self, h, x):
        x = self.conv_transpose(x)
        x = torch.cat([x, h], dim=1)
        x = self.double_conv(x)
        return x


class SegmentationLightningModule(pl.LightningModule):
    """Shared Lightning logic for binary deforestation segmentation."""

    def _make_metrics(self, prefix):
        return nn.ModuleDict(
            {
                f"{prefix}_precision": torchmetrics.Precision(task="binary", threshold=0.5),
                f"{prefix}_recall": torchmetrics.Recall(task="binary", threshold=0.5),
                f"{prefix}_f1_score": torchmetrics.F1Score(task="binary", threshold=0.5),
                f"{prefix}_iou": torchmetrics.JaccardIndex(task="binary"),
            }
        )

    def _init_metrics(self):
        self.train_metrics = self._make_metrics("train")
        self.val_metrics = self._make_metrics("val")
        self.test_metrics = self._make_metrics("test")

    def _common_step(self, batch, batch_idx, metrics):
        images, masks = batch
        logits = self(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, masks)

        out = {"loss": loss}
        for name, metric in metrics.items():
            out[name] = metric(preds, masks.int())
        return out

    def _log_step(self, out, prefix):
        log_values = {f"{prefix}_loss": out["loss"]}
        for key, value in out.items():
            if key != "loss":
                log_values[key] = value
        self.log_dict(
            log_values, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True
        )

    def training_step(self, batch, batch_idx):
        out = self._common_step(batch, batch_idx, self.train_metrics)
        self._log_step(out, "train")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._common_step(batch, batch_idx, self.val_metrics)
        self._log_step(out, "val")
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self._common_step(batch, batch_idx, self.test_metrics)
        self._log_step(out, "test")
        return out["loss"]

    def predict_step(self, batch, batch_idx):
        images, _ = batch
        logits = self(images)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).float()


class LitUNet(SegmentationLightningModule):
    def __init__(self, num_inputs, num_outputs, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = lr
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
        self._init_metrics()

    def forward(self, x):
        h1, x = self.downsample1(x)
        h2, x = self.downsample2(x)
        h3, x = self.downsample3(x)
        h4, x = self.downsample4(x)
        x = self.bottleneck(x)
        x = self.upsample1(h4, x)
        x = self.upsample2(h3, x)
        x = self.upsample3(h2, x)
        x = self.upsample4(h1, x)
        return self.conv2nd(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class AttentionGate(nn.Module):
    def __init__(self, n_input_x, n_input_g):
        super().__init__()
        self.conv_x = nn.Conv2d(n_input_x, n_input_x, kernel_size=1, stride=2, padding=0)
        self.conv_g = nn.Conv2d(n_input_g, n_input_x, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv_attn = nn.Conv2d(n_input_x, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, g):
        y = self.relu(self.conv_x(x) + self.conv_g(g))
        attention = self.upsample(self.sigmoid(self.conv_attn(y)))
        return attention * x


class LitAttentionUNet(SegmentationLightningModule):
    def __init__(self, num_inputs, num_outputs, lr=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = lr
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
        self._init_metrics()

    def forward(self, x):
        g1, x = self.downsample1(x)
        g2, x = self.downsample2(x)
        g3, x = self.downsample3(x)
        g4, x = self.downsample4(x)
        x = self.bottleneck(x)
        x = self.upsample1(self.ag1(g4, x), x)
        x = self.upsample2(self.ag2(g3, x), x)
        x = self.upsample3(self.ag3(g2, x), x)
        x = self.upsample4(self.ag4(g1, x), x)
        return self.conv2nd(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def get_model(model_name, num_inputs=4, num_outputs=1, lr=None):
    """Create a model by name."""
    if model_name == "unet":
        kwargs = {"num_inputs": num_inputs, "num_outputs": num_outputs}
        return LitUNet(**kwargs) if lr is None else LitUNet(**kwargs, lr=lr)
    if model_name in {"attn_unet", "attention_unet"}:
        kwargs = {"num_inputs": num_inputs, "num_outputs": num_outputs}
        return LitAttentionUNet(**kwargs) if lr is None else LitAttentionUNet(**kwargs, lr=lr)
    raise ValueError(f"Unknown model name: {model_name}")


def load_model_from_checkpoint(model_name, checkpoint_path, num_inputs=4, num_outputs=1):
    """Load a Lightning checkpoint by model name."""
    if model_name == "unet":
        return LitUNet.load_from_checkpoint(
            checkpoint_path, num_inputs=num_inputs, num_outputs=num_outputs
        )
    if model_name in {"attn_unet", "attention_unet"}:
        return LitAttentionUNet.load_from_checkpoint(
            checkpoint_path, num_inputs=num_inputs, num_outputs=num_outputs
        )
    raise ValueError(f"Unknown model name: {model_name}")
