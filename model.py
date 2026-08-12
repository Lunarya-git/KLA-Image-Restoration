"""
RestorationNet: a single fully-convolutional model that handles denoising
(speckle + Gaussian) and 2x super-resolution simultaneously, for both the
128->256 and 256->512 cases, without any explicit scale-conditioning input --
the same weights work on either input size because there are no dense/FC
layers anywhere in the network.

Architecture: conv stem -> N residual blocks -> PixelShuffle 2x upsample
-> output conv -> sigmoid.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class RestorationNet(nn.Module):
    """
    Fully convolutional restoration network.

    Args:
        in_channels: input channel count (1 for grayscale)
        out_channels: output channel count (1 for grayscale)
        base_channels: channel width used throughout the trunk
        num_residual_blocks: number of residual blocks in the trunk (6-8 per
            the locked protocol, but left configurable for later experiments)
        upscale_factor: spatial upscaling applied via PixelShuffle (2 per
            protocol; the same architecture would support 4 by chaining two
            2x PixelShuffle stages, though that is not needed here)
        output_activation: "sigmoid" (default) or "clamp" -- both guarantee
            output in [0, 1] since GT is always a valid image range
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_residual_blocks: int = 8,
        upscale_factor: int = 2,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        if output_activation not in ("sigmoid", "clamp"):
            raise ValueError(f"output_activation must be 'sigmoid' or 'clamp', got '{output_activation}'")

        self.upscale_factor = upscale_factor
        self.output_activation = output_activation

        # Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Residual trunk
        self.trunk = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_residual_blocks)])

        # Fuse trunk output with a skip from the stem (classic SRResNet-style
        # global residual connection), then upsample via PixelShuffle.
        self.trunk_fuse = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        upsample_channels = base_channels * (upscale_factor ** 2)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, upsample_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True),
        )

        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_out = self.stem(x)
        trunk_out = self.trunk(stem_out)
        fused = self.trunk_fuse(trunk_out) + stem_out
        up = self.upsample(fused)
        out = self.output_conv(up)

        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        else:
            out = torch.clamp(out, 0.0, 1.0)
        return out


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_from_config(config: dict) -> RestorationNet:
    m_cfg = config["model"]
    return RestorationNet(
        in_channels=m_cfg.get("in_channels", 1),
        out_channels=m_cfg.get("out_channels", 1),
        base_channels=m_cfg.get("base_channels", 64),
        num_residual_blocks=m_cfg.get("num_residual_blocks", 8),
        upscale_factor=m_cfg.get("upscale_factor", 2),
        output_activation=m_cfg.get("output_activation", "sigmoid"),
    )


if __name__ == "__main__":
    # Quick shape sanity check for both configured resolutions.
    net = RestorationNet(num_residual_blocks=8, base_channels=64, upscale_factor=2)
    print(f"Parameters: {count_parameters(net):,}")
    for h in (128, 256):
        x = torch.randn(2, 1, h, h)
        y = net(x)
        print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")
