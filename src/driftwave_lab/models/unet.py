"""CNN / U-Net next-step surrogate model.

A lightweight U-Net for next-step prediction of Hasegawa-Wakatani fields.
Serves as the baseline against which the FNO hero model is compared.

Architecture
------------
Encoder: 3 down-blocks (Conv-BN-ReLU x2 + MaxPool)
Bottleneck: Conv-BN-ReLU x2
Decoder: 3 up-blocks (ConvTranspose + skip-cat + Conv-BN-ReLU x2)
Head: 1x1 conv

Tensor conventions
------------------
- Input  : ``(B, C_in, H, W)``
- Output : ``(B, C_out, H, W)``

Periodic padding is used to respect the doubly-periodic domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Module):
    """Two consecutive Conv2d-BN-ReLU blocks with circular padding."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    """Downsampling: MaxPool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class _Up(nn.Module):
    """Upsampling: ConvTranspose2d + skip concatenation + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """Lightweight U-Net for 2D field next-step prediction.

    Parameters
    ----------
    in_channels : int
        Input channels (default 2: n, phi).
    out_channels : int
        Output channels (default 2: n, phi).
    base_filters : int
        Number of filters in the first encoder stage (default 32).
        Doubles at each subsequent stage.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_filters: int = 32,
    ) -> None:
        super().__init__()
        f = base_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters

        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _Down(f, f * 2)
        self.enc3 = _Down(f * 2, f * 4)
        self.bottleneck = _Down(f * 4, f * 8)

        self.dec3 = _Up(f * 8, f * 4)
        self.dec2 = _Up(f * 4, f * 2)
        self.dec1 = _Up(f * 2, f)

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor  (B, C_in, H, W)

        Returns
        -------
        Tensor  (B, C_out, H, W)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)

    def count_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
