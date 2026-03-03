"""2D Fourier Neural Operator (FNO) surrogate model.

Implements the FNO architecture from Li et al. (2021) adapted for
next-step prediction of Hasegawa-Wakatani turbulence fields on a
doubly periodic 2D domain.

Architecture
------------
1. **Lifting** : pointwise linear  (C_in -> width)
2. **Fourier layers** x N : spectral convolution + pointwise linear + GeLU
3. **Projection** : pointwise linear  (width -> C_out)

Tensor conventions
------------------
- Input  : ``(B, C_in, H, W)``  -- e.g. C_in=2 for ``[n, phi]``
- Output : ``(B, C_out, H, W)`` -- e.g. C_out=2 for ``[n, phi]``

References
----------
Li, Z. et al. (2021). "Fourier Neural Operator for Parametric Partial
Differential Equations." ICLR 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral convolution layer
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """2D Fourier-space convolution (complex multiply in truncated modes).

    Learns a complex weight tensor of shape ``(in_ch, out_ch, modes_x, modes_y)``
    and multiplies it with the truncated FFT of the input.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    # Complex multiply-sum over in_channels
    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # inp: (B, C_in, kx, ky), weights: (C_in, C_out, kx, ky)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: FFT -> truncate -> multiply -> pad -> IFFT."""
        B, _C, H, W = x.shape
        # Real FFT (last dim halved)
        x_ft = torch.fft.rfft2(x)

        # Allocate output in Fourier space
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )

        # Positive kx modes
        out_ft[:, :, : self.modes_x, : self.modes_y] = self._compl_mul2d(
            x_ft[:, :, : self.modes_x, : self.modes_y], self.weights1
        )
        # Negative kx modes (wrap from end)
        out_ft[:, :, -self.modes_x :, : self.modes_y] = self._compl_mul2d(
            x_ft[:, :, -self.modes_x :, : self.modes_y], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(H, W))


# ---------------------------------------------------------------------------
# Single Fourier layer
# ---------------------------------------------------------------------------


class FourierLayer(nn.Module):
    """One FNO block: spectral conv + pointwise linear + residual + activation."""

    def __init__(self, width: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.linear = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.linear(x))


# ---------------------------------------------------------------------------
# Full FNO 2D model
# ---------------------------------------------------------------------------


class FNO2d(nn.Module):
    """2D Fourier Neural Operator for next-step field prediction.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 2: n, phi).
    out_channels : int
        Number of output channels (default 2: n, phi).
    modes : int
        Number of Fourier modes to keep per spatial dimension (default 12).
    width : int
        Hidden channel width (default 32).
    n_layers : int
        Number of Fourier layers (default 4).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # Lifting
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        # Fourier layers
        self.layers = nn.ModuleList([FourierLayer(width, modes, modes) for _ in range(n_layers)])

        # Projection (two-stage for expressivity)
        self.proj1 = nn.Conv2d(width, width, kernel_size=1)
        self.proj2 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor  (B, C_in, H, W)

        Returns
        -------
        Tensor  (B, C_out, H, W)
        """
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        x = F.gelu(self.proj1(x))
        return self.proj2(x)

    def count_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
