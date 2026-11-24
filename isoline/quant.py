"""BatchNorm folding and power-of-two int8 quantisation-aware training.

Weights and activations use symmetric per-tensor int8 with power-of-two
scales, so every requantisation in RTL is a pure arithmetic shift. Biases stay
high precision here (they become int32 at export). Export to .hex plus an
integer-only reference forward is the next step once accuracy gates are met.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import StudentSiliconTCN


def fold_bn(w: torch.Tensor, bn: nn.BatchNorm1d):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return w * scale.view(-1, 1, 1), bn.bias - bn.running_mean * scale


def fq_weight(w: torch.Tensor) -> torch.Tensor:
    amax = w.detach().abs().max().clamp(min=1e-8)
    scale = 2.0 ** torch.ceil(torch.log2(amax / 127.0))
    q = torch.clamp(torch.round(w / scale), -128, 127) * scale
    return w + (q - w).detach()                      # straight-through estimator


class ActFQ(nn.Module):
    """int8 activation fake-quant with a power-of-two scale from a running max."""

    def __init__(self, momentum: float = 0.01):
        super().__init__()
        self.register_buffer("absmax", torch.zeros(()))
        self.momentum = momentum
        self.enabled = False

    def forward(self, x):
        if self.training:
            m = x.detach().abs().max()
            self.absmax.copy_(m if float(self.absmax) == 0.0
                              else (1 - self.momentum) * self.absmax + self.momentum * m)
        if not self.enabled or float(self.absmax) == 0.0:
            return x
        scale = 2.0 ** torch.ceil(torch.log2(self.absmax / 127.0))
        q = torch.clamp(torch.round(x / scale), -128, 127) * scale
        return x + (q - x).detach()


def _conv_with_bias(conv: nn.Conv1d, w, b) -> nn.Conv1d:
    c = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size[0],
                  conv.stride[0], conv.padding[0], conv.dilation[0], conv.groups, bias=True)
    c.weight = nn.Parameter(w.detach().clone())
    c.bias = nn.Parameter(b.detach().clone())
    return c


class FoldedStudent(nn.Module):
    """Student with BN folded into the convolutions; optional fake-quant."""

    def __init__(self, st: StudentSiliconTCN):
        super().__init__()
        self.qat = False
        self.stem = _conv_with_bias(st.stem, *fold_bn(st.stem.weight, st.bn0))
        self.dw = nn.ModuleList(_conv_with_bias(b.dw, *fold_bn(b.dw.weight, b.bn1)) for b in st.blocks)
        self.pw = nn.ModuleList(_conv_with_bias(b.pw, *fold_bn(b.pw.weight, b.bn2)) for b in st.blocks)
        self.sfc = copy.deepcopy(st.sfc)
        self.head = copy.deepcopy(st.head)
        n = len(st.blocks)
        self.fq_in, self.fq_stem, self.fq_pool = ActFQ(), ActFQ(), ActFQ()
        self.fq_s_in, self.fq_s = ActFQ(), ActFQ()
        self.fq_dw = nn.ModuleList(ActFQ() for _ in range(n))
        self.fq_blk = nn.ModuleList(ActFQ() for _ in range(n))

    def set_qat(self, on: bool):
        self.qat = on
        for m in self.modules():
            if isinstance(m, ActFQ):
                m.enabled = on

    def _conv(self, conv, x):
        w = fq_weight(conv.weight) if self.qat else conv.weight
        return F.conv1d(x, w, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)

    def _lin(self, lin, x):
        w = fq_weight(lin.weight) if self.qat else lin.weight
        return F.linear(x, w, lin.bias)

    def forward(self, x, sc):
        x = self.fq_in(x)
        h = self.fq_stem(F.relu(self._conv(self.stem, x)))
        for i in range(len(self.dw)):
            y = self.fq_dw[i](F.relu(self._conv(self.dw[i], h)))
            y = self._conv(self.pw[i], y)
            h = self.fq_blk[i](F.relu(h + y))
        h = self.fq_pool(h.mean(-1))
        s = self.fq_s(F.relu(self._lin(self.sfc, self.fq_s_in(sc))))
        return self._lin(self.head, torch.cat([h, s], 1))

    def footprint_bytes(self) -> int:
        """int8 weights + int32 biases of every conv / linear layer."""
        total = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                total += m.weight.numel() + 4 * m.bias.numel()
        return total
