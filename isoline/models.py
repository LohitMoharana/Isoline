"""Teacher (SE-ResNet1D, dense window) and silicon student (dilated DS-TCN).

Input scaling uses fixed power-of-two divisors only (no data-dependent
normalisation) so the same scaling is a plain right-shift in RTL.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIST_DIV = 16.0      # spike-count channels  (counts / 16)
ACC_DIV = 32.0       # leaky-integrator channel (int8 / 32)
SCAL_DIV = torch.tensor([32.0, 32.0, 32.0, 128.0, 128.0, 32.0, 32.0,   # rr, ewma, d_rr, spikes, spike_ewma, ratios
                         32.0, 32.0, 64.0, 64.0, 64.0, 128.0,         # prev_rr, ratio_prev, distances
                         32.0, 32.0, 32.0])                            # post_rr, post_ratio, post_pre_ratio (delayed mode only)
DENSE_DIV = 2.0      # mV -> network units for the teacher


def _scale3(a: np.ndarray):
    x = torch.from_numpy(a.astype(np.float32))
    return torch.cat([x[:, :2] / HIST_DIV, x[:, 2:] / ACC_DIV], dim=1)


def prep_spike(hist: np.ndarray, res: np.ndarray, scal: np.ndarray):
    x = torch.cat([_scale3(hist), _scale3(res)], dim=1)       # 6 channels
    s = torch.from_numpy(scal.astype(np.float32)) / SCAL_DIV[: scal.shape[1]]
    return x, s


def prep_dense(filt: np.ndarray):
    return torch.from_numpy(filt.astype(np.float32))[:, None, :] / DENSE_DIV


# ------------------------------------------------------------------ student
class DSBlock(nn.Module):
    """Dilated depthwise-separable residual block (BN folded at export)."""

    def __init__(self, c: int, dil: int):
        super().__init__()
        self.dw = nn.Conv1d(c, c, 3, padding=dil, dilation=dil, groups=c, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.dw(x)))
        y = self.bn2(self.pw(y))
        return F.relu(x + y)


class StudentSiliconTCN(nn.Module):
    def __init__(self, width: int = 40, dils=(1, 2, 4, 8), nscal: int = 13,
                 nscal_hidden: int = 16, ncls: int = 4):
        super().__init__()
        self.stem = nn.Conv1d(6, width, 1, bias=False)
        self.bn0 = nn.BatchNorm1d(width)
        self.blocks = nn.ModuleList(DSBlock(width, d) for d in dils)
        self.sfc = nn.Linear(nscal, nscal_hidden)
        self.head = nn.Linear(width + nscal_hidden, ncls)

    def forward(self, x, sc):
        h = F.relu(self.bn0(self.stem(x)))
        for b in self.blocks:
            h = b(h)
        h = h.mean(-1)                     # 32 positions -> sum then >> 5
        z = F.relu(self.sfc(sc))
        return self.head(torch.cat([h, z], 1))


# ------------------------------------------------------------------ teacher
class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(c // r, 4))
        self.fc2 = nn.Linear(max(c // r, 4), c)

    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(-1)))))
        return x * w.unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.se = SE(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm1d(cout))
                     if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.se(self.b2(self.c2(y)))
        return F.relu(y + self.skip(x))


class TeacherSEResNet1D(nn.Module):
    def __init__(self, nscal: int = 13, ch=(24, 48, 72, 96), ncls: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, ch[0], 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(ch[0]), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(ch[0], ch[0], 1), ResBlock(ch[0], ch[1], 2),
                                    ResBlock(ch[1], ch[2], 2), ResBlock(ch[2], ch[3], 2))
        self.sfc = nn.Linear(nscal, 32)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch[3] + 32, ncls))

    def forward(self, xd, sc):
        h = self.blocks(self.stem(xd)).mean(-1)
        return self.head(torch.cat([h, F.relu(self.sfc(sc))], 1))
