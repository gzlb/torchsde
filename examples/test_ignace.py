import logging
import os
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal

import torchsde


class SDE(torch.nn.Module):

    def __init__(self, a, b):
        super().__init__()
        self.noise_type="diagonal"
        self.sde_type = "ito"

        self.a = a
        self.b = b

    def f(self, t, y):
        x1, x2 = torch.split(y, split_size_or_sections=(1, 1), dim=1)
        a1, a2, a3 = self.a
        f1 = a1 * x1
        f2 = a2 * (a3 - x2)
        return torch.cat([f1, f2], dim=1)

    def g(self, t, y):
        x1, x2 = torch.split(y, split_size_or_sections=(1, 1), dim=1)
        b1 = self.b
        g1 = np.sqrt(x1)
        g2 = b1 * np.sqrt(x1)
        return torch.cat([g1, g2], dim=1)

batch_size, state_size, brownian_size = 32, 4, 3
y0 = torch.randn(2, 1)
print(y0)
ts = torch.tensor([0., 1.])
bm_vis = torchsde.BrownianInterval(
        t0=0, t1=1, size=(batch_size, state_size,), device='cpu', levy_area_approximation="space-time")
sde = SDE(a=(1,2,3),b=(2))
print('======')
ys = torchsde.sdeint(sde=sde, y0=y0, ts=ts)