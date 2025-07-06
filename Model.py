import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import os
from time import time
from utils import generate_and_plot, show_generated
from My_Dataset import MNISTTwoDigitDataset

# Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, 1)
        else:
            self.shortcut = nn.Identity()

        # Embedding projection
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_c)
        )

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Process embedding
        emb_out = self.emb_proj(emb)
        scale, shift = emb_out.chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim

        # Time and relation embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.relation_embed = nn.Embedding(4, emb_dim)


        self.enc1 = ResidualBlock(2, 64, emb_dim)
        self.enc2 = ResidualBlock(64, 128, emb_dim)
        self.enc3 = ResidualBlock(128, 256, emb_dim)
        self.enc4 = ResidualBlock(256, 512, emb_dim)
        self.mid = ResidualBlock(512, 512, emb_dim)

        # Decoder
        self.dec4 = ResidualBlock(512 + 512, 256, emb_dim)
        self.dec3 = ResidualBlock(256 + 256, 128, emb_dim)
        self.dec2 = ResidualBlock(128 + 128, 64, emb_dim)
        self.dec1 = ResidualBlock(64 + 64, 64, emb_dim)
        self.out_conv = nn.Conv2d(64, out_channels, 1)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))

    def forward(self, x, cond_img, t, relation):
        t_emb = self.time_mlp(t)
        rel_emb = self.relation_embed(relation)
        total_emb = t_emb + rel_emb

        # Combine input with condition image
        x = torch.cat([x, cond_img], dim=1)

        # Encoder with skip connections
        e1 = self.enc1(x, total_emb)        # 64x32
        e2 = self.enc2(self.down(e1), total_emb)  # 32x16
        e3 = self.enc3(self.down(e2), total_emb)  # 16x8
        e4 = self.enc4(self.down(e3), total_emb)  # 8x4


        e4_pooled = self.adaptive_pool(e4)  # Ensure 4x2
        m = self.mid(self.down(e4_pooled), total_emb)  # 4x2 -> 2x1


        m_up = F.interpolate(m, size=e4_pooled.size()[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([m_up, e4_pooled], dim=1), total_emb)

        d4_up = F.interpolate(d4, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d4_up, e3], dim=1), total_emb)

        d3_up = F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1), total_emb)
        d2_up = F.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1), total_emb)

        return self.out_conv(d1)


# Diffusion Functions
def q_sample(x_0, t, noise=None):
    """Forward diffusion process"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000  # diffusion steps
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0).to(device)
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_hat_t = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise, noise

@torch.no_grad()
def p_sample(model, x_cond, relation, shape):
    """Reverse diffusion sampling"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(shape, device=device)
    T = 1000  # diffusion steps
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0).to(device)
    for t_step in reversed(range(T)):
        t = torch.full((shape[0],), t_step, device=device, dtype=torch.long)
        z = torch.randn_like(x) if t_step > 0 else 0

        # Predict noise
        pred_noise = model(x, x_cond, t, relation)

        # Compute coefficients
        alpha_t = alpha[t][:, None, None, None]
        alpha_hat_t = alpha_hat[t][:, None, None, None]
        beta_t = beta[t][:, None, None, None]

        # Update x
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
        ) + torch.sqrt(beta_t) * z

    return x