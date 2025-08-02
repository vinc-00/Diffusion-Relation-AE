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
from utils import generate_and_plot, show_generated, q_sample, p_sample
from My_Dataset import MNISTTwoDigitDataset
from Model import UNet, ResidualBlock, TimeEmbedding

@torch.no_grad()
def test_model(model, test_loader, alpha_hybrid=0.8):
    model.eval()
    total_loss = 0.0
    total_noise_loss = 0.0
    total_psnr = 0.0
    num_samples = 0

    print("\nTesting model on test set...")
    for batch_idx, (x, relation, target) in enumerate(test_loader):
        x, relation, target = x.to(device), relation.to(device), target.to(device)
        batch_size = x.size(0)

        timesteps = torch.tensor([0, 250, 500, 750, 999], device=device).repeat(batch_size, 1).T.flatten()
        batch_rep = x.repeat(5, 1, 1, 1)
        relation_rep = relation.repeat(5)
        target_rep = target.repeat(5, 1, 1, 1)

        x_t, noise = q_sample(target_rep, timesteps)

        pred_noise = model(x_t, batch_rep, timesteps, relation_rep)

        sqrt_alpha_hat_t = torch.sqrt(alpha_hat[timesteps])[:, None, None, None]
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat[timesteps])[:, None, None, None]
        pred_x0 = (x_t - sqrt_one_minus_alpha_hat_t * pred_noise) / sqrt_alpha_hat_t

        noise_loss = F.mse_loss(pred_noise, noise, reduction='none')
        noise_loss = noise_loss.view(5, batch_size, *noise_loss.shape[1:]).mean(dim=[2, 3, 4])


        loss = noise_loss

        total_loss += loss.sum().item()
        total_noise_loss += noise_loss.sum().item()

        if 0 in timesteps:
            zero_idx = (timesteps == 0)
            zero_x = batch_rep[zero_idx][:batch_size]
            zero_relation = relation_rep[zero_idx][:batch_size]

            generated = p_sample(model, zero_x, zero_relation, shape=(batch_size, 1, 32, 64))


            mse = F.mse_loss(generated, target, reduction='none')
            mse = mse.view(batch_size, -1).mean(dim=1)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.sum().item()
            num_samples += batch_size

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx+1} batches...")

    avg_loss = total_loss / (len(test_loader.dataset) * 5)
    avg_noise_loss = total_noise_loss / (len(test_loader.dataset) * 5)
    avg_psnr = total_psnr / num_samples

    print("\nTest Results:")
    print(f"  Total Loss: {avg_loss:.5f}")
    print(f"  Noise Loss: {avg_noise_loss:.5f}")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")

    show_generated(model, test_loader, "final_test")