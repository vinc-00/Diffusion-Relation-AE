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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000  # diffusion steps
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0).to(device)
SAVE_DIR = "diffusion_model_weights"
os.makedirs(SAVE_DIR, exist_ok=True)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(shape, device=device)
    T = 1000  # diffusion steps
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0).to(device)
    for t_step in reversed(range(T)):
        t = torch.full((shape[0],), t_step, device=device, dtype=torch.long)
        z = torch.randn_like(x) if t_step > 0 else 0

        pred_noise = model(x, x_cond, t, relation)
        beta_t = beta[t][:, None, None, None]
        sqrt_alpha_t = torch.sqrt(alpha[t])[:, None, None, None]
        sqrt_alpha_hat_t = torch.sqrt(alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
        sigma_t = torch.sqrt(beta_t)

        x = (1 / sqrt_alpha_t) * (
            x - (beta_t / sqrt_one_minus_alpha_hat_t) * pred_noise
        ) + sigma_t * z

    return x



'''
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
'''


@torch.no_grad()
def show_generated(model, loader, epoch):
    model.eval()
    batch = next(iter(loader))

    x = batch['condition'][:4].to(device)
    relation = batch['relation'][:4].to(device)
    target = batch['target'][:4].to(device)
    is_valid = batch['is_valid'][:4].to(device)

    gen = p_sample(model, x, relation, shape=(4, 1, 32, 64)).cpu().clamp(-1, 1)

    def process_image(tensor):
        """Remove padding and normalize from [-1,1] to [0,1]"""
        tensor = tensor[:, :, 2:-2, 4:-4]  # Remove padding
        return (tensor * 0.5) + 0.5 

    x_processed = process_image(x.cpu())
    target_processed = process_image(target.cpu())
    gen_processed = process_image(gen)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    rel_names = {0: "predecessor (-1)",
                 1: "successor (+1)",
                 2: "-12",
                 3: "+12",
                 4: "-51",
                 5: "+51"}

    for i in range(4):
        axes[0, i].imshow(x_processed[i].squeeze(), cmap='gray', vmin=0, vmax=1)

        target_img = target_processed[i].squeeze()
        if not is_valid[i]:
            target_img = torch.zeros_like(target_img)
        axes[1, i].imshow(target_img, cmap='gray', vmin=0, vmax=1)

        axes[2, i].imshow(gen_processed[i].squeeze(), cmap='gray', vmin=0, vmax=1)

        if i == 0:
            axes[0, i].set_ylabel("Input", fontsize=12)
            axes[1, i].set_ylabel("Target", fontsize=12)
            axes[2, i].set_ylabel("Generated", fontsize=12)

        rel_type = rel_names[relation[i].item()]
        src_num = batch['src_num'][i].item()
        tgt_num = batch['tgt_num'][i].item() if is_valid[i] else "Invalid"
        axes[0, i].set_title(f"{src_num} → {tgt_num}\nRelation: {rel_type}", fontsize=10)

    plt.suptitle(f"Epoch {epoch} - Generated Number Relations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"generated_epoch_{epoch}.png"))
    plt.show()

@torch.no_grad()
def generate_and_plot(model, number, relation, device):
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True)
    digit_to_indices = {}
    for i in range(10):
        digit_to_indices[i] = [idx for idx, (_, digit) in enumerate(mnist_dataset) if digit == i]

    first_digit = number // 10
    first_idx = random.choice(digit_to_indices[first_digit])
    first_img, _ = mnist_dataset[first_idx]

    second_digit = number % 10
    second_idx = random.choice(digit_to_indices[second_digit])
    second_img, _ = mnist_dataset[second_idx]

    input_img = Image.new('L', (64, 32))

    first_pos = (16 - first_img.width // 2, 16 - first_img.height // 2)
    second_pos = (48 - second_img.width // 2, 16 - second_img.height // 2)

    input_img.paste(first_img, first_pos)
    input_img.paste(second_img, second_pos)

    input_tensor = transform(input_img).unsqueeze(0).to(device)
    relation_tensor = torch.tensor([relation], device=device, dtype=torch.long)

    generated = p_sample(
        model,
        input_tensor,
        relation_tensor,
        shape=(1, 1, 32, 64)
    ).cpu().clamp(-1, 1)

    def process_image(tensor):
        tensor = (tensor * 0.5) + 0.5
        return tensor.squeeze()

    input_display = process_image(input_tensor.cpu())
    generated_display = process_image(generated)

    if relation == 0:
        target_number = (number - 1) % 100
        rel_str = "predecessor"
    elif relation == 1:
        target_number = (number + 1) % 100
        rel_str = "successor"
    elif relation == 2:
        target_number = (number + 12) % 100
        rel_str = "+12"
    elif relation == 3:
        target_number = (number - 12) % 100
        rel_str = "-12"
    else:
        target_number = 0
        rel_str = 'Unknown'

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(input_display, cmap='gray')
    ax[0].set_title(f"Input: {number}")

    ax[1].imshow(generated_display, cmap='gray')
    title = f"Generated: {target_number} ({rel_str})"
    ax[1].set_title(title)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

    return generated_display
