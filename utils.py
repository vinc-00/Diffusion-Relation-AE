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
from Model import p_sample, q_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000  # diffusion steps
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0).to(device)
SAVE_DIR = "diffusion_model_weights"
os.makedirs(SAVE_DIR, exist_ok=True)

@torch.no_grad()
def show_generated(model, loader, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x, relation, target = next(iter(loader))
    x, relation = x[:4].to(device), relation[:4].to(device)

    # Generate images (32x64)
    gen = p_sample(model, x, relation, shape=(4, 1, 32, 64)).cpu().clamp(-1, 1)

    def process_image(tensor):
        tensor = tensor[:, :, 2:-2, 4:-4]
        return (tensor * 0.5) + 0.5

    x_processed = process_image(x.cpu())
    target_processed = process_image(target.cpu())
    gen_processed = process_image(gen)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(4):
        axes[0, i].imshow(x_processed[i].squeeze(), cmap='gray')

        axes[1, i].imshow(target_processed[i].squeeze(), cmap='gray')

        axes[2, i].imshow(gen_processed[i].squeeze(), cmap='gray')

        # Set titles
        if i == 0:
            axes[0, i].set_ylabel("Input", fontsize=12)
            axes[1, i].set_ylabel("Target", fontsize=12)
            axes[2, i].set_ylabel("Generated", fontsize=12)

        # Update relation labels for new types
        rel_type = ["predecessor", "successor", "+12", "-12"][relation[i].item()]
        axes[0, i].set_title(f"Relation: {rel_type}", fontsize=10)

    plt.suptitle(f"Epoch {epoch} - Generated Number Relations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"generated_epoch_{epoch}.png"))
    plt.show()

@torch.no_grad()
def generate_and_plot(model, number, relation, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Create the input image for the given number
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset for digit images
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

    # Concatenate images horizontally
    width = first_img.width + second_img.width
    height = max(first_img.height, second_img.height)
    input_img = first_img.copy()
    input_img = input_img.resize((width, height))
    input_img.paste(second_img, (first_img.width, 0))

    # Pad and normalize
    padded_img = transforms.Pad((16, 2, 16, 2))(input_img)
    input_tensor = transform(padded_img).unsqueeze(0).to(device)
    relation_tensor = torch.tensor([relation], device=device, dtype=torch.long)

    # Generate image
    generated = p_sample(
        model,
        input_tensor,
        relation_tensor,
        shape=(1, 1, 32, 64)
    ).cpu().clamp(-1, 1)

    def process_image(tensor):
        tensor = (tensor * 0.5) + 0.5
        return tensor[:, :, 4:-4, 2:-2].squeeze()

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