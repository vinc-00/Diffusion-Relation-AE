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


def train_diffusion(v_epochs=400, v_lr=1e-4, v_patience=25, v_model_name='Last_versione_DAE.pt', v_samples_per_pair=400):
    full_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_dataset = MNISTTwoDigitDataset(
        mnist_data=train_subset,
        samples_per_pair=v_samples_per_pair,
        train=True,
        min_num=0,
        max_num=99,
        relations=[-1, 1, -12, 12, -51, 51]
    )

    val_dataset = MNISTTwoDigitDataset(
        mnist_data=val_subset,
        samples_per_pair=v_samples_per_pair,
        train=True,
        min_num=0,
        max_num=99,
        relations=[-1, 1, -12, 12, -51, 51]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=v_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=v_patience, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    train_losses = []
    train_noise_losses = []
    val_losses = []
    val_noise_losses = []

    for epoch in range(v_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        model.train()
        epoch_train_loss = 0.0
        epoch_train_noise_loss = 0.0
        start_time = time()

        for batch in train_loader:
            x = batch['condition'].to(device)
            relation = batch['relation'].to(device)
            target = batch['target'].to(device)
            is_valid = batch['is_valid'].to(device)

            t = torch.randint(0, T, (x.size(0),), device=device)

            x_t, noise = q_sample(target, t)

            pred_noise = model(x_t, x, t, relation)

            noise_loss = F.mse_loss(pred_noise, noise)

            sqrt_alpha_hat_t = torch.sqrt(alpha_hat[t])[:, None, None, None]
            sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
            pred_x0 = (x_t - sqrt_one_minus_alpha_hat_t * pred_noise) / sqrt_alpha_hat_t

            loss = noise_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * x.size(0)
            epoch_train_noise_loss += noise_loss.item() * x.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        epoch_train_noise_loss /= len(train_loader.dataset)

        train_losses.append(epoch_train_loss)
        train_noise_losses.append(epoch_train_noise_loss)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_noise_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['condition'].to(device)
                relation = batch['relation'].to(device)
                target = batch['target'].to(device)
                is_valid = batch['is_valid'].to(device)

                t = torch.randint(0, T, (x.size(0),), device=device)

                x_t, noise = q_sample(target, t)
                pred_noise = model(x_t, x, t, relation)

                noise_loss = F.mse_loss(pred_noise, noise)

                sqrt_alpha_hat_t = torch.sqrt(alpha_hat[t])[:, None, None, None]
                sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
                pred_x0 = (x_t - sqrt_one_minus_alpha_hat_t * pred_noise) / sqrt_alpha_hat_t

                loss = noise_loss

                epoch_val_loss += loss.item() * x.size(0)
                epoch_val_noise_loss += noise_loss.item() * x.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_noise_loss /= len(val_loader.dataset)

        val_losses.append(epoch_val_loss)
        val_noise_losses.append(epoch_val_noise_loss)

        scheduler.step(epoch_val_loss)

        elapsed = time() - start_time
        print(f"Epoch {epoch}/{epochs} - {elapsed:.1f}s")
        print(f"  Train Loss: {epoch_train_loss:.5f} (Noise: {epoch_train_noise_loss:.5f})")
        print(f"  Val Loss:   {epoch_val_loss:.5f} (Noise: {epoch_val_noise_loss:.5f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.7f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, v_model_name))
            print(f"  Saved new best model with val loss: {best_val_loss:.5f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                early_stop = True

        if epoch % 5 == 0 or epoch == epochs or early_stop:
            test_vis_dataset = MNISTTwoDigitDataset(
                mnist_data=test_dataset,
                samples_per_pair=4,
                train=False,
                min_num=0,
                max_num=99,
                relations=[-1, 1, -12, 12, -51, 51]
            )
            test_vis_loader = DataLoader(test_vis_dataset, batch_size=4, shuffle=True)
            show_generated(model, test_vis_loader, epoch)

    print("Training complete.")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('Training and Validation Total Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_noise_losses, label='Train Noise Loss', linestyle='--')
    plt.plot(val_noise_losses, label='Val Noise Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Component Loss')
    plt.title('Component Losses')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curves.png"))
    plt.show()
    return model
