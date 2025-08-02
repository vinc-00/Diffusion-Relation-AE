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
from Train import train_diffusion
from Test import test_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000  # diffusion steps
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0).to(device)
SAVE_DIR = "diffusion_model_weights"
os.makedirs(SAVE_DIR, exist_ok=True)

train_diffusion(v_epochs=400, v_lr=1e-4, v_patience=25, v_model_name='Last_versione_DAE.pt')


