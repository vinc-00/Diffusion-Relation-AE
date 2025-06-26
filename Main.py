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
import utils
import My_Dataset
import Model
import Train
import Test


test_dataset = datasets.MNIST(root='./data', train=False, download=True)
test_relation_dataset = MNISTTwoDigitDataset(test_dataset, train=False)
test_loader = DataLoader(test_relation_dataset, batch_size=32, shuffle=False)

train_diffusion(epochs=40, patience=10)

model = UNet().to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
model.eval()

test_model(model, test_loader)

print("\nGenerating custom examples...")

print("Generating successor of 42...")
generate_and_plot(model, 42, 1, device)
print("Generating predecessor of 42...")
generate_and_plot(model, 42, 0, device)

print("Generating +12 for 42...")
generate_and_plot(model, 42, 2, device)
print("Generating -12 for 42...")
generate_and_plot(model, 42, 3, device)

print("Generating +12 for 90 (should wrap to 2)...")
generate_and_plot(model, 90, 2, device)
print("Generating -12 for 5 (should wrap to 93)...")
generate_and_plot(model, 5, 3, device)