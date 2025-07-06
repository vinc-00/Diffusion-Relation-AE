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


class MNISTTwoDigitDataset(Dataset):
    def __init__(self, mnist_data, samples_per_pair=None, train=True):
        self.data = mnist_data
        self.train = train
        self.tensors = [transforms.ToTensor()(img) for img, _ in mnist_data]
        
        self.digit_to_indices = self._create_digit_index()
        self.numbers = list(range(100))

        self.samples_per_pair = samples_per_pair
        if samples_per_pair is not None:
            self.examples = []
            for number in self.numbers:
                for relation in [0, 1, 2, 3]:
                    if self.train and relation in [2, 3]:
                        if relation == 2 and number + 12 > 99:
                            continue
                        if relation == 3 and number - 12 < 0:
                            continue
                    for _ in range(samples_per_pair):
                        self.examples.append((number, relation))
            self.length = len(self.examples)
        else:
            self.examples = None
            self.length = 10000

    def _create_digit_index(self):
        index = {i: [] for i in range(10)}
        for idx, (_, digit) in enumerate(self.data):
            index[digit].append(idx)
        return index

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.examples is not None:
            number, relation = self.examples[idx]
        else:
            number = random.choice(self.numbers)
            relation = random.choice([0, 1, 2, 3])

        # Calculate target number based on relation
        if relation == 0:  # Predecessor (-1)
            target_number = (number - 1) % 100
        elif relation == 1:  # Successor (+1)
            target_number = (number + 1) % 100
        elif relation == 2:  # +12
            target_number = number + 12 if self.train else (number + 12) % 100
        elif relation == 3:  # -12
            target_number = number - 12 if self.train else (number - 12) % 100
        
        # Ensure target is within valid range [0, 99]
        target_number = target_number % 100

        # Split into digits
        first_digit = number // 10
        second_digit = number % 10
        target_first = target_number // 10
        target_second = target_number % 10

        # Get tensors directly
        first_idx = random.choice(self.digit_to_indices[first_digit])
        first_tensor = self.tensors[first_idx]
        
        second_idx = random.choice(self.digit_to_indices[second_digit])
        second_tensor = self.tensors[second_idx]
        
        target_first_idx = random.choice(self.digit_to_indices[target_first])
        target_first_tensor = self.tensors[target_first_idx]
        
        target_second_idx = random.choice(self.digit_to_indices[target_second])
        target_second_tensor = self.tensors[target_second_idx]

        condition_tensor = torch.cat([first_tensor, second_tensor], dim=2)
        condition_tensor = F.pad(condition_tensor, (4, 4, 2, 2), "constant", 0)
        condition_tensor = (condition_tensor - 0.5) / 0.5  # Normalize to [-1,1]

        target_tensor = torch.cat([target_first_tensor, target_second_tensor], dim=2)
        target_tensor = F.pad(target_tensor, (4, 4, 2, 2), "constant", 0)
        target_tensor = (target_tensor - 0.5) / 0.5  # Normalize to [-1,1]

        return condition_tensor, torch.tensor(relation, dtype=torch.long), target_tensor