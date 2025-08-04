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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000  # diffusion steps
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0).to(device)
SAVE_DIR = "diffusion_model_weights"
os.makedirs(SAVE_DIR, exist_ok=True)


class MNISTTwoDigitDataset(Dataset):
    def __init__(self, mnist_data, samples_per_pair=None, train=True, min_num=0, max_num=99, relations=[-1, 1, -12, 12]):
        self.data = mnist_data
        self.train = train
        self.samples_per_pair = samples_per_pair
        self.min_num = min_num
        self.max_num = max_num

        self.digit_to_indices = self._create_digit_index()

        self.relation_map = {-1: 0, 1: 1, -12: 2, 12: 3, -51: 4, 51: 5}
        self.relations = relations

        self._create_pairs()

        self.tensors = [transforms.ToTensor()(img) for img, _ in mnist_data]

        self.black_image = torch.zeros(1, 28, 28)

    def _create_digit_index(self):
        index = {i: [] for i in range(10)}
        for idx, (_, digit) in enumerate(self.data):
            index[digit].append(idx)
        return index

    def _create_pairs(self):
        self.numbers = list(range(self.min_num, self.max_num + 1))
        self.pairs = []
        self.relation_labels = []
        self.actual_relations = []
        self.is_valid_target = [] 

        if self.samples_per_pair is not None:
            self.examples = []
            for number in self.numbers:
                for rel in self.relations:
                    relation_label = self.relation_map[rel]
                    target_num = number + rel

                    is_valid = (0 <= target_num <= 99 and
                               self._has_valid_digits(number) and
                               self._has_valid_digits(target_num))
                    
                    for _ in range(self.samples_per_pair):
                        self.examples.append((number, relation_label))
                        self.pairs.append((number, target_num))
                        self.relation_labels.append(relation_label)
                        self.actual_relations.append(rel)
                        self.is_valid_target.append(is_valid)
            self.length = len(self.examples)
        else:
            self.examples = None
            for number in self.numbers:
                for rel in self.relations:
                    target_num = number + rel
                    relation_label = self.relation_map[rel]

                    is_valid = (0 <= target_num <= 99 and
                               self._has_valid_digits(number) and
                               self._has_valid_digits(target_num))

                    self.pairs.append((number, target_num))
                    self.relation_labels.append(relation_label)
                    self.actual_relations.append(rel)
                    self.is_valid_target.append(is_valid)
            self.length = len(self.pairs)

    def _has_valid_digits(self, number):
        if number < 0 or number > 99: 
            return False
        tens = number // 10
        units = number % 10
        return tens in self.digit_to_indices and units in self.digit_to_indices

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.examples is not None:
            number, relation = self.examples[idx]
            target_number = self.pairs[idx][1]  
            is_valid = self.is_valid_target[idx]
        else:
            number, target_number = self.pairs[idx]
            relation = self.relation_labels[idx]
            is_valid = self.is_valid_target[idx]

        first_digit = number // 10
        second_digit = number % 10

        first_idx = random.choice(self.digit_to_indices[first_digit])
        first_tensor = self.tensors[first_idx]

        second_idx = random.choice(self.digit_to_indices[second_digit])
        second_tensor = self.tensors[second_idx]

        condition_tensor = torch.cat([first_tensor, second_tensor], dim=2)
        condition_tensor = F.pad(condition_tensor, (4, 4, 2, 2), "constant", 0)
        condition_tensor = (condition_tensor - 0.5) / 0.5  

        if is_valid:
            target_first = target_number // 10
            target_second = target_number % 10

            target_first_idx = random.choice(self.digit_to_indices[target_first])
            target_first_tensor = self.tensors[target_first_idx]

            target_second_idx = random.choice(self.digit_to_indices[target_second])
            target_second_tensor = self.tensors[target_second_idx]

            target_tensor = torch.cat([target_first_tensor, target_second_tensor], dim=2)
            tgt_img = torch.cat([target_first_tensor, target_second_tensor], dim=2)
        else:
            target_tensor = torch.cat([self.black_image, self.black_image], dim=2)
            tgt_img = torch.cat([self.black_image, self.black_image], dim=2)

        target_tensor = F.pad(target_tensor, (4, 4, 2, 2), "constant", 0)
        target_tensor = (target_tensor - 0.5) / 0.5 

        src_img = torch.cat([first_tensor, second_tensor], dim=2)

        return {
            'condition': condition_tensor,
            'relation': torch.tensor(relation, dtype=torch.long),
            'target': target_tensor,
            'src_img': src_img.unsqueeze(0),
            'tgt_img': tgt_img.unsqueeze(0),
            'src_num': number,
            'tgt_num': target_number if is_valid else -1,
            'actual_rel': self.actual_relations[idx],
            'is_valid': is_valid
        }