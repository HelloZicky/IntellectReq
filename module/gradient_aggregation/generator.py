import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader


class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.gen_func = nn.Tanh()
        self.gen_layer1 = nn.Linear(input_size, input_size // 2)
        self.gen_layer2 = nn.Linear(input_size // 2, input_size)

    def forward(self, feature):
        feature = self.gen_func(self.gen_layer1(feature))
        noisy_feature = self.gen_func(self.gen_layer2(feature))
        return noisy_feature
