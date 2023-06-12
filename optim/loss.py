import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, data_dict):
        return None