import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.model_single_layer_forward import ONN_Single

class ONN_multi(nn.Module):
    def __init__(self):
        super(ONN_multi, self).__init__()
        self.channels = 64
        self.voltages = nn.Parameter(torch.rand(4, 28), requires_grad=True)
        self.layer_num = 4
        self.layers_4 = nn.ModuleList(
            [ONN_Single() for _ in range(self.layer_num)]) 

    def forward(self, inputs):
        # inputs: (bs, self.channels)
        hidden = inputs
        for i in range(self.layer_num):
            hidden = self.layers_4[i](self.voltages[i], hidden)
            hidden = hidden / torch.abs(hidden).max(dim=1, keepdim=True)[0] ################

        output = torch.square(torch.abs(hidden)).squeeze()
        return output, hidden