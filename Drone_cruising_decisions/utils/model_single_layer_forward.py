import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.model_unit_forward import ONN_Layer

class ONN_Single(nn.Module):
    def __init__(self):
        super(ONN_Single, self).__init__()
        self.channels = 64
        self.channel_per_mode = 16
        self.device = 'cuda'
        self.layers = nn.ModuleList([
            ONN_Layer(input_dim=20), #1-1
            ONN_Layer(input_dim=24), #1-2
            ONN_Layer(input_dim=24), #1-3
            ONN_Layer(input_dim=20), #1-4
            ONN_Layer(input_dim=20), #1-5
            ONN_Layer(input_dim=24), #1-6
            ONN_Layer(input_dim=20)  #1-7
            ]) 
        

    def forward(self, voltages, inputs):
        voltage_1_1 = voltages[0     :4  + 1]
        voltage_1_2 = voltages[4 - 1 :8  + 1]
        voltage_1_3 = voltages[8 - 1 :12 + 1]
        voltage_1_4 = voltages[12 - 1:16    ]
        voltage_1_5 = voltages[16    :20 + 1]
        voltage_1_6 = voltages[20 - 1:24 + 1]
        voltage_1_7 = voltages[24 - 1:28    ]
        outputs_1_1, hidden_1_1 = self.layers[0](inputs[:, 0:16], voltage_1_1)
        _          , hidden_1_2 = self.layers[1](inputs[:,16:32], voltage_1_2)
        _          , hidden_1_3 = self.layers[2](inputs[:,32:48], voltage_1_3)
        outputs_1_4, hidden_1_4 = self.layers[3](inputs[:,48:64], voltage_1_4)
        hidden_1_5_in = torch.cat([hidden_1_1[:, 8:], hidden_1_2[:, 0:8]], dim=1)
        hidden_1_6_in = torch.cat([hidden_1_2[:, 8:], hidden_1_3[:, 0:8]], dim=1)
        hidden_1_7_in = torch.cat([hidden_1_3[:, 8:], hidden_1_4[:, 0:8]], dim=1)

        _, hidden_1_5_out = self.layers[4](hidden_1_5_in, voltage_1_5)
        _, hidden_1_6_out = self.layers[5](hidden_1_6_in, voltage_1_6)
        _, hidden_1_7_out = self.layers[6](hidden_1_7_in, voltage_1_7)
        outputs = torch.cat([hidden_1_1[:,0:8], hidden_1_5_out, hidden_1_6_out, hidden_1_7_out, hidden_1_4[:, 8:]], dim=1)
        return outputs