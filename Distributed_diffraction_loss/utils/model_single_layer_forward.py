import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.model_unit_forward import ONN_Layer
from utils.optics import ASM_propagate

class ONN_Single(nn.Module):
    def __init__(self, diffraction_size=1024, z=0.2, pixelsize=8, lamda=532):
        super(ONN_Single, self).__init__()
        self.channels = 64
        self.channel_per_mode = 16
        self.device = 'cuda'
        self.layers = nn.ModuleList([
            ONN_Layer(diffraction_size=diffraction_size, z=z, pixelsize=pixelsize, lamda=lamda), #1-1
            ONN_Layer(diffraction_size=diffraction_size, z=z, pixelsize=pixelsize, lamda=lamda), #1-2
            ONN_Layer(diffraction_size=diffraction_size, z=z, pixelsize=pixelsize, lamda=lamda), #1-3
            ONN_Layer(diffraction_size=diffraction_size, z=z, pixelsize=pixelsize, lamda=lamda), #1-4
            ONN_Layer(diffraction_size=diffraction_size, z=z, pixelsize=pixelsize, lamda=lamda), #1-5
            ONN_Layer(diffraction_size=diffraction_size, z=z, pixelsize=pixelsize, lamda=lamda), #1-6
            ONN_Layer(diffraction_size=diffraction_size, z=z, pixelsize=pixelsize, lamda=lamda)  #1-7
            ]) 
        

    def forward(self, inputs):
        outputs_1_1, hidden_1_1 = self.layers[0](inputs[:, 0:16])
        _          , hidden_1_2 = self.layers[1](inputs[:,16:32])
        _          , hidden_1_3 = self.layers[2](inputs[:,32:48])
        outputs_1_4, hidden_1_4 = self.layers[3](inputs[:,48:64])
        hidden_1_5_in = torch.cat([hidden_1_1[:, 8:], hidden_1_2[:, 0:8]], dim=1)
        hidden_1_6_in = torch.cat([hidden_1_2[:, 8:], hidden_1_3[:, 0:8]], dim=1)
        hidden_1_7_in = torch.cat([hidden_1_3[:, 8:], hidden_1_4[:, 0:8]], dim=1)

        _, hidden_1_5_out = self.layers[4](hidden_1_5_in)
        _, hidden_1_6_out = self.layers[5](hidden_1_6_in)
        _, hidden_1_7_out = self.layers[6](hidden_1_7_in)
        outputs = torch.cat([hidden_1_1[:,0:8], hidden_1_5_out, hidden_1_6_out, hidden_1_7_out, hidden_1_4[:, 8:]], dim=1)
        return outputs
