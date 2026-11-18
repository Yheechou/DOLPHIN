import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.model_single_layer_forward import ONN_Single
from utils.optics import ASM_propagate

class ONN_multi(nn.Module):
    def __init__(self, layer_num=4, diffraction_size=1024, z=0.2, pixelsize=8, lamda=532, energy_decay_db=0.0):
        super(ONN_multi, self).__init__()
        self.channels = 64
        self.layer_num = layer_num
        self.register_buffer(
            "energy_decay_power",
            torch.tensor(10 ** (-energy_decay_db / 10), dtype=torch.float32),
        )
        self.layers_4 = nn.ModuleList(
            [
                ONN_Single(
                    diffraction_size=diffraction_size,
                    z=z,
                    pixelsize=pixelsize,
                    lamda=lamda,
                )
                for _ in range(self.layer_num)
            ]) 

    def _total_energy(self, field):
        return torch.sum(torch.square(torch.abs(field)), dim=1, keepdim=True)

    def _apply_energy_decay(self, inputs_field, outputs_field, eps=1e-12):
        input_energy = self._total_energy(inputs_field)
        output_energy = self._total_energy(outputs_field)
        target_energy = input_energy * self.energy_decay_power
        scale = torch.sqrt(target_energy / (output_energy + eps))
        return outputs_field * scale

    def forward(self, inputs):
        # inputs: (bs, self.channels)
        hidden = inputs
        for i in range(self.layer_num):
            input_field = hidden
            hidden = self.layers_4[i](hidden)
            hidden = self._apply_energy_decay(input_field, hidden)
            
        output = torch.square(torch.abs(hidden)).squeeze()
        return output
