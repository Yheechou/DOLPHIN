import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.optics import ASM_propagate

class ONN_Layer(nn.Module):
    def __init__(self, input_dim=None, diffraction_size=1024, z=0.2, pixelsize=8, lamda=532):
        super(ONN_Layer, self).__init__()
        self.input_size = 16
        self.diffraction_size = diffraction_size
        self.device = 'cuda'
        
        # input modulation phase
        self.modulated_phase = nn.Parameter(
            torch.rand(self.input_size, device=self.device) * torch.pi * 2,
            requires_grad=True,
        )
        # diffraction: trainable 1D phase mask followed by ASM propagation
        self.phase_mask = nn.Parameter(
            torch.rand(1, self.diffraction_size, device=self.device) * torch.pi * 2,
            requires_grad=True,
        )
        self.asm_propagate = ASM_propagate(
            device=self.device,
            input_size=[1, self.diffraction_size],
            z=z,
            pixelsize=pixelsize,
            lamda=lamda,
        )

    def _interpolate_complex_field(self, field, output_size):
        real = F.interpolate(
            field.real.unsqueeze(1),
            size=output_size,
            mode='linear',
            align_corners=False,
        ).squeeze(1)
        imag = F.interpolate(
            field.imag.unsqueeze(1),
            size=output_size,
            mode='linear',
            align_corners=False,
        ).squeeze(1)
        return torch.complex(real, imag)


    def forward(self, inputs_field):
        # 1st layer
        hidden = torch.complex(torch.cos(self.modulated_phase), torch.sin(self.modulated_phase))
        hidden = hidden * inputs_field

        hidden = self._interpolate_complex_field(hidden, self.diffraction_size)
        phase_mask = torch.complex(torch.cos(self.phase_mask), torch.sin(self.phase_mask))
        hidden = hidden * phase_mask
        hidden = self.asm_propagate(hidden.unsqueeze(1)).squeeze(1)
        hidden = self._interpolate_complex_field(hidden, self.input_size)

        # output layer
        outputs = torch.square(torch.abs(hidden))

        fields = hidden
        return outputs, fields
