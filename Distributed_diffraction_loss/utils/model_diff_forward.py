import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.optics import ASM_propagate


class ONN_single_Layer(nn.Module):
    def __init__(self, propagation_size, z=0.2, pixelsize=8, lamda=532, energy_decay_db=0.0):
        super(ONN_single_Layer, self).__init__()
        self.propagation_size = propagation_size
        self.device = "cuda"
        self.register_buffer(
            "energy_decay_power",
            torch.tensor(10 ** (-energy_decay_db / 10), dtype=torch.float32),
        )

        self.phase_mask = nn.Parameter(
            torch.rand(1, self.propagation_size, device=self.device) * torch.pi * 2,
            requires_grad=True,
        )
        self.asm_propagate = ASM_propagate(
            device=self.device,
            input_size=[1, self.propagation_size],
            z=z,
            pixelsize=pixelsize,
            lamda=lamda,
        )

    def _total_energy(self, field):
        return torch.sum(torch.square(torch.abs(field)), dim=1, keepdim=True)

    def _apply_energy_decay(self, inputs_field, outputs_field, eps=1e-12):
        input_energy = self._total_energy(inputs_field)
        output_energy = self._total_energy(outputs_field)
        target_energy = input_energy * self.energy_decay_power
        scale = torch.sqrt(target_energy / (output_energy + eps))
        return outputs_field * scale

    def forward(self, inputs_field):
        input_field = inputs_field
        phase_mask = torch.complex(torch.cos(self.phase_mask), torch.sin(self.phase_mask))
        hidden = inputs_field * phase_mask
        hidden = self.asm_propagate(hidden.unsqueeze(1)).squeeze(1)
        hidden = self._apply_energy_decay(input_field, hidden)
        return hidden


class ONN_diff(nn.Module):
    def __init__(self, layer_num=4, diffraction_size=640, z=0.2, pixelsize=8, lamda=532, energy_decay_db=0.0):
        super(ONN_diff, self).__init__()
        self.channels = 64
        self.layer_num = layer_num
        self.propagation_size = diffraction_size
        self.layers = nn.ModuleList([
            ONN_single_Layer(
                propagation_size=self.propagation_size,
                z=z,
                pixelsize=pixelsize,
                lamda=lamda,
                energy_decay_db=energy_decay_db,
            )
            for _ in range(self.layer_num)
        ])

    def _to_complex(self, field):
        if torch.is_complex(field):
            return field
        return torch.complex(field, torch.zeros_like(field))

    def _interpolate_complex_field(self, field, output_size):
        field = self._to_complex(field)
        real = F.interpolate(
            field.real.unsqueeze(1),
            size=output_size,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        imag = F.interpolate(
            field.imag.unsqueeze(1),
            size=output_size,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return torch.complex(real, imag)

    def forward(self, inputs):
        hidden = self._interpolate_complex_field(inputs, self.propagation_size)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self._interpolate_complex_field(hidden, self.channels)
        output = torch.square(torch.abs(hidden)).squeeze()
        return output
