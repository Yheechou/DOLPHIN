import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ONN_Layer(nn.Module):
    def __init__(self, input_dim):
        super(ONN_Layer, self).__init__()
        self.input_size = 16
        self.device = 'cuda'
        
        # model: a * sin(b * v^2 + e * v + c)
        self.input_pha = nn.Parameter(torch.rand(self.input_size, device=self.device) * torch.pi * 2, requires_grad=False)
        self.modulate_b = nn.Parameter(torch.ones(self.input_size, device=self.device) * 0.7, requires_grad=False)
        self.input_dim = input_dim
        self.modulate_fc = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
        )
        for param in self.modulate_fc.parameters():
            param.requires_grad = False
        # diffraction: linear
        self.diff_real = nn.Parameter(torch.rand(self.input_size, self.input_size, device=self.device) / self.input_size, requires_grad=False)
        self.diff_imag = nn.Parameter(torch.rand(self.input_size, self.input_size, device=self.device) / self.input_size, requires_grad=False)
        self.energy_decay = nn.Parameter(torch.ones(self.input_size, device=self.device), requires_grad=False)

        # # output
        self.output_fc = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        ) 
        for param in self.output_fc.parameters():
            param.requires_grad = False
 

    def forward(self, inputs_field, voltages):
        # modulate voltage (V)
        voltages = voltages * 1e-3
        voltages = F.interpolate(voltages.unsqueeze(0).unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)
        voltages_inputs = voltages ** 2

        # 1st layer
        modulated_phase = self.modulate_b * self.modulate_fc(voltages_inputs) + self.input_pha
        hidden = torch.complex(torch.cos(modulated_phase), torch.sin(modulated_phase))
        hidden = hidden * inputs_field

        diff_matrix = torch.complex(self.diff_real, self.diff_imag)
        hidden = hidden @ diff_matrix.T
        hidden = hidden * self.energy_decay

        # output layer
        outputs = torch.square(torch.abs(hidden))

        fields = hidden
        return outputs, fields

