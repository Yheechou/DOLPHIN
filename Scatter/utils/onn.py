import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.optics import ASM_propagate

class ONN(nn.Module):
    def __init__(self):
        super(ONN, self).__init__()
        self.input_size = 500
        self.pad = self.input_size // 2
        self.wavelength = 532
        self.pixel_size = 8
        self.device = 'cuda'

        self.propagate = {}
        for distance in range(35):
            self.propagate['{:.2f}'.format(0.01+0.01*distance)] = ASM_propagate(z=0.01+0.01*distance, device=self.device, input_size=[self.input_size + self.pad * 2, self.input_size + self.pad * 2], lamda=self.wavelength, pixelsize=self.pixel_size)
        self.propagate_decoder = ASM_propagate(z=0.10, device=self.device, input_size=[self.input_size + self.pad * 2, self.input_size + self.pad * 2], lamda=self.wavelength, pixelsize=self.pixel_size)
        self.encoder_mask = nn.Parameter(torch.rand(1, 500, 500, device=self.device) * torch.pi, requires_grad=True)
        self.decoder_mask = nn.Parameter(torch.rand(self.input_size, self.input_size, device=self.device) * torch.pi, requires_grad=True)

    def forward(self, inputs):
        objects, distance = inputs
        distance = int(distance)
        objects = F.interpolate(input=objects, size=self.input_size, mode='nearest').squeeze()

        rand_phase = torch.rand(objects.shape, device=self.device) * 1.0 * torch.pi
        rand_phase = F.conv2d(rand_phase.unsqueeze(1), weight=torch.ones(1, 1, 3, 3, device=self.device) / 9, padding=1).squeeze(1)
        rand_phase = torch.complex(torch.cos(rand_phase), torch.sin(rand_phase))

        # emitter
        encoder_mask = F.interpolate(self.encoder_mask.unsqueeze(1), size=self.input_size, mode='nearest').squeeze()
        encoder_mask = torch.complex(torch.cos(encoder_mask), torch.sin(encoder_mask))
        hidden = torch.ones(objects.shape, device=self.device) * encoder_mask
        hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        hidden = self.propagate['{:.2f}'.format(0.01*distance)](hidden)
        hidden = hidden[:, self.pad:-self.pad, self.pad:-self.pad]

        # object scatter & reflect
        hidden = hidden * objects * rand_phase
        hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        hidden = self.propagate['{:.2f}'.format(0.01*distance)](hidden)
        hidden = hidden[:, self.pad:-self.pad, self.pad:-self.pad]

        # decoder
        decoder_mask = torch.complex(torch.cos(self.decoder_mask), torch.sin(self.decoder_mask))
        hidden = hidden * decoder_mask
        hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        hidden = self.propagate_decoder(hidden)
        hidden = hidden[:, self.pad:-self.pad, self.pad:-self.pad]

        output = torch.square(torch.abs(hidden))
        measurement = output[:, 250, 12 + 25 * np.arange(20)]
        measurement = measurement / measurement.sum(dim=1, keepdim=True)
        measurement = torch.log(measurement)
        return measurement