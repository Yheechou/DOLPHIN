import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import matplotlib.pyplot as plt
import numpy as np
from utils.optics import ASM_propagate

class Decoder_ONN(nn.Module):
    def __init__(self,
        d_decoder2receiver, 
        d_object2decoder,
        wavelength,
        pixel_size,
        prop_size,
        receive_size
        ):
        super(Decoder_ONN, self).__init__()
        self.device = 'cuda'
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.prop_size = prop_size
        self.receive_size = receive_size

        self.prop_decoder2receiver = ASM_propagate(z=d_decoder2receiver, device=self.device, input_size=[self.prop_size, self.prop_size], lamda=self.wavelength, pixelsize=self.pixel_size)
        self.prop_object2decoder = ASM_propagate(z=d_object2decoder, device=self.device, input_size=[self.prop_size, self.prop_size], lamda=self.wavelength, pixelsize=self.pixel_size)

    def forward(self, inputs, decoder_phase, distance=None):
        if distance is not None:
            self.prop_object2decoder = ASM_propagate(z=distance, device=self.device, input_size=[self.prop_size, self.prop_size], lamda=self.wavelength, pixelsize=self.pixel_size)
        # object to decoder
        pad_xx = (self.prop_size - inputs.shape[1]) // 2
        pad_yy = (self.prop_size - inputs.shape[2]) // 2
        if pad_xx < 0 or pad_yy < 0:
            raise ValueError(
                f"prop_size ({self.prop_size}) must be >= input field size "
                f"({inputs.shape[1]}, {inputs.shape[2]})."
            )
        hidden = F.pad(inputs, (pad_yy, pad_yy, pad_xx, pad_xx), mode='constant', value=0)
        hidden = self.prop_object2decoder(hidden)

        # decoder to receiver
        pad = (self.prop_size - decoder_phase.shape[1]) // 2
        if pad < 0:
            raise ValueError(
                f"prop_size ({self.prop_size}) must be >= decoder mask size ({decoder_phase.shape[1]})."
            )
        decoder_mask = F.pad(decoder_phase.to(inputs.device), (pad, pad, pad, pad), mode='constant', value=0)
        decoder_mask = torch.complex(torch.cos(decoder_mask), torch.sin(decoder_mask))
        hidden = hidden * decoder_mask
        pad = (hidden.shape[1] - self.receive_size) // 2
        hidden = self.prop_decoder2receiver(hidden)[:, pad:-pad, pad:-pad]
        return hidden
    
