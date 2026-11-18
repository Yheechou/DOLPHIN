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
        wavelength,
        pixel_size,
        prop_size,
        receive_size,
        propgate
        ):
        super(Decoder_ONN, self).__init__()
        self.device = 'cuda'
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.prop_size = prop_size
        self.receive_size = receive_size

        self.pad_cam_hl, self.pad_cam_hr, self.pad_cam_wl, self.pad_cam_wr = 500, 490, 420, 410
        self.pad_slm = (1760 - 1024) // 2
        self.propagate = propgate

        self.prop_decoder2receiver = ASM_propagate(z=d_decoder2receiver, device=self.device, input_size=[self.prop_size, self.prop_size], lamda=self.wavelength, pixelsize=self.pixel_size)
    

    def forward(self, inputs, decoder_phase, dis_num, receiver_phase=None):
        # object to decoder
        pad_xx = (self.prop_size - inputs.shape[1]) // 2
        pad_yy = (self.prop_size - inputs.shape[2]) // 2
        hidden = F.pad(inputs, (pad_yy, pad_yy, pad_xx, pad_xx), mode='constant', value=0)
        hidden = self.propagate[dis_num](hidden)

        # decoder to receiver
        pad = (self.prop_size - decoder_phase.shape[1]) // 2
        decoder_mask = F.pad(decoder_phase.cuda(), (self.pad_slm, self.pad_slm, self.pad_slm, self.pad_slm), mode='constant', value=0)
        decoder_mask = torch.complex(torch.cos(decoder_mask), torch.sin(decoder_mask))
        hidden = hidden * decoder_mask
        pad = (hidden.shape[1] - self.receive_size) // 2
        hidden = self.prop_decoder2receiver(hidden)[:, pad:-pad, pad:-pad]

        # receiver
        receiver_pattern = self.complex_interpolate(hidden.unsqueeze(1), size=8, mode='bilinear').squeeze(1)
        receiver_pattern = receiver_pattern / torch.amax(torch.abs(receiver_pattern), dim=(-2, -1), keepdim=True)
        output = receiver_pattern
        if receiver_phase is not None:
            receiver_phase = torch.complex(torch.cos(receiver_phase), torch.sin(receiver_phase)).reshape(-1, 8, 8)
            output = receiver_pattern * receiver_phase
        return output
    
    def complex_interpolate(self, input, size, mode):
        real_part = torch.real(input)
        imag_part = torch.imag(input)
        real_interp = F.interpolate(real_part, size=size, mode=mode)
        imag_interp = F.interpolate(imag_part, size=size, mode=mode)
        output = torch.complex(real_interp, imag_interp)
        return output
    

