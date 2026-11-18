import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.optics import ASM_propagate
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

count = 0

class Oixel_8by8(nn.Module):
    #8*8滑窗实现的“准卷积”，无下采样，后续用maxpool实现下采样
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        self.UnFold = nn.Unfold(kernel_size=8, stride=8, padding=0)
        self.device = 'cuda'

        self.w_mzi = nn.Parameter(torch.randn(self.ch_out, 64, 64))
        self.b_mzi = nn.Parameter(torch.randn(self.ch_out, 1, 64))
        
    def forward(self, x): # (batch, ch_in, H, W) 
        batch, ch, H, W = x.shape
        x = self.conv(x) # (batch, ch_out, H, W) 
        x = self.UnFold(x) # (batch, ch_out*8*8, H/8*W/8)
        x = torch.permute(x, [0, 2, 1]).reshape(batch*H//8*W//8, self.ch_out, 8*8) # (batch*H/8*W/8, ch_out, 8*8)
        # MZI fully connected 

        x = (torch.matmul(x[..., :, None, :], self.w_mzi) + self.b_mzi).squeeze()
        x = torch.permute(x.reshape(batch, H//8, W//8, self.ch_out, 8, 8), [0, 3, 1, 4, 2, 5])# (batch, ch, H//8, 8, W//8, 8)
        x = x.reshape(-1, self.ch_out, H, W)# (batch, ch_out, H, W) 
        return x

class ONN_deep_fc(nn.Module):
    def __init__(self):
        super(ONN_deep_fc, self).__init__()
        self.partfc_layers = nn.ModuleList([
            Oixel_8by8(ch_in=1, ch_out=8),
            Oixel_8by8(ch_in=8, ch_out=16),
            Oixel_8by8(ch_in=16, ch_out=32),
        ])
        self.conv1by1 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.output_layers = nn.Sequential(
            nn.Linear(16*16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        #input:(b, 128, 128)
        x = self.partfc_layers[0](x) 
        x = self.pool(F.relu(x)) #8*64*64
        x = self.partfc_layers[1](x) 
        x = self.pool(F.relu(x)) #16*32*32
        x = self.partfc_layers[2](x) 
        x = self.pool(F.relu(x)) #32*16*16
        x = self.conv1by1(x) #1*16*16
        x = self.output_layers(x.view(-1,16*16))

        return x

class ONN_2class(nn.Module):
    def __init__(self, device='cuda'):
        super(ONN_2class, self).__init__()
        self.mask_size = 508
        self.pad = self.mask_size
        self.wavelength = 1550
        self.pixel_size = 2
        self.device = device
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Linear((8 // 8) * (8 // 8) * 32, 2)
        self.fc2= nn.Linear(128, 2)
        self.part_fclayers = ONN_deep_fc()

        self.propagate_fix = ASM_propagate(z=0.01, device=self.device, input_size=[self.mask_size + self.pad * 2, self.mask_size + self.pad * 2], lamda=self.wavelength, pixelsize=self.pixel_size)
        self.propagate_decoder = ASM_propagate(z=0.01, device=self.device, input_size=[self.mask_size + self.pad * 2, self.mask_size + self.pad * 2], lamda=self.wavelength, pixelsize=self.pixel_size)
        self.propagate_encoder = ASM_propagate(z=0.01, device=self.device, input_size=[self.mask_size + self.pad * 2, self.mask_size + self.pad * 2], lamda=self.wavelength, pixelsize=self.pixel_size)
        self.encoder_mask = nn.Parameter(torch.rand(1, 1, self.mask_size, self.mask_size, device=self.device) * torch.pi, requires_grad=True)
        self.decoder_mask = nn.Parameter(torch.rand(1, 1, self.mask_size, self.mask_size, device=self.device) * torch.pi, requires_grad=True)
        self.emission_pattern = nn.Parameter(torch.rand(1, 1, 8, 8, device=self.device) * torch.pi, requires_grad=True)
        self.fc_layer = nn.Linear(in_features=64, out_features=20)

    def forward(self, inputs):
        objects = inputs
        scenes = F.interpolate(input=objects, size=self.mask_size, mode='nearest')

        # # scatter
        
        emission_pattern = torch.complex(torch.cos(self.emission_pattern), torch.sin(self.emission_pattern))
        init_input = self.complex_interpolate(emission_pattern, size=self.mask_size, mode='nearest')
        hidden = self.forward_propagate(init_input, scenes, self.encoder_mask, self.decoder_mask)

        # 浅脑+深脑
   

        # 浅脑

        # 浅脑+部分全连接
        output = torch.square(torch.abs(hidden))
        output = F.interpolate(output, size=(128, 128), mode='nearest')
        x = self.part_fclayers(output)
        return x
    

    
    def forward_propagate(self, input, objects, encoder_phase, decoder_phase):
        # emitter to obejct
        hidden = F.pad(input, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        hidden = self.propagate_encoder(hidden)
        hidden = hidden[:, :, self.pad:-self.pad, self.pad:-self.pad]
        encoder_mask = torch.complex(torch.cos(encoder_phase), torch.sin(encoder_phase))
        hidden = hidden * encoder_mask
        hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        hidden = self.propagate_fix(hidden)
        hidden = hidden[:, :, self.pad:-self.pad, self.pad:-self.pad]

        # object scatter & reflect
        hidden = hidden * objects
        hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        hidden = self.propagate_fix(hidden)
        hidden = hidden[:, :, self.pad:-self.pad, self.pad:-self.pad]

        # deocder receive
        decoder_mask = torch.complex(torch.cos(decoder_phase), torch.sin(decoder_phase))
        hidden = hidden * decoder_mask
        hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        hidden = self.propagate_decoder(hidden)
        hidden = hidden[:, :, self.pad:-self.pad, self.pad:-self.pad]
        
        return hidden

    def complex_interpolate(self, input, size, mode):
        real_part = torch.real(input)
        imag_part = torch.imag(input)
        real_interp = F.interpolate(real_part, size=size, mode=mode)
        imag_interp = F.interpolate(imag_part, size=size, mode=mode)
        output = torch.complex(real_interp, imag_interp)
        return output
    
if __name__ == '__main__':

    model = ONN_2class().to("cuda")
    input = torch.rand(1,1,512,512).to("cuda")
    output = model(input)
    print(output.shape)
    print("count:", count)
