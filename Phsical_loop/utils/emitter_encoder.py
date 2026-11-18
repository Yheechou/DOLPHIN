import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import matplotlib.pyplot as plt
import numpy as np
from utils.optics import ASM_propagate

class Encoder_ONN(nn.Module):
    def __init__(self,
            d_emitter2encoder, 
            wavelength,
            pixel_size,
            mask_row,
            mask_col,
            tile_row,
            tile_col,
            cell_row,  
            cell_col,
            prop_size,
            object_size,
            propgate
            ):
        super(Encoder_ONN, self).__init__()
        self.device = 'cuda'

        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.mask_row, self.mask_col = [mask_row, mask_col]
        self.tile_row, self.tile_col = [tile_row, tile_col]
        self.cell_row, self.cell_col = [cell_row, cell_col]
        self.prop_size = prop_size
        self.obeject_size = object_size
        self.propagate = propgate

        self.encode_matrix = torch.ones((1, self.tile_row, self.tile_col, 1), device=self.device)
        self.encode_co = torch.ones((1, self.tile_row, self.tile_col, 1), device=self.device)
        self.input_size = self.mask_row * self.cell_row
        self.prop_emitter2encoder = ASM_propagate(z=d_emitter2encoder, device=self.device, input_size=[self.prop_size, self.prop_size], lamda=self.wavelength, pixelsize=self.pixel_size)
    
    def forward(self, phase, slm_phase, dis_num):
        
        # emitter to encoder
        if not torch.is_complex(phase):
            mask_phase = torch.complex(torch.cos(phase), torch.sin(phase))
        else:
            mask_phase = phase
        mask_phase = torch.flip(mask_phase, dims=[1, 2])
        inputs = torch.ones(mask_phase.shape[0], self.mask_row, self.mask_col).to(self.device)
        inputs = inputs * mask_phase


        hidden = self.input_encoding(inputs)
        pad = (self.prop_size - hidden.shape[1]) // 2
        hidden = F.pad(hidden, (pad, pad, pad, pad), mode='constant', value=0)
        hidden = self.prop_emitter2encoder(hidden)

        # encoder to object
        pad = (self.prop_size - slm_phase.shape[1]) // 2
        slm_mask = F.pad(slm_phase.cuda(), (pad, pad, pad, pad), mode='constant', value=0)
        slm_mask = torch.complex(torch.cos(slm_mask), torch.sin(slm_mask))

        hidden = hidden * slm_mask
        hidden = self.propagate[dis_num](hidden)
        xx = (hidden.shape[1] - self.obeject_size[0]) // 2
        yy = (hidden.shape[1] - self.obeject_size[1]) // 2
        hidden = hidden[:, xx:-xx, yy:-yy]

        return hidden
    
    def input_encoding(self, input):
        input = torch.reshape(input, (-1, 1, 1, self.mask_row * self.mask_col))
        input_tile = input * self.encode_matrix
        input_tile = input_tile * self.encode_co

        input_paddings = F.pad(input_tile,
                               (0, 0,
                                int((self.cell_col - self.tile_col) / 2), int((self.cell_col - self.tile_col) / 2),
                                int((self.cell_row - self.tile_row) / 2), int((self.cell_row - self.tile_row) / 2),
                                0, 0),
                               mode='constant',
                               value=0.0)

        input = input_paddings.reshape(-1, self.cell_row, self.cell_col, self.mask_row, self.mask_col)
        input = torch.permute(input, (0, 3, 1, 4, 2))
        input = input.reshape((-1, self.mask_row * self.cell_row, self.mask_col * self.cell_col))
        return input
    

