import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import matplotlib.pyplot as plt
import numpy as np

from utils.emitter_encoder import Encoder_ONN
from utils.model_multi_layer_forward import ONN_multi
from utils.decoder_receiver import Decoder_ONN

class Classification(nn.Module):
    def __init__(self, object_size, device='cuda', distance_scale=20, model_config=None):
        super(Classification, self).__init__()
        self.device = device
        
        # 参数
        self.d_emitter2encoder = 0.0940
        self.d_encoder2object = 0.30
        self.d_decoder2receiver = 0.0940
        self.d_object2decoder = 0.30
        self.wavelength = 1550
        self.pixel_size = 8
        self.mask_row, self.mask_col = [8, 8]
        self.tile_row, self.tile_col = [8, 8]
        self.cell_row, self.cell_col = [32, 32]
        self.prop_size = 1760 #prop传播时的长宽
        self.object_size = object_size
        self.receive_size = 760 #接收时的长宽
        self.num_classes = 3
        self.receiver_output_size = 8
        self.phase_scale = 1
        self.diff_calibration_root = "assets/diffraction_calibration"
        self.coder_size = self.mask_row * self.cell_row
        if model_config is not None:
            self.prop_size = model_config.get("prop_size", self.prop_size)
            self.receive_size = model_config.get("receive_size", self.receive_size)
            self.coder_size = model_config.get("coder_size", self.coder_size)
        self._validate_sizes()

        # emitter encoder model
        self.model_emitter_encoder = Encoder_ONN(
            d_emitter2encoder=self.d_emitter2encoder, 
            d_encoder2object=self.d_encoder2object,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            mask_row=self.mask_row,
            mask_col=self.mask_col,
            tile_row=self.tile_row,
            tile_col=self.tile_col,
            cell_row=self.cell_row,
            cell_col=self.cell_col,
            prop_size=self.prop_size,
            object_size=self.object_size
            )
        
        # decoder receiver model
        self.model_decoder_receiver = Decoder_ONN(
            d_decoder2receiver = self.d_decoder2receiver, 
            d_object2decoder = self.d_object2decoder,
            wavelength = self.wavelength,
            pixel_size = self.pixel_size,
            prop_size=self.prop_size,
            receive_size=self.receive_size
            )
        
        self.depth = Depth2obstacle(distance_scale=distance_scale)
        # parameter
        self.emitter_phase = nn.Parameter(torch.rand(1, self.mask_row, self.mask_col, device=self.device) * torch.pi * self.phase_scale)
        self.encoder_phase = nn.Parameter(torch.rand(self.coder_size, self.coder_size, device=self.device) * torch.pi * self.phase_scale)
        self.decoder_phase = nn.Parameter(torch.rand(self.coder_size, self.coder_size, device=self.device) * torch.pi * self.phase_scale)
        self.x = nn.Parameter(torch.rand(1, self.num_classes, device=self.device), requires_grad=True)

        # diffraction model
        trainable_para = ["diff_real", "diff_imag", "voltages"]# 
        self.model_diffraction = self.init_Diff_ONN(trainable_para)

    def _validate_sizes(self):
        limits = {
            "object_height": self.object_size[0],
            "object_width": self.object_size[1],
            "receive_size": self.receive_size,
            "coder_size": self.coder_size,
        }
        for name, size in limits.items():
            if self.prop_size < size:
                raise ValueError(
                    f"prop_size ({self.prop_size}) must be >= {name} ({size}). "
                    "Increase prop_size or reduce the corresponding mask/object/receiver size."
                )

    def init_Diff_ONN(self, trainable_para):
        # diffraction model
        model_diffraction = ONN_multi()
        model_diffraction = model_diffraction.to(self.device)
        ckpt_1_1 = torch.load(os.path.join(self.diff_calibration_root, 'ckpt_1_1.pth'), map_location=self.device)
        ckpt_1_2 = torch.load(os.path.join(self.diff_calibration_root, 'ckpt_1_2.pth'), map_location=self.device)
        ckpt_1_3 = torch.load(os.path.join(self.diff_calibration_root, 'ckpt_1_3.pth'), map_location=self.device)
        ckpt_1_4 = torch.load(os.path.join(self.diff_calibration_root, 'ckpt_1_4.pth'), map_location=self.device)
        ckpt_1_5 = torch.load(os.path.join(self.diff_calibration_root, 'ckpt_3unit_1_5.pth'), map_location=self.device)
        ckpt_1_6 = torch.load(os.path.join(self.diff_calibration_root, 'ckpt_3unit_1_6.pth'), map_location=self.device)
        ckpt_1_7 = torch.load(os.path.join(self.diff_calibration_root, 'ckpt_3unit_1_7.pth'), map_location=self.device)

        module_list = [ckpt_1_1, ckpt_1_2, ckpt_1_3, ckpt_1_4]
        module_list_3unit = [ckpt_1_5, ckpt_1_6, ckpt_1_7]

        new_dict = {}
        for layer_num in range(4):
            for i, module in enumerate(module_list):
                for k, v in module["model_state"].items():
                        new_k = "layers_4.{}.layers.{}.".format(layer_num, i) + k
                        new_dict[new_k] = v
            for i, module in enumerate(module_list_3unit):
                for k, v in module["model_state"].items():
                    if k.startswith("layers.2"):
                        new_k = k.replace("layers.2", "layers_4.{}.layers.{}".format(layer_num, i+4), 1)
                        new_dict[new_k] = v
        model_diffraction.load_state_dict(new_dict, strict=False)

        # 设置可训练参数
        for name, param in model_diffraction.named_parameters():
            for para_name in trainable_para:
                if para_name in name:
                    param.requires_grad = True
                    break
           
        return model_diffraction


    def forward(self, objects):
        # emitter and encoder
        objects, distance = self.depth(objects)
        if distance == -1:
            return self.x
        else:
            # emitter and encoder
            encoder_phase = self.encoder_phase
            emitter_pattern = self.model_emitter_encoder(self.emitter_phase, encoder_phase, distance)  

            # object reflection    
            reflection_pattern = objects * emitter_pattern

            # decoder and receiver
            decoder_phase = self.decoder_phase
            receiver_pattern = self.model_decoder_receiver(reflection_pattern, decoder_phase, distance)
            receiver_pattern = self.complex_interpolate(receiver_pattern.unsqueeze(1), size=self.receiver_output_size, mode='bilinear').squeeze(1)
            receiver_pattern = receiver_pattern / torch.amax(torch.abs(receiver_pattern), dim=(-2, -1), keepdim=True)

            # diffraction model
            diffractive_pattern, diff_hidden = self.model_diffraction(receiver_pattern.reshape(-1, self.receiver_output_size ** 2))
            output = F.interpolate(
                diffractive_pattern.unsqueeze(0).unsqueeze(0),  # [16,1,64]
                size=self.num_classes,
                mode='nearest'
            ).squeeze(1)
            return output
    
    def complex_interpolate(self, input, size, mode):
        real_part = torch.real(input)
        imag_part = torch.imag(input)
        real_interp = F.interpolate(real_part, size=size, mode=mode)
        imag_interp = F.interpolate(imag_part, size=size, mode=mode)
        output = torch.complex(real_interp, imag_interp)
        return output
    
class Depth2obstacle(object):
    def __init__(self, ref=0.8, distance_scale=20):
        self.ref = ref
        self.distance_scale = distance_scale

    def __call__(self, img_dep):
        depth = img_dep * 15
        distance = torch.mean(depth[depth != 15]) / self.distance_scale
        if torch.isnan(distance).any():
            distance = torch.tensor(-1)
        img_obs = 1 - img_dep
        img_obs[img_obs > 0] = self.ref
        img_obs = img_obs.squeeze(1)
        return img_obs, distance.item()
