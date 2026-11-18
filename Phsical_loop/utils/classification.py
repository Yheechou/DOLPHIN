import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import matplotlib.pyplot as plt
import numpy as np
from utils.emitter_encoder import Encoder_ONN
from utils.model_multi_layer_forward import ONN_multi
from utils.decoder_receiver import Decoder_ONN
from utils.plot import plot_img
from utils.optics import ASM_propagate

class Classification(nn.Module):
    def __init__(self, object_size, num_classes=20):
        super(Classification, self).__init__()
        self.device = 'cuda'
        self.num_classes = num_classes

        # 标定参数
        self.d_emitter2encoder = 0.0940
        self.d_decoder2receiver = 0.0940
        self.wavelength = 1550
        self.pixel_size = 8
        self.mask_row, self.mask_col = [ 8,  8]
        self.tile_row, self.tile_col = [ 8,  8]
        self.cell_row, self.cell_col = [32, 32]
        self.prop_size = 1760 #prop传播时的长宽
        self.object_size = object_size
        self.receive_size = 760 #接收时的长宽
        self.input_size = self.mask_row * self.cell_row

        # distance参数
        self.n_distance = 51
        self.dis_step = 0.4
        self.propagate = {}
        for i in range(self.n_distance):
            self.propagate[i] = ASM_propagate(z=0.1+i*self.dis_step, device=self.device, input_size=[self.prop_size, self.prop_size], lamda=self.wavelength, pixelsize=self.pixel_size)
        self.loop_num = 5

        # emitter encoder model
        self.model_emitter_encoder = Encoder_ONN(
            d_emitter2encoder=self.d_emitter2encoder, 
            wavelength=self.wavelength,
            pixel_size=self.pixel_size,
            mask_row=self.mask_row,
            mask_col=self.mask_col,
            tile_row=self.tile_row,
            tile_col=self.tile_col,
            cell_row=self.cell_row,
            cell_col=self.cell_col,
            prop_size=self.prop_size,
            object_size=self.object_size,
            propgate=self.propagate 
            )
        self.model_emitter_encoder = self.model_emitter_encoder.to(self.device)

        # decoder receiver model
        self.model_decoder_receiver = Decoder_ONN(
            d_decoder2receiver = self.d_decoder2receiver, 
            wavelength = self.wavelength,
            pixel_size = self.pixel_size,
            receive_size = self.receive_size,
            prop_size =self.prop_size,
            propgate=self.propagate 
            )
        
        self.decoder_phase = torch.rand(256, 256, device=self.device) * torch.pi * 1
        self.decoder_phase = nn.Parameter(self.decoder_phase)
        self.encoder_phase = torch.rand(256, 256, device=self.device) * torch.pi * 1
        self.encoder_phase = nn.Parameter(self.encoder_phase)
        self.receiver_phase = nn.Parameter(torch.rand(1, 8, 8, device=self.device) * 1 * torch.pi)
        self.emmiter_phase = nn.Parameter(torch.rand(1, 8, 8, device=self.device) * 1 * torch.pi)

        # diffraction model
        trainable_para = ["diff_real", "diff_imag", "voltages"]
        self.diff_classifier = self.init_Diff_ONN(trainable_para)
        self.diff_loop_connect = nn.ModuleList([self.init_Diff_ONN(trainable_para) for _ in range(self.loop_num * 2 - 1)])

    def init_Diff_ONN(self, trainable_para):
        # diffraction model
        model_diffraction = ONN_multi()
        model_diffraction = model_diffraction.to(self.device)
        ckpt_1_1 = torch.load('assets/depth/ckpt/ckpt_1_1.pth')
        ckpt_1_2 = torch.load('assets/depth/ckpt/ckpt_1_2.pth')
        ckpt_1_3 = torch.load('assets/depth/ckpt/ckpt_1_3.pth')
        ckpt_1_4 = torch.load('assets/depth/ckpt/ckpt_1_4.pth')
        ckpt_1_5 = torch.load('assets/depth/ckpt/ckpt_3unit_1_5.pth')
        ckpt_1_6 = torch.load('assets/depth/ckpt/ckpt_3unit_1_6.pth')
        ckpt_1_7 = torch.load('assets/depth/ckpt/ckpt_3unit_1_7.pth')

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
        objects = objects.squeeze()
        dis_num = torch.randint(0, self.n_distance, (1,)).item()
        for i in range(self.loop_num):
            # emitter and encoder
            encoder_phase = F.interpolate(self.encoder_phase.unsqueeze(0).unsqueeze(1), size=1024, mode='nearest').squeeze()
            if i == 0:
                emmiter_phase = self.emmiter_phase
            emitter_pattern = self.model_emitter_encoder(emmiter_phase, encoder_phase, dis_num)  
                
            # object reflection  
            reflection_pattern = objects * emitter_pattern

            # decoder and receiver
            decoder_phase = F.interpolate(self.decoder_phase.unsqueeze(0).unsqueeze(1), size=1024, mode='nearest').squeeze()
            if i == 0:
                receiver_phase = self.receiver_phase
            receiver_pattern = self.model_decoder_receiver(reflection_pattern, decoder_phase, dis_num, receiver_phase)

            if i == self.loop_num - 1:
                break
            # modulate next loop
            emmiter_phase, _ = self.diff_loop_connect[2*i](receiver_pattern.reshape(-1, 64))
            emmiter_phase = emmiter_phase.reshape(-1, 8, 8) * torch.pi
            receiver_phase, _ = self.diff_loop_connect[2*i+1](receiver_pattern.reshape(-1, 64))
            receiver_phase = receiver_phase.reshape(-1, 8, 8) * torch.pi
        output, _ = self.diff_classifier(receiver_pattern.reshape(-1, 64))
        output = F.interpolate(
            output.unsqueeze(1),  # [16,1,64]
            size=self.num_classes,
            mode='nearest'
        ).squeeze(1)
        
        return output
    
    def random_partition(self, S, n):
        """ generate n numbers with summation S """
        assert S >= n, "S mast >= n"
        
        cuts = torch.randperm(S - 1)[:n-1] + 1
        cuts, _ = torch.sort(cuts)
        segments = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([S])]))
        return segments.tolist()

    def split(self, N, count):
        d = N / (count + 1)
        arr = np.arange(1, count + 1) * d
        return arr, d

    
