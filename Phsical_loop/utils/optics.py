import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASM_propagate(nn.Module):
    def __init__(self, device, input_size=[400, 400], lamda=532, pixelsize=8, z=0.2, refidx=1, train=False): 
        super(ASM_propagate, self).__init__()
        self.input_size = input_size            # input_size * input_size neurons in one layer
        self.z = z                              # distance bewteen two layers
        self.pixelsize = pixelsize * 1e-6       # pixel size
        self.lamda = lamda * 1e-9               # wavelength

        NFv, NFh = self.input_size[0], self.input_size[1]
        Fs = 1 / self.pixelsize
        Fh = Fs / NFh * np.arange((-np.ceil((NFh - 1) / 2)), np.floor((NFh - 1) / 2) + 0.5)
        Fv = Fs / NFv * np.arange((-np.ceil((NFv - 1) / 2)), np.floor((NFv - 1) / 2) + 0.5)
        [Fhh, Fvv] = np.meshgrid(Fh, Fv)
        
        np_H = self.PropGeneral(Fhh, Fvv, self.lamda, refidx, self.z)
        np_freqmask = self.BandLimitTransferFunction(self.pixelsize, self.z, self.lamda, Fvv, Fhh)
        
        self.H = torch.tensor(np_H, dtype=torch.complex64).to(device)
        self.freqmask = torch.tensor(np_freqmask, dtype=torch.complex64).to(device)
    
    def PropGeneral(self, Fhh, Fvv, lamda, refidx, z):
        DiffLimMat = np.ones(Fhh.shape)
        lamdaeff = lamda / refidx
        DiffLimMat[(Fhh ** 2.0 + Fvv ** 2.0) >= (1.0 / lamdaeff ** 2.0)] = 0.0

        temp1 = 2.0 * math.pi * z / lamdaeff
        temp3 = (lamdaeff * Fvv) ** 2.0
        temp4 = (lamdaeff * Fhh) ** 2.0
        temp2 = np.complex128(1.0 - temp3 - temp4) ** 0.5
        H = np.exp(1j * temp1 * temp2)
        H[np.logical_not(DiffLimMat)] = 0
        return H

    
    def BandLimitTransferFunction(self, pixelsize, z, lamda, Fvv, Fhh):
        hSize, vSize = Fvv.shape
        dU = (hSize * pixelsize) ** -1.0
        dV = (vSize * pixelsize) ** -1.0
        Ulimit = ((2.0 * dU * z) ** 2.0 + 1.0) ** -0.5 / lamda
        Vlimit = ((2.0 * dV * z) ** 2.0 + 1.0) ** -0.5 / lamda
        freqmask = ((Fvv ** 2.0 / (Ulimit ** 2.0) + Fhh ** 2.0 * (lamda ** 2.0)) <= 1.0) & ((Fvv ** 2.0 * (lamda ** 2.0) + Fhh ** 2.0 / (Vlimit ** 2.0)) <= 1.0)
        return freqmask


    def forward(self, waves, use_freqmask=True):
        spectrum = torch.fft.fftshift(torch.fft.fft2(waves))
        spectrum_z = torch.mul(spectrum, self.H)
        if use_freqmask is True:
            spectrum_z = torch.mul(spectrum_z, self.freqmask)
        wave_z = torch.fft.ifft2(torch.fft.ifftshift(spectrum_z))
        return wave_z


#         # 频率轴（不shift）

#         # 传播传递函数（不shift）


#         # waves: (..., H, W) complex
#         # 频域平移相位（空域平移 dx,dy 像素）

