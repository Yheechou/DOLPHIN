import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_bar(save_path, data1, data2):
    data1 = data1.cpu().detach().squeeze().numpy()
    data2 = data2.cpu().detach().squeeze().numpy()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(len(data1)), data1)
    plt.ylim([0, data2.max() + 0.2])
    plt.title('sim')
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(len(data2)), data2)
    plt.ylim([0, data2.max() + 0.2])
    plt.title('exp')
    plt.savefig(save_path, dpi=400)
    return