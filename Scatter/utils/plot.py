import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_img(path, img, clean=False):
    plt.figure()
    img = img.squeeze().cpu().detach().numpy()
    plt.imshow(img, cmap='gray')
    if clean:
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(path)

def plot_line(path, img):
    plt.figure()
    img = img.squeeze().cpu().detach().numpy()
    plt.plot(img)
    plt.savefig(path)

def plot_bar(path, img):
    plt.figure()
    img = img.squeeze().cpu().detach().numpy()
    plt.bar(np.arange(len(img)), img)
    plt.ylabel('phase (pi)')
    plt.savefig(path)