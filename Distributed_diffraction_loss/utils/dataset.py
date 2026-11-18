import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Distributed_Diff_Datasets(Dataset):
    """Custom PyTorch Dataset for loading data from .npy files"""

    def __init__(self, data_path, label_path):
        """
        Args:
            data_path (str): Path to the .npy file containing input data
            label_path (str): Path to the .npy file containing labels
        """
        # Load the numpy arrays
        self.data = np.load(data_path)
        self.labels = np.load(label_path)

        # Convert them to float32 tensors (common for neural networks)
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()

        # Check dimension consistency
        assert len(self.data) == len(self.labels), "Data and labels must have the same length"

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Return one sample and its corresponding label"""
        return self.data[idx], self.labels[idx]
