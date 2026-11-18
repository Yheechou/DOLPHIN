import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.onn_prop import ONN_2class
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

patch_size = 128
model_class = 'sidewalk'
model_type = 'prop_conv_128*128'
print(f"{model_class}_model_{patch_size}_{model_type}")

class CustomDataset(Dataset):
    def __init__(self, data_dir, balance=True):
        self.data = []
        self.labels = []

        class_0_dir = os.path.join(data_dir, '0')
        class_1_dir = os.path.join(data_dir, '1')
    
        class_0_files = sorted(os.listdir(class_0_dir))
        class_1_files = sorted(os.listdir(class_1_dir))

        num_class_0 = len(class_0_files)
        num_class_1 = len(class_1_files)

        for file in class_0_files:
            path = os.path.join(class_0_dir, file)
            self.data.append(path)
            self.labels.append(0)
        
        for file in class_1_files:
            for times in range(num_class_0 // num_class_1):
                path = os.path.join(class_1_dir, file)
                self.data.append(path)
                self.labels.append(1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(np.load(self.data[idx]), dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

class ContrastEnhancementTransform:
    """
    自定义对比度增强 Transform
    """
    def __init__(self, new_min=0, new_max=1):
        """
        初始化函数
        :param new_min: 新的最小值
        :param new_max: 新的最大值
        """
        super().__init__()
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, image_tensor):
        """
        前向传播函数
        :param image_tensor: 输入的图像张量，形状为 (C, H, W)
        :return: 对比度增强后的图像张量
        """
        # 计算当前图像的最小值和最大值
        min_pixel = image_tensor.min()
        max_pixel = image_tensor.max()
        
        # 线性变换
        enhanced_image = (image_tensor - min_pixel) / (max_pixel - min_pixel) * (self.new_max - self.new_min) + self.new_min
        return enhanced_image


# parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
epochs = 100
learning_rate = 0.0001

# dataloader
dataset = CustomDataset(data_dir="assets/cityscapes_seg/{}/{}".format(model_class, patch_size))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

# model
model = ONN_2class().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
file_path = os.path.join('data/tracking', "checkpoints", model_class,f"model_{patch_size}_{model_type}.txt")
os.makedirs(os.path.dirname(file_path), exist_ok=True)

load_ckpt = False
if load_ckpt:
    checkpoint = torch.load( f'checkpoints/{model_class}/model_{patch_size}_{model_type}.pth')
    model.load_state_dict(checkpoint)

with open(file_path, "w", encoding="utf-8") as file:
        file.write(f'lr: {learning_rate}')
# training
best_accuracy = 0.
os.makedirs(f'checkpoints/{model_class}',exist_ok=True)
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix(loss=total_loss / total, accuracy=correct / total)

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f'Train epoch [{epoch+1}/{epochs}], Accuracy: {correct / total:.4f}, Average Loss: {total_loss / total:.4f}\n')

    if correct / total > best_accuracy:
        best_accuracy = correct / total
        torch.save(model.state_dict(), f'checkpoints/{model_class}/model_{patch_size}_{model_type}.pth')