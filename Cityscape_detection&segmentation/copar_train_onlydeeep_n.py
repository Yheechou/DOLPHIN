########Final#######

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from utils.onn_partfc import Oixel_8by8

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#用于训练patch32，先推理出来前脑存着，单独训练深脑
patch_size = 32
model_class = 'car'
model_type = 'fixprop128_partfc'
print(f"{model_class}_model_{patch_size}_{model_type}")


class SavedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_0 = torch.load(os.path.join(data_dir, '0.pt'), weights_only=True)
        self.data_1 = torch.load(os.path.join(data_dir, '1.pt'), weights_only=True)
    
        self.l_0 = self.data_0.shape[0]
        self.l_1 = self.data_1.shape[0]


    def __len__(self):
        return self.l_0 + self.l_1
    
    def __getitem__(self, idx):
        if idx < self.l_0:
            return self.data_0[idx], 0
        else:
            return self.data_1[idx - self.l_0], 1


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
        x = x.unsqueeze(1)
        x = self.partfc_layers[0](x) 
        x = self.pool(F.relu(x)) #8*64*64dag
        x = self.partfc_layers[1](x) 
        x = self.pool(F.relu(x)) #16*32*32
        x = self.partfc_layers[2](x) 
        x = self.pool(F.relu(x)) #32*16*16
        x = self.conv1by1(x) #1*16*16
        x = self.output_layers(x.view(-1,16*16))
        return x


# parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 100
learning_rate = 0.0005
shallow_name = "model_32_prop_conv_128*128"

dataset = SavedDataset(data_dir="assets/cityscapes_shallowfix_pt/{}/{}".format(model_class, shallow_name))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

model = ONN_deep_fc().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
file_path = os.path.join('data/tracking', "checkpoints", model_class,f"model_{patch_size}_{model_type}.txt")
with open(file_path, "w", encoding="utf-8") as file:
        file.write(f'lr: {learning_rate}')

load_ckpt = False
if load_ckpt:
    checkpoint = torch.load( f'checkpoints/{model_class}/model_{patch_size}_{model_type}.pth')
    model.load_state_dict(checkpoint)

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


