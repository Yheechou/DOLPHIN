import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.onn_prop_partfc import ONN_2class

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

count = 0
# Network
class SimpleCNN(nn.Module):
    def __init__(self, patch_size=64):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute the final feature map size
        num_pooling_layers = 3  # 3 pooling layers halve the size each time
        final_size = patch_size // (2 ** num_pooling_layers)
        self.fc1 = nn.Linear(64 * final_size * final_size, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load numpy image
def load_image(image_path, depth_path):
    #Crop RGB img 
    image = Image.open(image_path).convert("RGB") 
    # 获取图片的宽度和高度
    width, height = image.size

    left = (width - 768) / 2
    top = (height - 768) / 2
    right = (width + 768) / 2
    bottom = (height + 768) / 2

    image = image.crop((left, top, right, bottom))
    img = FF.rgb_to_grayscale(image, 1)
    gray_image_np = np.array(img) / 255
    
    #Crop depth map
    disp = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    height, width = disp.shape[:2]
    startx = (width - 768) // 2
    starty = (height - 768) // 2
    disp = disp[starty:starty + 768, startx:startx + 768]
    return gray_image_np, disp, image

# Recursive segmentation function
def segment(image, models, thresholds, patch_size, x, y, mask):
    """
    image: 2048x1024 input image
    models: models for different patch sizes
    thresholds: confidence thresholds for each model
    patch_size: current block size
    x, y: top-left coordinates of the current block
    mask: 768x768 output mask
    """
    model = models[patch_size]
    patch = image[y:y+patch_size, x:x+patch_size]
    print(patch_size)
    patch_tensor = torch.tensor(patch, dtype=torch.float32).view(1, 1, patch_size, patch_size)
    patch_tensor = patch_tensor.to(next(model.parameters()).device)
    global count
    count += 1
    
    with torch.no_grad():
        output = model(patch_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()  # Get probability of class 1
    
    if prob > thresholds[patch_size]:
        if patch_size > 32:
            new_size = patch_size // 2
            for dx in [0, new_size]:
                for dy in [0, new_size]:
                    segment(image, models, thresholds, new_size, x + dx, y + dy, mask)
        else:
            mask[y:y+patch_size, x:x+patch_size] = 1  # Mark as vehicle region

def quantization_depth(depth, range, stride):
    disp =  torch.tensor(depth) / 256.0
    patch_size = 16
    H, W = disp.shape
    depth_patches = disp.view(H // patch_size, patch_size, W // patch_size, patch_size)
    patch_means = depth_patches.mean(dim=(1, 3))
    disp_tensor_smoothed = patch_means.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    disp = (2262.52 * 0.209) / (disp_tensor_smoothed + 1e-6)
    boundaries = torch.arange(1, range, stride)
    indices = torch.bucketize(disp, boundaries)
    depth_q = boundaries[indices - 1]
    depth_q = (depth_q - depth_q.min()) / (depth_q.max() - depth_q.min())
    depth_q = (depth_q * 255).byte()
    depth_q = depth_q.cpu().numpy()
    return depth_q

def get_main_area(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_label = 0
    max_area = 0

    for label in range(1, num_labels):
        # 获取当前连通区域的面积
        area = stats[label, cv2.CC_STAT_AREA]
        # 更新最大连通区域
        if area > max_area:
            max_area = area
            max_label = label

    # 创建一个掩膜，只包含最大连通区域
    max_component_mask = np.zeros_like(mask)
    max_component_mask[labels == max_label] = 1
    max_centroid = centroids[max_label]
    return max_component_mask, max_centroid

def get_car_center(depth_q, car_mask, num_bins, threshold):
    #按连通区分切分
    num_labels, labels = cv2.connectedComponents(car_mask, connectivity=8)
    single_car_mask_and_centers = []

    for k in range(1, num_labels):
        component_mask = np.zeros_like(car_mask)
        component_mask[labels == k] = 1
        # 计算直方图，将像素值分为num_bin个区间
        hist = cv2.calcHist([depth_q], [0], component_mask, [num_bins], [0, 256])
        
        # 绘制直方图
        bin_centers = np.linspace(0, 256, num_bins, endpoint=False) + 256 / (num_bins *2)
        plt.figure()
        plt.title("Grayscale Histogram with {} Bins".format(num_bins))
        plt.xlabel("Pixel Value Range")
        plt.ylabel("Pixel num")
        plt.bar(bin_centers, hist[:, 0], width=256/num_bins, align='center', color='blue')
        bin_edges = np.linspace(0, 256, num_bins, endpoint=False)
        tick_labels = ['' if i % 4 != 0 else f"{num:.1f}" for i, num in enumerate(bin_edges)]
        plt.xticks(bin_edges, tick_labels)
        plt.xlim([0, 256])
        plt.show()
        plt.savefig('result/hisk_{}.png'.format(k))  # 保存图像
        plt.figure()
        plt.imshow(component_mask, cmap='gray')
        plt.axis("off")
        plt.savefig('result/mask_{}.png'.format(k))

        # 查找连续3个或以上非零区间
        min_continuous_bins = 3
        start = -1
        pixel_num = 0
        continuous_ranges = []
        end = num_bins - 1#不考虑最后一个量化区间
        for i in range(end):
            if hist[i, 0] > 256:
                if start == -1:
                    start = i
                pixel_num += hist[i, 0]
            else:
                if start != -1 and i - start >= min_continuous_bins and pixel_num > threshold:
                    continuous_ranges.append((start, i - 1))
                start = -1
                pixel_num = 0
        if start != -1 and (end) - start >= min_continuous_bins and pixel_num > threshold:
            continuous_ranges.append((start, end-1))
        for (start, end) in continuous_ranges:
            # 将像素映射到区间
            min_val = bin_edges[start]
            if end == num_bins - 1:
                max_val = 256
            else:
                max_val = bin_edges[end+1]  
            print(min_val,max_val)  
            mask = cv2.inRange(depth_q * component_mask, min_val, max_val)
            mask, centroid = get_main_area(mask)
            single_car_mask_and_centers.append((mask, centroid))
    return single_car_mask_and_centers


# Evaluation function
def evaluate(image_path, depth_path, models, thresholds, model_class):
    image, depth, true_rgb = load_image(image_path, depth_path)  # Load 768x768 image
    colored_mask = np.ones((*image.shape, 3), dtype=np.uint8) * 70
    alpha_mask = np.zeros(image.shape[:2], dtype=np.float32)
    seg_masks = {}
    colors = {
        'car': (142, 0, 0), 'human': (60, 20, 220), 'traffic': (220, 220, 0), 'sky': (70, 130, 180),
        'road': (128, 64, 128), 'building': (70, 70, 70), 'vegetation': (107, 142, 35), 'sidewalk': (244, 35, 232)
    }
    
    for idx, cls in enumerate(model_class):
        mask = np.zeros_like(image, dtype=np.uint8)
        for y in range(0, 768, 128): #full 1024
            for x in range(0, 768, 128): #full 2048
                print(cls)
                segment(image, models[idx], thresholds, 128, x, y, mask)
        mask = smooth_mask(mask)
        seg_masks[cls] = mask 
        color = colors[cls]
        for c in range(3):
            colored_mask[:, :, c][mask == 1] = color[c]
        alpha_mask[mask == 1] = 1.0
    
    image_rgb = np.stack([image] * 3, axis=-1) if len(image.shape) == 2 else image
    blended_image = image_rgb.copy()
    
    for c in range(3):
        blended_image[:, :, c] = (1 - alpha_mask) * image_rgb[:, :, c] + colored_mask[:, :, c]
    
    blended_image = blended_image.astype(np.uint8)

    
    range_max = 51
    stride = 2
    threshold = 1000
    #Quantization depth map
    depth_q = quantization_depth(depth, range_max, stride // 2)
    #Found center
    mask_and_center = get_car_center(depth_q, seg_masks["car"], int((range_max - 1) // stride), threshold)

    # 显示对比图
    
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128)]
    
    #Plot point
    image_rgb = np.array(true_rgb).astype(np.uint8)
    image_rgb_c = image_rgb.copy()
    for idx, (mask, center) in enumerate(mask_and_center):
        color = color_list[idx % len(color_list)]
        cv2.circle(image_rgb, (int(center[0]), int(center[1])), 10, (255,0,0), -1)
        overlay = np.zeros_like(image_rgb, dtype=np.uint8)
        alpha = 0.6  # 透明度
        cv2.circle(overlay, (int(center[0]), int(center[1])), 80,(150,0,0) , -1)
        image_rgb = cv2.addWeighted(image_rgb, 1, overlay, alpha, 0)
        image_rgb_c[mask > 0] = color
    #Plot transparent mask
    for key, mask in seg_masks.items():
        if key != "car" and key != "human" and key != "building" and key != "sidewalk":
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            overlay = np.zeros_like(image_rgb, dtype=np.uint8)
            color = colors[key]
            alpha = 0.8  # 透明度
            overlay[mask == 1] = color
            image_rgb = cv2.addWeighted(image_rgb, 1, overlay, alpha, 0)
        elif key == "human" and mask.max() != 0:
            mask_human, center = get_main_area(mask)
            cv2.circle(image_rgb, (int(center[0]), int(center[1])), 10, (0,0,255), -1)
            overlay = np.zeros_like(image_rgb, dtype=np.uint8)
            alpha = 0.8  # 透明度
            cv2.circle(overlay, (int(center[0]), int(center[1])), 60,(0,0,159) , -1)
            image_rgb = cv2.addWeighted(image_rgb, 1, overlay, alpha, 0)
            image_rgb_c[mask_human > 0] = color



    plt.figure(dpi=400)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.savefig('result/result1.png', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.figure(dpi=400)
    plt.imshow(image_rgb_c)
    plt.axis("off")
    plt.savefig('result/result2.png', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.figure(dpi=400)
    plt.imshow(depth_q,cmap="gray")
    plt.axis("off")
    plt.savefig('result/result3.png', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.figure(dpi=400)
    plt.imshow(blended_image)
    plt.axis("off")
    plt.savefig('result/result4.png', dpi=400, bbox_inches='tight', pad_inches=0)


def smooth_mask(mask):
    mask = mask.astype(np.float32)
    kernel = np.ones((30, 30), np.uint8) # kernel size
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_mask = np.zeros_like(mask)
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(smoothed_mask, [smoothed_contour], -1, 1, thickness=cv2.FILLED)
    return smoothed_mask

# Load models
def load_models(model_class):
    patch_sizes = [128, 64, 32]
    models = {}
    for size in patch_sizes:
        model = ONN_2class()
        print(f'checkpoints_f/{model_class}/model_{size}.pth')
        model.load_state_dict(torch.load(f'checkpoints_f/{model_class}/model_{size}.pth'), strict=True)
        model.eval()
        model.to('cuda')
        models[size] = model
    return models

def draw_star(img, center, radius, color=(0, 255, 0), thickness=2):
    """
    在指定位置绘制一个星号。
    :param img: 目标图像
    :param center: 星号的中心点坐标 (x, y)
    :param radius: 星号的外半径
    :param color: 星号的颜色 (B, G, R)
    :param thickness: 线段的宽度
    """
    # 计算星号的五个顶点
    angles = np.deg2rad([0, 72, 144, 216, 288])
    points = np.array([
        (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
        for angle in angles
    ], dtype=np.int32)

    # 绘制星号的线段
    for i in range(5):
        start_point = points[i]
        end_point = points[(i + 2) % 5]
        cv2.line(img, tuple(start_point), tuple(end_point), color, thickness)


if __name__ == "__main__":
    model_class = ['sky', 'building', 'vegetation', 'road', 'car', 'human', 'traffic','sidewalk']
    models = []
    for class_ in model_class:
        models.append(load_models(class_))
    image_path = "data/cityscapes/leftImg8bit/train/aachen/aachen_000171_000019_leftImg8bit.png" 
    depth_path = "data/cityscapes/disparity/train/aachen/aachen_000171_000019_disparity.png"
    

    thresholds = {128: 0.5, 64: 0.5, 32: 0.5, 16: 0.5}  # Classification threshold for each model
    evaluate(image_path, depth_path, models, thresholds, model_class)
    print(count)