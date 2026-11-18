import gzip
import os
import struct

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.model_diff_forward import ONN_diff


class PCAMNISTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def read_idx_images(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist_raw(root):
    raw_dir = os.path.join(root, "MNIST", "raw")
    train_images = read_idx_images(os.path.join(raw_dir, "train-images-idx3-ubyte"))
    train_labels = read_idx_labels(os.path.join(raw_dir, "train-labels-idx1-ubyte"))
    test_images = read_idx_images(os.path.join(raw_dir, "t10k-images-idx3-ubyte"))
    test_labels = read_idx_labels(os.path.join(raw_dir, "t10k-labels-idx1-ubyte"))
    return train_images, train_labels, test_images, test_labels


def build_pca_datasets(data_root, n_components=64):
    train_images, train_labels, test_images, test_labels = load_mnist_raw(data_root)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=43)
    train_features = pca.fit_transform(train_images).astype(np.float32)
    test_features = pca.transform(test_images).astype(np.float32)

    scale = np.max(np.abs(train_features)) + 1e-8
    train_features = train_features / scale
    test_features = test_features / scale

    train_dataset = PCAMNISTDataset(train_features, train_labels)
    test_dataset = PCAMNISTDataset(test_features, test_labels)
    return train_dataset, test_dataset


def onn_outputs_to_logits(outputs, eps=1e-8):
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    outputs = outputs / (torch.amax(outputs, dim=1, keepdim=True) + eps)
    return F.interpolate(
        outputs.unsqueeze(1),
        size=10,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


asm_z = 32 * 1e-6 # 单位m
asm_pixelsize = 50 * 1e-3 # 单位um
asm_lamda = 1550
diffraction_size = 640
energy_decay_db = 0
layer_num = 10

root = os.path.dirname(os.path.abspath(__file__))
exp_name = "mnist_diffraction"
record_dir = os.path.join(root, "record", exp_name, f"layers_{layer_num}")
save_dir = os.path.join(record_dir, "ckpt")
record_path = os.path.join(record_dir, "record.txt")
data_root = "assets/attention/data"
lr = 1e-3
batch_size = 256
num_epochs = 100
load_ckpt = False
ckpt_name = "ckpt_best.pth"

os.makedirs(record_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

if not load_ckpt:
    with open(record_path, "w", encoding="utf-8") as file:
        file.write(f"exp_name: {exp_name}\n")
        file.write(f"data_root: {data_root}\n")
        file.write(f"asm_z: {asm_z}\n")
        file.write(f"asm_pixelsize: {asm_pixelsize}\n")
        file.write(f"asm_lamda: {asm_lamda}\n")
        file.write(f"diffraction_size: {diffraction_size}\n")
        file.write(f"propagation_width: {diffraction_size * 4}\n")
        file.write(f"energy_decay_db: {energy_decay_db}\n")
        file.write(f"layer_num: {layer_num}\n")
        file.write(f"lr: {lr}\n")
        file.write(f"batch_size: {batch_size}\n")
        file.write(f"num_epochs: {num_epochs}\n")
        file.write("pca_components: 64\n")
        file.write("\n")

model = ONN_diff(
    layer_num=layer_num,
    diffraction_size=diffraction_size,
    z=asm_z,
    pixelsize=asm_pixelsize,
    lamda=asm_lamda,
    energy_decay_db=energy_decay_db,
)
model = model.cuda()

for name, param in model.named_parameters():
    param.requires_grad = "phase_mask" in name

if load_ckpt:
    ckpt_path = os.path.join(save_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    with open(record_path, "a", encoding="utf-8") as file:
        file.write(f"\nLoad checkpoint: {ckpt_path}\n")

trainable_named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
trainable_params = [param for _, param in trainable_named_params]
optimizer = torch.optim.Adam(trainable_params, lr=lr)
criterion = nn.CrossEntropyLoss()

num_trainable_params = sum(param.numel() for param in trainable_params)
print(f"Trainable parameters: {num_trainable_params}")
with open(record_path, "a", encoding="utf-8") as file:
    file.write(f"trainable_parameters: {num_trainable_params}\n")

for name, param in trainable_named_params:
    print(f"{name:40s}  shape={tuple(param.shape)}")

torch.manual_seed(43)
np.random.seed(43)

train_dataset, test_dataset = build_pca_datasets(data_root, n_components=64)
trainloader = DataLoader(train_dataset, num_workers=1, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, num_workers=1, batch_size=batch_size, shuffle=False)

torch.autograd.set_detect_anomaly(True)
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
        for i, (inputs, labels) in pbar:
            optimizer.zero_grad()
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            logits = onn_outputs_to_logits(outputs)
            loss = criterion(logits, labels)
            if epoch == 0 and i == 0 and trainable_named_params:
                debug_name, debug_param = trainable_named_params[0]
                debug_param_before = debug_param.detach().clone()
            loss.backward()
            if epoch == 0 and i == 0:
                grad_norms = [
                    param.grad.detach().norm().item()
                    for _, param in trainable_named_params
                    if param.grad is not None
                ]
                max_grad_norm = max(grad_norms) if grad_norms else 0.0
                mean_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            optimizer.step()
            if epoch == 0 and i == 0 and trainable_named_params:
                param_delta = (debug_param.detach() - debug_param_before).abs().max().item()
                debug_msg = (
                    f"First batch debug: loss={loss.item():.6f}, "
                    f"logit_std={logits.detach().std().item():.6e}, "
                    f"mean_grad_norm={mean_grad_norm:.6e}, "
                    f"max_grad_norm={max_grad_norm:.6e}, "
                    f"{debug_name}_max_delta={param_delta:.6e}"
                )
                print(debug_msg)
                with open(record_path, "a", encoding="utf-8") as file:
                    file.write(debug_msg + "\n")

            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "Current Loss": f"{loss.item():.4f}",
                "Running Loss": f"{running_loss / (i + 1):.4f}",
                "Train Acc": f"{running_correct / running_total:.4f}",
            })

    train_loss = running_loss / len(trainloader)
    train_accuracy = running_correct / running_total

    model.eval()
    running_test_loss = 0.0
    running_test_correct = 0
    running_test_total = 0

    with torch.no_grad():
        with tqdm(enumerate(testloader, 0), total=len(testloader), desc="Evaluating", leave=False) as test_pbar:
            for i, (inputs, labels) in test_pbar:
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                logits = onn_outputs_to_logits(outputs)
                loss = criterion(logits, labels)

                probs = torch.softmax(logits, dim=1)
                predicted = torch.argmax(probs, dim=1)
                running_test_loss += loss.item()
                running_test_total += labels.size(0)
                running_test_correct += (predicted == labels).sum().item()

                test_pbar.set_postfix({
                    "Test Loss": f"{running_test_loss / (i + 1):.4f}",
                    "Test Acc": f"{running_test_correct / running_test_total:.4f}",
                })

    average_loss = running_test_loss / len(testloader)
    accuracy = running_test_correct / running_test_total
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {average_loss:.4f}, Test Acc: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "epoch": epoch + 1,
        }
        torch.save(state, os.path.join(save_dir, ckpt_name))
        with open(record_path, "a", encoding="utf-8") as file:
            file.write(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Test Loss: {average_loss:.6f}, "
                f"Test Acc: {accuracy:.4f}, "
                f"Best Acc: {best_accuracy:.4f}, "
                f"saved: {ckpt_name}\n"
            )
    else:
        with open(record_path, "a", encoding="utf-8") as file:
            file.write(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Test Loss: {average_loss:.6f}, "
                f"Test Acc: {accuracy:.4f}, "
                f"Best Acc: {best_accuracy:.4f}\n"
            )
